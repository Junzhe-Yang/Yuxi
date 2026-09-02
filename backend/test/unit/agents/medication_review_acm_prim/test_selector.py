from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    AtlasDocumentCard,
    AtlasTopicCue,
    CorpusAtlas,
)
from yuxi.agents.buildin.medication_review_acm_prim.selector import (
    select_companion_cues,
    selector_prompt_hash,
)
from yuxi.agents.buildin.medication_review_lite.models import PlanAnchor
from yuxi.agents.buildin.medication_review_prim.models import (
    InvestigationItem,
    QueryRecord,
)


class FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.responses.pop(0))


def _atlas() -> CorpusAtlas:
    cue = AtlasTopicCue(
        cue_id="AT-CUE",
        cue_text="联用与低血压",
        source_chunk_ids=["chunk-1"],
    )
    return CorpusAtlas(
        builder_version="acm-atlas-v1",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-01-01T00:00:00Z",
        document_cards=[
            AtlasDocumentCard(
                doc_id="file-2",
                file_name="共识.md",
                title="共识",
                scope_summary="涵盖联用风险。",
                topic_cues=[cue],
            )
        ],
    )


def _state() -> dict:
    return {
        "raw_case_text": "患者使用方案甲，已有头晕。",
        "plan_anchors": [
            PlanAnchor(
                element_id="PE001",
                source_span="方案甲",
                source_start=4,
                source_end=7,
                label="方案甲",
                kind="explicit_regimen_or_other",
            )
        ],
        "patient_modifiers": [],
        "query_records": [
            QueryRecord(
                query_id="Q-1",
                tool_call_id="call-1",
                query_text="方案甲适应证",
                reason="调查适应证",
                started_at="2026-01-01T00:00:00Z",
                elapsed_ms=1,
                status="success",
                evidence_ids=["EV-1"],
            )
        ],
        "investigations": [],
        "evidence_store": {
            "EV-1": {
                "evidence_id": "EV-1",
                "content_hash": "hash",
                "raw_text": "内容",
                "file_id": "file-1",
                "occurrences": [
                    {
                        "record_id": "Q-1",
                        "tool_call_id": "call-1",
                        "source_method": "search",
                        "query_text": "方案甲适应证",
                        "reason": "调查适应证",
                        "shown_excerpt": "内容",
                    }
                ],
            }
        },
    }


def test_selector_hash_covers_input_contract(monkeypatch) -> None:
    from yuxi.agents.buildin.medication_review_acm_prim import selector

    original = selector_prompt_hash()
    monkeypatch.setattr(
        selector,
        "SELECTOR_INPUT_CONTRACT_VERSION",
        "changed-input-contract",
    )

    assert selector_prompt_hash() != original


async def test_selector_derives_documents_and_removes_unknown_case_ids() -> None:
    response = json.dumps(
        {
            "cues": [
                {
                    "question_hint": "方案甲是否涉及联用低血压风险？",
                    "linked_plan_ids": ["PE001", "PE999"],
                    "linked_modifier_ids": [],
                    "atlas_cue_ids": ["AT-CUE"],
                    "novelty_explanation": "现有查询只调查了适应证。",
                }
            ]
        },
        ensure_ascii=False,
    )

    selection = await select_companion_cues(
        model=FakeModel([response]),
        state=_state(),
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.selector_audit.status == "success"
    assert selection.selector_audit.retrieved_document_ids == ["file-1"]
    cue = selection.companion_cues[0]
    assert cue.companion_id == "AC01"
    assert cue.linked_plan_ids == ["PE001"]
    assert cue.suggested_doc_ids == ["file-2"]
    assert selection.selector_audit.dropped_items[0]["reason"] == (
        "unknown_case_ids_removed"
    )


async def test_selector_input_excludes_later_query_and_investigation_state() -> None:
    state = _state()
    state["query_records"].append(
        QueryRecord(
            query_id="Q-LATER",
            tool_call_id="call-later",
            query_text="后续查询",
            reason="不应进入首次选择器",
            started_at="2026-01-01T00:00:02Z",
            elapsed_ms=1,
            status="success",
            evidence_ids=["EV-LATER"],
        )
    )
    state["investigations"] = [
        InvestigationItem(
            investigation_id="INV-MERGED",
            question="首轮与后续查询共同调查的问题",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:02Z",
            query_ids=["Q-1", "Q-LATER"],
            candidate_evidence_ids=["EV-1", "EV-LATER"],
        )
    ]
    state["evidence_store"]["EV-LATER"] = {
        "evidence_id": "EV-LATER",
        "content_hash": "later",
        "raw_text": "后续证据",
        "file_id": "file-later",
        "occurrences": [
            {
                "record_id": "Q-LATER",
                "tool_call_id": "call-later",
                "source_method": "search",
                "query_text": "后续查询",
                "reason": "不应进入首次选择器",
                "shown_excerpt": "后续证据",
            }
        ],
    }
    model = FakeModel(['{"cues":[]}'])

    selection = await select_companion_cues(
        model=model,
        state=state,
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    selector_payload = json.loads(
        model.calls[0][-1].content.split("选择器输入：\n", 1)[1].split(
            "\n\n只返回一个符合",
            1,
        )[0]
    )
    assert selection.observed_query_ids == ["Q-1"]
    assert [
        value["query_id"] for value in selector_payload["query_records"]
    ] == ["Q-1"]
    assert selector_payload["retrieved_document_ids"] == ["file-1"]
    assert selector_payload["investigations"][0]["query_ids"] == [
        "Q-1"
    ]
    assert selector_payload["investigations"][0][
        "candidate_evidence_ids"
    ] == ["EV-1"]
    assert "EV-LATER" not in json.dumps(selector_payload, ensure_ascii=False)


async def test_selector_input_includes_failed_peer_from_first_parallel_batch() -> None:
    state = _state()
    state["query_records"].insert(
        0,
        QueryRecord(
            query_id="Q-FAILED",
            tool_call_id="call-failed",
            query_text="首轮并行但技术失败的查询",
            reason="同批已尝试方向",
            started_at="2025-12-31T23:59:59Z",
            elapsed_ms=1,
            status="technical_failed",
        ),
    )
    state["messages"] = [
        {
            "type": "ai",
            "tool_calls": [
                {"id": "call-failed"},
                {"id": "call-1"},
            ],
        }
    ]
    model = FakeModel(['{"cues":[]}'])

    selection = await select_companion_cues(
        model=model,
        state=state,
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.observed_query_ids == ["Q-FAILED", "Q-1"]
    payload = json.loads(
        model.calls[0][-1].content.split("选择器输入：\n", 1)[1].split(
            "\n\n只返回一个符合",
            1,
        )[0]
    )
    assert [value["query_id"] for value in payload["query_records"]] == [
        "Q-FAILED",
        "Q-1",
    ]
    assert payload["retrieved_document_ids"] == ["file-1"]


async def test_selector_accepts_empty_list_without_repair() -> None:
    model = FakeModel(['{"cues":[]}'])

    selection = await select_companion_cues(
        model=model,
        state=_state(),
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.selector_audit.status == "empty"
    assert selection.companion_cues == []
    assert len(model.calls) == 1


async def test_selector_repairs_unknown_atlas_id_once() -> None:
    invalid = json.dumps(
        {
            "cues": [
                {
                    "question_hint": "未知",
                    "linked_plan_ids": [],
                    "linked_modifier_ids": [],
                    "atlas_cue_ids": ["AT-UNKNOWN"],
                    "novelty_explanation": "未知",
                }
            ]
        },
        ensure_ascii=False,
    )
    valid = json.dumps(
        {
            "cues": [
                {
                    "question_hint": "联用风险？",
                    "linked_plan_ids": ["PE001"],
                    "linked_modifier_ids": [],
                    "atlas_cue_ids": ["AT-CUE"],
                    "novelty_explanation": "尚未调查联用风险。",
                }
            ]
        },
        ensure_ascii=False,
    )
    model = FakeModel([invalid, valid])

    selection = await select_companion_cues(
        model=model,
        state=_state(),
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.selector_audit.status == "repaired"
    assert len(model.calls) == 2


async def test_selector_records_repair_even_when_repaired_result_is_empty() -> None:
    model = FakeModel(["not-json", '{"cues":[]}'])

    selection = await select_companion_cues(
        model=model,
        state=_state(),
        atlas=_atlas(),
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.selector_audit.status == "repaired"
    assert selection.companion_cues == []
    assert len(model.calls) == 2


async def test_selector_keeps_six_cues_above_soft_budget_without_repair() -> None:
    atlas = _atlas()
    source_cue = atlas.document_cards[0].topic_cues[0]
    atlas_cues = [
        source_cue.model_copy(update={"cue_id": f"AT-CUE-{index}"})
        for index in range(7)
    ]
    atlas = atlas.model_copy(
        update={
            "document_cards": [
                atlas.document_cards[0].model_copy(
                    update={
                        "topic_cues": atlas_cues,
                    }
                )
            ]
        }
    )
    too_many = json.dumps(
        {
            "cues": [
                {
                    "question_hint": f"补充调查 {index}？",
                    "linked_plan_ids": ["PE001"],
                    "linked_modifier_ids": [],
                    "atlas_cue_ids": [f"AT-CUE-{index}"],
                    "novelty_explanation": "现有调查未涉及。",
                }
                for index in range(7)
            ]
        },
        ensure_ascii=False,
    )
    model = FakeModel([too_many])

    selection = await select_companion_cues(
        model=model,
        state=_state(),
        atlas=atlas,
        trigger_type="after_query",
        created_after_query_id="Q-1",
        technical_retry_limit=0,
    )

    assert selection.selector_audit.status == "success"
    assert len(selection.companion_cues) == 6
    assert len(model.calls) == 1
    assert selection.selector_audit.dropped_items[-1]["reason"] == (
        "soft_budget_exceeded"
    )
