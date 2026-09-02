from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
)
from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas import (
    AtlasStore,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    CorpusAtlas,
)
from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    AcmReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_prim.models import (
    MedicationReviewAcmPrimState,
)
from yuxi.agents.buildin.medication_review_acm_prim.tools import (
    adaptive_coverage_checkpoint,
    extend_investigation_agenda,
    open_atlas_document,
    search_review_kb_acm_dispatch,
    set_investigation_agenda,
    submit_coverage_gap_assessment,
    update_acm_investigation,
    v7_effort_checkpoint,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    content_identity_hash,
    evidence_id_from_hash,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrieverSelection,
    coverage_reflection,
    ensure_runtime_resources,
    open_review_evidence,
)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ToolCallingFakeModel:
        del tools, tool_choice, kwargs
        return self


def _atlas() -> CorpusAtlas:
    return CorpusAtlas(
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash="atlas-adaptive",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-08-25T00:00:00Z",
        prompt_versions={"document_extract": "whole-document-v2"},
        prompt_hashes={"document_extract": "extract-hash"},
        document_cards=[],
    )


def _coverage_audit(investigation_id: str) -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            "status": "covered" if index == 0 else "not_applicable",
            "investigation_ids": [investigation_id] if index == 0 else [],
            "rationale": "已覆盖" if index == 0 else "当前病例不适用",
        }
        for index, dimension in enumerate(ADAPTIVE_AUDIT_DIMENSIONS)
    ]


def _graph(model: ToolCallingFakeModel):
    return create_agent(
        model=model,
        tools=[
            search_review_kb_acm_dispatch,
            set_investigation_agenda,
            extend_investigation_agenda,
            submit_coverage_gap_assessment,
            open_atlas_document,
            open_review_evidence,
            update_acm_investigation,
            coverage_reflection,
            v7_effort_checkpoint,
            adaptive_coverage_checkpoint,
        ],
        middleware=[AcmReviewHarnessMiddleware(model=model)],
        state_schema=MedicationReviewAcmPrimState,
        context_schema=MedicationReviewAcmPrimContext,
    )


@pytest.mark.asyncio
async def test_adaptive_graph_completes_dynamic_contract_and_trace(
    monkeypatch,
) -> None:
    atlas = _atlas()
    monkeypatch.setattr(
        AtlasStore,
        "load_current",
        lambda _self, _db_id: atlas,
    )

    async def no_runtime_validation(_self, _atlas_value):
        return None

    monkeypatch.setattr(
        CorpusAtlasBuilder,
        "validate_runtime",
        no_runtime_validation,
    )

    raw_case = "患者正在使用方案甲，请审查其适用性。"
    question = "方案甲是否适合当前患者？"
    investigation_id = "INV-ACM-" + hashlib.sha256(f"call-agenda\0{1}\0{question}".encode()).hexdigest()[:12].upper()
    raw_evidence = "现有证据支持在满足监测条件时使用方案甲。"
    evidence_id = evidence_id_from_hash(
        content_identity_hash(
            raw_text=raw_evidence,
            db_id="db-1",
            file_id="file-1",
            chunk_id="chunk-1",
            chunk_index=1,
        )
    )
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content=('{"anchors":[{"source_span":"方案甲","label":"方案甲","kind":"explicit_regimen_or_other"}]}')
            ),
            AIMessage(content='{"modifiers":[]}'),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "set_investigation_agenda",
                        "args": {
                            "items": [
                                {
                                    "question": question,
                                    "why_it_matters": "会改变对原方案的最终判断",
                                    "distinct_scope": "当前方案适用性",
                                    "decision_tags": ["current_regimen"],
                                    "investigation_kind": "current_regimen_review",
                                    "focus_plan_ids": ["PE001"],
                                    "evidence_obligations": ["方案甲的适用条件"],
                                }
                            ]
                        },
                        "id": "call-agenda",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_review_kb",
                        "args": {
                            "query_text": "方案甲 当前患者 适用性",
                            "reason": "发现回答当前方案适用性的来源",
                            "investigation_id": investigation_id,
                            "uncovered_aspect": "方案甲的适用条件",
                            "retrieval_intent": "source_discovery",
                        },
                        "id": "call-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_investigation",
                        "args": {
                            "investigation_id": investigation_id,
                            "status": "answered",
                            "selected_evidence_ids": [evidence_id],
                            "working_note": "证据回答了方案甲的主要适用条件",
                            "obligation_supports": [
                                {
                                    "obligation": "方案甲的适用条件",
                                    "evidence_ids": [evidence_id],
                                }
                            ],
                            "review_outcome": "appropriate",
                            "resolved_aspects": ["方案甲的适用条件"],
                            "remaining_aspects": [],
                        },
                        "id": "call-close",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_coverage_gap_assessment",
                        "args": {
                            "material_gap_found": False,
                            "rationale": "当前病例只有方案甲，适用性调查已闭合",
                            "coverage_audit": _coverage_audit(investigation_id),
                        },
                        "id": "call-gap",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "②【逐项判断】\n"
                    f"■ 【PE001】方案甲\n判断：在监测条件下可用。[{evidence_id}]\n\n"
                    "③【正面判断汇总】\n方案甲有证据支持。\n\n"
                    "④【负面判断汇总】\n无。\n\n"
                    "⑤【综合建议】\n结合患者情况持续监测。"
                )
            ),
        ]
    )
    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        acm_protocol="adaptive_coverage",
        v7_retrieval_depth="shadow_top25",
        max_search_calls=10,
    )
    ensure_runtime_resources(context)

    async def retriever(_query: str, **_kwargs: Any):
        return [
            {
                "content": raw_evidence,
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                },
                "score": 0.9,
            }
        ]

    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1"}

    result = await _graph(model).ainvoke(
        {"messages": [HumanMessage(content=raw_case)]},
        context=context,
    )

    terminal = [value for value in result["messages"] if isinstance(value, AIMessage) and not value.tool_calls]
    tool_names = [value.name for value in result["messages"] if isinstance(value, ToolMessage)]
    assert tool_names == [
        "set_investigation_agenda",
        "search_review_kb",
        "update_investigation",
        "submit_coverage_gap_assessment",
    ]
    assert len(terminal) == 1
    trace = terminal[0].additional_kwargs["medication_review_trace"]
    assert trace["schema_version"] == "12.0"
    assert trace["method_family"] == "acm-prim-rag-v8"
    assert trace["method_version"] == ("acm-prim-rag-v8-adaptive-two-track-evidence-v5-query-focus-shadow_top25-vector")
    assert trace["adaptive_coverage_report"]["status"] == "completed"
    assert trace["adaptive_coverage_report"]["actual_investigation_count"] == 1
    assert trace["adaptive_coverage_report"]["executed_search_calls"] == 1
    assert trace["investigation_agenda"]["revision"] == 1
    assert trace["probe_records"][0]["retrieval_intent"] == "source_discovery"
    assert trace["gap_assessments"][0]["material_gap_found"] is False
    assert trace["retrieval_records"][0]["fetch_k"] == 25
    assert trace["retrieval_records"][0]["visible_k"] == 10
