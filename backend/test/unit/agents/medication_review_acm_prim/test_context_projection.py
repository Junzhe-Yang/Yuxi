from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
)
from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    AcmReviewHarnessMiddleware,
    _build_selected_evidence_memory,
)


def _context(*, protocol: str = "adaptive_coverage") -> MedicationReviewAcmPrimContext:
    return MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        acm_protocol=protocol,
        v7_retrieval_depth=("shadow_top25" if protocol == "adaptive_coverage" else "top10"),
    )


def _tool_call(name: str, tool_call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )


def _search_message(tool_call_id: str, content: str) -> ToolMessage:
    return ToolMessage(
        id=f"message-{tool_call_id}",
        content=content,
        tool_call_id=tool_call_id,
        name="search_review_kb",
    )


def _query(tool_call_id: str, query_id: str, evidence_id: str) -> dict:
    return {
        "query_id": query_id,
        "tool_call_id": tool_call_id,
        "investigation_id": "INV-1",
        "query_text": f"查询 {query_id}",
        "reason": "定位证据",
        "started_at": "2026-08-31T00:00:00Z",
        "elapsed_ms": 1,
        "status": "success",
        "evidence_ids": [evidence_id],
        "new_evidence_ids": [evidence_id],
    }


def test_projection_cools_only_old_consumed_knowledge_results() -> None:
    old_content = "旧候选原文" * 1000
    overlap_content = "最近已读候选原文" * 1000
    unread_content = "尚未处理候选原文" * 1000
    messages = [
        HumanMessage(content="病例"),
        _tool_call("search_review_kb", "call-old"),
        _search_message("call-old", old_content),
        _tool_call("search_review_kb", "call-overlap"),
        _search_message("call-overlap", overlap_content),
        _tool_call("search_review_kb", "call-unread"),
        _search_message("call-unread", unread_content),
    ]
    state = {
        "query_records": [
            _query("call-old", "Q-OLD", "EV-OLD"),
            _query("call-overlap", "Q-OVERLAP", "EV-OVERLAP"),
            _query("call-unread", "Q-UNREAD", "EV-UNREAD"),
        ]
    }

    projected = AcmReviewHarnessMiddleware(model=object()).project_model_messages(
        messages=messages,
        state=state,
        context=_context(),
    )

    assert projected is not messages
    assert projected[2].content.startswith("[ACM 历史知识结果已冷却]")
    assert "query_id=Q-OLD" in projected[2].content
    assert "EV-OLD" in projected[2].content
    assert old_content not in projected[2].content
    assert projected[4].content == overlap_content
    assert projected[6].content == unread_content
    assert projected[2].id == messages[2].id
    assert projected[2].tool_call_id == messages[2].tool_call_id
    assert messages[2].content == old_content


def test_projection_bounds_long_candidate_history_to_two_full_results() -> None:
    messages = [HumanMessage(content="病例")]
    queries = []
    for index in range(10):
        tool_call_id = f"call-{index}"
        evidence_id = f"EV-{index}"
        messages.extend(
            [
                _tool_call("search_review_kb", tool_call_id),
                _search_message(tool_call_id, f"FULL-{index}-" + "证据" * 4000),
            ]
        )
        queries.append(_query(tool_call_id, f"Q-{index}", evidence_id))

    projected = AcmReviewHarnessMiddleware(model=object()).project_model_messages(
        messages=messages,
        state={"query_records": queries},
        context=_context(),
    )

    projected_tool_contents = [value.content for value in projected if isinstance(value, ToolMessage)]
    assert sum(value.startswith("FULL-") for value in projected_tool_contents) == 2
    assert sum(value.startswith("[ACM 历史知识结果已冷却]") for value in projected_tool_contents) == 8
    assert sum(len(value) for value in projected_tool_contents) < (
        sum(len(value.content) for value in messages if isinstance(value, ToolMessage)) * 0.3
    )


@pytest.mark.parametrize(
    ("tool_name", "state", "expected_fragments"),
    [
        (
            "open_review_evidence",
            {
                "open_records": [
                    {
                        "record_id": "OPEN-1",
                        "tool_call_id": "call-old",
                        "parent_evidence_id": "EV-PARENT",
                        "investigation_id": "INV-1",
                        "reason": "补充相邻原文",
                        "window_before": 1,
                        "window_after": 1,
                        "started_at": "2026-08-31T00:00:00Z",
                        "elapsed_ms": 1,
                        "status": "success",
                        "evidence_ids": ["EV-OPEN"],
                    }
                ]
            },
            ["record_id=OPEN-1", "parent_evidence_id=EV-PARENT", "EV-OPEN"],
        ),
        (
            "open_atlas_document",
            {
                "atlas_document_open_records": [
                    {
                        "record_id": "ATLAS-1",
                        "tool_call_id": "call-old",
                        "doc_id": "DOC-1",
                        "title": "指南文档",
                        "reason": "查看主题",
                        "topic_count": 2,
                        "cue_ids": ["CUE-1"],
                    }
                ]
            },
            ["record_id=ATLAS-1", "doc_id=DOC-1", "CUE-1"],
        ),
    ],
)
def test_projection_receipts_keep_open_result_provenance(
    tool_name: str,
    state: dict,
    expected_fragments: list[str],
) -> None:
    old_result = ToolMessage(
        content="待冷却的完整原文" * 1000,
        tool_call_id="call-old",
        name=tool_name,
    )
    overlap_result = _search_message("call-overlap", "保留的重叠结果")
    messages = [
        HumanMessage(content="病例"),
        _tool_call(tool_name, "call-old"),
        old_result,
        _tool_call("search_review_kb", "call-overlap"),
        overlap_result,
        AIMessage(content="继续调查"),
    ]

    projected = AcmReviewHarnessMiddleware(model=object()).project_model_messages(
        messages=messages,
        state=state,
        context=_context(),
    )

    assert projected[2].content.startswith("[ACM 历史知识结果已冷却]")
    assert all(value in projected[2].content for value in expected_fragments)
    assert projected[4].content == overlap_result.content
    assert old_result.content.startswith("待冷却的完整原文")


def test_selected_evidence_memory_keeps_exact_excerpt_without_mutating_store() -> None:
    exact_excerpt = "  原样保留的 Evidence 片段 <tag>\n第二行  "
    fallback_raw_text = "没有 occurrence 时保留的完整原文"
    state = {
        "investigations": [
            {
                "investigation_id": "INV-1",
                "question": "问题一",
                "status": "answered",
                "query_ids": ["Q-1"],
                "candidate_evidence_ids": ["EV-1", "EV-2"],
                "selected_evidence_ids": ["EV-1"],
                "created_at": "2026-08-31T00:00:00Z",
                "updated_at": "2026-08-31T00:00:01Z",
            }
        ],
        "adaptive_investigation_meta": [
            {
                "investigation_id": "INV-1",
                "obligation_supports": [
                    {
                        "obligation": "直接证据义务",
                        "evidence_ids": ["EV-2"],
                    }
                ],
                "updated_at": "2026-08-31T00:00:01Z",
            }
        ],
        "evidence_store": {
            "EV-1": {
                "evidence_id": "EV-1",
                "content_hash": "hash-1",
                "raw_text": "EV-1 的完整 chunk 原文",
                "source_document": "指南一.md",
                "file_id": "file-1",
                "chunk_id": "chunk-1",
                "chunk_index": 1,
                "occurrences": [
                    {
                        "record_id": "Q-1",
                        "tool_call_id": "call-1",
                        "source_method": "search",
                        "query_text": "查询一",
                        "reason": "定位证据",
                        "shown_excerpt": exact_excerpt,
                    }
                ],
            },
            "EV-2": {
                "evidence_id": "EV-2",
                "content_hash": "hash-2",
                "raw_text": fallback_raw_text,
                "source_document": "指南二.md",
                "file_id": "file-2",
                "chunk_id": "chunk-2",
                "chunk_index": 2,
            },
        },
    }

    memory = _build_selected_evidence_memory(state)

    assert exact_excerpt in memory
    assert fallback_raw_text in memory
    assert memory.count("[EV-1]") == 1
    assert memory.count("[EV-2]") == 1
    assert state["evidence_store"]["EV-1"]["raw_text"] == "EV-1 的完整 chunk 原文"
    assert state["evidence_store"]["EV-1"]["occurrences"][0]["shown_excerpt"] == exact_excerpt


def test_legacy_protocol_does_not_project_messages() -> None:
    messages = [
        HumanMessage(content="病例"),
        _tool_call("search_review_kb", "call-old"),
        _search_message("call-old", "旧实验原始结果"),
        AIMessage(content="继续"),
    ]

    projected = AcmReviewHarnessMiddleware(model=object()).project_model_messages(
        messages=messages,
        state={},
        context=_context(protocol="legacy_v7"),
    )

    assert projected is messages
