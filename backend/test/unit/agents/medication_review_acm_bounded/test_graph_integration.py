from __future__ import annotations

import hashlib
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_acm_bounded.context import (
    MedicationReviewAcmBoundedContext,
)
from yuxi.agents.buildin.medication_review_acm_bounded.context_view import (
    AcmBoundedModelViewMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_bounded.controller import (
    AcmBoundedControllerMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_bounded.generation import (
    AcmGenerationGuardMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_bounded.harness import (
    AcmBoundedHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_bounded.models import (
    MedicationReviewAcmBoundedState,
)
from yuxi.agents.buildin.medication_review_acm_bounded.tools import BOUNDED_TOOLS
from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas import (
    AtlasStore,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    CorpusAtlas,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    content_identity_hash,
    evidence_id_from_hash,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrieverSelection,
    ensure_runtime_resources,
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
        snapshot_hash="atlas-bounded",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-09-02T00:00:00Z",
        prompt_versions={"document_extract": "whole-document-v2"},
        prompt_hashes={"document_extract": "extract-hash"},
        document_cards=[],
    )


def _audit_entries() -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            "status": "covered" if index == 0 else "not_applicable",
            "investigation_aliases": ["I1"] if index == 0 else [],
            "rationale": "已覆盖" if index == 0 else "当前病例不适用",
        }
        for index, dimension in enumerate(ADAPTIVE_AUDIT_DIMENSIONS)
    ]


def _graph(model: ToolCallingFakeModel):
    return create_agent(
        model=model,
        tools=BOUNDED_TOOLS,
        middleware=[
            AcmBoundedHarnessMiddleware(model=model),
            AcmBoundedControllerMiddleware(),
            AcmBoundedModelViewMiddleware(),
            AcmGenerationGuardMiddleware(),
        ],
        state_schema=MedicationReviewAcmBoundedState,
        context_schema=MedicationReviewAcmBoundedContext,
    )


@pytest.mark.asyncio
async def test_bounded_graph_completes_and_emits_trace_13(monkeypatch) -> None:
    atlas = _atlas()
    monkeypatch.setattr(AtlasStore, "load_current", lambda _self, _db_id: atlas)

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
                        "name": "propose_initial_agenda",
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
                        "name": "search_active_obligation",
                        "args": {"query_text": "方案甲 适用条件"},
                        "id": "call-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record_active_obligation_support",
                        "args": {
                            "evidence_aliases": ["E1"],
                            "verdict": "SUPPORTED",
                            "rationale": "该证据直接说明方案甲的适用条件",
                        },
                        "id": "call-support",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "close_current_regimen_investigation",
                        "args": {
                            "status": "answered",
                            "conclusion": "在满足监测条件时方案甲可用",
                            "review_outcome": "appropriate",
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
                        "name": "audit_coverage",
                        "args": {
                            "entries": _audit_entries(),
                            "rationale": "当前病例的必要维度均已审计",
                        },
                        "id": "call-audit",
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
    context = MedicationReviewAcmBoundedContext(
        knowledges=["知识库"],
        max_search_calls=10,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
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
        config={"recursion_limit": 50},
    )

    terminal = [value for value in result["messages"] if isinstance(value, AIMessage) and not value.tool_calls]
    tool_names = [value.name for value in result["messages"] if isinstance(value, ToolMessage)]
    assert tool_names == [
        "propose_initial_agenda",
        "search_active_obligation",
        "record_active_obligation_support",
        "close_current_regimen_investigation",
        "audit_coverage",
    ]
    assert len(terminal) == 1
    trace = terminal[0].additional_kwargs["medication_review_trace"]
    assert trace["schema_version"] == "13.0"
    assert trace["method_family"] == "acm-prim-rag-v9"
    assert trace["adaptive_coverage_report"]["status"] == "completed"
    assert trace["model_context_window_tokens"] == 262_144
    assert len(trace["context_manifests"]) == 6
    assert trace["context_atoms"]
    assert all(
        value["projected_total_tokens"] <= value["configured_context_window"] for value in trace["context_manifests"]
    )
    assert len(trace["tool_outcomes"]) == 5
    assert all(value["semantic_outcome"] == "SUCCESS" for value in trace["tool_outcomes"])
    assert trace["generation_abort_records"] == []
    assert trace["obligation_judgments"][0]["evidence_ids"] == [evidence_id]
    assert trace["evidence_store"][0]["raw_text"] == raw_evidence
    assert trace["citation_verification"]["status"] == "ready"
    assert trace["citation_verification"]["evidence_snapshots"][0]["raw_text"] == raw_evidence
    assert trace["directives"][-1]["phase"] == "DRAFT_FINAL"
    assert trace["investigation_agenda"]["items"][0]["investigation_id"] == (investigation_id)


@pytest.mark.asyncio
async def test_bounded_graph_stops_after_repeated_semantic_failure(monkeypatch) -> None:
    atlas = _atlas()
    monkeypatch.setattr(AtlasStore, "load_current", lambda _self, _db_id: atlas)

    async def no_runtime_validation(_self, _atlas_value):
        return None

    monkeypatch.setattr(
        CorpusAtlasBuilder,
        "validate_runtime",
        no_runtime_validation,
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
                        "name": "propose_initial_agenda",
                        "args": {
                            "items": [
                                {
                                    "question": "方案甲是否适合当前患者？",
                                    "why_it_matters": "会改变对原方案的最终判断",
                                    "distinct_scope": "当前方案适用性",
                                    "decision_tags": ["current_regimen"],
                                    "investigation_kind": "current_regimen_review",
                                    "focus_plan_ids": ["PE001"],
                                    "evidence_obligations": ["方案甲的适用条件"],
                                }
                            ]
                        },
                        "id": "call-agenda-failure",
                        "type": "tool_call",
                    }
                ],
            ),
            *[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_active_obligation",
                            "args": {"query_text": "方案甲 剂量 监测"},
                            "id": f"call-invalid-{index}",
                            "type": "tool_call",
                        }
                    ],
                )
                for index in (1, 2)
            ],
            AIMessage(content="当前检索动作的语义修复已耗尽，请查看 Trace 后重试。"),
        ]
    )
    context = MedicationReviewAcmBoundedContext(
        knowledges=["知识库"],
        max_search_calls=10,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
    )
    ensure_runtime_resources(context)

    async def retriever_must_not_run(_query: str, **_kwargs: Any):
        raise AssertionError("invalid query must be rejected before retrieval")

    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever_must_not_run,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1"}

    result = await _graph(model).ainvoke(
        {"messages": [HumanMessage(content="患者正在使用方案甲，请审查其适用性。")]},
        context=context,
        config={"recursion_limit": 30},
    )

    terminal = [value for value in result["messages"] if isinstance(value, AIMessage) and not value.tool_calls]
    trace = terminal[0].additional_kwargs["medication_review_trace"]
    search_outcomes = [value for value in trace["tool_outcomes"] if value["tool_name"] == "search_active_obligation"]
    assert [value["reason_code"] for value in search_outcomes] == [
        "QUERY_SHAPE_INVALID",
        "REPEATED_REJECTED_ACTION",
    ]
    assert trace["run_status"] == "partial"
    assert trace["completion_reason"] == "semantic_repair_exhausted"
    assert trace["directives"][-1]["phase"] == "FAIL_EXPLICIT"
