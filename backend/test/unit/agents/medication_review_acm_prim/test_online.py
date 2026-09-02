from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    CorpusAtlas,
)
from yuxi.agents.buildin.medication_review_acm_prim.graph import (
    MedicationReviewAcmPrimAgent,
)
from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    AcmReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_prim.memory import (
    ATLAS_NAVIGATION_PROMPT_HASH,
)
from yuxi.agents.buildin.medication_review_acm_prim.tools import (
    open_atlas_document,
)
from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    disabled_anchor_audit,
)
from yuxi.agents.buildin.medication_review_prim.extraction import (
    disabled_modifier_audit,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimTrace,
    PrimCoverageReport,
    ReflectionReport,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    search_review_kb_b1,
    search_review_kb_relation,
)


def _atlas() -> CorpusAtlas:
    return CorpusAtlas(
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash="atlas",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-01-01T00:00:00Z",
        prompt_versions={"document_extract": "whole-document-v2"},
        prompt_hashes={"document_extract": "extract-hash"},
        document_cards=[
            {
                "doc_id": "file-atlas",
                "file_name": "共识.md",
                "title": "共识",
                "scope_summary": "覆盖方案选择。",
                "topic_cues": [
                    {
                        "cue_id": "AT-ONE",
                        "cue_text": "某方案需要监测。",
                        "source_chunk_ids": ["chunk-secret"],
                    }
                ],
            }
        ],
    )


def _context_with_atlas() -> MedicationReviewAcmPrimContext:
    context = MedicationReviewAcmPrimContext(knowledges=["知识库"])
    context._acm_prim_atlas = _atlas()
    return context


@pytest.mark.asyncio
async def test_open_atlas_document_returns_topics_without_source_chunks() -> None:
    result = await open_atlas_document.coroutine(
        doc_id="file-atlas",
        reason="查看治疗主题",
        runtime=SimpleNamespace(
            context=_context_with_atlas(),
            state={},
            tool_call_id="atlas-open-1",
        ),
    )

    content = result.update["messages"][0].content
    assert "某方案需要监测" in content
    assert "chunk-secret" not in content
    assert "不是 Evidence" in content
    assert "file_id='file-atlas'" in content
    record = result.update["atlas_document_open_records"][0]
    assert record.doc_id == "file-atlas"
    assert record.topic_count == 1
    assert record.cue_ids == ["AT-ONE"]


@pytest.mark.asyncio
async def test_adaptive_open_atlas_document_keeps_global_discovery_available() -> None:
    context = _context_with_atlas()
    context.acm_protocol = "adaptive_coverage"
    context.v7_retrieval_depth = "shadow_top25"

    result = await open_atlas_document.coroutine(
        doc_id="file-atlas",
        reason="判断是否需要在已知来源内定位",
        runtime=SimpleNamespace(
            context=context,
            state={},
            tool_call_id="atlas-open-adaptive",
        ),
    )

    content = result.update["messages"][0].content
    assert "within_document_localization" in content
    assert "source_discovery" in content
    assert "retrieval_scope='global'" in content


@pytest.mark.asyncio
async def test_open_unknown_atlas_document_returns_available_ids() -> None:
    result = await open_atlas_document.coroutine(
        doc_id="missing",
        reason="查看主题",
        runtime=SimpleNamespace(
            context=_context_with_atlas(),
            state={},
            tool_call_id="atlas-open-2",
        ),
    )

    content = result.update["messages"][0].content
    assert "missing" in content
    assert "file-atlas" in content
    assert "atlas_document_open_records" not in result.update


def test_memory_contains_document_overview_not_all_topic_cues() -> None:
    middleware = AcmReviewHarnessMiddleware(model=object())
    memory = middleware.augment_investigation_memory(
        state={},
        context=_context_with_atlas(),
        memory_text="PRIM 调查记忆",
    )

    assert "PRIM 调查记忆" in memory
    assert "覆盖方案选择" in memory
    assert "open_atlas_document" in memory
    assert "某方案需要监测" not in memory
    assert "先调用" in memory


def test_online_path_keeps_prim_search_tools_without_hidden_selector() -> None:
    middleware = AcmReviewHarnessMiddleware(model=object())

    assert middleware.project_search_tool("full", state={}) is search_review_kb_relation
    assert middleware.project_search_tool("b1", state={}) is search_review_kb_b1


@pytest.mark.asyncio
async def test_online_model_hook_does_not_run_hidden_selector() -> None:
    update = await AcmReviewHarnessMiddleware(model=object()).abefore_model(
        {"query_records": [{"status": "success"}]},
        SimpleNamespace(context=_context_with_atlas()),
    )

    assert update is None


@pytest.mark.asyncio
async def test_candidate_hook_does_not_run_selector() -> None:
    update = await AcmReviewHarnessMiddleware(
        model=object()
    ).prepare_candidate_state(
        state={},
        context=_context_with_atlas(),
        candidate_body="第一版答案",
    )

    assert update == {}


@pytest.mark.asyncio
async def test_graph_exposes_atlas_navigation_and_prim_tools(monkeypatch) -> None:
    captured: dict = {}
    built_graph = object()
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_acm_prim.graph.load_chat_model",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_acm_prim.graph.create_agent",
        lambda **kwargs: captured.update(kwargs) or built_graph,
    )
    agent = MedicationReviewAcmPrimAgent()

    async def no_checkpointer():
        return None

    monkeypatch.setattr(agent, "_get_checkpointer", no_checkpointer)
    result = await agent.get_graph(
        context=MedicationReviewAcmPrimContext(knowledges=[]),
    )

    assert result is built_graph
    tool_names = {value.name for value in captured["tools"]}
    assert tool_names == {
        "search_review_kb",
        "set_investigation_agenda",
        "extend_investigation_agenda",
        "submit_coverage_gap_assessment",
        "open_atlas_document",
        "open_review_evidence",
        "update_investigation",
        "coverage_reflection",
        "v7_effort_checkpoint",
        "adaptive_coverage_checkpoint",
    }


def test_finalize_trace_records_navigation_without_selector_fields() -> None:
    base = MedicationReviewPrimTrace(
        method_version="prim-rag-v2-full-vector-top10",
        requested_profile="full",
        effective_profile="full",
        run_status="completed",
        completion_reason="model_final",
        review_run_id="run-1",
        raw_question_hash="hash",
        plan_extraction=disabled_anchor_audit(),
        modifier_extraction=disabled_modifier_audit(),
        coverage_report=PrimCoverageReport(),
        reflection_report=ReflectionReport(enabled=True),
        final_answer_hash="answer-hash",
    )
    trace = AcmReviewHarnessMiddleware(model=object()).finalize_trace(
        base_trace=base,
        state={
            "atlas_snapshot": {"snapshot_hash": "atlas"},
            "atlas_document_open_records": [
                {
                    "record_id": "AOPEN-call-1",
                    "tool_call_id": "call-1",
                    "doc_id": "file-atlas",
                    "title": "共识",
                    "reason": "查看主题",
                    "topic_count": 1,
                }
            ],
        },
        context=_context_with_atlas(),
    )

    payload = trace.model_dump(mode="json")
    assert trace.schema_version == "10.0"
    assert trace.method_family == "acm-prim-rag-v3"
    assert trace.prompt_hashes["atlas_navigation"] == ATLAS_NAVIGATION_PROMPT_HASH
    assert trace.atlas_document_open_records[0].doc_id == "file-atlas"
    assert "companion_selection" not in payload
    assert trace.query_records == base.query_records
