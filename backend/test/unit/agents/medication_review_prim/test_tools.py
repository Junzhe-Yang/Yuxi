from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    InvestigationItem,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrievalOutcome,
    RetrieverSelection,
    _search_review_kb_impl,
    ensure_runtime_resources,
    search_review_kb_investigation,
    search_tool_for_profile,
    update_investigation,
)


def _context(retriever, *, max_search_calls: int = 40):
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
        max_search_calls=max_search_calls,
    )
    ensure_runtime_resources(context)
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1", "file-2"}
    return context


def _runtime(context, *, tool_call_id="call-1", state=None):
    return SimpleNamespace(
        context=context,
        state=state
        or {
            "plan_anchors": [{"element_id": "PE001"}],
            "patient_modifiers": [{"modifier_id": "PM001"}],
            "investigations": [],
            "evidence_store": {},
            "search_count": 0,
            "open_count": 0,
        },
        tool_call_id=tool_call_id,
    )


@pytest.mark.parametrize(
    ("profile", "expected", "excluded"),
    [
        (
            "b1",
            {"query_text", "reason", "retrieval_scope", "file_id"},
            {"focus_plan_ids", "question", "investigation_id"},
        ),
        (
            "m1",
            {"query_text", "reason", "retrieval_scope", "file_id", "focus_plan_ids"},
            {"focus_modifier_ids", "question"},
        ),
        (
            "m2",
            {
                "query_text",
                "reason",
                "retrieval_scope",
                "file_id",
                "focus_plan_ids",
                "focus_modifier_ids",
            },
            {"question", "investigation_id"},
        ),
        (
            "full",
            {
                "query_text",
                "reason",
                "retrieval_scope",
                "file_id",
                "focus_plan_ids",
                "focus_modifier_ids",
                "question",
                "investigation_id",
            },
            {"relation_question", "relation_id"},
        ),
    ],
)
def test_search_schema_is_projected_by_profile(profile, expected, excluded):
    properties = set(
        search_tool_for_profile(profile).tool_call_schema.model_json_schema()[
            "properties"
        ]
    )
    assert expected <= properties
    assert not (excluded & properties)
    assert "runtime" not in properties


@pytest.mark.asyncio
async def test_search_creates_open_investigation_even_with_results() -> None:
    async def retriever(_query: str, **_kwargs):
        return [
            {
                "content": "肾功能减退时方案甲需要调整。",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                },
                "score": 0.9,
            }
        ]

    result = await search_review_kb_investigation.coroutine(
        query_text="方案甲 肾功能 剂量调整",
        reason="核验适用条件",
        question="肾功能减退是否改变方案甲？",
        focus_plan_ids=["PE001"],
        focus_modifier_ids=["PM001"],
        runtime=_runtime(_context(retriever)),
    )

    query = result.update["query_records"][0]
    investigation = result.update["investigations"][0]
    assert query.investigation_id == investigation.investigation_id
    assert investigation.investigation_id.startswith("INV-")
    assert investigation.status == "open"
    assert investigation.selected_evidence_ids == []
    assert investigation.candidate_evidence_ids == query.evidence_ids
    assert investigation.candidate_file_ids == ["file-1"]


@pytest.mark.asyncio
async def test_document_scope_passes_filter_file_ids() -> None:
    calls = []

    async def retriever(query: str, **kwargs):
        calls.append((query, kwargs))
        return []

    result = await _search_review_kb_impl(
        query_text="剂量 给药间隔",
        reason="在已定位共识中找具体阈值",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(_context(retriever)),
    )

    assert result.update["query_records"][0].retrieval_scope == "document"
    assert calls[0][1]["filter_file_ids"] == ["file-1"]
    assert calls[0][1]["final_top_k"] == 10


@pytest.mark.asyncio
async def test_invalid_document_never_falls_back_to_global() -> None:
    calls = 0

    async def retriever(_query: str, **_kwargs):
        nonlocal calls
        calls += 1
        return []

    result = await _search_review_kb_impl(
        query_text="剂量",
        reason="补查",
        retrieval_scope="document",
        file_id="file-outside",
        runtime=_runtime(_context(retriever)),
    )

    assert calls == 0
    record = result.update["query_records"][0]
    assert record.status == "technical_failed"
    assert "不属于当前知识库" in (record.error_message or "")


@pytest.mark.asyncio
async def test_document_scope_accepts_file_id_from_current_evidence_when_catalog_is_unavailable(
    monkeypatch,
) -> None:
    calls = []

    async def retriever(query: str, **kwargs):
        calls.append((query, kwargs))
        return []

    async def unavailable_catalog(_db_id: str):
        raise RuntimeError("metadata service unavailable")

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_prim.tools.knowledge_base.get_database_info",
        unavailable_catalog,
    )
    context = _context(retriever)
    context._prim_allowed_file_ids = None
    state = {
        "investigations": [],
        "search_count": 0,
        "evidence_store": {
            "EV-ONE": {
                "evidence_id": "EV-ONE",
                "content_hash": "hash",
                "raw_text": "此前从当前知识库召回的内容",
                "file_id": "file-1",
                "chunk_id": "chunk-1",
                "chunk_index": 1,
            }
        },
    }

    result = await _search_review_kb_impl(
        query_text="剂量 给药间隔",
        reason="继续深挖已经召回的文档",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state=state),
    )

    assert result.update["query_records"][0].status == "success_empty"
    assert calls[0][1]["filter_file_ids"] == ["file-1"]


@pytest.mark.asyncio
async def test_second_knowledge_call_in_same_model_turn_is_deferred() -> None:
    calls = 0

    async def retriever(_query: str, **_kwargs):
        nonlocal calls
        calls += 1
        return []

    context = _context(retriever)
    state = {
        "messages": [
            {
                "type": "ai",
                "tool_calls": [
                    {"id": "call-first", "name": "search_review_kb"},
                    {"id": "call-second", "name": "search_review_kb"},
                ],
            }
        ],
        "investigations": [],
        "evidence_store": {},
        "search_count": 0,
    }
    result = await _search_review_kb_impl(
        query_text="第二个查询",
        reason="不应在本轮执行",
        runtime=_runtime(
            context,
            tool_call_id="call-second",
            state=state,
        ),
    )

    assert calls == 0
    assert "query_records" not in result.update
    assert "search_count" not in result.update
    assert result.update["deferred_knowledge_calls"][0].reason == "same_model_turn"


@pytest.mark.asyncio
async def test_update_requires_real_candidate_and_closes_explicitly() -> None:
    investigation = InvestigationItem(
        investigation_id="INV-ONE",
        question="问题",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        candidate_evidence_ids=["EV-ONE"],
    )
    state = {
        "investigations": [investigation],
        "evidence_store": {
            "EV-ONE": {
                "evidence_id": "EV-ONE",
                "content_hash": "hash",
                "raw_text": "内容",
            }
        },
    }
    runtime = SimpleNamespace(state=state, tool_call_id="update-1")

    invalid = await update_investigation.coroutine(
        investigation_id="INV-ONE",
        status="answered",
        selected_evidence_ids=["EV-UNKNOWN"],
        runtime=runtime,
    )
    assert "investigations" not in invalid.update

    valid = await update_investigation.coroutine(
        investigation_id="INV-ONE",
        status="answered",
        selected_evidence_ids=["ev-one"],
        working_note="已有足够证据形成有边界回答。",
        runtime=runtime,
    )
    updated = valid.update["investigations"][0]
    assert updated.status == "answered"
    assert updated.selected_evidence_ids == ["EV-ONE"]


@pytest.mark.asyncio
async def test_custom_strategy_still_emits_diagnostic() -> None:
    async def retriever(_query: str, **_kwargs):
        raise AssertionError("custom strategy should replace flat retrieval")

    captured = []

    async def strategy(request):
        captured.append(request)
        return RetrievalOutcome(
            chunks=[],
            returned_count=0,
            diagnostic_record={"retrieval_record_id": "RR-1"},
        )

    result = await _search_review_kb_impl(
        query_text="方案甲 肾功能 调整",
        reason="兼容历史诊断",
        focus_plan_ids=["PE001"],
        runtime=_runtime(_context(retriever)),
        retrieval_strategy=strategy,
    )

    assert captured[0].focus_plan_ids == ["PE001"]
    assert result.update["routed_retrieval_records"] == [
        {"retrieval_record_id": "RR-1"}
    ]
