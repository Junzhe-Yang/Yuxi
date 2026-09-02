from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
    build_adaptive_coverage_report,
    build_adaptive_investigation_memory,
    build_adaptive_state_fingerprint,
)
from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
    validate_acm_context,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    CorpusAtlas,
)
from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    ADAPTIVE_CONTEXT_PROJECTION_HASH,
    ADAPTIVE_CONTEXT_PROJECTION_VERSION,
    AcmReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_prim.memory import (
    ADAPTIVE_ATLAS_NAVIGATION_PROMPT_HASH,
    ADAPTIVE_ATLAS_NAVIGATION_PROMPT_VERSION,
    build_adaptive_atlas_document_memory,
)
from yuxi.agents.buildin.medication_review_acm_prim.models import (
    MedicationReviewAcmAdaptiveTrace,
    V7AgendaItemDraft,
    merge_adaptive_agenda,
)
from yuxi.agents.buildin.medication_review_acm_prim.tools import (
    extend_investigation_agenda,
    search_review_kb_adaptive,
    set_investigation_agenda,
    submit_coverage_gap_assessment,
    update_acm_investigation,
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
    RetrieverSelection,
    ensure_runtime_resources,
)


def _context(*, retriever=None, max_search_calls: int = 20):
    async def empty_retriever(_query: str, **_kwargs):
        return []

    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        acm_protocol="adaptive_coverage",
        v7_retrieval_depth="shadow_top25",
        max_search_calls=max_search_calls,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
    )
    validate_acm_context(context)
    ensure_runtime_resources(context)
    context._prim_retrieval_fetch_k = 25
    context._prim_retrieval_visible_k = 10
    context._prim_retrieval_diagnostic_state_key = "adaptive_retrieval_records"
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever or empty_retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1", "file-2"}
    return context


def _runtime(context, state, tool_call_id: str):
    return SimpleNamespace(
        context=context,
        state=state,
        tool_call_id=tool_call_id,
    )


def _base_state() -> dict[str, Any]:
    return {
        "plan_anchors": [{"element_id": "PE001"}],
        "patient_modifiers": [{"modifier_id": "PM001"}],
        "investigations": [],
        "query_records": [],
        "evidence_store": {},
        "adaptive_probe_records": [],
        "adaptive_recovery_requirements": [],
        "adaptive_investigation_meta": [],
        "adaptive_gap_assessments": [],
        "adaptive_checkpoint_records": [],
        "search_count": 0,
    }


def _merge_by_id(left, right, field: str):
    values = {}
    for value in [*(left or []), *(right or [])]:
        key = value.get(field) if isinstance(value, dict) else getattr(value, field)
        values[key] = value
    return list(values.values())


def _apply(state: dict[str, Any], update: dict[str, Any]) -> None:
    id_fields = {
        "investigations": "investigation_id",
        "query_records": "query_id",
        "adaptive_probe_records": "probe_record_id",
        "adaptive_recovery_requirements": "recovery_id",
        "adaptive_investigation_meta": "investigation_id",
        "adaptive_gap_assessments": "assessment_id",
        "adaptive_checkpoint_records": "checkpoint_id",
    }
    for key, value in update.items():
        if key in {"messages", "warnings", "technical_attempts"}:
            continue
        if key == "search_count":
            state[key] = int(state.get(key) or 0) + int(value or 0)
        elif key == "evidence_store":
            state[key] = {**(state.get(key) or {}), **(value or {})}
        elif key in id_fields:
            state[key] = _merge_by_id(
                state.get(key),
                value,
                id_fields[key],
            )
        else:
            state[key] = value


def _draft(
    question: str,
    scope: str,
    *evidence_obligations: str,
) -> V7AgendaItemDraft:
    return V7AgendaItemDraft(
        question=question,
        why_it_matters=f"{question}可能改变最终判断",
        distinct_scope=scope,
        focus_plan_ids=["PE001"],
        focus_modifier_ids=["PM001"],
        decision_tags=["patient_preference"],
        investigation_kind="current_regimen_review",
        evidence_obligations=list(evidence_obligations) or [f"{question}的主要依据"],
    )


def _coverage_audit(
    investigation_id: str,
    *,
    gap_dimension: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "dimension": dimension,
            "status": "gap" if dimension == gap_dimension else "covered" if index == 0 else "not_applicable",
            "investigation_ids": [investigation_id] if index == 0 else [],
            "rationale": (
                "该维度仍缺直接证据"
                if dimension == gap_dimension
                else "已由调查覆盖"
                if index == 0
                else "当前病例不适用"
            ),
        }
        for index, dimension in enumerate(ADAPTIVE_AUDIT_DIMENSIONS)
    ]


def test_adaptive_context_requires_shadow_top25() -> None:
    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        acm_protocol="adaptive_coverage",
        v7_retrieval_depth="top10",
    )

    with pytest.raises(ValueError, match="shadow_top25"):
        validate_acm_context(context)


def test_adaptive_atlas_memory_does_not_force_document_scope() -> None:
    memory = build_adaptive_atlas_document_memory(
        [
            {
                "title": "治疗共识",
                "doc_id": "file-1",
                "scope_summary": "覆盖治疗选择与监测。",
            }
        ]
    )

    assert "source_discovery/global" in memory
    assert "within_document_localization/document" in memory
    assert "不会强制" in memory


def test_public_adaptive_tool_schema_has_no_fixed_agenda_count() -> None:
    agenda_schema = set_investigation_agenda.tool_call_schema.model_json_schema()
    search_schema = search_review_kb_adaptive.tool_call_schema.model_json_schema()

    items_schema = agenda_schema["properties"]["items"]
    assert items_schema["minItems"] == 1
    assert "maxItems" not in items_schema
    assert "evidence_obligations" in agenda_schema["$defs"]["AcmAgendaItemDraft"]["properties"]
    assert {
        "query_text",
        "reason",
        "investigation_id",
        "uncovered_aspect",
        "retrieval_intent",
    }.issubset(search_schema["required"])
    query_description = search_schema["properties"]["query_text"]["description"]
    assert "单一" in query_description
    assert "6" in query_description


@pytest.mark.asyncio
async def test_adaptive_search_rejects_multi_axis_query_without_spending_budget() -> None:
    retriever_calls = 0

    async def retriever(_query: str, **_kwargs):
        nonlocal retriever_calls
        retriever_calls += 1
        return []

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("依托考昔剂量审查", "急性期推荐剂量", "依托考昔推荐剂量")],
        runtime=_runtime(context, state, "agenda-query-contract"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    result = await search_review_kb_adaptive.coroutine(
        query_text="依托考昔 痛风急性期 推荐剂量 心血管风险 监测",
        reason="同时查询剂量、安全和监测",
        investigation_id=investigation_id,
        uncovered_aspect="依托考昔推荐剂量",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-multi-axis"),
    )

    assert "单一待查属性" in result.update["messages"][0].content
    assert "query_records" not in result.update
    assert "search_count" not in result.update
    assert retriever_calls == 0


@pytest.mark.asyncio
async def test_adaptive_search_rejects_more_than_six_query_concepts_without_spending_budget() -> None:
    retriever_calls = 0

    async def retriever(_query: str, **_kwargs):
        nonlocal retriever_calls
        retriever_calls += 1
        return []

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("文拉法辛剂量审查", "起始与滴定剂量", "文拉法辛起始与滴定剂量")],
        runtime=_runtime(context, state, "agenda-query-width"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    result = await search_review_kb_adaptive.coroutine(
        query_text="老年 抑郁 文拉法辛 起始剂量 加量 滴定 目标剂量",
        reason="复制完整义务形成宽查询",
        investigation_id=investigation_id,
        uncovered_aspect="文拉法辛起始与滴定剂量",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-query-width"),
    )

    assert "最多 6 个" in result.update["messages"][0].content
    assert "query_records" not in result.update
    assert "search_count" not in result.update
    assert retriever_calls == 0


@pytest.mark.asyncio
async def test_adaptive_search_memory_expands_only_the_active_obligation() -> None:
    context = _context()
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一", "当前必须定位的剂量义务", "稍后处理的监测义务"),
            _draft("问题二不应在当前展开", "范围二", "另一个调查的安全义务"),
        ],
        runtime=_runtime(context, state, "agenda-focused-memory"),
    )
    _apply(state, agenda.update)

    memory = build_adaptive_investigation_memory(
        state,
        build_adaptive_coverage_report(state, context),
    )

    assert "当前阶段：定向检索" in memory
    assert "当前必须定位的剂量义务" in memory
    assert "稍后处理的监测义务" not in memory
    assert "问题二不应在当前展开" not in memory
    assert "另一个调查的安全义务" not in memory
    assert "【其他调查状态】" in memory


@pytest.mark.asyncio
async def test_adaptive_review_memory_restores_all_obligations_before_close() -> None:
    context = _context()
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("完整审查", "剂量与监测", "完整剂量义务", "完整监测义务")],
        runtime=_runtime(context, state, "agenda-review-memory"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id
    state["query_records"] = [
        {
            "query_id": "Q-DOSE",
            "tool_call_id": "call-dose",
            "investigation_id": investigation_id,
            "query_text": "方案甲 推荐剂量",
            "reason": "定位剂量",
            "started_at": "2026-08-31T00:00:00Z",
            "elapsed_ms": 1,
            "status": "success_empty",
        },
        {
            "query_id": "Q-MONITOR",
            "tool_call_id": "call-monitor",
            "investigation_id": investigation_id,
            "query_text": "方案甲 监测要求",
            "reason": "定位监测",
            "started_at": "2026-08-31T00:00:01Z",
            "elapsed_ms": 1,
            "status": "success_empty",
        },
    ]
    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q-DOSE",
            "query_id": "Q-DOSE",
            "investigation_id": investigation_id,
            "uncovered_aspect": "完整剂量义务",
            "retrieval_intent": "source_discovery",
            "route_key": "route-dose",
            "status": "success_empty",
        },
        {
            "probe_record_id": "APROBE-Q-MONITOR",
            "query_id": "Q-MONITOR",
            "investigation_id": investigation_id,
            "uncovered_aspect": "完整监测义务",
            "retrieval_intent": "source_discovery",
            "route_key": "route-monitor",
            "status": "success_empty",
        },
    ]
    state["search_count"] = 2

    memory = build_adaptive_investigation_memory(
        state,
        build_adaptive_coverage_report(state, context),
    )

    assert "当前阶段：证据核对与关闭" in memory
    assert "完整剂量义务" in memory
    assert "完整监测义务" in memory
    assert memory.count("完整剂量义务") == 1
    assert memory.count("完整监测义务") == 1


def test_adaptive_memory_replaces_duplicate_base_investigation_blocks_with_case_nodes() -> None:
    context = _context()
    context._acm_prim_atlas = CorpusAtlas(
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash="atlas",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-08-31T00:00:00Z",
        document_cards=[],
    )
    state = _base_state()
    state.update(
        {
            "effective_profile": "full",
            "plan_anchors": [
                {
                    "element_id": "PE001",
                    "source_span": "文拉法辛缓释片 75mg 每日一次",
                    "source_start": 0,
                    "source_end": 19,
                    "label": "文拉法辛缓释片",
                    "kind": "medication_order",
                }
            ],
            "patient_modifiers": [
                {
                    "modifier_id": "PM001",
                    "source_span": "既往高血压",
                    "source_start": 20,
                    "source_end": 26,
                }
            ],
        }
    )

    memory = AcmReviewHarnessMiddleware(model=object()).augment_investigation_memory(
        state=state,
        context=context,
        memory_text="【当前待解决的证据问题】\nBASE-DUPLICATE-BLOCK",
    )

    assert "BASE-DUPLICATE-BLOCK" not in memory
    assert "文拉法辛缓释片 75mg 每日一次" in memory
    assert "既往高血压" in memory
    assert memory.count("【明确治疗方案要素】") == 1


def test_adaptive_fingerprint_is_order_stable_and_state_sensitive() -> None:
    meta = [
        {
            "investigation_id": "INV-1",
            "resolved_aspects": ["适用性"],
            "updated_at": "2026-08-25T00:00:00Z",
        },
        {
            "investigation_id": "INV-2",
            "remaining_aspects": ["监测"],
            "updated_at": "2026-08-25T00:00:01Z",
        },
    ]
    recoveries = [
        {
            "recovery_id": "RECOVERY-1",
            "investigation_id": "INV-1",
            "source_query_id": "Q1",
            "uncovered_aspect": "来源",
            "status": "resolved",
            "created_at": "2026-08-25T00:00:00Z",
        },
        {
            "recovery_id": "RECOVERY-2",
            "investigation_id": "INV-2",
            "source_query_id": "Q2",
            "uncovered_aspect": "边界",
            "status": "pending",
            "created_at": "2026-08-25T00:00:01Z",
        },
    ]
    left = {
        **_base_state(),
        "adaptive_investigation_meta": meta,
        "adaptive_recovery_requirements": recoveries,
        "evidence_store": {"E1": {}, "E2": {}},
    }
    right = {
        **_base_state(),
        "adaptive_investigation_meta": list(reversed(meta)),
        "adaptive_recovery_requirements": list(reversed(recoveries)),
        "evidence_store": {"E2": {}, "E1": {}},
    }

    assert build_adaptive_state_fingerprint(left) == build_adaptive_state_fingerprint(right)
    right["adaptive_recovery_requirements"][0] = {
        **right["adaptive_recovery_requirements"][0],
        "status": "closed_insufficient",
    }
    assert build_adaptive_state_fingerprint(left) != build_adaptive_state_fingerprint(right)


@pytest.mark.asyncio
async def test_adaptive_agenda_has_no_fixed_business_count_and_can_extend() -> None:
    context = _context()
    state = _base_state()
    result = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一"),
            _draft("问题二", "范围二"),
            _draft("问题三", "范围三"),
            _draft("问题四", "范围四"),
        ],
        runtime=_runtime(context, state, "agenda-1"),
    )
    _apply(state, result.update)

    assert len(state["adaptive_agenda"].items) == 4
    assert state["adaptive_agenda"].revision == 1
    assert state["adaptive_agenda"].items[0].decision_tags == ["patient_preference"]
    original_agenda = state["adaptive_agenda"]

    extension = await extend_investigation_agenda.coroutine(
        items=[_draft("问题五", "范围五")],
        reason="关闭前发现新的跨方案缺口",
        runtime=_runtime(context, state, "agenda-2"),
    )
    _apply(state, extension.update)

    assert len(state["adaptive_agenda"].items) == 5
    assert state["adaptive_agenda"].revision == 2
    assert merge_adaptive_agenda(original_agenda, state["adaptive_agenda"]) == state["adaptive_agenda"]

    rewritten_item = state["adaptive_agenda"].items[0].model_copy(update={"question": "被改写的问题"})
    rewritten_agenda = state["adaptive_agenda"].model_copy(
        update={"items": [rewritten_item, *state["adaptive_agenda"].items[1:]]}
    )
    with pytest.raises(ValueError, match="append-only"):
        merge_adaptive_agenda(state["adaptive_agenda"], rewritten_agenda)


@pytest.mark.asyncio
async def test_adaptive_agenda_rejects_missing_plan_and_duplicate_but_not_budget_overflow() -> None:
    context = _context(max_search_calls=2)
    state = _base_state()
    missing_plan = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="只调查患者因素",
                why_it_matters="可能影响安全性",
                distinct_scope="患者因素",
                focus_modifier_ids=["PM001"],
            )
        ],
        runtime=_runtime(context, state, "agenda-missing-plan"),
    )
    assert "尚未建立 current_regimen_review 的 PlanAnchor" in missing_plan.update["messages"][0].content

    missing_obligation = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一").model_copy(update={"evidence_obligations": []})],
        runtime=_runtime(context, state, "agenda-missing-obligation"),
    )
    assert "必须列出 evidence_obligations" in missing_obligation.update["messages"][0].content

    duplicate = await set_investigation_agenda.coroutine(
        items=[_draft("同一问题", "范围一"), _draft("同一问题", "范围二")],
        runtime=_runtime(context, state, "agenda-duplicate"),
    )
    assert "完全重复" in duplicate.update["messages"][0].content

    overflow = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一"),
            _draft("问题二", "范围二"),
            _draft("问题三", "范围三"),
        ],
        runtime=_runtime(context, state, "agenda-overflow"),
    )
    assert len(overflow.update["adaptive_agenda"].items) == 3
    assert "剩余搜索容量不足" not in overflow.update["messages"][0].content


@pytest.mark.asyncio
async def test_adaptive_scheduler_prioritizes_every_unprobed_investigation() -> None:
    async def retriever(query: str, **_kwargs):
        return [
            {
                "content": f"{query} 的证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一", "问题一的主要依据", "问题一的边界"),
            _draft("问题二", "范围二"),
        ],
        runtime=_runtime(context, state, "agenda-priority"),
    )
    _apply(state, agenda_result.update)
    first_id, second_id = [value.investigation_id for value in state["adaptive_agenda"].items]

    first = await search_review_kb_adaptive.coroutine(
        query_text="问题一 来源",
        reason="首次来源发现",
        investigation_id=first_id,
        uncovered_aspect="问题一的主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-first"),
    )
    _apply(state, first.update)

    rejected = await search_review_kb_adaptive.coroutine(
        query_text="问题一 边界",
        reason="过早继续同一调查",
        investigation_id=first_id,
        uncovered_aspect="问题一的边界",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-rejected"),
    )

    assert second_id in rejected.update["messages"][0].content
    assert "query_records" not in rejected.update


@pytest.mark.asyncio
async def test_successful_query_text_can_be_reused_by_a_different_investigation() -> None:
    calls = 0

    async def retriever(query: str, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            {
                "content": f"{query} 的共享证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-shared",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一", "调查一主要依据"),
            _draft("问题二", "范围二", "调查二主要依据"),
        ],
        runtime=_runtime(context, state, "agenda-query-scope"),
    )
    _apply(state, agenda.update)
    first_id, second_id = [value.investigation_id for value in state["adaptive_agenda"].items]

    first = await search_review_kb_adaptive.coroutine(
        query_text="共享来源查询",
        reason="调查一来源发现",
        investigation_id=first_id,
        uncovered_aspect="调查一主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-shared-first"),
    )
    _apply(state, first.update)
    second = await search_review_kb_adaptive.coroutine(
        query_text="共享来源查询",
        reason="调查二独立来源发现",
        investigation_id=second_id,
        uncovered_aspect="调查二主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-shared-second"),
    )

    assert second.update["query_records"][0].status == "success"
    assert calls == 2


@pytest.mark.asyncio
async def test_one_broad_hit_cannot_support_two_distinct_obligations() -> None:
    async def retriever(query: str, **_kwargs):
        if "安全" in query:
            return []
        return [
            {
                "content": "只能直接回答剂量与疗程的证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-dose",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("完整评估问题一", "剂量与安全边界", "剂量与疗程", "患者安全性")],
        runtime=_runtime(context, state, "agenda-two-obligations"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    dose = await search_review_kb_adaptive.coroutine(
        query_text="方案甲 剂量 疗程",
        reason="定位剂量与疗程",
        investigation_id=investigation_id,
        uncovered_aspect="剂量与疗程",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-dose-obligation"),
    )
    _apply(state, dose.update)
    evidence_id = dose.update["query_records"][0].evidence_ids[0]
    safety = await search_review_kb_adaptive.coroutine(
        query_text="方案甲 患者安全性",
        reason="定位患者特异风险",
        investigation_id=investigation_id,
        uncovered_aspect="患者安全性",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-safety-obligation"),
    )
    _apply(state, safety.update)

    closed = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="answered",
        selected_evidence_ids=[evidence_id],
        working_note="尝试用一条宽证据支持两个义务",
        obligation_supports=[
            {"obligation": "剂量与疗程", "evidence_ids": [evidence_id]},
            {"obligation": "患者安全性", "evidence_ids": [evidence_id]},
        ],
        resolved_aspects=["剂量与疗程", "患者安全性"],
        remaining_aspects=[],
        runtime=_runtime(context, state, "close-broad-evidence"),
    )

    assert "义务 [患者安全性]" in closed.update["messages"][0].content
    assert "由该义务定向 probe" in closed.update["messages"][0].content
    assert state["investigations"][0].status == "open"


@pytest.mark.asyncio
async def test_adjusted_current_regimen_requires_linked_improvement_plan() -> None:
    async def retriever(_query: str, **_kwargs):
        return [
            {
                "content": "现方案需要调整的直接证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-adjust",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("现方案是否需调整", "现用药逐项审查", "现方案的调整判断")],
        runtime=_runtime(context, state, "agenda-adjust"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id
    searched = await search_review_kb_adaptive.coroutine(
        query_text="现方案 患者条件 调整",
        reason="判定现用药是否合理",
        investigation_id=investigation_id,
        uncovered_aspect="现方案的调整判断",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-adjust"),
    )
    _apply(state, searched.update)
    evidence_id = searched.update["query_records"][0].evidence_ids[0]
    closed = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="answered",
        selected_evidence_ids=[evidence_id],
        working_note="直接证据表明现方案需调整",
        obligation_supports=[
            {
                "obligation": "现方案的调整判断",
                "evidence_ids": [evidence_id],
            }
        ],
        review_outcome="adjust",
        resolved_aspects=["现方案的调整判断"],
        remaining_aspects=[],
        runtime=_runtime(context, state, "close-adjust"),
    )
    _apply(state, closed.update)

    report = build_adaptive_coverage_report(state, context)
    assert report.action_required_plan_ids == ["PE001"]
    assert report.missing_improvement_plan_ids == ["PE001"]
    assert "improvement_plan_missing" in report.incomplete_reasons

    improvement = _draft(
        "现方案应如何具体调整",
        "纠正、替代与随访",
        "可执行的替代或纠正方案",
    ).model_copy(update={"investigation_kind": "improvement_plan"})
    extended = await extend_investigation_agenda.coroutine(
        items=[improvement],
        reason="现用药审查已确认需调整",
        runtime=_runtime(context, state, "agenda-improvement"),
    )
    _apply(state, extended.update)

    updated_report = build_adaptive_coverage_report(state, context)
    assert updated_report.missing_improvement_plan_ids == []
    assert updated_report.improvement_plan_covered_plan_ids == ["PE001"]


@pytest.mark.asyncio
async def test_parent_evidence_allows_first_document_localization() -> None:
    calls: list[dict[str, Any]] = []

    async def retriever(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return []

    context = _context(retriever=retriever)
    state = _base_state()
    state["evidence_store"] = {
        "E-PARENT": {
            "evidence_id": "E-PARENT",
            "content_hash": "parent-hash",
            "raw_text": "父证据",
            "file_id": "file-1",
        }
    }
    agenda = await set_investigation_agenda.coroutine(
        items=[
            _draft("从父证据继续定位", "衍生文档范围", "具体剂量边界").model_copy(
                update={"parent_evidence_ids": ["E-PARENT"]}
            )
        ],
        runtime=_runtime(context, state, "agenda-parent"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    result = await search_review_kb_adaptive.coroutine(
        query_text="父文档内具体边界",
        reason="沿父证据定位",
        investigation_id=investigation_id,
        uncovered_aspect="具体剂量边界",
        retrieval_intent="within_document_localization",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state, "search-parent"),
    )

    assert result.update["query_records"][0].status == "success_empty"
    assert calls and calls[0]["filter_file_ids"] == ["file-1"]


@pytest.mark.asyncio
async def test_search_intent_and_successful_duplicate_routes_are_rejected() -> None:
    call_count = 0

    async def retriever(query: str, **_kwargs):
        nonlocal call_count
        call_count += 1
        return [
            {
                "content": f"{query} 的原始检索内容",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一", "主要依据")],
        runtime=_runtime(context, state, "agenda-routes"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    invalid_intent = await search_review_kb_adaptive.coroutine(
        query_text="错误范围",
        reason="错误地在文档内发现来源",
        investigation_id=investigation_id,
        uncovered_aspect="主要依据",
        retrieval_intent="source_discovery",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state, "search-invalid-intent"),
    )
    assert "source_discovery 必须" in invalid_intent.update["messages"][0].content
    assert call_count == 0

    adjacent = await search_review_kb_adaptive.coroutine(
        query_text="相邻原文",
        reason="错误使用搜索读取相邻原文",
        investigation_id=investigation_id,
        uncovered_aspect="主要依据",
        retrieval_intent="adjacent_context",
        runtime=_runtime(context, state, "search-adjacent"),
    )
    assert "open_review_evidence" in adjacent.update["messages"][0].content
    assert call_count == 0

    successful = await search_review_kb_adaptive.coroutine(
        query_text="首次全库查询",
        reason="来源发现",
        investigation_id=investigation_id,
        uncovered_aspect="主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-success"),
    )
    assert "原始检索内容" in successful.update["messages"][0].content
    _apply(state, successful.update)
    assert call_count == 1

    exact_query = await search_review_kb_adaptive.coroutine(
        query_text="  首次全库查询 ",
        reason="重复查询",
        investigation_id=investigation_id,
        uncovered_aspect="主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-exact-query"),
    )
    assert "已成功执行过" in exact_query.update["messages"][0].content

    duplicate_route = await search_review_kb_adaptive.coroutine(
        query_text="不同查询文本",
        reason="重复相同路线",
        investigation_id=investigation_id,
        uncovered_aspect="主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-duplicate-route"),
    )
    assert "路线已成功执行" in duplicate_route.update["messages"][0].content
    assert call_count == 1


@pytest.mark.asyncio
async def test_empty_global_probe_allows_atomic_query_rewrite() -> None:
    call_count = 0

    async def retriever(query: str, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return []
        return [
            {
                "content": f"{query} 命中的直接证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一", "非药物替代")],
        runtime=_runtime(context, state, "agenda-empty-rewrite"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    empty = await search_review_kb_adaptive.coroutine(
        query_text="宽泛替代方案",
        reason="首次来源发现",
        investigation_id=investigation_id,
        uncovered_aspect="非药物替代",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-empty"),
    )
    _apply(state, empty.update)
    assert state["query_records"][0].status == "success_empty"

    rewritten = await search_review_kb_adaptive.coroutine(
        query_text="认知行为治疗 正念 接受承诺疗法",
        reason="把非药物义务改写为原子查询",
        investigation_id=investigation_id,
        uncovered_aspect="非药物替代",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-rewritten"),
    )

    assert rewritten.update["query_records"][0].status == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_document_zero_novelty_requires_successful_global_recovery() -> None:
    async def retriever(_query: str, **_kwargs):
        return [
            {
                "content": "同一个已知证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一", "剂量边界")],
        runtime=_runtime(context, state, "agenda-recovery"),
    )
    _apply(state, agenda_result.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    initial = await search_review_kb_adaptive.coroutine(
        query_text="首次来源查询",
        reason="发现来源",
        investigation_id=investigation_id,
        uncovered_aspect="剂量边界",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-source"),
    )
    _apply(state, initial.update)
    localized = await search_review_kb_adaptive.coroutine(
        query_text="文档内边界查询",
        reason="定位边界",
        investigation_id=investigation_id,
        uncovered_aspect="剂量边界",
        retrieval_intent="within_document_localization",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state, "search-document"),
    )
    _apply(state, localized.update)

    assert state["adaptive_recovery_requirements"][0].status == "pending"
    memory = build_adaptive_investigation_memory(
        state,
        build_adaptive_coverage_report(state, context),
    )
    assert "当前强制全库恢复任务" in memory
    assert investigation_id in memory
    assert "file-1" in memory
    blocked = await search_review_kb_adaptive.coroutine(
        query_text="另一文档内查询",
        reason="试图继续局部搜索",
        investigation_id=investigation_id,
        uncovered_aspect="剂量边界",
        retrieval_intent="within_document_localization",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state, "search-blocked"),
    )
    assert "source_discovery/global" in blocked.update["messages"][0].content

    recovery = await search_review_kb_adaptive.coroutine(
        query_text="全库恢复查询",
        reason="避免在单一文档中局部循环",
        investigation_id=investigation_id,
        uncovered_aspect="剂量边界",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-recovery"),
    )

    assert recovery.update["adaptive_recovery_requirements"][0].status == "resolved"


@pytest.mark.asyncio
async def test_document_evidence_is_novel_per_investigation_not_global_store() -> None:
    async def retriever(query: str, **_kwargs):
        if query == "第二调查来源":
            return [
                {
                    "content": "第二调查自己的来源",
                    "metadata": {
                        "source": "共识.md",
                        "file_id": "file-1",
                        "chunk_id": "chunk-second",
                    },
                }
            ]
        return [
            {
                "content": "已由另一调查召回的共享证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-shared",
                },
            }
        ]

    context = _context(retriever=retriever)
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[
            _draft("问题一", "范围一", "第一调查主要依据"),
            _draft("问题二", "范围二", "第二调查主要依据", "第二调查局部边界"),
        ],
        runtime=_runtime(context, state, "agenda-investigation-novelty"),
    )
    _apply(state, agenda.update)
    first_id, second_id = [value.investigation_id for value in state["adaptive_agenda"].items]

    first = await search_review_kb_adaptive.coroutine(
        query_text="第一调查来源",
        reason="建立全局共享证据",
        investigation_id=first_id,
        uncovered_aspect="第一调查主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-investigation-first"),
    )
    _apply(state, first.update)
    second = await search_review_kb_adaptive.coroutine(
        query_text="第二调查来源",
        reason="建立第二调查来源",
        investigation_id=second_id,
        uncovered_aspect="第二调查主要依据",
        retrieval_intent="source_discovery",
        runtime=_runtime(context, state, "search-investigation-second"),
    )
    _apply(state, second.update)

    localized = await search_review_kb_adaptive.coroutine(
        query_text="第二调查文档定位",
        reason="定位另一调查已见、但本调查尚未纳入的证据",
        investigation_id=second_id,
        uncovered_aspect="第二调查局部边界",
        retrieval_intent="within_document_localization",
        retrieval_scope="document",
        file_id="file-1",
        runtime=_runtime(context, state, "search-investigation-document"),
    )

    assert localized.update["query_records"][0].new_evidence_ids == []
    assert localized.update["adaptive_probe_records"][0].redundant is False
    assert localized.update["adaptive_recovery_requirements"] == []


@pytest.mark.asyncio
async def test_success_empty_completes_probe_and_recovery_preempts_unprobed() -> None:
    context = _context()
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一"), _draft("问题二", "范围二")],
        runtime=_runtime(context, state, "agenda-statuses"),
    )
    _apply(state, agenda.update)
    first_id, second_id = [value.investigation_id for value in state["adaptive_agenda"].items]
    state["query_records"] = [
        {
            "query_id": "Q-EMPTY",
            "tool_call_id": "call-empty",
            "investigation_id": first_id,
            "query_text": "空结果查询",
            "reason": "真实执行",
            "started_at": "2026-08-25T00:00:00Z",
            "elapsed_ms": 1,
            "status": "success_empty",
        },
        {
            "query_id": "Q-FAILED",
            "tool_call_id": "call-failed",
            "investigation_id": second_id,
            "query_text": "技术失败查询",
            "reason": "真实执行失败",
            "started_at": "2026-08-25T00:00:01Z",
            "elapsed_ms": 1,
            "status": "technical_failed",
        },
    ]
    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q-EMPTY",
            "query_id": "Q-EMPTY",
            "investigation_id": first_id,
            "uncovered_aspect": "问题一的主要依据",
            "retrieval_intent": "source_discovery",
            "route_key": "route-empty",
            "status": "success_empty",
        },
        {
            "probe_record_id": "APROBE-Q-FAILED",
            "query_id": "Q-FAILED",
            "investigation_id": second_id,
            "uncovered_aspect": "问题二的主要依据",
            "retrieval_intent": "source_discovery",
            "route_key": "route-failed",
            "status": "technical_failed",
        },
    ]
    state["search_count"] = 2

    report = build_adaptive_coverage_report(state, context)
    assert report.unprobed_investigation_ids == [second_id]
    assert report.eligible_investigation_ids == [second_id]

    state["adaptive_recovery_requirements"] = [
        {
            "recovery_id": "RECOVERY-1",
            "investigation_id": first_id,
            "source_query_id": "Q-EMPTY",
            "uncovered_aspect": "局部检索零新增",
            "status": "pending",
            "created_at": "2026-08-25T00:00:02Z",
        }
    ]
    recovery_report = build_adaptive_coverage_report(state, context)
    assert recovery_report.eligible_investigation_ids == [first_id]


@pytest.mark.asyncio
async def test_close_and_current_no_gap_assessment_complete_contract() -> None:
    context = _context()
    state = _base_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一")],
        runtime=_runtime(context, state, "agenda-close"),
    )
    _apply(state, agenda_result.update)
    investigation = state["investigations"][0]
    investigation_id = investigation.investigation_id
    investigation = investigation.model_copy(
        update={
            "query_ids": ["Q1"],
            "candidate_evidence_ids": ["E1"],
        }
    )
    state["investigations"] = [investigation]
    state["evidence_store"] = {
        "E1": {
            "evidence_id": "E1",
            "content_hash": "hash-e1",
            "raw_text": "回答问题一的证据",
        }
    }
    state["query_records"] = [
        {
            "query_id": "Q1",
            "tool_call_id": "query-1",
            "investigation_id": investigation_id,
            "query_text": "查询一",
            "reason": "首次检索",
            "started_at": "2026-08-25T00:00:00Z",
            "elapsed_ms": 1,
            "status": "success",
            "evidence_ids": ["E1"],
            "new_evidence_ids": ["E1"],
        }
    ]
    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q1",
            "query_id": "Q1",
            "investigation_id": investigation_id,
            "uncovered_aspect": "问题一的主要依据",
            "retrieval_intent": "source_discovery",
            "route_key": "route-q1",
            "status": "success",
        }
    ]
    state["search_count"] = 1

    invalid = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="answered",
        selected_evidence_ids=["E1"],
        working_note="已有证据",
        runtime=_runtime(context, state, "close-invalid"),
    )
    assert "未绑定直接 Evidence" in invalid.update["messages"][0].content

    closed = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="answered",
        selected_evidence_ids=["E1"],
        working_note="已有证据回答主要问题",
        obligation_supports=[
            {
                "obligation": "问题一的主要依据",
                "evidence_ids": ["E1"],
            }
        ],
        review_outcome="appropriate",
        resolved_aspects=["问题一的主要依据"],
        remaining_aspects=[],
        runtime=_runtime(context, state, "close-valid"),
    )
    _apply(state, closed.update)
    assert state["investigations"][0].status == "answered", closed.update
    incomplete_audit = await submit_coverage_gap_assessment.coroutine(
        material_gap_found=False,
        rationale="仅审计了一个维度",
        coverage_audit=_coverage_audit(investigation_id)[:1],
        runtime=_runtime(context, state, "gap-incomplete"),
    )
    assert "coverage_audit 缺少维度" in incomplete_audit.update["messages"][0].content

    assessment = await submit_coverage_gap_assessment.coroutine(
        material_gap_found=False,
        rationale="全部可能改变最终判断的问题均已覆盖",
        coverage_audit=_coverage_audit(investigation_id),
        runtime=_runtime(context, state, "gap-clear"),
    )
    _apply(state, assessment.update)

    report = build_adaptive_coverage_report(state, context)
    assert report.status == "completed", report.model_dump()
    assert report.actual_investigation_count == 1
    assert report.executed_search_calls == 1
    assert (
        await AcmReviewHarnessMiddleware(model=object()).prepare_pre_final_interruption(
            state=state,
            context=context,
            candidate_body="覆盖完成后的答案",
        )
        is None
    )

    state["evidence_store"]["E2"] = {
        "evidence_id": "E2",
        "content_hash": "hash-e2",
        "raw_text": "审计后新打开的相邻证据",
    }
    stale = build_adaptive_coverage_report(state, context)
    assert stale.gap_assessment_status == "stale"
    assert stale.status == "incomplete"


@pytest.mark.asyncio
async def test_insufficient_and_dismissed_require_explicit_justification() -> None:
    context = _context()
    state = _base_state()
    agenda = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一")],
        runtime=_runtime(context, state, "agenda-closure-rules"),
    )
    _apply(state, agenda.update)
    investigation_id = state["adaptive_agenda"].items[0].investigation_id

    no_probe = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="insufficient",
        residual_uncertainty="知识库没有明确边界",
        closure_reason="已尝试检索但不足",
        runtime=_runtime(context, state, "insufficient-no-probe"),
    )
    assert "真实检索尝试" in no_probe.update["messages"][0].content

    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q1",
            "query_id": "Q1",
            "investigation_id": investigation_id,
            "uncovered_aspect": "问题一的主要依据",
            "retrieval_intent": "source_discovery",
            "route_key": "route-1",
            "status": "technical_failed",
        }
    ]
    insufficient = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="insufficient",
        residual_uncertainty="技术失败后仍无法确定",
        closure_reason="记录技术失败并保留不确定性",
        runtime=_runtime(context, state, "insufficient-valid"),
    )
    _apply(state, insufficient.update)
    assert state["investigations"][0].status == "insufficient"

    missing_reason = await update_acm_investigation.coroutine(
        investigation_id=investigation_id,
        status="dismissed",
        closure_reason="",
        runtime=_runtime(context, state, "dismissed-no-reason"),
    )
    assert "dismissed 必须" in missing_reason.update["messages"][0].content


@pytest.mark.asyncio
async def test_gap_assessment_atomically_appends_new_investigation() -> None:
    context = _context()
    state = _base_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[_draft("问题一", "范围一")],
        runtime=_runtime(context, state, "agenda-gap"),
    )
    _apply(state, agenda_result.update)
    investigation = state["investigations"][0]
    investigation_id = investigation.investigation_id
    state["investigations"] = [investigation.model_copy(update={"status": "insufficient"})]
    state["query_records"] = [
        {
            "query_id": "Q1",
            "tool_call_id": "query-1",
            "investigation_id": investigation_id,
            "query_text": "查询一",
            "reason": "首次检索",
            "started_at": "2026-08-25T00:00:00Z",
            "elapsed_ms": 1,
            "status": "success_empty",
        }
    ]
    obligation = state["adaptive_agenda"].items[0].evidence_obligations[0]
    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q1",
            "query_id": "Q1",
            "investigation_id": investigation_id,
            "uncovered_aspect": obligation,
            "retrieval_intent": "source_discovery",
            "route_key": "route-q1",
            "status": "success_empty",
        }
    ]
    state["adaptive_investigation_meta"] = [
        {
            "investigation_id": investigation_id,
            "remaining_aspects": [obligation],
            "updated_at": "2026-08-25T00:00:01Z",
        }
    ]
    state["search_count"] = 1

    result = await submit_coverage_gap_assessment.coroutine(
        material_gap_found=True,
        rationale="发现尚未评估的长期监测问题",
        coverage_audit=_coverage_audit(
            investigation_id,
            gap_dimension="monitoring_followup_and_stop_rules",
        ),
        unsupported_investigation_ids=[investigation_id],
        unsupported_obligations={investigation_id: [obligation]},
        proposed_items=[_draft("长期监测如何安排", "长期随访范围")],
        runtime=_runtime(context, state, "gap-append"),
    )
    _apply(state, result.update)

    assessment = state["adaptive_gap_assessments"][0]
    assert state["adaptive_agenda"].revision == 2
    assert len(state["adaptive_agenda"].items) == 2
    assert len(assessment.proposed_investigation_ids) == 1
    reopened = next(value for value in state["investigations"] if value.investigation_id == investigation_id)
    reopened_meta = next(
        value for value in state["adaptive_investigation_meta"] if value.investigation_id == investigation_id
    )
    assert reopened.status == "open"
    assert obligation in reopened_meta.remaining_aspects
    report = build_adaptive_coverage_report(state, context)
    assert report.gap_assessment_status == "stale"
    assert report.unprobed_investigation_ids == assessment.proposed_investigation_ids


@pytest.mark.asyncio
async def test_adaptive_checkpoint_does_not_loop_without_state_progress() -> None:
    context = _context()
    middleware = AcmReviewHarnessMiddleware(model=object())
    state = _base_state()
    state["review_run_id"] = "run-1"

    first = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="准备提前回答",
    )
    assert first is not None
    assert first.tool_name == "adaptive_coverage_checkpoint"
    state["adaptive_checkpoint_records"] = first.state_update["adaptive_checkpoint_records"]

    second = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="仍然决定回答",
    )
    assert second is None

    state["evidence_store"]["E-PROGRESS"] = {
        "evidence_id": "E-PROGRESS",
        "content_hash": "progress",
        "raw_text": "新证据使状态指纹变化",
    }
    third = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="状态变化后再次提前回答",
    )
    assert third is not None


@pytest.mark.asyncio
async def test_search_guard_exhaustion_preserves_candidate_with_warning() -> None:
    context = _context(max_search_calls=1)
    middleware = AcmReviewHarnessMiddleware(model=object())
    state = _base_state()
    state["search_count"] = 1

    interruption = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="保护预算耗尽后的回答",
    )
    warning = await middleware.prepare_candidate_state(
        state=state,
        context=context,
        candidate_body="保护预算耗尽后的回答",
    )

    assert interruption is None
    assert "contract incomplete" in warning["warnings"][0]


def test_harness_projects_only_adaptive_protocol_tools() -> None:
    context = _context()
    middleware = AcmReviewHarnessMiddleware(model=object())

    assert middleware.project_search_tool("full", context, {}) is search_review_kb_adaptive
    assert middleware.tool_is_visible(
        tool_name="set_investigation_agenda",
        state={},
        context=context,
    )
    assert not middleware.tool_is_visible(
        tool_name="search_review_kb",
        state={},
        context=context,
    )
    assert not middleware.tool_is_visible(
        tool_name="update_investigation",
        state={},
        context=context,
    )
    assert not middleware.tool_is_visible(
        tool_name="extend_investigation_agenda",
        state={},
        context=context,
    )
    assert not middleware.tool_is_visible(
        tool_name="submit_coverage_gap_assessment",
        state={},
        context=context,
    )
    assert not middleware.tool_is_visible(
        tool_name="adaptive_coverage_checkpoint",
        state={},
        context=context,
    )


def test_adaptive_trace_uses_schema_12_and_keeps_actual_counts() -> None:
    context = _context()
    context._acm_prim_atlas = CorpusAtlas(
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash="atlas",
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
        state=_base_state(),
        context=context,
    )

    assert isinstance(trace, MedicationReviewAcmAdaptiveTrace)
    assert trace.schema_version == "12.0"
    assert trace.method_family == "acm-prim-rag-v8"
    assert trace.method_version == ("acm-prim-rag-v8-adaptive-two-track-evidence-v5-query-focus-shadow_top25-vector")
    assert trace.protocol == "adaptive_coverage"
    assert trace.adaptive_coverage_report.actual_investigation_count == 0
    assert trace.budgets["minimum_search_calls"] == 0
    assert trace.budgets["retrieval_fetch_k"] == 25
    assert trace.budgets["agent_visible_k"] == 10
    assert trace.prompt_versions["atlas_navigation"] == ADAPTIVE_ATLAS_NAVIGATION_PROMPT_VERSION
    assert trace.prompt_hashes["atlas_navigation"] == ADAPTIVE_ATLAS_NAVIGATION_PROMPT_HASH
    assert trace.prompt_versions["context_projection"] == ADAPTIVE_CONTEXT_PROJECTION_VERSION
    assert trace.prompt_hashes["context_projection"] == ADAPTIVE_CONTEXT_PROJECTION_HASH
