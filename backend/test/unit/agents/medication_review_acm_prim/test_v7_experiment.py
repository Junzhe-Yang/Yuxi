from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
    validate_acm_context,
)
from yuxi.agents.buildin.medication_review_acm_prim.experiment import (
    build_v7_contract_memory,
    build_v7_contract_report,
    configure_v7_retrieval,
)
from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    AcmReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_prim.models import (
    V7AgendaItemDraft,
    V7CheckpointRecord,
    V7ProbeRecord,
)
from yuxi.agents.buildin.medication_review_acm_prim.tools import (
    search_review_kb_acm_dispatch,
    set_investigation_agenda,
    update_acm_investigation,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrieverSelection,
    ensure_runtime_resources,
)


def _context(
    *,
    arm: str = "a1",
    depth: str = "shadow_top25",
    retriever=None,
) -> MedicationReviewAcmPrimContext:
    async def empty_retriever(_query: str, **_kwargs):
        return []

    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        v7_experiment_arm=arm,
        v7_retrieval_depth=depth,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
    )
    configure_v7_retrieval(context)
    ensure_runtime_resources(context)
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever or empty_retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1"}
    return context


def _runtime(context, state, tool_call_id="call-1"):
    return SimpleNamespace(
        context=context,
        state=state,
        tool_call_id=tool_call_id,
    )


def _agenda_state() -> tuple[MedicationReviewAcmPrimContext, dict]:
    context = _context(arm="a2_k2", depth="top10")
    state = {
        "plan_anchors": [
            {"element_id": "PE001"},
            {"element_id": "PE002"},
        ],
        "patient_modifiers": [{"modifier_id": "PM001"}],
        "investigations": [],
        "query_records": [],
        "evidence_store": {},
        "search_count": 0,
    }
    return context, state


def test_acm_defaults_keep_a0_but_raise_runtime_ceiling() -> None:
    context = MedicationReviewAcmPrimContext(knowledges=["知识库"])

    validate_acm_context(context)

    assert context.v7_experiment_arm == "a0"
    assert context.v7_retrieval_depth == "top10"
    assert context.max_search_calls == 50


def test_v7_contract_rejects_budget_below_minimum() -> None:
    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        v7_experiment_arm="a1",
        max_search_calls=5,
    )

    with pytest.raises(ValueError, match="大于等于 6"):
        validate_acm_context(context)


def test_v7_context_accepts_high_runtime_ceiling() -> None:
    context = MedicationReviewAcmPrimContext(
        knowledges=["知识库"],
        v7_experiment_arm="a2_k3",
        max_search_calls=100,
    )

    validate_acm_context(context)

    assert context.max_search_calls == 100


def test_retrieval_depth_prompt_is_independent_of_effort_contract() -> None:
    context = _context(arm="a0", depth="visible_top25")

    report = build_v7_contract_report({}, context)

    assert report.status == "disabled"
    assert "展示 25 个候选块" in build_v7_contract_memory(report)


@pytest.mark.asyncio
async def test_shadow_top25_keeps_tail_out_of_evidence_store() -> None:
    async def retriever(_query: str, **kwargs):
        assert kwargs["final_top_k"] == 25
        return [
            {
                "content": f"片段 {index}",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": f"chunk-{index}",
                    "chunk_index": index,
                },
                "score": 1 - index / 100,
            }
            for index in range(1, 26)
        ]

    context = _context(retriever=retriever)
    state = {
        "plan_anchors": [],
        "patient_modifiers": [],
        "investigations": [],
        "evidence_store": {},
        "search_count": 0,
    }
    result = await search_review_kb_acm_dispatch.coroutine(
        query_text="治疗方案 调整",
        reason="核查依据",
        runtime=_runtime(context, state),
    )

    query = result.update["query_records"][0]
    diagnostic = result.update["v7_retrieval_records"][0]
    assert query.retained_count == 10
    assert len(result.update["evidence_store"]) == 10
    assert diagnostic["fetch_k"] == 25
    assert diagnostic["visible_k"] == 10
    assert len(diagnostic["candidates"]) == 25
    assert diagnostic["candidates"][10]["chunk_id"] == "chunk-11"


@pytest.mark.asyncio
async def test_visible_top25_exposes_all_candidates() -> None:
    async def retriever(_query: str, **_kwargs):
        return [
            {
                "content": f"片段 {index}",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": f"chunk-{index}",
                },
            }
            for index in range(1, 26)
        ]

    context = _context(depth="visible_top25", retriever=retriever)
    result = await search_review_kb_acm_dispatch.coroutine(
        query_text="治疗方案 调整",
        reason="可见深度消融",
        runtime=_runtime(
            context,
            {
                "plan_anchors": [],
                "patient_modifiers": [],
                "investigations": [],
                "evidence_store": {},
                "search_count": 0,
            },
        ),
    )

    assert result.update["query_records"][0].retained_count == 25
    assert len(result.update["evidence_store"]) == 25
    assert result.update["v7_retrieval_records"][0]["visible_k"] == 25


@pytest.mark.asyncio
async def test_agenda_requires_exact_k_and_all_plan_anchors() -> None:
    context, state = _agenda_state()
    incomplete = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="核查方案一",
                focus_plan_ids=["PE001"],
                distinct_scope="方案选择",
            ),
            V7AgendaItemDraft(
                question="核查患者因素",
                focus_modifier_ids=["PM001"],
                distinct_scope="患者条件",
            ),
        ],
        runtime=_runtime(context, state, "agenda-bad"),
    )
    assert "尚未覆盖 PlanAnchor：PE002" in incomplete.update["messages"][0].content
    assert "v7_agenda" not in incomplete.update

    complete = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="核查方案一的适用性",
                focus_plan_ids=["PE001"],
                distinct_scope="方案一",
            ),
            V7AgendaItemDraft(
                question="核查方案二及患者条件",
                focus_plan_ids=["PE002"],
                focus_modifier_ids=["PM001", "PM-UNKNOWN"],
                distinct_scope="方案二与患者条件",
            ),
        ],
        runtime=_runtime(context, state, "agenda-good"),
    )
    assert complete.update["v7_agenda"].required_count == 2
    assert len(complete.update["investigations"]) == 2
    assert complete.update["v7_agenda"].items[1].focus_modifier_ids == [
        "PM001"
    ]


@pytest.mark.asyncio
async def test_a2_enforces_breadth_first_probe_order() -> None:
    context, state = _agenda_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="问题一",
                focus_plan_ids=["PE001"],
                distinct_scope="方面一",
            ),
            V7AgendaItemDraft(
                question="问题二",
                focus_plan_ids=["PE002"],
                distinct_scope="方面二",
            ),
        ],
        runtime=_runtime(context, state, "agenda-order"),
    )
    agenda = agenda_result.update["v7_agenda"]
    first_id, second_id = [value.investigation_id for value in agenda.items]
    state.update(
        {
            "v7_agenda": agenda,
            "investigations": agenda_result.update["investigations"],
            "v7_probe_records": [
                V7ProbeRecord(
                    probe_record_id="PROBE-Q1",
                    query_id="Q1",
                    investigation_id=first_id,
                    probe_pass="initial_probe",
                    status="success",
                )
            ],
            "query_records": [
                {
                    "query_id": "Q1",
                    "tool_call_id": "call-q1",
                    "investigation_id": first_id,
                    "query_text": "查询一",
                    "reason": "初始探查",
                    "started_at": "2026-01-01T00:00:00Z",
                    "elapsed_ms": 1,
                    "status": "success",
                }
            ],
            "search_count": 1,
        }
    )

    rejected = await search_review_kb_acm_dispatch.coroutine(
        query_text="问题一 另一表述",
        reason="过早进入互补轮",
        investigation_id=first_id,
        probe_pass="complementary_probe",
        uncovered_aspect="另一个方面",
        runtime=_runtime(context, state, "search-rejected"),
    )

    assert "当前应执行 initial_probe" in rejected.update["messages"][0].content
    assert second_id in rejected.update["messages"][0].content
    assert "query_records" not in rejected.update


@pytest.mark.asyncio
async def test_complementary_probe_rejects_exact_initial_query() -> None:
    context, state = _agenda_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="问题一",
                focus_plan_ids=["PE001"],
                distinct_scope="方面一",
            ),
            V7AgendaItemDraft(
                question="问题二",
                focus_plan_ids=["PE002"],
                distinct_scope="方面二",
            ),
        ],
        runtime=_runtime(context, state, "agenda-duplicate"),
    )
    agenda = agenda_result.update["v7_agenda"]
    first_id, second_id = [value.investigation_id for value in agenda.items]
    state.update(
        {
            "v7_agenda": agenda,
            "investigations": agenda_result.update["investigations"],
            "v7_probe_records": [
                V7ProbeRecord(
                    probe_record_id="PROBE-Q1",
                    query_id="Q1",
                    investigation_id=first_id,
                    probe_pass="initial_probe",
                    status="success",
                ),
                V7ProbeRecord(
                    probe_record_id="PROBE-Q2",
                    query_id="Q2",
                    investigation_id=second_id,
                    probe_pass="initial_probe",
                    status="success_empty",
                ),
            ],
            "query_records": [
                {
                    "query_id": "Q1",
                    "tool_call_id": "call-q1",
                    "investigation_id": first_id,
                    "query_text": "同一 条 查询",
                    "reason": "首轮",
                    "started_at": "2026-01-01T00:00:01Z",
                    "elapsed_ms": 1,
                    "status": "success",
                },
                {
                    "query_id": "Q2",
                    "tool_call_id": "call-q2",
                    "investigation_id": second_id,
                    "query_text": "另一个查询",
                    "reason": "首轮",
                    "started_at": "2026-01-01T00:00:02Z",
                    "elapsed_ms": 1,
                    "status": "success_empty",
                },
            ],
            "search_count": 2,
        }
    )

    rejected = await search_review_kb_acm_dispatch.coroutine(
        query_text="  同一 条   查询 ",
        reason="互补检索",
        investigation_id=first_id,
        probe_pass="complementary_probe",
        uncovered_aspect="需要核查另一边界",
        runtime=_runtime(context, state, "search-duplicate"),
    )

    assert "不能逐字重复" in rejected.update["messages"][0].content
    assert "query_records" not in rejected.update


@pytest.mark.asyncio
async def test_required_investigation_cannot_close_before_two_probes() -> None:
    context, state = _agenda_state()
    agenda_result = await set_investigation_agenda.coroutine(
        items=[
            V7AgendaItemDraft(
                question="问题一",
                focus_plan_ids=["PE001"],
                distinct_scope="方面一",
            ),
            V7AgendaItemDraft(
                question="问题二",
                focus_plan_ids=["PE002"],
                distinct_scope="方面二",
            ),
        ],
        runtime=_runtime(context, state, "agenda-close"),
    )
    agenda = agenda_result.update["v7_agenda"]
    first_id = agenda.items[0].investigation_id
    state.update(
        {
            "v7_agenda": agenda,
            "investigations": agenda_result.update["investigations"],
            "v7_probe_records": [],
        }
    )

    result = await update_acm_investigation.coroutine(
        investigation_id=first_id,
        status="insufficient",
        working_note="想提前关闭",
        runtime=_runtime(context, state, "close-early"),
    )

    assert "尚未完成 initial_probe" in result.update["messages"][0].content
    assert "investigations" not in result.update


def test_contract_counts_success_empty_but_not_technical_failure() -> None:
    context = _context(arm="a1", depth="top10")
    records = []
    for index, status in enumerate(
        ["success", "success_empty", "technical_failed"],
        start=1,
    ):
        records.append(
            {
                "query_id": f"Q{index}",
                "tool_call_id": f"call-{index}",
                "query_text": f"查询 {index}",
                "reason": "核查",
                "started_at": f"2026-01-01T00:00:0{index}Z",
                "elapsed_ms": 1,
                "status": status,
            }
        )

    report = build_v7_contract_report(
        {"query_records": records, "search_count": 3},
        context,
    )

    assert report.executed_search_calls == 3
    assert report.successful_search_calls == 2
    assert report.incomplete_reasons == ["minimum_search_calls_not_met"]


@pytest.mark.asyncio
async def test_checkpoint_is_recoverable_and_never_loops_without_progress() -> None:
    context = _context(arm="a1", depth="top10")
    middleware = AcmReviewHarnessMiddleware(model=object())
    state = {
        "review_run_id": "run-1",
        "query_records": [],
        "search_count": 0,
    }

    first = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="准备提前回答",
    )
    assert first is not None
    assert first.tool_name == "v7_effort_checkpoint"

    checkpoint: V7CheckpointRecord = first.state_update[
        "v7_checkpoint_records"
    ][0]
    state["v7_checkpoint_records"] = [checkpoint]
    second = await middleware.prepare_pre_final_interruption(
        state=state,
        context=context,
        candidate_body="仍然决定回答",
    )

    assert second is None
    warning_update = await middleware.prepare_candidate_state(
        state=state,
        context=context,
        candidate_body="仍然决定回答",
    )
    assert "contract incomplete" in warning_update["warnings"][0]
