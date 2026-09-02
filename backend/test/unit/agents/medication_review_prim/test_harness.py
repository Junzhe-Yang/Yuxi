from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

from yuxi.agents.buildin.medication_review_lite.models import (
    AnchorExtractionAudit,
    PlanAnchor,
)
from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
)
from yuxi.agents.buildin.medication_review_prim.extraction import (
    disabled_modifier_audit,
)
from yuxi.agents.buildin.medication_review_prim.harness import (
    ReviewHarnessMiddleware,
    _effective_profile,
    _run_status,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    ModifierExtractionAudit,
    PrimCoverageReport,
    ReflectionReport,
)
from yuxi.agents.buildin.medication_review_prim.tools import coverage_reflection
from yuxi.agents.buildin.medication_review_prim.tools import (
    open_review_evidence,
    search_review_kb_relation,
    update_investigation,
)


def _anchor(element_id: str, span: str, start: int) -> PlanAnchor:
    return PlanAnchor(
        element_id=element_id,
        source_span=span,
        source_start=start,
        source_end=start + len(span),
        label=span,
        kind="explicit_regimen_or_other",
    )


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="方案甲和方案乙")],
        "review_run_id": "run-1",
        "requested_profile": "full",
        "effective_profile": "full",
        "raw_case_text": "方案甲和方案乙",
        "raw_question_hash": "hash",
        "plan_anchors": [
            _anchor("PE001", "方案甲", 0),
            _anchor("PE002", "方案乙", 3),
        ],
        "plan_extraction": AnchorExtractionAudit(
            status="success",
            started_at="2026-01-01T00:00:00Z",
        ),
        "patient_modifiers": [],
        "modifier_extraction": disabled_modifier_audit(),
        "evidence_store": {},
        "query_records": [],
        "investigations": [],
        "deferred_knowledge_calls": [],
        "open_records": [],
        "knowledge_base_snapshot": {"name": "知识库"},
        "search_count": 0,
        "open_count": 0,
        "technical_attempts": 0,
        "reflection_attempted": False,
        "reflection_report": ReflectionReport(enabled=True),
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_default_extension_hooks_are_noops() -> None:
    middleware = ReviewHarnessMiddleware(model=object())
    context = MedicationReviewPrimContext(knowledges=["知识库"])
    state = _state()

    assert await middleware.augment_initial_state(
        state=state,
        update={},
        runtime=SimpleNamespace(context=context),
    ) == {}
    assert await middleware.prepare_candidate_state(
        state=state,
        context=context,
        candidate_body="答案",
    ) == {}
    assert middleware.augment_investigation_memory(
        state=state,
        context=context,
        memory_text="原记忆",
    ) == "原记忆"
    assert middleware.project_model_messages(
        messages=state["messages"],
        state=state,
        context=context,
    ) is state["messages"]
    assert middleware.build_reflection_supplement(
        state=state,
        context=context,
    ) == ""
    assert middleware.reflection_state_update(
        state=state,
        context=context,
    ) == {}
    assert middleware.project_search_tool(
        "full",
        context=context,
        state=state,
    ) is search_review_kb_relation


@pytest.mark.asyncio
async def test_model_message_projection_is_applied_only_to_request_copy() -> None:
    class ProjectingHarness(ReviewHarnessMiddleware):
        def project_model_messages(self, *, messages, state, context):
            del messages, state, context
            return [HumanMessage(content="投影后的模型上下文")]

    context = MedicationReviewPrimContext(knowledges=["知识库"])
    state = _state()
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=SimpleNamespace(context=context),
        tools=[update_investigation],
    )

    async def handler(model_request):
        assert model_request.messages[0].content == "投影后的模型上下文"
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_investigation",
                            "args": {},
                            "id": "update-one",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    await ProjectingHarness(model=object()).awrap_model_call(request, handler)

    assert state["messages"][0].content == "方案甲和方案乙"


@pytest.mark.asyncio
async def test_full_profile_uses_uninvestigated_plan_for_soft_reflection() -> None:
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
    )
    state = _state()
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=SimpleNamespace(context=context),
        tools=[coverage_reflection],
    )
    seen_tools = []

    async def handler(model_request):
        seen_tools.extend(model_request.tools)
        return ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "②【逐项判断】\n"
                        "■ 【PE001】方案甲\n判断：证据不足。\n\n"
                        "③【正面判断汇总】\n无。\n\n"
                        "④【负面判断汇总】\n无。\n\n"
                        "⑤【综合建议】\n复核。"
                    )
                )
            ]
        )

    result = await ReviewHarnessMiddleware(model=object()).awrap_model_call(
        request,
        handler,
    )

    assert isinstance(result, ExtendedModelResponse)
    synthetic = result.model_response.result[0]
    assert synthetic.tool_calls[0]["name"] == "coverage_reflection"
    assert result.command.update["reflection_attempted"] is True
    report = result.command.update["reflection_report"]
    assert report.uninvestigated_plan_ids_before == ["PE001", "PE002"]
    assert seen_tools == []


@pytest.mark.asyncio
async def test_degraded_section_parse_still_triggers_soft_reflection() -> None:
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
    )
    state = _state()
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=SimpleNamespace(context=context),
        tools=[coverage_reflection],
    )

    async def handler(_model_request):
        return ModelResponse(
            result=[AIMessage(content=("未按章节输出，但包含一个标题：\n" "■ 【PE001】方案甲\n判断：证据不足。"))]
        )

    result = await ReviewHarnessMiddleware(model=object()).awrap_model_call(
        request,
        handler,
    )

    assert isinstance(result, ExtendedModelResponse)
    synthetic = result.model_response.result[0]
    assert synthetic.tool_calls[0]["name"] == "coverage_reflection"
    assert result.command.update["reflection_report"].triggered is True


def test_empty_modifier_result_keeps_full_investigation_profile() -> None:
    plan_audit = AnchorExtractionAudit(
        status="success",
        started_at="2026-01-01T00:00:00Z",
    )
    modifier_audit = ModifierExtractionAudit(
        status="no_valid_modifier",
        started_at="2026-01-01T00:00:00Z",
    )

    assert _effective_profile(
        requested="full",
        plan_audit=plan_audit,
        modifier_audit=modifier_audit,
    ) == "full"
    assert _run_status(
        answer_body="有效回答",
        plan_audit=plan_audit,
        modifier_audit=modifier_audit,
        coverage=PrimCoverageReport(),
        errors=[],
    ) == "completed"


def test_failed_auxiliary_extraction_does_not_disable_full_agent_loop() -> None:
    failed_plan = AnchorExtractionAudit(
        status="failed",
        started_at="2026-01-01T00:00:00Z",
        error_type="ProviderError",
        error_message="temporary failure",
    )
    failed_modifier = ModifierExtractionAudit(
        status="failed",
        started_at="2026-01-01T00:00:00Z",
        error_type="ProviderError",
        error_message="temporary failure",
    )

    assert _effective_profile(
        requested="full",
        plan_audit=failed_plan,
        modifier_audit=failed_modifier,
    ) == "full"
    assert _effective_profile(
        requested="m3",
        plan_audit=failed_plan,
        modifier_audit=failed_modifier,
    ) == "m3"


@pytest.mark.asyncio
async def test_exhausted_knowledge_budgets_hide_read_tools_from_next_model_round() -> None:
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
        max_search_calls=8,
        max_open_calls=2,
    )
    state = _state()
    state.update({"search_count": 8, "open_count": 2})
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=SimpleNamespace(context=context),
        tools=[
            search_review_kb_relation,
            open_review_evidence,
            update_investigation,
            coverage_reflection,
        ],
    )
    visible_names = []

    async def handler(model_request):
        visible_names.extend(value.name for value in model_request.tools)
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_investigation",
                            "args": {
                                "investigation_id": "INV-ONE",
                                "status": "insufficient",
                            },
                            "id": "update-one",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    await ReviewHarnessMiddleware(model=object()).awrap_model_call(
        request,
        handler,
    )

    assert visible_names == ["update_investigation"]
