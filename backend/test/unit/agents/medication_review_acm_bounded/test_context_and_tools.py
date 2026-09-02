from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_acm_bounded.context import (
    MODEL_CONTEXT_WINDOW_TOKENS,
    MedicationReviewAcmBoundedContext,
    validate_bounded_context,
)
from yuxi.agents.buildin.medication_review_acm_bounded.context_view import (
    AcmBoundedModelViewMiddleware,
    ContextCapacityError,
)
from yuxi.agents.buildin.medication_review_acm_bounded import graph as graph_module
from yuxi.agents.buildin.medication_review_acm_bounded.graph import (
    MedicationReviewAcmBoundedAgent,
    _declare_context_window,
)
from yuxi.agents.buildin.medication_review_acm_bounded.models import (
    ActionDirective,
)
from yuxi.agents.buildin.medication_review_acm_bounded.tools import (
    propose_initial_agenda,
    record_active_obligation_support,
    search_active_obligation,
)


def _context() -> MedicationReviewAcmBoundedContext:
    return MedicationReviewAcmBoundedContext(
        knowledges=["知识库"],
        max_search_calls=50,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
    )


def _directive(**updates) -> ActionDirective:
    values = {
        "directive_id": "DIR-TEST",
        "phase": "SEARCH_ACTIVE_OBLIGATION",
        "state_version": 0,
        "state_fingerprint": "state",
        "active_investigation_id": "INV-1",
        "active_obligation": "当前药物的推荐剂量",
        "allowed_actions": ["search_active_obligation"],
        "allowed_routes": ["global"],
        "bound_retrieval_scope": "global",
        "bound_retrieval_intent": "source_discovery",
    }
    values.update(updates)
    return ActionDirective(**values)


def test_context_has_one_physical_capacity_and_no_phase_admission_limit() -> None:
    context = _context()

    validate_bounded_context(context)

    assert context.model_context_window_tokens == MODEL_CONTEXT_WINDOW_TOKENS
    assert MODEL_CONTEXT_WINDOW_TOKENS == 262_144
    assert not hasattr(context, "admission_line_tokens")
    assert not hasattr(context, "phase_hard_limits")
    assert context.phase_observation_targets["SEARCH_ACTIVE_OBLIGATION"] == 32_000


def test_model_context_declaration_preserves_verified_smaller_provider_limit() -> None:
    context = _context()
    model = SimpleNamespace(profile={"max_input_tokens": 131_072})

    declared = _declare_context_window(model, context)

    assert declared.profile["max_input_tokens"] == 131_072
    assert context._acm_bounded_provider_context_window_tokens == 131_072
    assert context._acm_bounded_context_window_verified is True


def test_model_context_declaration_uses_256k_when_provider_profile_is_missing() -> None:
    context = _context()
    model = SimpleNamespace()

    declared = _declare_context_window(model, context)

    assert declared.profile["max_input_tokens"] == 262_144
    assert context._acm_bounded_provider_context_window_tokens == 262_144
    assert context._acm_bounded_context_window_verified is False


@pytest.mark.asyncio
async def test_graph_can_be_rebuilt_without_runtime_context_for_state_reads(
    monkeypatch,
) -> None:
    """Chat persistence rebuilds the graph without the execution AgentConfig."""
    agent = object.__new__(MedicationReviewAcmBoundedAgent)

    async def no_checkpointer():
        return None

    monkeypatch.setattr(agent, "_get_checkpointer", no_checkpointer)
    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda _model: SimpleNamespace(profile={}),
    )
    monkeypatch.setattr(graph_module, "create_agent", lambda **kwargs: kwargs)

    graph = await agent.get_graph()

    assert graph["context_schema"] is MedicationReviewAcmBoundedContext
    assert any(middleware.__class__.__name__ == "AcmBoundedHarnessMiddleware" for middleware in graph["middleware"])

    with pytest.raises(ValueError, match="必须且只能选择一个 Milvus 知识库"):
        validate_bounded_context(MedicationReviewAcmBoundedContext())


def test_bounded_tool_schemas_do_not_make_model_echo_controller_state() -> None:
    agenda_schema = propose_initial_agenda.tool_call_schema.model_json_schema()
    search_schema = search_active_obligation.tool_call_schema.model_json_schema()
    support_schema = record_active_obligation_support.tool_call_schema.model_json_schema()

    assert agenda_schema["properties"]["items"]["minItems"] == 1
    assert "maxItems" not in agenda_schema["properties"]["items"]
    assert set(search_schema["required"]) == {"query_text"}
    assert set(search_schema["properties"]) == {"query_text"}
    assert not {
        "investigation_id",
        "uncovered_aspect",
        "retrieval_intent",
        "file_id",
    }.intersection(search_schema["properties"])
    assert set(support_schema["required"]) == {
        "evidence_aliases",
        "verdict",
        "rationale",
    }
    assert "investigation_id" not in support_schema["properties"]
    assert "obligation" not in support_schema["properties"]


@pytest.mark.asyncio
async def test_phase_target_only_warns_and_does_not_block_model_call() -> None:
    context = _context()
    directive = _directive(
        phase="PROPOSE_INITIAL_AGENDA",
        allowed_actions=["propose_initial_agenda"],
        active_investigation_id=None,
        active_obligation=None,
        allowed_routes=[],
        bound_retrieval_scope=None,
        bound_retrieval_intent=None,
    )
    request = ModelRequest(
        model=SimpleNamespace(profile={"max_input_tokens": 262_144}),
        messages=[HumanMessage(content="病例事实" * 30_000)],
        tools=[propose_initial_agenda],
        state={"action_directive": directive},
        runtime=SimpleNamespace(context=context),
    )
    called = False

    async def handler(model_request):
        nonlocal called
        called = True
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "propose_initial_agenda",
                            "args": {"items": []},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    response = await AcmBoundedModelViewMiddleware().awrap_model_call(
        request,
        handler,
    )

    assert called is True
    manifest = response.command.update["bounded_context_manifests"][0]
    assert manifest.above_observation_target is True
    assert manifest.projected_total_tokens < manifest.configured_context_window
    assert manifest.reduction_actions == []


@pytest.mark.asyncio
async def test_verified_smaller_provider_capacity_is_enforced_explicitly() -> None:
    context = _context()
    directive = _directive(
        phase="PROPOSE_INITIAL_AGENDA",
        allowed_actions=["propose_initial_agenda"],
        active_investigation_id=None,
        active_obligation=None,
        allowed_routes=[],
        bound_retrieval_scope=None,
        bound_retrieval_intent=None,
    )
    request = ModelRequest(
        model=SimpleNamespace(profile={"max_input_tokens": 1_000}),
        messages=[HumanMessage(content="病例事实" * 2_000)],
        tools=[propose_initial_agenda],
        state={"action_directive": directive},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_request):
        raise AssertionError("capacity failure must happen before provider call")

    with pytest.raises(ContextCapacityError, match="cannot fit provider capacity"):
        await AcmBoundedModelViewMiddleware().awrap_model_call(request, handler)


@pytest.mark.asyncio
async def test_semantic_rejection_is_error_status_and_same_action_is_blocked() -> None:
    context = _context()
    directive = _directive()
    state = {
        "action_directive": directive,
        "bounded_state_version": 0,
        "bounded_action_attempts": [],
    }
    runtime = SimpleNamespace(
        context=context,
        state=state,
        tool_call_id="call-first",
    )

    first = await search_active_obligation.coroutine(
        query_text="药物 剂量 监测",
        runtime=runtime,
    )
    first_message = first.update["messages"][0]
    first_outcome = first.update["bounded_tool_outcomes"][0]
    assert isinstance(first_message, ToolMessage)
    assert first_message.status == "error"
    assert first_outcome.transport_status == "COMPLETED"
    assert first_outcome.semantic_outcome == "INVALID_ARGUMENT"
    assert first_outcome.reason_code == "QUERY_SHAPE_INVALID"
    assert first_outcome.executed_backend is False

    state["bounded_action_attempts"] = first.update["bounded_action_attempts"]
    runtime.tool_call_id = "call-second"
    second = await search_active_obligation.coroutine(
        query_text="药物 剂量 监测",
        runtime=runtime,
    )
    second_outcome = second.update["bounded_tool_outcomes"][0]
    assert second_outcome.reason_code == "REPEATED_REJECTED_ACTION"
    assert second_outcome.executed_backend is False
