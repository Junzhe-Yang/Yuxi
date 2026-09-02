from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_acm_bounded.citations import (
    build_citation_verification,
)
from yuxi.agents.buildin.medication_review_acm_bounded.context import (
    MedicationReviewAcmBoundedContext,
)
from yuxi.agents.buildin.medication_review_acm_bounded.context_view import (
    project_messages,
)
from yuxi.agents.buildin.medication_review_acm_bounded.generation import (
    AcmGenerationGuardMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_bounded.graph import (
    MedicationReviewAcmBoundedAgent,
)
from yuxi.agents.buildin.medication_review_acm_bounded.models import (
    ActionDirective,
    ToolOutcome,
)
from yuxi.agents.buildin.medication_review_acm_bounded.prompt import (
    build_bounded_prompt,
)
from yuxi.agents.buildin.medication_review_acm_bounded.tools import (
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
        "directive_id": "DIR-ACTIVE",
        "phase": "REVIEW_ACTIVE_OBLIGATION",
        "state_version": 3,
        "state_fingerprint": "state-3",
        "active_investigation_id": "INV-1",
        "active_obligation": "义务一",
        "allowed_actions": [
            "record_active_obligation_support",
            "read_active_evidence",
            "open_active_evidence",
        ],
        "evidence_aliases": {"E1": "EV-1"},
    }
    values.update(updates)
    return ActionDirective(**values)


def _interaction(call_id: str, result: str) -> list:
    return [
        AIMessage(
            content="这段过程自述不应保留",
            tool_calls=[
                {
                    "name": "search_active_obligation",
                    "args": {"query_text": call_id},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=result,
            tool_call_id=call_id,
            name="search_active_obligation",
        ),
    ]


def _outcome(call_id: str, investigation_id: str, obligation: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=call_id,
        directive_id="DIR-OLD",
        tool_name="search_active_obligation",
        transport_status="COMPLETED",
        semantic_outcome="SUCCESS",
        state_changed=True,
        executed_backend=True,
        state_version_before=1,
        state_version_after=2,
        message_for_model="完成",
        investigation_id=investigation_id,
        obligation=obligation,
        evidence_ids=["EV-1"],
    )


def test_projection_prioritizes_active_scope_and_preserves_pairs() -> None:
    messages = [HumanMessage(content="原始病例")]
    for index in range(5):
        messages.extend(_interaction(f"call-{index}", f"RESULT-{index}-" + "原文" * 200))
    messages.append(AIMessage(content="Let me invoke search " * 200))
    state = {
        "bounded_tool_outcomes": [
            _outcome("call-0", "INV-1", "义务一"),
            *[_outcome(f"call-{index}", "INV-X", "其它义务") for index in range(1, 5)],
        ]
    }

    projected = project_messages(
        messages=messages,
        state=state,
        directive=_directive(),
    )

    call_ids = {call["id"] for value in projected.messages if isinstance(value, AIMessage) for call in value.tool_calls}
    result_ids = {value.tool_call_id for value in projected.messages if isinstance(value, ToolMessage)}
    assert "call-0" in call_ids
    assert "call-1" not in call_ids
    assert call_ids == result_ids
    assert all(value.content == "" for value in projected.messages if isinstance(value, AIMessage))
    assert not any("Let me invoke" in str(value.content) for value in projected.messages)
    assert next(value for value in projected.entries if value.source_id == "call-0").retention_reason == "ACTIVE_SCOPE"
    assert any(value.tool_call_id == "call-0" for value in projected.atoms)


def test_projection_rejects_orphan_tool_call() -> None:
    messages = [
        HumanMessage(content="病例"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_active_obligation",
                    "args": {"query_text": "药物 剂量"},
                    "id": "orphan-call",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    with pytest.raises(ValueError, match="orphan tool-call/result"):
        project_messages(
            messages=messages,
            state={},
            directive=_directive(),
        )


def test_review_prompt_rehydrates_exact_evidence_raw_text_without_mutation() -> None:
    exact = "  精确 Evidence <table>\n第二行，空格也保留。  "
    state = {
        "evidence_store": {
            "EV-1": {
                "evidence_id": "EV-1",
                "content_hash": "hash-1",
                "raw_text": exact,
                "source_document": "指南.md",
                "file_id": "file-1",
                "chunk_id": "chunk-1",
                "chunk_index": 1,
            }
        }
    }

    prompt = build_bounded_prompt(
        state=state,
        context=_context(),
        directive=_directive(),
    )

    assert exact in prompt.evidence_memory
    assert state["evidence_store"]["EV-1"]["raw_text"] == exact
    assert "hash=hash-1" in prompt.evidence_memory

    final_prompt = build_bounded_prompt(
        state=state,
        context=_context(),
        directive=_directive(
            phase="DRAFT_FINAL",
            active_investigation_id=None,
            active_obligation=None,
            allowed_actions=[],
            expected_output_kind="final_answer",
        ),
    )
    assert exact in final_prompt.evidence_memory


def test_citation_rehydration_preserves_exact_raw_text_and_hash() -> None:
    exact = "  最终引用对应的完整原文\n不得压缩。  "
    state = {
        "evidence_store": {
            "EV-1": {
                "evidence_id": "EV-1",
                "content_hash": "hash-1",
                "raw_text": exact,
                "source_document": "指南.md",
                "file_id": "file-1",
                "chunk_id": "chunk-1",
                "chunk_index": 1,
            }
        }
    }

    verification = build_citation_verification(
        "该方案在监测条件下可用。[EV-1]",
        state,
    )

    assert verification.status == "ready"
    assert verification.claims[0].evidence_ids == ["EV-1"]
    assert verification.evidence_snapshots[0].raw_text == exact
    assert verification.evidence_snapshots[0].content_hash == "hash-1"
    assert len(verification.evidence_snapshots[0].raw_text_sha256) == 64


@pytest.mark.asyncio
async def test_generation_guard_isolates_first_output_and_repairs_once() -> None:
    context = _context()
    directive = _directive(
        phase="SEARCH_ACTIVE_OBLIGATION",
        allowed_actions=["search_active_obligation"],
        expected_output_kind="tool_call",
        evidence_aliases={},
    )
    request = ModelRequest(
        model=SimpleNamespace(profile={"max_input_tokens": 262_144}),
        messages=[HumanMessage(content="病例")],
        tools=[search_active_obligation],
        state={"action_directive": directive},
        runtime=SimpleNamespace(context=context),
    )
    responses = iter(
        [
            ModelResponse(result=[AIMessage(content="我将搜索。")]),
            ModelResponse(
                result=[
                    AIMessage(
                        content="修复轮仍然错误地输出了一段结论。",
                        tool_calls=[
                            {
                                "name": "search_active_obligation",
                                "args": {"query_text": "药物 推荐剂量"},
                                "id": "call-repaired",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            ),
        ]
    )

    async def handler(_request):
        return next(responses)

    result = await AcmGenerationGuardMiddleware().awrap_model_call(
        request,
        handler,
    )

    assert isinstance(result, ExtendedModelResponse)
    assert result.model_response.result[0].tool_calls[0]["id"] == "call-repaired"
    assert result.model_response.result[0].content == ""
    aborts = result.command.update["bounded_generation_aborts"]
    assert len(aborts) == 2
    assert aborts[0].raw_output == "我将搜索。"
    assert aborts[0].reason_codes == ["MISSING_REQUIRED_TOOL_CALL"]
    assert aborts[1].reason_codes == ["ACTION_CONTENT_SUPPRESSED"]
    assert "修复轮仍然错误地输出" in aborts[1].raw_output
    assert len(result.command.update["bounded_context_manifests"]) == 1


@pytest.mark.asyncio
async def test_generation_guard_suppresses_valid_action_narration_without_retry() -> None:
    context = _context()
    directive = _directive(
        phase="SEARCH_ACTIVE_OBLIGATION",
        allowed_actions=["search_active_obligation"],
        expected_output_kind="tool_call",
        evidence_aliases={},
    )
    request = ModelRequest(
        model=SimpleNamespace(profile={"max_input_tokens": 262_144}),
        messages=[HumanMessage(content="病例")],
        tools=[search_active_obligation],
        state={"action_directive": directive},
        runtime=SimpleNamespace(context=context),
    )
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        return ModelResponse(
            result=[
                AIMessage(
                    content="调查尚未完成，但这里错误地给出了一大段最终结论。",
                    additional_kwargs={"reasoning_content": "内部过程也不应透出"},
                    tool_calls=[
                        {
                            "name": "search_active_obligation",
                            "args": {"query_text": "方案甲 适用条件"},
                            "id": "call-valid",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    result = await AcmGenerationGuardMiddleware().awrap_model_call(
        request,
        handler,
    )

    assert isinstance(result, ExtendedModelResponse)
    assert calls == 1
    clean_message = result.model_response.result[0]
    assert clean_message.content == ""
    assert clean_message.tool_calls[0]["id"] == "call-valid"
    assert "reasoning_content" not in clean_message.additional_kwargs
    aborts = result.command.update["bounded_generation_aborts"]
    assert len(aborts) == 1
    assert aborts[0].reason_codes == ["ACTION_CONTENT_SUPPRESSED"]
    assert "错误地给出了一大段最终结论" in aborts[0].raw_output
    assert "内部过程也不应透出" in aborts[0].raw_output


def test_live_stream_projection_hides_action_prose_but_keeps_terminal_text() -> None:
    agent = object.__new__(MedicationReviewAcmBoundedAgent)
    action = _directive(
        phase="SEARCH_ACTIVE_OBLIGATION",
        allowed_actions=["search_active_obligation"],
        expected_output_kind="tool_call",
        evidence_aliases={},
    )
    chunk = AIMessageChunk(
        content="调查未完成时生成的最终结论",
        additional_kwargs={"reasoning_content": "不应输出的内部过程"},
    )

    projected = agent.project_stream_message(
        chunk,
        {"action_directive": action},
    )

    assert projected.content == ""
    assert "reasoning_content" not in projected.additional_kwargs

    runtime_context = _context()
    setattr(runtime_context, "_acm_bounded_directive", action)
    assert (
        agent.project_stream_message(
            chunk,
            {},
            context=runtime_context,
        ).content
        == ""
    )

    final = AIMessageChunk(content="真正的最终结论")
    final_directive = _directive(
        phase="DRAFT_FINAL",
        active_investigation_id=None,
        active_obligation=None,
        allowed_actions=[],
        expected_output_kind="final_answer",
    )
    assert (
        agent.project_stream_message(
            final,
            {"action_directive": final_directive},
        ).content
        == "真正的最终结论"
    )

    failure = AIMessageChunk(
        content="流程已明确停止",
        additional_kwargs={"acm_bounded_terminal_failure": True},
    )
    assert (
        agent.project_stream_message(
            failure,
            {"action_directive": action},
        ).content
        == "流程已明确停止"
    )


@pytest.mark.asyncio
async def test_generation_guard_stops_explicitly_after_one_failed_repair() -> None:
    context = _context()
    directive = _directive(
        phase="SEARCH_ACTIVE_OBLIGATION",
        allowed_actions=["search_active_obligation"],
        expected_output_kind="tool_call",
        evidence_aliases={},
    )
    request = ModelRequest(
        model=SimpleNamespace(profile={"max_input_tokens": 262_144}),
        messages=[HumanMessage(content="病例")],
        tools=[search_active_obligation],
        state={"action_directive": directive},
        runtime=SimpleNamespace(context=context),
    )

    async def handler(_request):
        return ModelResponse(result=[AIMessage(content="稍后调用工具。")])

    result = await AcmGenerationGuardMiddleware().awrap_model_call(
        request,
        handler,
    )

    assert isinstance(result, ExtendedModelResponse)
    assert "流程已明确停止" in result.model_response.result[0].content
    assert result.model_response.result[0].additional_kwargs["acm_bounded_terminal_failure"] is True
    assert len(result.command.update["bounded_generation_aborts"]) == 2
    assert context._acm_bounded_generation_exhausted is True
