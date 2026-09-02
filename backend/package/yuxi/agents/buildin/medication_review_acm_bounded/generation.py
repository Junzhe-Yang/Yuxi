from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    message_text,
)
from yuxi.utils.datetime_utils import utc_isoformat

from .context import MedicationReviewAcmBoundedContext
from .context_view import build_request_manifest
from .models import (
    ActionDirective,
    GenerationAbortRecord,
    MedicationReviewAcmBoundedState,
)

GENERATION_GUARD_VERSION = "acm-bounded-generation-guard-v1"


def _last_ai_message(response: ModelResponse) -> AIMessage | None:
    return next(
        (value for value in reversed(response.result) if isinstance(value, AIMessage)),
        None,
    )


def _raw_output(message: AIMessage | None) -> str:
    if message is None:
        return ""
    body = message_text(message)
    reasoning = {
        key: message.additional_kwargs[key]
        for key in ("reasoning_content", "reasoning", "analysis")
        if message.additional_kwargs.get(key) not in (None, "")
    }
    if not message.tool_calls and not getattr(message, "invalid_tool_calls", None) and not reasoning:
        return body
    payload = {
        "content": body,
        "tool_calls": message.tool_calls,
        "invalid_tool_calls": getattr(message, "invalid_tool_calls", []),
        **reasoning,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _output_tokens(message: AIMessage | None) -> int:
    if message is None:
        return 0
    usage = message.usage_metadata
    if isinstance(usage, dict):
        for key in ("output_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
    return count_tokens_approximately([message])


def _has_action_narration(
    message: AIMessage | None,
    directive: ActionDirective,
) -> bool:
    if message is None or directive.expected_output_kind != "tool_call" or not message.tool_calls:
        return False
    if message_text(message).strip():
        return True
    return any(
        isinstance(message.additional_kwargs.get(key), str) and message.additional_kwargs[key].strip()
        for key in ("reasoning_content", "reasoning", "analysis")
    )


def _without_action_narration(
    response: ModelResponse,
    message: AIMessage,
) -> ModelResponse:
    additional_kwargs = dict(message.additional_kwargs)
    for key in ("reasoning_content", "reasoning", "analysis"):
        additional_kwargs.pop(key, None)
    sanitized = message.model_copy(update={"content": "", "additional_kwargs": additional_kwargs})
    return ModelResponse(
        result=[sanitized if value is message else value for value in response.result],
        structured_response=response.structured_response,
    )


def _repetition_reason(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 600:
        return None

    lines = [re.sub(r"\s+", " ", value).strip() for value in text.splitlines()]
    meaningful_lines = [value for value in lines if len(value) >= 12]
    if meaningful_lines and Counter(meaningful_lines).most_common(1)[0][1] >= 5:
        return "REPEATED_LINE"

    shingles = [compact[index : index + 24] for index in range(0, len(compact) - 23, 8)]
    if len(shingles) >= 60:
        counts = Counter(shingles)
        highest = counts.most_common(1)[0][1]
        diversity = len(counts) / len(shingles)
        if highest >= 12 or diversity < 0.35:
            return "REPETITIVE_CHARACTER_PATTERN"
    return None


def generation_anomalies(
    message: AIMessage | None,
    directive: ActionDirective,
    context: MedicationReviewAcmBoundedContext,
) -> list[str]:
    reasons: list[str] = []
    if message is None:
        return ["MISSING_AI_MESSAGE"]

    tool_calls = list(message.tool_calls or [])
    invalid_calls = list(getattr(message, "invalid_tool_calls", None) or [])
    if invalid_calls:
        reasons.append("INVALID_TOOL_CALL_ENCODING")
    if directive.expected_output_kind == "tool_call" and not tool_calls:
        reasons.append("MISSING_REQUIRED_TOOL_CALL")
    if directive.expected_output_kind == "final_answer" and tool_calls:
        reasons.append("TOOL_CALL_DURING_FINAL_DRAFT")
    if len(tool_calls) > 1:
        reasons.append("MULTIPLE_TOOL_CALLS")
    illegal = [
        str(value.get("name") or "")
        for value in tool_calls
        if str(value.get("name") or "") not in directive.allowed_actions
    ]
    if illegal:
        reasons.append("ILLEGAL_TOOL_CALL:" + ",".join(illegal))

    output_tokens = _output_tokens(message)
    output_limit = (
        context.final_output_tokens if directive.phase == "DRAFT_FINAL" else context.action_output_absolute_limit
    )
    if output_tokens > output_limit:
        reasons.append("OUTPUT_BUDGET_EXCEEDED")
    repetition = _repetition_reason(message_text(message))
    if repetition:
        reasons.append(repetition)
    return reasons


def _abort_record(
    *,
    message: AIMessage | None,
    directive: ActionDirective,
    model_call_id: str,
    reasons: list[str],
    retry_number: int,
) -> GenerationAbortRecord:
    return GenerationAbortRecord(
        abort_id=f"ABORT-{uuid.uuid4().hex[:16].upper()}",
        model_call_id=model_call_id,
        directive_id=directive.directive_id,
        phase=directive.phase,
        reason_codes=reasons,
        output_tokens=_output_tokens(message),
        retry_number=retry_number,
        raw_output=_raw_output(message),
        created_at=utc_isoformat(),
    )


def _repair_prompt(request: ModelRequest, directive: ActionDirective) -> SystemMessage:
    original = request.system_message
    original_text = message_text(original) if original is not None else ""
    if directive.expected_output_kind == "tool_call":
        repair = (
            "【生成修复】上一响应已被隔离。不要解释、复述计划或输出思维过程；"
            "只生成一次当前允许的合法工具调用。allowed_actions=" + ",".join(directive.allowed_actions)
        )
    else:
        repair = (
            "【生成修复】上一响应已被隔离。直接输出一次完整、无循环的最终审查正文；不要调用工具，不要输出内部过程。"
        )
    return SystemMessage(content=f"{original_text}\n\n{repair}".strip())


class AcmGenerationGuardMiddleware(AgentMiddleware[MedicationReviewAcmBoundedState, MedicationReviewAcmBoundedContext]):
    """Isolate one malformed generation, repair once, then fail explicitly."""

    state_schema = MedicationReviewAcmBoundedState

    async def awrap_model_call(
        self,
        request: ModelRequest[MedicationReviewAcmBoundedContext],
        handler: Callable[
            [ModelRequest[MedicationReviewAcmBoundedContext]],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse | ExtendedModelResponse:
        context = request.runtime.context
        raw_directive = (request.state.get("action_directive") if isinstance(request.state, dict) else None) or getattr(
            context, "_acm_bounded_directive", None
        )
        if raw_directive is None:
            raise RuntimeError("ACM Bounded generation guard 缺少 ActionDirective")
        directive = (
            raw_directive
            if isinstance(raw_directive, ActionDirective)
            else ActionDirective.model_validate(raw_directive)
        )
        setattr(context, "_acm_bounded_generation_exhausted", False)
        setattr(context, "_acm_bounded_pending_aborts", [])

        response = await handler(request)
        first_message = _last_ai_message(response)
        first_reasons = generation_anomalies(first_message, directive, context)
        if not first_reasons:
            if first_message is not None and _has_action_narration(first_message, directive):
                pending_manifests = list(getattr(context, "_acm_bounded_pending_manifests", []) or [])
                model_call_id = (
                    pending_manifests[-1].model_call_id
                    if pending_manifests
                    else f"MC-UNKNOWN-{directive.directive_id[4:12]}"
                )
                suppression = _abort_record(
                    message=first_message,
                    directive=directive,
                    model_call_id=model_call_id,
                    reasons=["ACTION_CONTENT_SUPPRESSED"],
                    retry_number=0,
                )
                setattr(context, "_acm_bounded_pending_aborts", [suppression])
                return ExtendedModelResponse(
                    model_response=_without_action_narration(response, first_message),
                    command=Command(
                        update={
                            "bounded_generation_aborts": [suppression],
                            "warnings": ["ACM Bounded suppressed narration from an action response"],
                        }
                    ),
                )
            return response

        pending_manifests = list(getattr(context, "_acm_bounded_pending_manifests", []) or [])
        first_model_call_id = (
            pending_manifests[-1].model_call_id if pending_manifests else f"MC-UNKNOWN-{directive.directive_id[4:12]}"
        )
        first_abort = _abort_record(
            message=first_message,
            directive=directive,
            model_call_id=first_model_call_id,
            reasons=first_reasons,
            retry_number=0,
        )

        retry_output_tokens = (
            min(4_096, context.final_output_tokens)
            if directive.phase == "DRAFT_FINAL"
            else min(1_024, context.action_output_tokens)
        )
        retry_settings = {
            **dict(request.model_settings or {}),
            "max_completion_tokens": retry_output_tokens,
        }
        retry_tool_choice = None
        if context.bounded_force_tool_choice and len(request.tools or []) == 1:
            retry_tool_choice = getattr((request.tools or [])[0], "name", None)
        retry_request = request.override(
            system_message=_repair_prompt(request, directive),
            model_settings=retry_settings,
            tool_choice=retry_tool_choice,
        )
        retry_manifest = build_request_manifest(
            request=retry_request,
            context=context,
            directive=directive,
            reduction_actions=["GENERATION_REPAIR_RETRY"],
            ordinal_offset=1,
        )
        pending_manifests.append(retry_manifest)
        setattr(context, "_acm_bounded_pending_manifests", pending_manifests)

        retry_response = await handler(retry_request)
        retry_message = _last_ai_message(retry_response)
        retry_reasons = generation_anomalies(retry_message, directive, context)
        if not retry_reasons:
            aborts = [first_abort]
            clean_retry_response = retry_response
            if retry_message is not None and _has_action_narration(retry_message, directive):
                aborts.append(
                    _abort_record(
                        message=retry_message,
                        directive=directive,
                        model_call_id=retry_manifest.model_call_id,
                        reasons=["ACTION_CONTENT_SUPPRESSED"],
                        retry_number=1,
                    )
                )
                clean_retry_response = _without_action_narration(retry_response, retry_message)
            setattr(context, "_acm_bounded_pending_aborts", aborts)
            return ExtendedModelResponse(
                model_response=clean_retry_response,
                command=Command(
                    update={
                        "bounded_generation_aborts": aborts,
                        "bounded_context_manifests": [retry_manifest],
                    }
                ),
            )

        second_abort = _abort_record(
            message=retry_message,
            directive=directive,
            model_call_id=retry_manifest.model_call_id,
            reasons=retry_reasons,
            retry_number=1,
        )
        aborts = [first_abort, second_abort]
        setattr(context, "_acm_bounded_pending_aborts", aborts)
        setattr(context, "_acm_bounded_generation_exhausted", True)
        failure = AIMessage(
            content=(
                "ACM Bounded 本轮生成保护已触发：初次输出和一次受限修复输出均不符合"
                f"动作协议，流程已明确停止。phase={directive.phase}；"
                "reason=" + ",".join(retry_reasons)
            ),
            additional_kwargs={"acm_bounded_terminal_failure": True},
        )
        return ExtendedModelResponse(
            model_response=ModelResponse(
                result=[failure],
                structured_response=retry_response.structured_response,
            ),
            command=Command(
                update={
                    "bounded_generation_aborts": aborts,
                    "bounded_context_manifests": [retry_manifest],
                    "warnings": ["ACM Bounded generation repair exhausted"],
                }
            ),
        )
