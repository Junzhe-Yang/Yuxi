from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import Command

from .context import MedicationReviewAcmBoundedContext
from .models import (
    ActionDirective,
    ContextAtom,
    ContextManifest,
    ContextManifestEntry,
    MedicationReviewAcmBoundedState,
    ToolOutcome,
)
from .prompt import BoundedPrompt, build_bounded_prompt

CONTEXT_VIEW_VERSION = "acm-bounded-context-view-v1"
_KNOWLEDGE_TOOLS = {
    "search_active_obligation",
    "read_active_evidence",
    "open_active_evidence",
    "open_active_atlas_document",
}
_REJECTION_OUTCOMES = {"REJECTED", "INVALID_ARGUMENT", "NEEDS_INPUT"}


class ContextCapacityError(RuntimeError):
    """The lossless mandatory view cannot fit the provider context."""


@dataclass(frozen=True)
class ProjectedMessages:
    messages: list[AnyMessage]
    entries: list[ContextManifestEntry]
    atoms: list[ContextAtom]
    excluded_counts: dict[str, int]


@dataclass(frozen=True)
class _Interaction:
    index: int
    assistant: AIMessage
    results: list[ToolMessage]
    call_ids: list[str]
    complete: bool


def _outcomes_from_state(state: dict[str, Any]) -> dict[str, ToolOutcome]:
    outcomes: dict[str, ToolOutcome] = {}
    for raw in state.get("bounded_tool_outcomes") or []:
        try:
            value = raw if isinstance(raw, ToolOutcome) else ToolOutcome.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        outcomes[value.call_id] = value
    return outcomes


def _interactions(messages: list[AnyMessage]) -> list[_Interaction]:
    result_by_call: dict[str, ToolMessage] = {
        value.tool_call_id: value for value in messages if isinstance(value, ToolMessage) and value.tool_call_id
    }
    interactions: list[_Interaction] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        call_ids = [str(value.get("id") or "") for value in message.tool_calls]
        call_ids = [value for value in call_ids if value]
        results = [result_by_call[value] for value in call_ids if value in result_by_call]
        interactions.append(
            _Interaction(
                index=index,
                assistant=message,
                results=results,
                call_ids=call_ids,
                complete=bool(call_ids) and len(results) == len(call_ids),
            )
        )
    return interactions


def _receipt(message: ToolMessage, outcome: ToolOutcome | None) -> str:
    payload: dict[str, Any] = {
        "receipt_version": "TOOL_RECEIPT_V1",
        "call_id": message.tool_call_id,
        "action": message.name or "unknown",
    }
    if outcome is not None:
        payload.update(
            {
                "outcome": outcome.semantic_outcome,
                "reason_code": outcome.reason_code,
                "state_changed": outcome.state_changed,
                "state_version_after": outcome.state_version_after,
                "evidence_refs": outcome.evidence_ids,
                "next_legal_actions": outcome.next_legal_actions,
                "investigation_id": outcome.investigation_id,
                "obligation": outcome.obligation,
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _interaction_scope_matches(
    interaction: _Interaction,
    directive: ActionDirective,
    outcomes: dict[str, ToolOutcome],
) -> bool:
    for call_id in interaction.call_ids:
        outcome = outcomes.get(call_id)
        if outcome is None:
            continue
        if directive.active_investigation_id and outcome.investigation_id == directive.active_investigation_id:
            return True
        if directive.active_obligation and outcome.obligation == directive.active_obligation:
            return True
        if outcome.directive_id == directive.directive_id and outcome.semantic_outcome in _REJECTION_OUTCOMES:
            return True
    return False


def project_messages(
    *,
    messages: list[AnyMessage],
    state: dict[str, Any],
    directive: ActionDirective,
    include_fallback_tail: bool = True,
) -> ProjectedMessages:
    outcomes = _outcomes_from_state(state)
    interactions = _interactions(messages)
    incomplete = [value for value in interactions if not value.complete]
    if incomplete:
        call_ids = [call_id for value in incomplete for call_id in value.call_ids]
        raise ValueError("ACM Bounded projection refuses orphan tool-call/result pairs: " + ",".join(call_ids))
    unresolved = {value.index for value in interactions if not value.complete}
    relevant = {value.index for value in interactions if _interaction_scope_matches(value, directive, outcomes)}
    latest_rejection_index = next(
        (
            value.index
            for value in reversed(interactions)
            if any(
                outcomes.get(call_id) is not None and outcomes[call_id].semantic_outcome in _REJECTION_OUTCOMES
                for call_id in value.call_ids
            )
        ),
        None,
    )
    selected = unresolved | relevant
    if latest_rejection_index is not None:
        selected.add(latest_rejection_index)
    if include_fallback_tail:
        selected.update(value.index for value in interactions[-3:])

    indexed_messages: list[tuple[int, AnyMessage]] = []
    entries: list[ContextManifestEntry] = []
    atoms: list[ContextAtom] = []
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            indexed_messages.append((index, message))
            entries.append(
                ContextManifestEntry(
                    source_id=str(message.id or f"human-{index}"),
                    component="USER_CASE",
                    representation="raw",
                    retention_reason="MANDATORY_CASE",
                    token_count=count_tokens_approximately([message]),
                    priority=0,
                )
            )
            atoms.append(
                ContextAtom(
                    atom_id=f"ATOM-H-{message.id or index}",
                    kind="USER_CASE",
                    phase=directive.phase,
                    state_version=directive.state_version,
                    token_count=count_tokens_approximately([message]),
                )
            )

    interaction_by_index = {value.index: value for value in interactions}
    for index in sorted(selected):
        interaction = interaction_by_index[index]
        priority = 0 if index in unresolved else 1 if index in relevant else 3 if index == latest_rejection_index else 4
        reason = (
            "UNRESOLVED_TOOL_FRONTIER"
            if index in unresolved
            else "ACTIVE_SCOPE"
            if index in relevant
            else "LATEST_RELEVANT_REJECTION"
            if index == latest_rejection_index
            else "RECENT_FALLBACK_TAIL"
        )
        assistant = interaction.assistant.model_copy(update={"content": ""})
        indexed_messages.append((index, assistant))
        interaction_tokens = count_tokens_approximately([assistant])
        for offset, tool_message in enumerate(interaction.results, start=1):
            outcome = outcomes.get(tool_message.tool_call_id)
            use_receipt = bool(
                interaction.complete and ((tool_message.name or "") in _KNOWLEDGE_TOOLS or priority >= 2)
            )
            projected = (
                tool_message.model_copy(update={"content": _receipt(tool_message, outcome)})
                if use_receipt
                else tool_message
            )
            indexed_messages.append((index + offset / 100, projected))
            interaction_tokens += count_tokens_approximately([projected])
        entries.append(
            ContextManifestEntry(
                source_id="+".join(interaction.call_ids) or f"interaction-{index}",
                component="TOOL_INTERACTION",
                representation=("receipt" if interaction.complete else "raw"),
                retention_reason=reason,
                scope_keys={
                    "investigation_id": directive.active_investigation_id or "",
                    "obligation": directive.active_obligation or "",
                },
                token_count=interaction_tokens,
                priority=priority,
            )
        )
        primary_outcome = next(
            (outcomes[call_id] for call_id in interaction.call_ids if call_id in outcomes),
            None,
        )
        atoms.append(
            ContextAtom(
                atom_id="ATOM-T-" + (interaction.call_ids[0] if interaction.call_ids else str(index)),
                kind=(
                    "REJECTION"
                    if primary_outcome is not None and primary_outcome.semantic_outcome in _REJECTION_OUTCOMES
                    else "TOOL_INTERACTION"
                ),
                phase=directive.phase,
                investigation_id=(primary_outcome.investigation_id if primary_outcome is not None else None),
                obligation=(primary_outcome.obligation if primary_outcome is not None else None),
                evidence_ids=(primary_outcome.evidence_ids if primary_outcome is not None else []),
                tool_call_id=(interaction.call_ids[0] if interaction.call_ids else None),
                state_version=(
                    primary_outcome.state_version_after if primary_outcome is not None else directive.state_version
                ),
                material_state_change=bool(primary_outcome and primary_outcome.state_changed),
                outcome=(primary_outcome.semantic_outcome if primary_outcome is not None else None),
                token_count=interaction_tokens,
                rehydration_handle=(
                    "evidence://" + ",".join(primary_outcome.evidence_ids)
                    if primary_outcome is not None and primary_outcome.evidence_ids
                    else None
                ),
            )
        )

    projected = [value for _, value in sorted(indexed_messages, key=lambda pair: pair[0])]
    selected_tool_indexes = selected
    process_ai_count = sum(isinstance(value, AIMessage) and not value.tool_calls for value in messages)
    excluded_interactions = len(interactions) - len(selected_tool_indexes)
    return ProjectedMessages(
        messages=projected,
        entries=entries,
        atoms=atoms,
        excluded_counts={
            "process_narration": process_ai_count,
            "unrelated_completed_interaction": max(excluded_interactions, 0),
            "aborted_generation": len(state.get("bounded_generation_aborts") or []),
        },
    )


def _tool_schema_tokens(tools: list[Any]) -> int:
    payloads: list[Any] = []
    for value in tools:
        if isinstance(value, dict):
            payloads.append(value)
            continue
        try:
            schema = value.get_input_schema().model_json_schema()
        except Exception:  # noqa: BLE001
            schema = {}
        payloads.append(
            {
                "name": getattr(value, "name", ""),
                "description": getattr(value, "description", ""),
                "schema": schema,
            }
        )
    text = json.dumps(payloads, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, (len(text) + 3) // 4) if text else 0


def _provider_context_window(request: ModelRequest, context: MedicationReviewAcmBoundedContext) -> int:
    configured = context.model_context_window_tokens
    profile = getattr(request.model, "profile", None)
    if isinstance(profile, Mapping):
        provider_value = profile.get("max_input_tokens")
        if isinstance(provider_value, int) and provider_value > 0:
            return min(configured, provider_value)
    return configured


def build_request_manifest(
    *,
    request: ModelRequest,
    context: MedicationReviewAcmBoundedContext,
    directive: ActionDirective,
    entries: list[ContextManifestEntry] | None = None,
    excluded_counts: dict[str, int] | None = None,
    reduction_actions: list[str] | None = None,
    ordinal_offset: int = 0,
    prompt: BoundedPrompt | None = None,
) -> ContextManifest:
    prior_count = len(request.state.get("bounded_context_manifests") or []) if isinstance(request.state, dict) else 0
    ordinal = prior_count + 1 + ordinal_offset
    model_call_id = f"MC-{ordinal:04d}-{directive.directive_id[4:12]}"
    messages = list(request.messages)
    system_message = request.system_message
    system_tokens = count_tokens_approximately([system_message]) if system_message is not None else 0
    message_tokens = count_tokens_approximately(messages)
    tool_tokens = _tool_schema_tokens(list(request.tools or []))
    projected_input = system_tokens + message_tokens + tool_tokens
    settings = dict(request.model_settings or {})
    reserved_output = int(
        settings.get("max_completion_tokens")
        or settings.get("max_tokens")
        or (context.final_output_tokens if directive.phase == "DRAFT_FINAL" else context.action_output_tokens)
    )
    window = _provider_context_window(request, context)
    target = context.phase_observation_targets.get(directive.phase)
    warnings: list[str] = []
    above_target = bool(target and projected_input > target)
    if above_target:
        warnings.append(
            f"projected_input_tokens={projected_input} exceeds observation target={target}; call remains allowed"
        )
    component_tokens = {
        "system_prompt": system_tokens,
        "tool_schema": tool_tokens,
        "messages": message_tokens,
    }
    if prompt is not None:
        component_tokens.update(
            {
                "case_memory": max(0, (len(prompt.case_memory) + 3) // 4),
                "working_ledger": max(0, (len(prompt.ledger_memory) + 3) // 4),
                "atlas_memory": max(0, (len(prompt.atlas_memory) + 3) // 4),
                "active_evidence": max(0, (len(prompt.evidence_memory) + 3) // 4),
            }
        )
    return ContextManifest(
        model_call_id=model_call_id,
        directive_id=directive.directive_id,
        phase=directive.phase,
        entries=list(entries or []),
        excluded_counts=dict(excluded_counts or {}),
        component_tokens=component_tokens,
        projected_input_tokens=projected_input,
        reserved_output_tokens=reserved_output,
        projected_total_tokens=projected_input + reserved_output,
        configured_context_window=window,
        phase_observation_target=target,
        above_observation_target=above_target,
        reduction_actions=list(reduction_actions or []),
        warnings=warnings,
    )


class AcmBoundedModelViewMiddleware(
    AgentMiddleware[MedicationReviewAcmBoundedState, MedicationReviewAcmBoundedContext]
):
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
        state = request.state if isinstance(request.state, dict) else {}
        raw_directive = state.get("action_directive") or getattr(
            context,
            "_acm_bounded_directive",
            None,
        )
        if raw_directive is None:
            raise RuntimeError("ACM Bounded model view 缺少 ActionDirective")
        directive = (
            raw_directive
            if isinstance(raw_directive, ActionDirective)
            else ActionDirective.model_validate(raw_directive)
        )
        allowed = set(directive.allowed_actions)
        visible_tools = [value for value in (request.tools or []) if getattr(value, "name", "") in allowed]
        projection = project_messages(
            messages=list(request.messages),
            state=state,
            directive=directive,
            include_fallback_tail=True,
        )
        prompt = build_bounded_prompt(
            state=state,
            context=context,
            directive=directive,
        )
        output_tokens = (
            context.final_output_tokens if directive.phase == "DRAFT_FINAL" else context.action_output_tokens
        )
        settings = {
            **dict(request.model_settings or {}),
            "max_completion_tokens": output_tokens,
        }
        tool_choice = None
        if context.bounded_force_tool_choice and len(visible_tools) == 1:
            tool_choice = getattr(visible_tools[0], "name", None)
        model_request = request.override(
            messages=projection.messages,
            system_message=SystemMessage(content=prompt.text),
            tools=visible_tools,
            tool_choice=tool_choice,
            model_settings=settings,
        )
        manifest = build_request_manifest(
            request=model_request,
            context=context,
            directive=directive,
            entries=projection.entries,
            excluded_counts=projection.excluded_counts,
            prompt=prompt,
        )
        if manifest.projected_total_tokens > manifest.configured_context_window:
            projection = project_messages(
                messages=list(request.messages),
                state=state,
                directive=directive,
                include_fallback_tail=False,
            )
            model_request = model_request.override(messages=projection.messages)
            manifest = build_request_manifest(
                request=model_request,
                context=context,
                directive=directive,
                entries=projection.entries,
                excluded_counts=projection.excluded_counts,
                reduction_actions=["REMOVE_RECENT_FALLBACK_TAIL"],
                prompt=prompt,
            )
        if manifest.projected_total_tokens > manifest.configured_context_window:
            raise ContextCapacityError(
                "ACM Bounded mandatory context cannot fit provider capacity: "
                f"input={manifest.projected_input_tokens}, "
                f"reserved_output={manifest.reserved_output_tokens}, "
                f"window={manifest.configured_context_window}"
            )

        setattr(context, "_acm_bounded_pending_manifests", [manifest])
        setattr(context, "_acm_bounded_pending_atoms", projection.atoms)
        response = await handler(model_request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "bounded_context_manifests": [manifest],
                    "bounded_context_atoms": projection.atoms,
                }
            ),
        )
