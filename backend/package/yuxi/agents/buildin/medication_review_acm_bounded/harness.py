from __future__ import annotations

import hashlib
from typing import Any

from yuxi.agents.buildin.medication_review_acm_prim.harness import (
    AcmReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_acm_prim.models import (
    MedicationReviewAcmAdaptiveTrace,
)
from yuxi.agents.buildin.medication_review_prim.harness import (
    PreFinalInterruption,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimTrace,
)

from .citations import (
    CITATION_REHYDRATION_VERSION,
    build_citation_verification,
)
from .context import (
    MedicationReviewAcmBoundedContext,
    validate_bounded_context,
)
from .context_view import CONTEXT_VIEW_VERSION
from .controller import CONTROLLER_VERSION
from .generation import GENERATION_GUARD_VERSION
from .models import (
    ActionAttempt,
    ActionDirective,
    BoundedObligationJudgment,
    CitationVerificationRecord,
    ContextAtom,
    ContextManifest,
    GenerationAbortRecord,
    MedicationReviewAcmBoundedState,
    MedicationReviewAcmBoundedTrace,
    ToolOutcome,
)
from .prompt import PROMPT_VERSION, build_bounded_prompt

METHOD_VERSION = "acm-prim-rag-v9-bounded-controller-v1-shadow_top25-vector"


def _typed_values[ModelT](
    values: list[Any] | None,
    model_type: type[ModelT],
    key: str,
) -> list[ModelT]:
    result: dict[str, ModelT] = {}
    for raw in values or []:
        try:
            value = raw if isinstance(raw, model_type) else model_type.model_validate(raw)
        except Exception:  # noqa: BLE001 - malformed audit data stays trace-local
            continue
        result[str(getattr(value, key))] = value
    return list(result.values())


class AcmBoundedHarnessMiddleware(AcmReviewHarnessMiddleware):
    state_schema = MedicationReviewAcmBoundedState

    async def abefore_agent(
        self,
        state: MedicationReviewAcmBoundedState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        validate_bounded_context(runtime.context)
        return await super().abefore_agent(state, runtime)

    async def augment_initial_state(
        self,
        *,
        state: dict[str, Any],
        update: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any]:
        inherited = await super().augment_initial_state(
            state=state,
            update=update,
            runtime=runtime,
        )
        initialized: dict[str, Any] = {
            "reflection_attempted": True,
        }
        defaults = {
            "bounded_state_version": 0,
            "bounded_directives": [],
            "bounded_context_manifests": [],
            "bounded_context_atoms": [],
            "bounded_tool_outcomes": [],
            "bounded_action_attempts": [],
            "bounded_obligation_judgments": [],
            "bounded_generation_aborts": [],
        }
        initialized.update({key: value for key, value in defaults.items() if key not in state})
        return {**inherited, **initialized}

    def augment_investigation_memory(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
        memory_text: str,
    ) -> str:
        del state, context, memory_text
        # The inner model-view middleware replaces the legacy prompt wholesale.
        return ""

    def project_model_messages(
        self,
        *,
        messages: list[Any],
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
    ) -> list[Any]:
        del state, context
        # Projection is centralized in AcmBoundedModelViewMiddleware.
        return messages

    def tool_is_visible(
        self,
        *,
        tool_name: str,
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
    ) -> bool:
        raw = state.get("action_directive") or getattr(
            context,
            "_acm_bounded_directive",
            None,
        )
        if raw is None:
            return False
        directive = raw if isinstance(raw, ActionDirective) else ActionDirective.model_validate(raw)
        return tool_name in directive.allowed_actions

    async def prepare_pre_final_interruption(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
        candidate_body: str,
    ) -> PreFinalInterruption | None:
        del state, context, candidate_body
        # Controller + generation guard own action-round liveness. Reusing the
        # legacy synthetic checkpoint here would reintroduce the old loop.
        return None

    async def prepare_candidate_state(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
        candidate_body: str,
    ) -> dict[str, Any]:
        del context
        return {
            "bounded_citation_verification": build_citation_verification(
                candidate_body,
                state,
            )
        }

    def finalize_trace(
        self,
        *,
        base_trace: MedicationReviewPrimTrace,
        state: dict[str, Any],
        context: MedicationReviewAcmBoundedContext,
    ) -> MedicationReviewAcmBoundedTrace:
        adaptive_trace = super().finalize_trace(
            base_trace=base_trace,
            state=state,
            context=context,
        )
        if not isinstance(adaptive_trace, MedicationReviewAcmAdaptiveTrace):
            raise TypeError("ACM Bounded 必须基于 adaptive trace 构建")

        pending_manifests = list(getattr(context, "_acm_bounded_pending_manifests", []) or [])
        pending_aborts = list(getattr(context, "_acm_bounded_pending_aborts", []) or [])
        pending_atoms = list(getattr(context, "_acm_bounded_pending_atoms", []) or [])
        current_directive = getattr(context, "_acm_bounded_directive", None)
        directives = _typed_values(
            [
                *(state.get("bounded_directives") or []),
                *([current_directive] if current_directive is not None else []),
            ],
            ActionDirective,
            "directive_id",
        )
        manifests = _typed_values(
            [
                *(state.get("bounded_context_manifests") or []),
                *pending_manifests,
            ],
            ContextManifest,
            "model_call_id",
        )
        aborts = _typed_values(
            [
                *(state.get("bounded_generation_aborts") or []),
                *pending_aborts,
            ],
            GenerationAbortRecord,
            "abort_id",
        )

        bounded_prompt_hash = ""
        if directives:
            bounded_prompt_hash = hashlib.sha256(
                build_bounded_prompt(
                    state=state,
                    context=context,
                    directive=directives[-1],
                ).text.encode("utf-8")
            ).hexdigest()
        payload = adaptive_trace.model_dump(
            exclude={
                "schema_version",
                "method_family",
                "method_version",
                "prompt_versions",
                "prompt_hashes",
                "budgets",
                "warnings",
                "run_status",
                "completion_reason",
            }
        )
        exhausted = bool(getattr(context, "_acm_bounded_generation_exhausted", False))
        semantic_exhausted = bool(current_directive is not None and current_directive.phase == "FAIL_EXPLICIT")
        context_window_verified = bool(getattr(context, "_acm_bounded_context_window_verified", False))
        provider_context_window = int(
            getattr(
                context,
                "_acm_bounded_provider_context_window_tokens",
                context.model_context_window_tokens,
            )
        )
        raw_citation = state.get("bounded_citation_verification")
        citation_verification = (
            raw_citation
            if isinstance(raw_citation, CitationVerificationRecord)
            else CitationVerificationRecord.model_validate(raw_citation)
            if raw_citation is not None
            else None
        )
        warnings = list(
            dict.fromkeys(
                [
                    *adaptive_trace.warnings,
                    *(["生成保护的一次修复已耗尽，流程显式终止"] if exhausted else []),
                    *(["同一状态下的两次语义修复均未推进 ledger，流程显式终止"] if semantic_exhausted else []),
                    *(
                        ["provider 未提供可验证的 max_input_tokens；按 262144 声明运行"]
                        if not context_window_verified
                        else []
                    ),
                ]
            )
        )
        return MedicationReviewAcmBoundedTrace(
            **payload,
            method_version=METHOD_VERSION,
            run_status="partial" if exhausted or semantic_exhausted else adaptive_trace.run_status,
            completion_reason=(
                "generation_guard_exhausted"
                if exhausted
                else "semantic_repair_exhausted"
                if semantic_exhausted
                else adaptive_trace.completion_reason
            ),
            prompt_versions={
                **adaptive_trace.prompt_versions,
                "bounded_agent": PROMPT_VERSION,
                "bounded_context_view": CONTEXT_VIEW_VERSION,
                "bounded_controller": CONTROLLER_VERSION,
                "bounded_generation_guard": GENERATION_GUARD_VERSION,
                "bounded_citation_rehydration": CITATION_REHYDRATION_VERSION,
            },
            prompt_hashes={
                **adaptive_trace.prompt_hashes,
                "bounded_agent_final": bounded_prompt_hash,
            },
            budgets={
                **dict(adaptive_trace.budgets),
                "model_context_window_tokens": context.model_context_window_tokens,
                "action_output_tokens": context.action_output_tokens,
                "action_output_absolute_limit": context.action_output_absolute_limit,
                "final_output_tokens": context.final_output_tokens,
            },
            warnings=warnings,
            controller_version=CONTROLLER_VERSION,
            context_view_version=CONTEXT_VIEW_VERSION,
            prompt_version=PROMPT_VERSION,
            model_context_window_tokens=context.model_context_window_tokens,
            provider_context_window_tokens=provider_context_window,
            context_window_verified=context_window_verified,
            directives=directives,
            context_manifests=manifests,
            context_atoms=_typed_values(
                [*(state.get("bounded_context_atoms") or []), *pending_atoms],
                ContextAtom,
                "atom_id",
            ),
            tool_outcomes=_typed_values(
                state.get("bounded_tool_outcomes"),
                ToolOutcome,
                "call_id",
            ),
            action_attempts=_typed_values(
                state.get("bounded_action_attempts"),
                ActionAttempt,
                "fingerprint",
            ),
            obligation_judgments=_typed_values(
                state.get("bounded_obligation_judgments"),
                BoundedObligationJudgment,
                "judgment_id",
            ),
            generation_abort_records=aborts,
            citation_verification=citation_verification,
        )
