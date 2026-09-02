from __future__ import annotations

import hashlib
import json
from typing import Any

from .context import MedicationReviewContext
from .models import (
    MedicationReviewTraceV3,
    STRUCTURED_IO_VERSION,
    V3_METHOD_FAMILY,
    V3_QUERY_POLICY,
)


def effective_profile(context: MedicationReviewContext) -> str:
    synthesis = context.synthesis_mode.replace("_", "-")
    return f"pea-rag-v2-{context.agenda_mode}-{synthesis}-vector-v1"


def safe_context_snapshot(context: MedicationReviewContext) -> dict[str, Any]:
    return {
        "model": context.model,
        "system_prompt_hash": hashlib.sha256(
            context.system_prompt.encode("utf-8")
        ).hexdigest(),
        "knowledges": list(context.knowledges or []),
        "run_mode": context.run_mode,
        "agenda_mode": context.agenda_mode,
        "synthesis_mode": context.synthesis_mode,
        "diagnostic_trace": context.diagnostic_trace,
        "retrieval_timeout_seconds": context.retrieval_timeout_seconds,
        "max_search_calls": context.max_search_calls,
        "max_open_calls": context.max_open_calls,
        "max_agent_steps": context.max_agent_steps,
        "retrieval_top_k": context.retrieval_top_k,
        "max_final_evidence_tokens": context.max_final_evidence_tokens,
        "max_review_questions": context.max_review_questions,
        "max_claim_evidence": context.max_claim_evidence,
        "max_subqueries_per_action": context.max_subqueries_per_action,
        "technical_retry_limit": context.technical_retry_limit,
        "plan_repair_limit": context.plan_repair_limit,
    }


def rendered_answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def build_trace(
    *,
    state: dict[str, Any],
    context: MedicationReviewContext,
) -> MedicationReviewTraceV3:
    profile = effective_profile(context)
    trace = MedicationReviewTraceV3(
        method_version=profile,
        method_family=V3_METHOD_FAMILY,
        review_run_id=state["review_run_id"],
        run_status=state.get("run_status", "failed"),
        run_mode=context.run_mode,
        agenda_mode=context.agenda_mode,
        synthesis_mode=context.synthesis_mode,
        effective_profile=profile,
        completion_reason=state.get("completion_reason", "unknown"),
        last_completed_stage=state.get("stage", "unknown"),
        case_id=(state.get("patient_case") or {}).get("case_id"),
        patient_case=state.get("patient_case"),
        patient_facts=state.get("patient_facts") or [],
        plan_extraction=state.get("plan_extraction") or {},
        plan_elements=state.get("plan_elements") or [],
        review_agenda=state.get("review_agenda") or [],
        agenda_audit=state.get("agenda_audit") or {},
        agent_steps=state.get("agent_steps") or [],
        search_records=state.get("search_records") or [],
        open_records=state.get("open_records") or [],
        finish_retrieval=state.get("finish_retrieval") or {},
        evidence=state.get("evidence") or [],
        evidence_selection=state.get("evidence_selection") or {},
        evidence_claims=state.get("evidence_claims") or [],
        claim_extraction=state.get("claim_extraction") or {},
        review_synthesis=state.get("review_synthesis") or {},
        local_validation_events=state.get("local_validation_events") or [],
        final_review=state.get("final_review"),
        rendered_answer_hash=(
            rendered_answer_hash(state["rendered_answer"])
            if state.get("rendered_answer")
            else None
        ),
        knowledge_base_snapshot=state.get("knowledge_base_snapshot") or {},
        agent_config_snapshot=safe_context_snapshot(context),
        prompt_versions={
            "structured_io": STRUCTURED_IO_VERSION,
            "query_policy": V3_QUERY_POLICY,
            "plan_extraction": "pea-plan-extraction-v5",
            "plan_verification": "pea-plan-verification-v2",
            "review_agenda": "pea-review-agenda-v1",
            "agent_decision": "pea-retrieval-agent-v1",
            "claim_extraction": "pea-claim-extraction-v1",
            "review_synthesis": "pea-review-synthesis-v1",
            "renderer": "pea-six-section-renderer-v3",
        },
        usage=state.get("usage") or {},
        warnings=state.get("warnings") or [],
        errors=state.get("errors") or [],
    )
    usage = dict(trace.usage)
    payload = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    usage["trace_bytes"] = len(payload.encode("utf-8"))
    return trace.model_copy(update={"usage": usage})
