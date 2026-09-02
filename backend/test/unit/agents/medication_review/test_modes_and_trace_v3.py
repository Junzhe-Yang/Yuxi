from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review.context import MedicationReviewContext
from yuxi.agents.buildin.medication_review.retrieval import (
    MedicationReviewConfigError,
    validate_v3_context,
)
from yuxi.agents.buildin.medication_review.trace import build_trace, effective_profile


def _context(**updates) -> MedicationReviewContext:
    values = {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "knowledges": ["知识库"],
    }
    values.update(updates)
    return MedicationReviewContext(**values)


def test_mode_combinations_are_validated():
    validate_v3_context(_context(run_mode="stop_after_agenda", agenda_mode="dynamic"))
    validate_v3_context(_context(run_mode="stop_after_claims", synthesis_mode="claims"))
    with pytest.raises(MedicationReviewConfigError):
        validate_v3_context(_context(run_mode="stop_after_agenda", agenda_mode="none"))
    with pytest.raises(MedicationReviewConfigError):
        validate_v3_context(
            _context(run_mode="stop_after_claims", synthesis_mode="direct_chunks")
        )


def test_trace_records_effective_profile_and_debug_stage():
    context = _context(
        run_mode="stop_after_retrieval",
        agenda_mode="none",
        synthesis_mode="direct_chunks",
    )
    state = {
        "review_run_id": "run-1",
        "run_status": "debug_stopped",
        "completion_reason": "stop_after_retrieval",
        "stage": "evidence_prepared",
        "rendered_answer": "诊断输出",
    }

    trace = build_trace(state=state, context=context)

    assert trace.schema_version == "3.0"
    assert trace.effective_profile == "pea-rag-v2-none-direct-chunks-vector-v1"
    assert trace.effective_profile == effective_profile(context)
    assert trace.last_completed_stage == "evidence_prepared"
    assert trace.usage["trace_bytes"] > 0
