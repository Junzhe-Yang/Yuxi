from __future__ import annotations

from types import SimpleNamespace

from yuxi.agents.buildin.medication_review_acm_bounded.context import (
    MedicationReviewAcmBoundedContext,
)
from yuxi.agents.buildin.medication_review_acm_bounded.controller import (
    build_action_directive,
)


def _context() -> MedicationReviewAcmBoundedContext:
    context = MedicationReviewAcmBoundedContext(
        knowledges=["知识库"],
        max_search_calls=50,
        evidence_excerpt_chars=600,
        technical_retry_limit=0,
    )
    context._acm_prim_atlas = SimpleNamespace(document_cards=[])
    return context


def _open_state() -> dict:
    now = "2026-09-02T00:00:00Z"
    return {
        "bounded_state_version": 1,
        "plan_anchors": [],
        "patient_modifiers": [],
        "adaptive_agenda": {
            "agenda_id": "AGENDA-1",
            "created_at": now,
            "updated_at": now,
            "revision": 1,
            "items": [
                {
                    "investigation_id": "INV-1",
                    "question": "当前药物需要怎样监测？",
                    "why_it_matters": "监测会改变安全性判断",
                    "distinct_scope": "药物监测",
                    "investigation_kind": "monitoring_followup",
                    "evidence_obligations": ["当前药物的监测频率"],
                    "created_at": now,
                    "revision": 1,
                }
            ],
            "revisions": [
                {
                    "revision": 1,
                    "tool_call_id": "agenda-call",
                    "created_at": now,
                    "reason": "initial",
                    "added_investigation_ids": ["INV-1"],
                }
            ],
        },
        "investigations": [
            {
                "investigation_id": "INV-1",
                "question": "当前药物需要怎样监测？",
                "status": "open",
                "candidate_file_ids": ["file-1"],
                "created_at": now,
                "updated_at": now,
            }
        ],
        "query_records": [],
        "evidence_store": {},
        "adaptive_probe_records": [],
        "adaptive_recovery_requirements": [],
        "adaptive_investigation_meta": [],
        "adaptive_gap_assessments": [],
        "search_count": 0,
    }


def test_controller_routes_each_coverage_state_to_one_narrow_phase() -> None:
    context = _context()
    assert build_action_directive({}, context).phase == "PROPOSE_INITIAL_AGENDA"

    state = _open_state()
    search = build_action_directive(state, context)
    assert search.phase == "SEARCH_ACTIVE_OBLIGATION"
    assert search.active_investigation_id == "INV-1"
    assert search.active_obligation == "当前药物的监测频率"
    assert search.allowed_actions == ["search_active_obligation"]
    assert search.bound_retrieval_scope == "global"

    state["query_records"] = [
        {
            "query_id": "Q-1",
            "tool_call_id": "search-call",
            "investigation_id": "INV-1",
            "query_text": "药物 监测频率",
            "reason": "当前药物的监测频率",
            "retrieval_scope": "global",
            "started_at": "2026-09-02T00:00:01Z",
            "elapsed_ms": 1,
            "status": "success",
            "evidence_ids": ["EV-1"],
            "new_evidence_ids": ["EV-1"],
        }
    ]
    state["adaptive_probe_records"] = [
        {
            "probe_record_id": "APROBE-Q-1",
            "query_id": "Q-1",
            "investigation_id": "INV-1",
            "uncovered_aspect": "当前药物的监测频率",
            "retrieval_intent": "source_discovery",
            "route_key": "INV-1:global",
            "status": "success",
        }
    ]
    state["evidence_store"] = {
        "EV-1": {
            "evidence_id": "EV-1",
            "content_hash": "hash-1",
            "raw_text": "建议每四周监测一次。",
            "source_document": "指南.md",
            "file_id": "file-1",
            "chunk_id": "chunk-1",
            "chunk_index": 1,
        }
    }
    state["investigations"][0].update(
        {
            "query_ids": ["Q-1"],
            "candidate_evidence_ids": ["EV-1"],
        }
    )
    review = build_action_directive(state, context)
    assert review.phase == "REVIEW_ACTIVE_OBLIGATION"
    assert review.evidence_aliases == {"E1": "EV-1"}
    assert "record_active_obligation_support" in review.allowed_actions

    state["bounded_obligation_judgments"] = [
        {
            "judgment_id": "JUDG-1",
            "tool_call_id": "judge-call",
            "investigation_id": "INV-1",
            "obligation": "当前药物的监测频率",
            "verdict": "INSUFFICIENT",
            "rationale": "当前块没有明确频率",
            "state_version_after": 1,
            "created_at": "2026-09-02T00:00:02Z",
        }
    ]
    document_search = build_action_directive(state, context)
    assert document_search.phase == "SEARCH_ACTIVE_OBLIGATION"
    assert document_search.bound_retrieval_scope == "document"
    assert document_search.bound_file_id == "file-1"
    assert document_search.allowed_routes == ["document"]

    state["bounded_state_version"] = 2
    state["query_records"].append(
        {
            "query_id": "Q-2",
            "tool_call_id": "document-search-call",
            "investigation_id": "INV-1",
            "query_text": "监测 频率",
            "reason": "当前药物的监测频率",
            "retrieval_scope": "document",
            "file_id": "file-1",
            "started_at": "2026-09-02T00:00:03Z",
            "elapsed_ms": 1,
            "status": "success",
            "evidence_ids": ["EV-2"],
            "new_evidence_ids": ["EV-2"],
        }
    )
    state["adaptive_probe_records"].append(
        {
            "probe_record_id": "APROBE-Q-2",
            "query_id": "Q-2",
            "investigation_id": "INV-1",
            "uncovered_aspect": "当前药物的监测频率",
            "retrieval_intent": "within_document_localization",
            "route_key": "INV-1:document:file-1",
            "status": "success",
        }
    )
    state["evidence_store"]["EV-2"] = {
        "evidence_id": "EV-2",
        "content_hash": "hash-2",
        "raw_text": "补充证据明确了监测频率。",
        "source_document": "指南.md",
        "file_id": "file-1",
        "chunk_id": "chunk-2",
        "chunk_index": 2,
    }
    state["investigations"][0]["query_ids"].append("Q-2")
    state["investigations"][0]["candidate_evidence_ids"].append("EV-2")

    rereview = build_action_directive(state, context)
    assert rereview.phase == "REVIEW_ACTIVE_OBLIGATION"
    assert rereview.evidence_aliases == {"E1": "EV-1", "E2": "EV-2"}


def test_controller_exits_after_two_semantic_repairs_without_state_change() -> None:
    state = _open_state()
    state["bounded_tool_outcomes"] = [
        {
            "call_id": f"call-{index}",
            "directive_id": "DIR-OLD",
            "tool_name": "search_active_obligation",
            "transport_status": "COMPLETED",
            "semantic_outcome": "INVALID_ARGUMENT",
            "reason_code": "QUERY_CONTRACT",
            "retryable_by_model": True,
            "technical_retryable": False,
            "state_changed": False,
            "executed_backend": False,
            "state_version_before": 1,
            "state_version_after": 1,
            "message_for_model": "修复 query",
        }
        for index in (1, 2)
    ]

    directive = build_action_directive(state, _context())

    assert directive.phase == "FAIL_EXPLICIT"
    assert directive.allowed_actions == []
    assert directive.expected_output_kind == "final_answer"
