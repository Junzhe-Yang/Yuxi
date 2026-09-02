from __future__ import annotations

import hashlib
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    adaptive_agenda_from_state,
    adaptive_meta_from_state,
    adaptive_probe_records_from_state,
    adaptive_recoveries_from_state,
    build_adaptive_coverage_report,
    build_adaptive_state_fingerprint,
    normalize_text,
)
from yuxi.agents.buildin.medication_review_lite.models import EvidenceItem, OpenRecord
from yuxi.agents.buildin.medication_review_prim.memory import (
    investigations_from_state,
    queries_from_state,
)

from .context import MedicationReviewAcmBoundedContext
from .models import (
    ActionDirective,
    BoundedObligationJudgment,
    MedicationReviewAcmBoundedState,
    ToolOutcome,
)

CONTROLLER_VERSION = "acm-bounded-controller-v1"


def _judgments_from_state(state: dict[str, Any]) -> list[BoundedObligationJudgment]:
    values: list[BoundedObligationJudgment] = []
    for raw in state.get("bounded_obligation_judgments") or []:
        try:
            values.append(
                raw if isinstance(raw, BoundedObligationJudgment) else BoundedObligationJudgment.model_validate(raw)
            )
        except Exception:  # noqa: BLE001 - malformed audit state must not drive routing
            continue
    return values


def _repair_failures_at_version(
    state: dict[str, Any],
    state_version: int,
) -> list[ToolOutcome]:
    failures: list[ToolOutcome] = []
    for raw in state.get("bounded_tool_outcomes") or []:
        try:
            value = raw if isinstance(raw, ToolOutcome) else ToolOutcome.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        if (
            value.state_version_before == state_version
            and value.state_version_after == state_version
            and not value.state_changed
            and value.semantic_outcome in {"REJECTED", "INVALID_ARGUMENT", "NEEDS_INPUT"}
        ):
            failures.append(value)
    return failures


def _latest_judgment(
    state: dict[str, Any],
    investigation_id: str,
    obligation: str,
) -> BoundedObligationJudgment | None:
    target = normalize_text(obligation)
    matches = [
        value
        for value in _judgments_from_state(state)
        if value.investigation_id == investigation_id and normalize_text(value.obligation) == target
    ]
    return matches[-1] if matches else None


def _evidence_store(state: dict[str, Any]) -> dict[str, EvidenceItem]:
    values: dict[str, EvidenceItem] = {}
    for evidence_id, raw in (state.get("evidence_store") or {}).items():
        try:
            values[str(evidence_id).upper()] = (
                raw if isinstance(raw, EvidenceItem) else EvidenceItem.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return values


def _direct_evidence_ids(
    state: dict[str, Any],
    investigation_id: str,
    obligation: str,
) -> list[str]:
    normalized = normalize_text(obligation)
    query_by_id = {value.query_id: value for value in queries_from_state(state)}
    evidence_ids: list[str] = []
    for probe in adaptive_probe_records_from_state(state):
        if probe.investigation_id != investigation_id or normalize_text(probe.uncovered_aspect) != normalized:
            continue
        query = query_by_id.get(probe.query_id)
        if query is not None:
            evidence_ids.extend(query.evidence_ids)

    changed = True
    while changed:
        changed = False
        known = set(evidence_ids)
        for raw in state.get("open_records") or []:
            try:
                record = raw if isinstance(raw, OpenRecord) else OpenRecord.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            if record.investigation_id != investigation_id or record.parent_evidence_id not in known:
                continue
            before = len(evidence_ids)
            evidence_ids.extend(record.evidence_ids)
            evidence_ids = list(dict.fromkeys(evidence_ids))
            changed = changed or len(evidence_ids) > before
    return list(dict.fromkeys(value.upper() for value in evidence_ids))


def _support_evidence_ids(state: dict[str, Any], investigation_id: str | None = None) -> list[str]:
    evidence_ids: list[str] = []
    for meta in adaptive_meta_from_state(state):
        if investigation_id is not None and meta.investigation_id != investigation_id:
            continue
        for support in meta.obligation_supports:
            evidence_ids.extend(support.evidence_ids)
    return list(dict.fromkeys(value.upper() for value in evidence_ids))


def _alias_map(prefix: str, values: list[str]) -> dict[str, str]:
    return {f"{prefix}{index}": value for index, value in enumerate(dict.fromkeys(values), start=1)}


def _atlas_aliases(context: MedicationReviewAcmBoundedContext, state: dict[str, Any]) -> dict[str, str]:
    atlas = getattr(context, "_acm_prim_atlas", None)
    cards = list(getattr(atlas, "document_cards", []) or [])
    opened = {
        str(raw.get("doc_id") if isinstance(raw, dict) else getattr(raw, "doc_id", ""))
        for raw in state.get("atlas_document_open_records") or []
    }
    doc_ids = [str(card.doc_id) for card in cards if str(card.doc_id) not in opened]
    return _alias_map("D", doc_ids)


def _document_route(
    state: dict[str, Any],
    investigation_id: str,
    obligation: str,
    candidate_file_ids: list[str],
) -> str | None:
    normalized = normalize_text(obligation)
    query_by_id = {value.query_id: value for value in queries_from_state(state)}
    attempted_files: set[str] = set()
    for probe in adaptive_probe_records_from_state(state):
        if probe.investigation_id != investigation_id or normalize_text(probe.uncovered_aspect) != normalized:
            continue
        query = query_by_id.get(probe.query_id)
        if query is not None and query.retrieval_scope == "document" and query.file_id:
            attempted_files.add(query.file_id)
    return next((value for value in candidate_file_ids if value not in attempted_files), None)


def build_action_directive(
    state: dict[str, Any],
    context: MedicationReviewAcmBoundedContext,
) -> ActionDirective:
    report = build_adaptive_coverage_report(state, context)
    agenda = adaptive_agenda_from_state(state)
    agenda_items = list(agenda.items) if agenda else []
    item_by_id = {value.investigation_id: value for value in agenda_items}
    investigation_by_id = {value.investigation_id: value for value in investigations_from_state(state)}
    version = int(state.get("bounded_state_version") or 0)
    phase = "DRAFT_FINAL"
    active_id: str | None = None
    active_obligation: str | None = None
    active_recovery_id: str | None = None
    bound_scope: str | None = None
    bound_intent: str | None = None
    bound_file_id: str | None = None

    if not report.agenda_created:
        phase = "PROPOSE_INITIAL_AGENDA"
    elif report.missing_current_regimen_review_plan_ids or report.missing_improvement_plan_ids:
        phase = "EXTEND_AGENDA"
    else:
        pending = [value for value in adaptive_recoveries_from_state(state) if value.status == "pending"]
        if pending:
            recovery = pending[0]
            phase = "SEARCH_ACTIVE_OBLIGATION"
            active_id = recovery.investigation_id
            active_obligation = recovery.uncovered_aspect
            active_recovery_id = recovery.recovery_id
            bound_scope = "global"
            bound_intent = "source_discovery"
        elif report.unprobed_evidence_obligations:
            active_id = next(
                (value for value in report.eligible_investigation_ids if value in report.unprobed_evidence_obligations),
                next(iter(report.unprobed_evidence_obligations)),
            )
            active_obligation = report.unprobed_evidence_obligations[active_id][0]
            phase = "SEARCH_ACTIVE_OBLIGATION"
            bound_scope = "global"
            bound_intent = "source_discovery"
        elif report.unsupported_evidence_obligations:
            active_id = next(
                (
                    value
                    for value in [*report.eligible_investigation_ids, *report.open_investigation_ids]
                    if value in report.unsupported_evidence_obligations
                ),
                next(iter(report.unsupported_evidence_obligations)),
            )
            active_obligation = report.unsupported_evidence_obligations[active_id][0]
            latest = _latest_judgment(state, active_id, active_obligation)
            investigation = investigation_by_id.get(active_id)
            if latest is None or latest.state_version_after < version:
                phase = "REVIEW_ACTIVE_OBLIGATION"
            elif latest.verdict == "INSUFFICIENT":
                candidate_files = list(investigation.candidate_file_ids if investigation else [])
                bound_file_id = _document_route(
                    state,
                    active_id,
                    active_obligation,
                    candidate_files,
                )
                if bound_file_id:
                    phase = "SEARCH_ACTIVE_OBLIGATION"
                    bound_scope = "document"
                    bound_intent = "within_document_localization"
                else:
                    phase = "CLOSE_ACTIVE_INVESTIGATION"
            else:
                phase = "REVIEW_ACTIVE_OBLIGATION"
        elif report.open_investigation_ids:
            active_id = report.open_investigation_ids[0]
            phase = "CLOSE_ACTIVE_INVESTIGATION"
        elif report.status != "completed":
            phase = "AUDIT_COVERAGE"

    if phase != "DRAFT_FINAL" and len(_repair_failures_at_version(state, version)) >= 2:
        phase = "FAIL_EXPLICIT"

    active_item = item_by_id.get(active_id or "")
    evidence_ids: list[str]
    if phase == "REVIEW_ACTIVE_OBLIGATION" and active_id and active_obligation:
        evidence_ids = _direct_evidence_ids(state, active_id, active_obligation)
    elif phase == "CLOSE_ACTIVE_INVESTIGATION":
        evidence_ids = _support_evidence_ids(state, active_id)
    elif phase == "DRAFT_FINAL":
        evidence_ids = _support_evidence_ids(state)
    else:
        evidence_ids = []
    evidence_store = _evidence_store(state)
    evidence_ids = [value for value in evidence_ids if value in evidence_store]

    candidate_files = list(
        investigation_by_id.get(active_id).candidate_file_ids
        if active_id and investigation_by_id.get(active_id)
        else []
    )
    if bound_file_id and bound_file_id not in candidate_files:
        candidate_files.insert(0, bound_file_id)

    action_by_phase = {
        "PROPOSE_INITIAL_AGENDA": ["propose_initial_agenda"],
        "EXTEND_AGENDA": ["extend_investigation_agenda_bounded"],
        "SEARCH_ACTIVE_OBLIGATION": ["search_active_obligation"],
        "REVIEW_ACTIVE_OBLIGATION": [
            "record_active_obligation_support",
            "read_active_evidence",
            "open_active_evidence",
        ],
        "CLOSE_ACTIVE_INVESTIGATION": [
            (
                "close_current_regimen_investigation"
                if active_item and active_item.investigation_kind == "current_regimen_review"
                else "close_active_investigation"
            )
        ],
        "AUDIT_COVERAGE": ["audit_coverage"],
        "DRAFT_FINAL": [],
        "FAIL_EXPLICIT": [],
    }
    atlas_aliases = _atlas_aliases(context, state) if phase == "SEARCH_ACTIVE_OBLIGATION" else {}
    if atlas_aliases:
        action_by_phase["SEARCH_ACTIVE_OBLIGATION"].append("open_active_atlas_document")
    if evidence_ids and phase == "SEARCH_ACTIVE_OBLIGATION":
        action_by_phase["SEARCH_ACTIVE_OBLIGATION"].append("read_active_evidence")

    state_fingerprint = build_adaptive_state_fingerprint(state)
    source = "\0".join(
        (
            CONTROLLER_VERSION,
            str(version),
            state_fingerprint,
            phase,
            active_id or "",
            active_obligation or "",
        )
    )
    directive_id = "DIR-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16].upper()
    investigation_aliases = _alias_map(
        "I",
        [value.investigation_id for value in agenda_items],
    )
    return ActionDirective(
        directive_id=directive_id,
        phase=phase,
        state_version=version,
        state_fingerprint=state_fingerprint,
        active_investigation_id=active_id,
        active_obligation=active_obligation,
        active_recovery_id=active_recovery_id,
        active_investigation_kind=(active_item.investigation_kind if active_item else None),
        allowed_actions=action_by_phase[phase],
        allowed_routes=([bound_scope] if bound_scope else []),
        bound_retrieval_scope=bound_scope,
        bound_retrieval_intent=bound_intent,
        bound_file_id=bound_file_id,
        evidence_aliases=_alias_map("E", evidence_ids),
        file_aliases=_alias_map("F", candidate_files),
        atlas_document_aliases=atlas_aliases,
        investigation_aliases=investigation_aliases,
        retry_policy={"model_repairs": 1, "technical_retries": context.technical_retry_limit},
        expected_output_kind=("final_answer" if phase in {"DRAFT_FINAL", "FAIL_EXPLICIT"} else "tool_call"),
    )


class AcmBoundedControllerMiddleware(
    AgentMiddleware[MedicationReviewAcmBoundedState, MedicationReviewAcmBoundedContext]
):
    state_schema = MedicationReviewAcmBoundedState

    async def abefore_model(
        self,
        state: MedicationReviewAcmBoundedState,
        runtime: Any,
    ) -> dict[str, Any]:
        context: MedicationReviewAcmBoundedContext = runtime.context
        directive = build_action_directive(dict(state), context)
        setattr(context, "_acm_bounded_directive", directive)
        return {
            "action_directive": directive,
            "bounded_directives": [directive],
        }
