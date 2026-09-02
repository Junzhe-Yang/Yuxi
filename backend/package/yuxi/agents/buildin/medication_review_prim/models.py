from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field

from yuxi.agents import BaseState
from yuxi.agents.buildin.medication_review_lite.models import (
    AnchorExtractionAudit,
    EvidenceItem,
    OpenRecord,
    PlanAnchor,
    TechnicalAttempt,
    TraceError,
    add_int,
    merge_evidence_store,
    merge_open_records,
    merge_snapshot,
    merge_unique_strings,
)

ExperimentProfile = Literal["b1", "m1", "m2", "m3", "full"]
RunStatus = Literal["completed", "partial", "failed"]
RecordStatus = Literal["success", "success_empty", "technical_failed"]
RetrievalScope = Literal["global", "document"]
InvestigationOrigin = Literal["agent", "atlas"]
InvestigationStatus = Literal["open", "answered", "insufficient", "dismissed"]
DeferredKnowledgeCallReason = Literal["same_model_turn", "budget_exhausted"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientModifierDraft(StrictModel):
    source_span: str = Field(min_length=1)


class PatientModifierEnvelope(StrictModel):
    modifiers: list[PatientModifierDraft] = Field(default_factory=list)


class PatientModifier(StrictModel):
    modifier_id: str
    source_span: str
    source_start: int
    source_end: int


class ModifierExtractionAudit(StrictModel):
    status: Literal[
        "disabled",
        "success",
        "repaired",
        "failed",
        "no_valid_modifier",
    ]
    schema_name: str = "PatientModifierEnvelope"
    started_at: str
    elapsed_ms: int = 0
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    dropped_drafts: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class QueryRecord(StrictModel):
    query_id: str
    tool_call_id: str
    investigation_id: str | None = None
    query_text: str
    reason: str
    retrieval_scope: RetrievalScope = "global"
    file_id: str | None = None
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    atlas_companion_ids: list[str] = Field(default_factory=list)
    started_at: str
    elapsed_ms: int
    status: RecordStatus
    returned_count: int = 0
    retained_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    attempts: list[TechnicalAttempt] = Field(default_factory=list)
    invalid_focus_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class InvestigationItem(StrictModel):
    investigation_id: str
    question: str
    origin: InvestigationOrigin = "agent"
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    atlas_companion_ids: list[str] = Field(default_factory=list)
    status: InvestigationStatus = "open"
    query_ids: list[str] = Field(default_factory=list)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    candidate_file_ids: list[str] = Field(default_factory=list)
    working_note: str = ""
    created_at: str
    updated_at: str
    warnings: list[str] = Field(default_factory=list)


class DeferredKnowledgeCall(StrictModel):
    tool_call_id: str
    tool_name: Literal["search_review_kb", "open_review_evidence"]
    reason: DeferredKnowledgeCallReason
    created_at: str


class PrimCoverageReport(StrictModel):
    expected_element_ids: list[str] = Field(default_factory=list)
    item_element_ids_before_reflection: list[str] = Field(default_factory=list)
    item_element_ids_after_reflection: list[str] = Field(default_factory=list)
    missing_before_reflection: list[str] = Field(default_factory=list)
    missing_after_reflection: list[str] = Field(default_factory=list)
    unknown_element_ids: list[str] = Field(default_factory=list)
    duplicate_item_element_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    section_parse_degraded: bool = False
    reflection_attempted: bool = False
    reflection_succeeded: bool = False
    warnings: list[str] = Field(default_factory=list)


class ReflectionReport(StrictModel):
    enabled: bool
    triggered: bool = False
    trigger_reason: str | None = None
    first_draft: str | None = None
    first_draft_hash: str | None = None
    first_draft_usage: dict[str, Any] = Field(default_factory=dict)
    second_draft: str | None = None
    second_draft_hash: str | None = None
    missing_before: list[str] = Field(default_factory=list)
    open_investigation_ids_before: list[str] = Field(default_factory=list)
    uninvestigated_plan_ids_before: list[str] = Field(default_factory=list)
    open_investigation_ids_after: list[str] = Field(default_factory=list)
    uninvestigated_plan_ids_after: list[str] = Field(default_factory=list)
    search_count_before: int = 0
    search_count_after: int = 0
    open_count_before: int = 0
    open_count_after: int = 0
    evidence_ids_before: list[str] = Field(default_factory=list)
    evidence_ids_added: list[str] = Field(default_factory=list)
    missing_after: list[str] = Field(default_factory=list)
    completed: bool = False
    fallback_to_first_draft: bool = False
    warnings: list[str] = Field(default_factory=list)


class MedicationReviewPrimTrace(StrictModel):
    schema_version: Literal["8.0"] = "8.0"
    method_family: Literal["prim-rag-v2"] = "prim-rag-v2"
    method_version: str
    requested_profile: ExperimentProfile
    effective_profile: ExperimentProfile
    run_status: RunStatus
    completion_reason: str
    review_run_id: str
    raw_question_hash: str
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    plan_anchors: list[PlanAnchor] = Field(default_factory=list)
    plan_extraction: AnchorExtractionAudit
    patient_modifiers: list[PatientModifier] = Field(default_factory=list)
    modifier_extraction: ModifierExtractionAudit
    knowledge_base_snapshot: dict[str, Any] = Field(default_factory=dict)
    query_records: list[QueryRecord] = Field(default_factory=list)
    investigations: list[InvestigationItem] = Field(default_factory=list)
    deferred_knowledge_calls: list[DeferredKnowledgeCall] = Field(
        default_factory=list
    )
    open_records: list[OpenRecord] = Field(default_factory=list)
    evidence_store: list[EvidenceItem] = Field(default_factory=list)
    coverage_report: PrimCoverageReport
    reflection_report: ReflectionReport
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    budgets: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[TraceError] = Field(default_factory=list)
    final_answer_hash: str


def _as_query(value: QueryRecord | dict[str, Any]) -> QueryRecord:
    return (
        value
        if isinstance(value, QueryRecord)
        else QueryRecord.model_validate(value)
    )


def _as_investigation(
    value: InvestigationItem | dict[str, Any],
) -> InvestigationItem:
    return (
        value
        if isinstance(value, InvestigationItem)
        else InvestigationItem.model_validate(value)
    )


def merge_query_records(
    left: list[QueryRecord] | None,
    right: list[QueryRecord] | None,
) -> list[QueryRecord]:
    by_id: dict[str, QueryRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = _as_query(raw)
        by_id[value.query_id] = value
    return sorted(
        by_id.values(),
        key=lambda value: (value.started_at, value.query_id),
    )


def merge_investigations(
    left: list[InvestigationItem] | None,
    right: list[InvestigationItem] | None,
) -> list[InvestigationItem]:
    by_id: dict[str, InvestigationItem] = {}
    for raw in [*(left or []), *(right or [])]:
        value = _as_investigation(raw)
        current = by_id.get(value.investigation_id)
        if current is None:
            by_id[value.investigation_id] = value
            continue
        by_id[value.investigation_id] = current.model_copy(
            update={
                "focus_plan_ids": list(
                    dict.fromkeys(
                        [*current.focus_plan_ids, *value.focus_plan_ids]
                    )
                ),
                "focus_modifier_ids": list(
                    dict.fromkeys(
                        [
                            *current.focus_modifier_ids,
                            *value.focus_modifier_ids,
                        ]
                    )
                ),
                "atlas_companion_ids": list(
                    dict.fromkeys(
                        [
                            *current.atlas_companion_ids,
                            *value.atlas_companion_ids,
                        ]
                    )
                ),
                "status": value.status,
                "query_ids": list(
                    dict.fromkeys([*current.query_ids, *value.query_ids])
                ),
                "candidate_evidence_ids": list(
                    dict.fromkeys(
                        [
                            *current.candidate_evidence_ids,
                            *value.candidate_evidence_ids,
                        ]
                    )
                ),
                # Selection is the Agent's latest explicit judgment. Unlike
                # candidates, it must be possible to replace or clear it.
                "selected_evidence_ids": list(value.selected_evidence_ids),
                "candidate_file_ids": list(
                    dict.fromkeys(
                        [
                            *current.candidate_file_ids,
                            *value.candidate_file_ids,
                        ]
                    )
                ),
                "working_note": value.working_note or current.working_note,
                "updated_at": max(current.updated_at, value.updated_at),
                "warnings": list(
                    dict.fromkeys([*current.warnings, *value.warnings])
                ),
            }
        )
    return sorted(
        by_id.values(),
        key=lambda value: (value.created_at, value.investigation_id),
    )


def merge_deferred_knowledge_calls(
    left: list[DeferredKnowledgeCall] | None,
    right: list[DeferredKnowledgeCall] | None,
) -> list[DeferredKnowledgeCall]:
    by_id: dict[str, DeferredKnowledgeCall] = {}
    for raw in [*(left or []), *(right or [])]:
        value = (
            raw
            if isinstance(raw, DeferredKnowledgeCall)
            else DeferredKnowledgeCall.model_validate(raw)
        )
        by_id[value.tool_call_id] = value
    return sorted(
        by_id.values(),
        key=lambda value: (value.created_at, value.tool_call_id),
    )


class MedicationReviewPrimState(BaseState, total=False):
    review_run_id: NotRequired[str]
    requested_profile: NotRequired[ExperimentProfile]
    effective_profile: NotRequired[ExperimentProfile]
    raw_case_text: NotRequired[str]
    raw_question_hash: NotRequired[str]
    plan_anchors: NotRequired[list[PlanAnchor]]
    plan_extraction: NotRequired[AnchorExtractionAudit]
    patient_modifiers: NotRequired[list[PatientModifier]]
    modifier_extraction: NotRequired[ModifierExtractionAudit]
    evidence_store: NotRequired[
        Annotated[dict[str, EvidenceItem], merge_evidence_store]
    ]
    query_records: NotRequired[
        Annotated[list[QueryRecord], merge_query_records]
    ]
    investigations: NotRequired[
        Annotated[list[InvestigationItem], merge_investigations]
    ]
    deferred_knowledge_calls: NotRequired[
        Annotated[
            list[DeferredKnowledgeCall],
            merge_deferred_knowledge_calls,
        ]
    ]
    open_records: NotRequired[
        Annotated[list[OpenRecord], merge_open_records]
    ]
    knowledge_base_snapshot: NotRequired[
        Annotated[dict[str, Any], merge_snapshot]
    ]
    search_count: NotRequired[Annotated[int, add_int]]
    open_count: NotRequired[Annotated[int, add_int]]
    technical_attempts: NotRequired[Annotated[int, add_int]]
    reflection_attempted: NotRequired[bool]
    reflection_report: NotRequired[ReflectionReport]
    draft_answer: NotRequired[str]
    coverage_report: NotRequired[PrimCoverageReport]
    final_answer: NotRequired[str]
    run_status: NotRequired[RunStatus]
    warnings: NotRequired[Annotated[list[str], merge_unique_strings]]
    errors: NotRequired[list[TraceError]]
