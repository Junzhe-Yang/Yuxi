from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing import NotRequired

from yuxi.agents import BaseState

ExperimentProfile = Literal["b1", "m1", "m2", "m3"]
AnchorKind = Literal[
    "medication_order",
    "explicit_duration_or_timing",
    "explicit_monitoring_or_followup",
    "explicit_regimen_or_other",
]
RunStatus = Literal["completed", "partial", "failed"]
RecordStatus = Literal[
    "success",
    "success_empty",
    "technical_failed",
    "invalid_source",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanAnchorDraft(StrictModel):
    source_span: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: AnchorKind


class PlanAnchorEnvelope(StrictModel):
    anchors: list[PlanAnchorDraft] = Field(default_factory=list)


class PlanAnchor(StrictModel):
    element_id: str
    source_span: str
    source_start: int
    source_end: int
    label: str
    kind: AnchorKind


class AnchorExtractionAudit(StrictModel):
    status: Literal["disabled", "success", "repaired", "failed", "no_valid_anchor"]
    schema_name: str = "PlanAnchorEnvelope"
    started_at: str
    elapsed_ms: int = 0
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    dropped_drafts: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class TechnicalAttempt(StrictModel):
    attempt: int
    started_at: str
    elapsed_ms: int
    status: Literal["success", "success_empty", "timeout", "embedding_error", "backend_error"]
    returned_count: int = 0
    error_type: str | None = None
    error_message: str | None = None


class EvidenceOccurrence(StrictModel):
    record_id: str
    tool_call_id: str
    source_method: Literal["search", "open"]
    query_text: str
    reason: str
    focus_element_ids: list[str] = Field(default_factory=list)
    rank: int | None = None
    score: float | None = None
    distance: float | None = None
    shown_excerpt: str
    excerpt_start: int = 0
    excerpt_end: int = 0
    excerpt_fallback: bool = False
    parent_evidence_id: str | None = None


class EvidenceItem(StrictModel):
    evidence_id: str
    content_hash: str
    raw_text: str
    source_document: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    occurrences: list[EvidenceOccurrence] = Field(default_factory=list)


class SearchRecord(StrictModel):
    record_id: str
    tool_call_id: str
    query_text: str
    reason: str
    focus_element_ids: list[str] = Field(default_factory=list)
    started_at: str
    elapsed_ms: int
    status: RecordStatus
    returned_count: int = 0
    retained_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    attempts: list[TechnicalAttempt] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class OpenRecord(StrictModel):
    record_id: str
    tool_call_id: str
    parent_evidence_id: str
    investigation_id: str | None = None
    reason: str
    window_before: int
    window_after: int
    started_at: str
    elapsed_ms: int
    status: RecordStatus
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    attempts: list[TechnicalAttempt] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class CoverageReport(StrictModel):
    expected_element_ids: list[str] = Field(default_factory=list)
    item_element_ids_before_patch: list[str] = Field(default_factory=list)
    item_element_ids_after_patch: list[str] = Field(default_factory=list)
    missing_before_patch: list[str] = Field(default_factory=list)
    missing_after_patch: list[str] = Field(default_factory=list)
    unknown_element_ids: list[str] = Field(default_factory=list)
    duplicate_item_element_ids: list[str] = Field(default_factory=list)
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    section_parse_degraded: bool = False
    patch_attempted: bool = False
    patch_succeeded: bool = False
    patch_usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TraceError(StrictModel):
    stage: str
    error_type: str
    message: str
    record_id: str | None = None


class MedicationReviewLiteTrace(StrictModel):
    schema_version: Literal["4.0"] = "4.0"
    method_family: Literal["pat-rag-v1"] = "pat-rag-v1"
    method_version: str
    experiment_profile: ExperimentProfile
    run_status: RunStatus
    completion_reason: str
    review_run_id: str
    raw_question_hash: str
    prompt_version: str
    prompt_hash: str
    plan_anchors: list[PlanAnchor] = Field(default_factory=list)
    anchor_extraction: AnchorExtractionAudit
    knowledge_base_snapshot: dict[str, Any] = Field(default_factory=dict)
    search_records: list[SearchRecord] = Field(default_factory=list)
    open_records: list[OpenRecord] = Field(default_factory=list)
    evidence_store: list[EvidenceItem] = Field(default_factory=list)
    coverage_report: CoverageReport
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    budgets: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[TraceError] = Field(default_factory=list)
    final_answer_hash: str


def _as_occurrence(value: EvidenceOccurrence | dict[str, Any]) -> EvidenceOccurrence:
    return value if isinstance(value, EvidenceOccurrence) else EvidenceOccurrence.model_validate(value)


def _as_evidence(value: EvidenceItem | dict[str, Any]) -> EvidenceItem:
    return value if isinstance(value, EvidenceItem) else EvidenceItem.model_validate(value)


def merge_evidence_store(
    left: dict[str, EvidenceItem] | None,
    right: dict[str, EvidenceItem] | None,
) -> dict[str, EvidenceItem]:
    merged = {key: _as_evidence(value) for key, value in (left or {}).items()}
    for evidence_id, raw_item in (right or {}).items():
        item = _as_evidence(raw_item)
        current = merged.get(evidence_id)
        if current is None:
            merged[evidence_id] = item
            continue
        occurrence_index = {
            (value.tool_call_id, value.record_id, value.rank, value.source_method): value
            for value in current.occurrences
        }
        for occurrence in item.occurrences:
            occurrence = _as_occurrence(occurrence)
            occurrence_index[
                (
                    occurrence.tool_call_id,
                    occurrence.record_id,
                    occurrence.rank,
                    occurrence.source_method,
                )
            ] = occurrence
        merged[evidence_id] = current.model_copy(
            update={
                "occurrences": sorted(
                    occurrence_index.values(),
                    key=lambda value: (
                        value.record_id,
                        value.rank if value.rank is not None else 10**9,
                        value.source_method,
                    ),
                )
            }
        )
    return merged


def _merge_records(
    left: list[SearchRecord | OpenRecord | dict[str, Any]] | None,
    right: list[SearchRecord | OpenRecord | dict[str, Any]] | None,
    model: type[SearchRecord] | type[OpenRecord],
) -> list[Any]:
    by_id: dict[str, Any] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, model) else model.model_validate(raw)
        by_id[value.record_id] = value
    return sorted(by_id.values(), key=lambda value: (value.started_at, value.record_id))


def merge_search_records(
    left: list[SearchRecord] | None,
    right: list[SearchRecord] | None,
) -> list[SearchRecord]:
    return _merge_records(left, right, SearchRecord)


def merge_open_records(
    left: list[OpenRecord] | None,
    right: list[OpenRecord] | None,
) -> list[OpenRecord]:
    return _merge_records(left, right, OpenRecord)


def merge_unique_strings(left: list[str] | None, right: list[str] | None) -> list[str]:
    return list(dict.fromkeys([*(left or []), *(right or [])]))


def add_int(left: int | None, right: int | None) -> int:
    return int(left or 0) + int(right or 0)


def merge_snapshot(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    return {**left, **right}


class MedicationReviewLiteState(BaseState, total=False):
    review_run_id: NotRequired[str]
    experiment_profile: NotRequired[ExperimentProfile]
    raw_case_text: NotRequired[str]
    raw_question_hash: NotRequired[str]
    plan_anchors: NotRequired[list[PlanAnchor]]
    anchor_extraction: NotRequired[AnchorExtractionAudit]
    evidence_store: NotRequired[
        Annotated[dict[str, EvidenceItem], merge_evidence_store]
    ]
    search_records: NotRequired[Annotated[list[SearchRecord], merge_search_records]]
    open_records: NotRequired[Annotated[list[OpenRecord], merge_open_records]]
    knowledge_base_snapshot: NotRequired[
        Annotated[dict[str, Any], merge_snapshot]
    ]
    search_count: NotRequired[Annotated[int, add_int]]
    open_count: NotRequired[Annotated[int, add_int]]
    technical_attempts: NotRequired[Annotated[int, add_int]]
    coverage_report: NotRequired[CoverageReport]
    final_answer: NotRequired[str]
    run_status: NotRequired[RunStatus]
    warnings: NotRequired[Annotated[list[str], merge_unique_strings]]
    errors: NotRequired[list[TraceError]]
