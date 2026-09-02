from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired

from pydantic import Field

from yuxi.agents.buildin.medication_review_lite.models import merge_snapshot
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimState,
    MedicationReviewPrimTrace,
    RecordStatus,
    StrictModel,
)

SelectorStatus = Literal["success", "repaired", "empty", "failed"]
SelectorTrigger = Literal["after_query", "first_draft"]
V7ProbePass = Literal[
    "initial_probe",
    "complementary_probe",
    "adaptive_probe",
]
AdaptiveRetrievalIntent = Literal[
    "source_discovery",
    "within_document_localization",
    "adjacent_context",
]
AdaptiveRecoveryStatus = Literal[
    "pending",
    "resolved",
    "closed_insufficient",
]
AdaptiveGapStatus = Literal["current", "stale", "missing"]
AdaptiveDecisionTag = Annotated[str, Field(min_length=1, max_length=80)]
AdaptiveAuditDimension = Literal[
    "indication_and_expected_benefit",
    "dose_route_frequency_duration_titration",
    "patient_specific_safety_contraindication_interaction",
    "monitoring_followup_and_stop_rules",
    "alternatives_and_missing_therapy",
    "cross_regimen_and_long_term_management",
]
AdaptiveAuditStatus = Literal["covered", "not_applicable", "gap"]
AdaptiveInvestigationKind = Literal[
    "current_regimen_review",
    "improvement_plan",
    "monitoring_followup",
    "cross_regimen_review",
]
AdaptiveReviewOutcome = Literal["appropriate", "adjust", "avoid"]


# These selector models are retained only for replaying historical V2 experiments.
class CompanionCue(StrictModel):
    companion_id: str
    question_hint: str
    linked_plan_ids: list[str] = Field(default_factory=list)
    linked_modifier_ids: list[str] = Field(default_factory=list)
    atlas_cue_ids: list[str] = Field(default_factory=list)
    suggested_doc_ids: list[str] = Field(default_factory=list)
    novelty_explanation: str


class CompanionSelectorAudit(StrictModel):
    status: SelectorStatus
    trigger_type: SelectorTrigger
    created_after_query_id: str | None = None
    observed_query_ids: list[str] = Field(default_factory=list)
    observed_investigation_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    selector_input_hash: str
    atlas_snapshot_hash: str
    started_at: str
    elapsed_ms: int
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    dropped_items: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class CompanionSelection(StrictModel):
    trigger_type: SelectorTrigger
    created_after_query_id: str | None = None
    observed_query_ids: list[str] = Field(default_factory=list)
    companion_cues: list[CompanionCue] = Field(default_factory=list)
    selector_audit: CompanionSelectorAudit


class AtlasDocumentOpenRecord(StrictModel):
    record_id: str
    tool_call_id: str
    doc_id: str
    title: str
    reason: str
    topic_count: int
    cue_ids: list[str] = Field(default_factory=list)


class V7AgendaItem(StrictModel):
    investigation_id: str
    question: str
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    distinct_scope: str


class AcmAgendaItemDraft(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    distinct_scope: str = Field(min_length=1, max_length=300)
    why_it_matters: str = Field(default="", max_length=500)
    decision_tags: list[AdaptiveDecisionTag] = Field(
        default_factory=list,
        description="开放审计标签；可使用协议示例标签，也可填写病例特有标签。",
    )
    investigation_kind: AdaptiveInvestigationKind | None = Field(
        default=None,
        description=("adaptive_coverage 下必填：逐项现用药审查、改进方案、监测随访或跨方案审查。"),
    )
    evidence_obligations: list[str] = Field(
        default_factory=list,
        description=(
            "该调查必须用直接证据逐项回答的最小义务；adaptive_coverage 下不能为空，数量由病例决定，不是固定槽位。"
        ),
    )
    parent_evidence_ids: list[str] = Field(default_factory=list)


class V7AgendaItemDraft(AcmAgendaItemDraft):
    """Backward-compatible name used by legacy V7 tests and callers."""


class AdaptiveAgendaItemDraft(AcmAgendaItemDraft):
    """Semantic name for adaptive agenda creation and append calls."""


class V7InvestigationAgenda(StrictModel):
    agenda_id: str
    experiment_arm: Literal["a2_k2", "a2_k3"]
    required_count: int
    created_at: str
    items: list[V7AgendaItem] = Field(default_factory=list)


class V7ProbeRecord(StrictModel):
    probe_record_id: str
    query_id: str
    investigation_id: str
    probe_pass: V7ProbePass
    uncovered_aspect: str = ""
    status: RecordStatus


class V7RetrievalCandidate(StrictModel):
    rank: int
    file_id: str | None = None
    source_document: str | None = None
    chunk_id: str | int | None = None
    chunk_index: int | str | None = None
    content_hash: str
    score: float | None = None
    distance: float | None = None


class V7RetrievalRecord(StrictModel):
    record_id: str
    query_id: str
    fetch_k: int
    visible_k: int
    returned_count: int
    candidates: list[V7RetrievalCandidate] = Field(default_factory=list)


class V7CheckpointRecord(StrictModel):
    checkpoint_id: str
    created_at: str
    executed_search_calls: int
    successful_search_calls: int
    contract_fingerprint: str
    incomplete_reasons: list[str] = Field(default_factory=list)
    candidate_answer_hash: str


class V7ContractReport(StrictModel):
    status: Literal["disabled", "completed", "incomplete"]
    experiment_arm: Literal["a0", "a1", "a2_k2", "a2_k3"]
    retrieval_depth: Literal["top10", "shadow_top25", "visible_top25"]
    minimum_search_calls: int = 0
    maximum_search_calls: int
    required_investigation_count: int = 0
    agenda_created: bool = False
    agenda_investigation_ids: list[str] = Field(default_factory=list)
    covered_plan_ids: list[str] = Field(default_factory=list)
    missing_plan_ids: list[str] = Field(default_factory=list)
    executed_search_calls: int = 0
    successful_search_calls: int = 0
    initial_probe_completed_ids: list[str] = Field(default_factory=list)
    complementary_probe_completed_ids: list[str] = Field(default_factory=list)
    missing_initial_probe_ids: list[str] = Field(default_factory=list)
    missing_complementary_probe_ids: list[str] = Field(default_factory=list)
    open_required_investigation_ids: list[str] = Field(default_factory=list)
    incomplete_reasons: list[str] = Field(default_factory=list)
    premature_final_attempts: int = 0
    fetch_k: int = 10
    visible_k: int = 10


class AdaptiveAgendaRevision(StrictModel):
    revision: int
    tool_call_id: str
    created_at: str
    reason: str
    added_investigation_ids: list[str] = Field(default_factory=list)


class AdaptiveAgendaItem(StrictModel):
    investigation_id: str
    question: str
    why_it_matters: str
    distinct_scope: str
    decision_tags: list[AdaptiveDecisionTag] = Field(default_factory=list)
    investigation_kind: AdaptiveInvestigationKind | None = None
    evidence_obligations: list[str] = Field(default_factory=list)
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    parent_evidence_ids: list[str] = Field(default_factory=list)
    created_at: str
    revision: int


class AdaptiveInvestigationAgenda(StrictModel):
    agenda_id: str
    created_at: str
    updated_at: str
    revision: int
    items: list[AdaptiveAgendaItem] = Field(default_factory=list)
    revisions: list[AdaptiveAgendaRevision] = Field(default_factory=list)


class AdaptiveObligationSupport(StrictModel):
    obligation: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1)


class AdaptiveCoverageAuditEntry(StrictModel):
    dimension: AdaptiveAuditDimension
    status: AdaptiveAuditStatus
    investigation_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=800)


class AdaptiveInvestigationMeta(StrictModel):
    investigation_id: str
    resolved_aspects: list[str] = Field(default_factory=list)
    remaining_aspects: list[str] = Field(default_factory=list)
    obligation_supports: list[AdaptiveObligationSupport] = Field(default_factory=list)
    review_outcome: AdaptiveReviewOutcome | None = None
    residual_uncertainty: str = ""
    closure_reason: str = ""
    updated_at: str


class AdaptiveProbeRecord(StrictModel):
    probe_record_id: str
    query_id: str
    investigation_id: str
    uncovered_aspect: str
    retrieval_intent: AdaptiveRetrievalIntent
    route_key: str
    status: RecordStatus
    redundant: bool = False


class AdaptiveRecoveryRequirement(StrictModel):
    recovery_id: str
    investigation_id: str
    source_query_id: str
    file_id: str | None = None
    uncovered_aspect: str
    status: AdaptiveRecoveryStatus = "pending"
    created_at: str
    resolved_by_query_id: str | None = None


class AdaptiveGapAssessment(StrictModel):
    assessment_id: str
    created_at: str
    state_fingerprint: str
    material_gap_found: bool
    rationale: str
    unsupported_investigation_ids: list[str] = Field(default_factory=list)
    unsupported_obligations: dict[str, list[str]] = Field(default_factory=dict)
    proposed_investigation_ids: list[str] = Field(default_factory=list)
    coverage_audit: list[AdaptiveCoverageAuditEntry] = Field(default_factory=list)


class AdaptiveCheckpointRecord(StrictModel):
    checkpoint_id: str
    created_at: str
    state_fingerprint: str
    incomplete_reasons: list[str] = Field(default_factory=list)
    candidate_answer_hash: str


class AdaptiveCoverageReport(StrictModel):
    status: Literal["disabled", "completed", "incomplete"]
    incomplete_reasons: list[str] = Field(default_factory=list)
    agenda_created: bool = False
    agenda_investigation_ids: list[str] = Field(default_factory=list)
    covered_plan_ids: list[str] = Field(default_factory=list)
    uncovered_plan_ids: list[str] = Field(default_factory=list)
    current_regimen_reviewed_plan_ids: list[str] = Field(default_factory=list)
    missing_current_regimen_review_plan_ids: list[str] = Field(default_factory=list)
    action_required_plan_ids: list[str] = Field(default_factory=list)
    improvement_plan_covered_plan_ids: list[str] = Field(default_factory=list)
    missing_improvement_plan_ids: list[str] = Field(default_factory=list)
    unprobed_investigation_ids: list[str] = Field(default_factory=list)
    open_investigation_ids: list[str] = Field(default_factory=list)
    pending_recovery_ids: list[str] = Field(default_factory=list)
    eligible_investigation_ids: list[str] = Field(default_factory=list)
    unprobed_evidence_obligations: dict[str, list[str]] = Field(default_factory=dict)
    unsupported_evidence_obligations: dict[str, list[str]] = Field(default_factory=dict)
    total_evidence_obligation_count: int = 0
    supported_evidence_obligation_count: int = 0
    gap_assessment_status: AdaptiveGapStatus = "missing"
    latest_gap_assessment_id: str | None = None
    state_fingerprint: str = ""
    actual_investigation_count: int = 0
    executed_search_calls: int = 0
    successful_search_calls: int = 0
    maximum_search_calls: int = 0
    fetch_k: int = 25
    visible_k: int = 10


def _merge_document_open_records(
    left: list[AtlasDocumentOpenRecord] | None,
    right: list[AtlasDocumentOpenRecord] | None,
) -> list[AtlasDocumentOpenRecord]:
    by_id: dict[str, AtlasDocumentOpenRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AtlasDocumentOpenRecord) else AtlasDocumentOpenRecord.model_validate(raw)
        by_id[value.record_id] = value
    return list(by_id.values())


def _merge_probe_records(
    left: list[V7ProbeRecord] | None,
    right: list[V7ProbeRecord] | None,
) -> list[V7ProbeRecord]:
    by_id: dict[str, V7ProbeRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, V7ProbeRecord) else V7ProbeRecord.model_validate(raw)
        by_id[value.probe_record_id] = value
    return list(by_id.values())


def _merge_retrieval_records(
    left: list[V7RetrievalRecord] | None,
    right: list[V7RetrievalRecord] | None,
) -> list[V7RetrievalRecord]:
    by_id: dict[str, V7RetrievalRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, V7RetrievalRecord) else V7RetrievalRecord.model_validate(raw)
        by_id[value.record_id] = value
    return list(by_id.values())


def _merge_checkpoint_records(
    left: list[V7CheckpointRecord] | None,
    right: list[V7CheckpointRecord] | None,
) -> list[V7CheckpointRecord]:
    by_id: dict[str, V7CheckpointRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, V7CheckpointRecord) else V7CheckpointRecord.model_validate(raw)
        by_id[value.checkpoint_id] = value
    return list(by_id.values())


def _merge_adaptive_meta(
    left: list[AdaptiveInvestigationMeta] | None,
    right: list[AdaptiveInvestigationMeta] | None,
) -> list[AdaptiveInvestigationMeta]:
    by_id: dict[str, AdaptiveInvestigationMeta] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AdaptiveInvestigationMeta) else AdaptiveInvestigationMeta.model_validate(raw)
        by_id[value.investigation_id] = value
    return list(by_id.values())


def merge_adaptive_agenda(
    left: AdaptiveInvestigationAgenda | dict[str, Any] | None,
    right: AdaptiveInvestigationAgenda | dict[str, Any] | None,
) -> AdaptiveInvestigationAgenda | None:
    """Accept full append-only agenda snapshots and reject history rewrites."""
    if left is None:
        if right is None:
            return None
        return (
            right
            if isinstance(right, AdaptiveInvestigationAgenda)
            else AdaptiveInvestigationAgenda.model_validate(right)
        )
    current = (
        left if isinstance(left, AdaptiveInvestigationAgenda) else AdaptiveInvestigationAgenda.model_validate(left)
    )
    if right is None:
        return current
    incoming = (
        right if isinstance(right, AdaptiveInvestigationAgenda) else AdaptiveInvestigationAgenda.model_validate(right)
    )
    if incoming.agenda_id != current.agenda_id or incoming.created_at != current.created_at:
        raise ValueError("adaptive agenda identity cannot change")
    if incoming.items[: len(current.items)] != current.items:
        raise ValueError("adaptive agenda items are append-only and immutable")
    if incoming.revisions[: len(current.revisions)] != current.revisions:
        raise ValueError("adaptive agenda revisions are append-only and immutable")
    if incoming.revision < current.revision:
        raise ValueError("adaptive agenda revision cannot move backwards")
    if len(incoming.items) > len(current.items) and incoming.revision <= current.revision:
        raise ValueError("adaptive agenda append must advance revision")
    return incoming


def _merge_adaptive_probe_records(
    left: list[AdaptiveProbeRecord] | None,
    right: list[AdaptiveProbeRecord] | None,
) -> list[AdaptiveProbeRecord]:
    by_id: dict[str, AdaptiveProbeRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AdaptiveProbeRecord) else AdaptiveProbeRecord.model_validate(raw)
        by_id[value.probe_record_id] = value
    return list(by_id.values())


def _merge_adaptive_recoveries(
    left: list[AdaptiveRecoveryRequirement] | None,
    right: list[AdaptiveRecoveryRequirement] | None,
) -> list[AdaptiveRecoveryRequirement]:
    by_id: dict[str, AdaptiveRecoveryRequirement] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AdaptiveRecoveryRequirement) else AdaptiveRecoveryRequirement.model_validate(raw)
        by_id[value.recovery_id] = value
    return list(by_id.values())


def _merge_adaptive_gap_assessments(
    left: list[AdaptiveGapAssessment] | None,
    right: list[AdaptiveGapAssessment] | None,
) -> list[AdaptiveGapAssessment]:
    by_id: dict[str, AdaptiveGapAssessment] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AdaptiveGapAssessment) else AdaptiveGapAssessment.model_validate(raw)
        by_id[value.assessment_id] = value
    return list(by_id.values())


def _merge_adaptive_checkpoint_records(
    left: list[AdaptiveCheckpointRecord] | None,
    right: list[AdaptiveCheckpointRecord] | None,
) -> list[AdaptiveCheckpointRecord]:
    by_id: dict[str, AdaptiveCheckpointRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, AdaptiveCheckpointRecord) else AdaptiveCheckpointRecord.model_validate(raw)
        by_id[value.checkpoint_id] = value
    return list(by_id.values())


class MedicationReviewAcmPrimState(MedicationReviewPrimState, total=False):
    atlas_snapshot: NotRequired[Annotated[dict[str, Any], merge_snapshot]]
    atlas_document_open_records: NotRequired[Annotated[list[AtlasDocumentOpenRecord], _merge_document_open_records]]
    v7_experiment_snapshot: NotRequired[Annotated[dict[str, Any], merge_snapshot]]
    v7_agenda: NotRequired[V7InvestigationAgenda]
    v7_probe_records: NotRequired[Annotated[list[V7ProbeRecord], _merge_probe_records]]
    v7_retrieval_records: NotRequired[Annotated[list[V7RetrievalRecord], _merge_retrieval_records]]
    v7_checkpoint_records: NotRequired[Annotated[list[V7CheckpointRecord], _merge_checkpoint_records]]
    adaptive_protocol_snapshot: NotRequired[Annotated[dict[str, Any], merge_snapshot]]
    adaptive_agenda: NotRequired[Annotated[AdaptiveInvestigationAgenda, merge_adaptive_agenda]]
    adaptive_investigation_meta: NotRequired[Annotated[list[AdaptiveInvestigationMeta], _merge_adaptive_meta]]
    adaptive_probe_records: NotRequired[Annotated[list[AdaptiveProbeRecord], _merge_adaptive_probe_records]]
    adaptive_recovery_requirements: NotRequired[
        Annotated[
            list[AdaptiveRecoveryRequirement],
            _merge_adaptive_recoveries,
        ]
    ]
    adaptive_gap_assessments: NotRequired[
        Annotated[
            list[AdaptiveGapAssessment],
            _merge_adaptive_gap_assessments,
        ]
    ]
    adaptive_checkpoint_records: NotRequired[
        Annotated[
            list[AdaptiveCheckpointRecord],
            _merge_adaptive_checkpoint_records,
        ]
    ]
    adaptive_retrieval_records: NotRequired[Annotated[list[V7RetrievalRecord], _merge_retrieval_records]]


class MedicationReviewAcmPrimTrace(MedicationReviewPrimTrace):
    schema_version: Literal["10.0"] = "10.0"
    method_family: Literal["acm-prim-rag-v3"] = "acm-prim-rag-v3"
    atlas_snapshot: dict[str, Any] = Field(default_factory=dict)
    atlas_document_open_records: list[AtlasDocumentOpenRecord] = Field(default_factory=list)


class MedicationReviewAcmV7Trace(MedicationReviewAcmPrimTrace):
    schema_version: Literal["11.0"] = "11.0"
    method_family: Literal["acm-prim-rag-v7"] = "acm-prim-rag-v7"
    experiment_arm: Literal["a1", "a2_k2", "a2_k3", "a0"]
    retrieval_depth: Literal["top10", "shadow_top25", "visible_top25"]
    contract_report: V7ContractReport
    investigation_agenda: V7InvestigationAgenda | None = None
    probe_records: list[V7ProbeRecord] = Field(default_factory=list)
    retrieval_records: list[V7RetrievalRecord] = Field(default_factory=list)
    checkpoint_records: list[V7CheckpointRecord] = Field(default_factory=list)


class MedicationReviewAcmAdaptiveTrace(MedicationReviewAcmPrimTrace):
    schema_version: Literal["12.0"] = "12.0"
    method_family: Literal["acm-prim-rag-v8"] = "acm-prim-rag-v8"
    protocol: Literal["adaptive_coverage"] = "adaptive_coverage"
    retrieval_depth: Literal["shadow_top25"] = "shadow_top25"
    adaptive_coverage_report: AdaptiveCoverageReport
    investigation_agenda: AdaptiveInvestigationAgenda | None = None
    investigation_meta: list[AdaptiveInvestigationMeta] = Field(default_factory=list)
    probe_records: list[AdaptiveProbeRecord] = Field(default_factory=list)
    recovery_requirements: list[AdaptiveRecoveryRequirement] = Field(default_factory=list)
    gap_assessments: list[AdaptiveGapAssessment] = Field(default_factory=list)
    retrieval_records: list[V7RetrievalRecord] = Field(default_factory=list)
    checkpoint_records: list[AdaptiveCheckpointRecord] = Field(default_factory=list)
