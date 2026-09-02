from __future__ import annotations

from typing import Any, Literal, NotRequired

from pydantic import BaseModel, ConfigDict, Field

from yuxi.agents import BaseState

SCHEMA_VERSION = "1.0"
METHOD_VERSION = "relation-coverage-a1-vector-atomic-v1"
QUERY_STYLE = "atomic_clinical_proposition_v1"
V2_SCHEMA_VERSION = "2.0"
V2_METHOD_VERSION = "pea-rag-mfull-vector-v1.4"
V2_QUERY_POLICY = "agent-clinical-proposition-vector-v1"
V3_SCHEMA_VERSION = "3.0"
V3_METHOD_FAMILY = "pea-rag-v2-vector-v1"
V3_DEFAULT_METHOD_VERSION = "pea-rag-v2-dynamic-claims-vector-v1"
V3_QUERY_POLICY = "agent-clinical-proposition-batch-vector-v1"
STRUCTURED_IO_VERSION = "json-schema-prompt-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Diagnosis(StrictModel):
    diagnosis_id: str | None = None
    name: str
    source_mention: str
    status: Literal["active", "history", "suspected", "unknown"] = "unknown"


class Medication(StrictModel):
    medication_id: str | None = None
    source_mention: str
    generic_name: str | None = None
    normalized_name: str | None = None
    normalization_source: Literal["input", "map", "model", "unresolved"] = "unresolved"
    dose: str | None = None
    dose_unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    status: Literal["current", "planned", "historical", "discontinued", "unknown"] = "unknown"


class LabValue(StrictModel):
    lab_id: str | None = None
    indicator: str
    value: str
    unit: str | None = None
    measured_at: str | None = None
    source_mention: str | None = None


class PatientCaseInput(StrictModel):
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    renal_function: LabValue | None = None
    hepatic_function: LabValue | None = None
    other_labs: list[LabValue] = Field(default_factory=list)
    clinical_risks: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)


class PatientCase(PatientCaseInput):
    case_id: str
    raw_question_hash: str


class ReviewSlot(StrictModel):
    slot_id: str
    slot_type: Literal[
        "medication_profile",
        "organ_function",
        "drug_drug",
        "drug_disease",
        "prescribing_omission",
        "duplication",
        "cumulative_burden",
    ]
    subject_ids: list[str] = Field(default_factory=list)
    target_ids: list[str] = Field(default_factory=list)
    patient_constraints: dict[str, Any] = Field(default_factory=dict)
    required_attributes: list[str] = Field(default_factory=list)
    priority: int = 1
    generation_reason: str
    applicability_status: Literal["queryable", "not_applicable", "unsupported"] = "queryable"
    covered_by_bundle_ids: list[str] = Field(default_factory=list)


class QueryBundle(StrictModel):
    bundle_id: str
    slot_ids: list[str]
    query_role: Literal["challenge"] = "challenge"
    retrieval_view: Literal["dense"] = "dense"
    query_style: Literal["atomic_clinical_proposition_v1"] = QUERY_STYLE
    query_text: str
    template_id: str
    template_version: str = "1"
    expected_evidence_types: list[str] = Field(default_factory=list)
    core_entity_ids: list[str] = Field(default_factory=list)
    validation_status: Literal["valid", "invalid"] = "valid"
    validation_errors: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceOccurrence(StrictModel):
    bundle_id: str
    slot_ids: list[str]
    rank: int
    score: float | None = None
    distance: float | None = None


class RetrievedEvidence(StrictModel):
    evidence_id: str
    raw_text: str
    source_document: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | str | None = None
    score_type: Literal["cosine_similarity"] = "cosine_similarity"
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    occurrences: list[EvidenceOccurrence] = Field(default_factory=list)


class RetrievalRecord(StrictModel):
    bundle_id: str
    query_text: str
    slot_ids: list[str]
    status: Literal[
        "success",
        "success_empty",
        "timeout",
        "embedding_error",
        "backend_error",
        "invalid_query",
        "skipped_budget",
    ]
    started_at: str
    elapsed_ms: int
    returned_count: int = 0
    retained_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class TraceError(StrictModel):
    type: str
    message: str
    stage: str


class MedicationReviewTrace(StrictModel):
    schema_version: str = SCHEMA_VERSION
    method_version: str = METHOD_VERSION
    review_run_id: str
    run_status: Literal[
        "completed",
        "partial",
        "parse_failed",
        "plan_failed",
        "invalid_config",
        "plan_budget_exceeded",
        "retrieval_failed",
    ]
    case_id: str | None = None
    patient_case: dict[str, Any] | None = None
    review_slots: list[dict[str, Any]] = Field(default_factory=list)
    query_bundles: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_records: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_base_snapshot: dict[str, Any] = Field(default_factory=dict)
    agent_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MedicationReviewState(BaseState, total=False):
    raw_question: NotRequired[str]
    raw_question_hash: NotRequired[str]
    review_run_id: NotRequired[str]
    run_status: NotRequired[str]
    patient_case: NotRequired[dict[str, Any] | None]
    review_slots: NotRequired[list[dict[str, Any]]]
    query_bundles: NotRequired[list[dict[str, Any]]]
    retrieval_records: NotRequired[list[dict[str, Any]]]
    evidence: NotRequired[list[dict[str, Any]]]
    knowledge_base_snapshot: NotRequired[dict[str, Any]]
    agent_config_snapshot: NotRequired[dict[str, Any]]
    usage: NotRequired[dict[str, Any]]
    warnings: NotRequired[list[str]]
    errors: NotRequired[list[dict[str, Any]]]
    extraction_mode: NotRequired[str]


PlanElementType = Literal[
    "medication_order",
    "combination_regimen",
    "treatment_intent",
    "treatment_phase",
    "duration_or_schedule",
    "evaluation_timing",
    "monitoring_plan",
    "follow_up_plan",
    "switch_stop_escalation_rule",
    "nonpharmacologic_plan",
    "other_explicit_plan",
]

PatientFactType = Literal[
    "demographic",
    "symptom_or_sign",
    "disease_status",
    "organ_function",
    "laboratory",
    "allergy",
    "history",
    "functional_status",
    "clinical_risk",
    "other",
]

ReviewTargetType = Literal[
    "indication_support",
    "geriatric_appropriateness",
    "dose_frequency",
    "duration",
    "drug_disease",
    "drug_drug",
    "organ_function",
    "duplication",
    "cumulative_burden",
    "monitoring_requirement",
    "prescribing_omission",
    "regimen_consistency",
    "timing_consistency",
    "other",
]

EvidenceRole = Literal["support", "challenge", "condition", "recommendation", "gap"]
Judgement = Literal[
    "appropriate",
    "appropriate_with_monitoring",
    "needs_adjustment",
    "inappropriate",
    "insufficient_evidence",
]


class PatientFactDraft(StrictModel):
    fact_type: PatientFactType
    source_span: str
    normalized_summary: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class PatientFact(PatientFactDraft):
    fact_id: str
    source_start: int
    source_end: int


class TreatmentPlanElementDraft(StrictModel):
    draft_id: str
    element_type: PlanElementType
    source_span: str
    normalized_summary: str
    parent_draft_id: str | None = None
    component_draft_ids: list[str] = Field(default_factory=list)
    target_diagnosis_mentions: list[str] = Field(default_factory=list)
    medication_mentions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class CasePlanExtractionDraft(StrictModel):
    patient_case: PatientCaseInput
    patient_facts: list[PatientFactDraft] = Field(default_factory=list)
    plan_elements: list[TreatmentPlanElementDraft] = Field(default_factory=list)


class TreatmentPlanElement(StrictModel):
    element_id: str
    element_type: PlanElementType
    source_span: str
    source_start: int
    source_end: int
    normalized_summary: str
    parent_element_id: str | None = None
    component_element_ids: list[str] = Field(default_factory=list)
    target_diagnosis_ids: list[str] = Field(default_factory=list)
    medication_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class PlanValidationIssue(StrictModel):
    issue_type: str
    message: str
    element_id: str | None = None
    draft_id: str | None = None
    source_span: str | None = None
    severity: Literal["error", "warning"] = "error"


class PlanRepairOperation(StrictModel):
    operation: Literal[
        "add_element",
        "update_element_type",
        "update_attributes",
        "update_relationship",
        "remove_duplicate",
        "remove_hallucinated",
    ]
    target_element_id: str | None = None
    element: TreatmentPlanElementDraft | None = None
    replacement_type: PlanElementType | None = None
    replacement_attributes: dict[str, Any] = Field(default_factory=dict)
    replacement_parent_element_id: str | None = None
    replacement_component_element_ids: list[str] = Field(default_factory=list)
    reason: str


class PlanVerificationResult(StrictModel):
    missing_explicit_spans: list[str] = Field(default_factory=list)
    duplicated_element_ids: list[str] = Field(default_factory=list)
    hallucinated_element_ids: list[str] = Field(default_factory=list)
    incorrect_element_types: list[str] = Field(default_factory=list)
    incorrect_relationships: list[str] = Field(default_factory=list)
    repair_operations: list[PlanRepairOperation] = Field(default_factory=list)


class StructuredInvocationAudit(StrictModel):
    stage: str
    schema_name: str
    started_at: str
    elapsed_ms: int
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    repaired: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    failure_kind: Literal["provider", "structured"] | None = None
    error_type: str | None = None
    error_message: str | None = None


class PlanExtractionAudit(StrictModel):
    extraction: StructuredInvocationAudit | None = None
    deterministic_issues: list[PlanValidationIssue] = Field(default_factory=list)
    verification: StructuredInvocationAudit | None = None
    verifier_result: PlanVerificationResult | None = None
    applied_repair_operations: list[PlanRepairOperation] = Field(default_factory=list)
    runtime_revisions: list[dict[str, Any]] = Field(default_factory=list)


class ReviewTarget(StrictModel):
    target_id: str
    target_type: ReviewTargetType
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_patient_fact_ids: list[str] = Field(default_factory=list)
    generation_source: Literal["deterministic", "agent", "evidence_triggered"]
    generation_reason: str
    priority: Literal["high", "medium", "low"]
    required_dimensions: list[str] = Field(default_factory=list)


class ReviewTargetState(StrictModel):
    target: ReviewTarget
    status: Literal[
        "unexamined",
        "searched",
        "evidence_found",
        "no_applicable_evidence",
        "finding_submitted",
        "insufficient",
    ] = "unexamined"
    query_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class SearchIntent(StrictModel):
    query_id: str
    query_text: str
    target_element_ids: list[str] = Field(default_factory=list)
    target_review_ids: list[str] = Field(default_factory=list)
    intended_evidence_role: EvidenceRole
    search_reason: str


class TechnicalAttempt(StrictModel):
    attempt: int
    started_at: str
    elapsed_ms: int
    status: Literal["success", "success_empty", "timeout", "embedding_error", "backend_error"]
    returned_count: int = 0
    error_type: str | None = None
    error_message: str | None = None


class EvidenceSearchRecord(StrictModel):
    intent: SearchIntent
    status: Literal[
        "success",
        "success_empty",
        "all_irrelevant",
        "all_not_applicable",
        "assessment_failed",
        "technical_failed",
        "invalid_query",
        "skipped_budget",
        "skipped_early_stop",
    ]
    attempts: list[TechnicalAttempt] = Field(default_factory=list)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    active_evidence_ids: list[str] = Field(default_factory=list)
    new_requirements_closed: list[str] = Field(default_factory=list)
    duplicate_ratio: float = 0.0


class EvidenceOpenRecord(StrictModel):
    open_id: str
    parent_evidence_id: str
    reason: str
    window_before: int
    window_after: int
    status: Literal["success", "success_empty", "timeout", "backend_error", "invalid_source"]
    started_at: str
    elapsed_ms: int
    evidence_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class EvidenceOccurrenceV2(StrictModel):
    query_id: str
    target_element_ids: list[str] = Field(default_factory=list)
    target_review_ids: list[str] = Field(default_factory=list)
    intended_evidence_role: EvidenceRole
    rank: int
    score: float | None = None
    distance: float | None = None


class EvidenceItem(StrictModel):
    evidence_id: str
    raw_text: str
    source_document: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    source_method: Literal["search", "open"] = "search"
    parent_evidence_id: str | None = None
    occurrences: list[EvidenceOccurrenceV2] = Field(default_factory=list)


class EvidenceBindingAssessment(StrictModel):
    element_id: str | None = None
    review_target_id: str | None = None
    relevance: Literal["direct", "partial", "background", "irrelevant"]
    applicability: Literal[
        "applicable",
        "partially_applicable",
        "not_applicable",
        "uncertain",
    ]
    polarity: Literal[
        "supports_appropriate",
        "supports_inappropriate",
        "conditional",
        "supports_recommendation",
        "neutral",
    ]
    covered_dimensions: list[str] = Field(default_factory=list)
    matched_patient_fact_ids: list[str] = Field(default_factory=list)
    mismatched_patient_fact_ids: list[str] = Field(default_factory=list)
    missing_patient_information: list[str] = Field(default_factory=list)
    supporting_span: str


class EvidenceAssessment(StrictModel):
    assessment_id: str
    evidence_id: str
    bindings: list[EvidenceBindingAssessment] = Field(default_factory=list)
    contains_numeric_threshold: bool = False
    contains_duration_rule: bool = False
    contains_exception: bool = False
    contains_monitoring: bool = False
    contains_recommendation: bool = False


class EvidenceAssessmentBatch(StrictModel):
    assessments: list[EvidenceAssessment] = Field(default_factory=list)


class DimensionState(StrictModel):
    name: str
    status: Literal[
        "unexamined",
        "searched",
        "satisfied",
        "contradicted",
        "blocked_missing_patient_info",
        "insufficient",
    ] = "unexamined"
    query_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ElementLedgerEntry(StrictModel):
    element_id: str
    judgement: Literal[
        "unresolved",
        "appropriate",
        "appropriate_with_monitoring",
        "needs_adjustment",
        "inappropriate",
        "insufficient_evidence",
    ] = "unresolved"
    support_evidence_ids: list[str] = Field(default_factory=list)
    challenge_evidence_ids: list[str] = Field(default_factory=list)
    condition_evidence_ids: list[str] = Field(default_factory=list)
    recommendation_evidence_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    dimension_states: list[DimensionState] = Field(default_factory=list)
    quantitative_issue: str | None = None
    qualitative_issue: str | None = None
    clinical_risk: str | None = None
    recommended_action: str | None = None
    monitoring_requirement: str | None = None
    patient_applicability_summary: str | None = None
    rationale: str | None = None
    missing_patient_information: list[str] = Field(default_factory=list)
    evidence_gap: str | None = None
    last_updated_step: int = 0


class ElementJudgement(StrictModel):
    element_id: str
    judgement: Judgement
    rationale: str
    patient_applicability_summary: str
    support_evidence_ids: list[str] = Field(default_factory=list)
    challenge_evidence_ids: list[str] = Field(default_factory=list)
    condition_evidence_ids: list[str] = Field(default_factory=list)
    recommendation_evidence_ids: list[str] = Field(default_factory=list)
    clinical_risk: str | None = None
    recommended_action: str | None = None
    monitoring_requirement: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class CrossElementFinding(StrictModel):
    finding_id: str
    finding_type: str
    linked_element_ids: list[str]
    judgement: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    recommended_action: str | None = None


class CompletionGap(StrictModel):
    element_id: str | None = None
    review_target_id: str | None = None
    gap_type: str
    description: str
    allowed_next_actions: list[str] = Field(default_factory=list)


class GapReport(StrictModel):
    report_id: str
    finalization_approved: bool
    gaps: list[CompletionGap] = Field(default_factory=list)
    remaining_budgets: dict[str, int] = Field(default_factory=dict)


class Recommendation(StrictModel):
    recommendation_id: str
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_finding_ids: list[str] = Field(default_factory=list)
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_scope: Literal["retrieved_corpus", "not_supported_by_retrieved_corpus"]


class AnswerElementNarrative(StrictModel):
    element_id: str
    rationale: str
    patient_applicability_summary: str
    clinical_risk: str | None = None
    recommended_action: str | None = None
    monitoring_requirement: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class AnswerNarrativeDraft(StrictModel):
    element_narratives: list[AnswerElementNarrative]
    cross_element_findings: list[CrossElementFinding] = Field(default_factory=list)
    integrated_recommendations: list[Recommendation] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)


class FinalReview(StrictModel):
    case_id: str
    plan_elements: list[TreatmentPlanElement]
    element_judgements: list[ElementJudgement]
    cross_element_findings: list[CrossElementFinding] = Field(default_factory=list)
    positive_element_ids: list[str] = Field(default_factory=list)
    negative_element_ids: list[str] = Field(default_factory=list)
    insufficient_element_ids: list[str] = Field(default_factory=list)
    integrated_recommendations: list[Recommendation] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)


class AgentDecisionRecord(StrictModel):
    step: int
    action: str
    tool_call_id: str | None = None
    started_at: str
    elapsed_ms: int
    status: Literal["success", "protocol_error", "model_error"]
    args: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class MedicationReviewTraceV2(StrictModel):
    schema_version: str = V2_SCHEMA_VERSION
    method_version: str = V2_METHOD_VERSION
    review_run_id: str
    run_status: Literal["completed", "partial", "failed"]
    completion_reason: str
    case_id: str | None = None
    patient_case: dict[str, Any] | None = None
    patient_facts: list[dict[str, Any]] = Field(default_factory=list)
    plan_extraction: dict[str, Any] = Field(default_factory=dict)
    plan_elements: list[dict[str, Any]] = Field(default_factory=list)
    retired_plan_elements: list[dict[str, Any]] = Field(default_factory=list)
    review_targets: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    search_records: list[dict[str, Any]] = Field(default_factory=list)
    open_records: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_assessments: list[dict[str, Any]] = Field(default_factory=list)
    evidence_assessment_audits: list[dict[str, Any]] = Field(default_factory=list)
    ledger: list[dict[str, Any]] = Field(default_factory=list)
    ledger_history: list[dict[str, Any]] = Field(default_factory=list)
    cross_element_findings: list[dict[str, Any]] = Field(default_factory=list)
    gap_reports: list[dict[str, Any]] = Field(default_factory=list)
    final_review: dict[str, Any] | None = None
    answer_generation: dict[str, Any] = Field(default_factory=dict)
    answer_validation: dict[str, Any] = Field(default_factory=dict)
    rendered_answer_hash: str | None = None
    knowledge_base_snapshot: dict[str, Any] = Field(default_factory=dict)
    agent_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MedicationReviewStateV2(BaseState, total=False):
    review_run_id: NotRequired[str]
    stage: NotRequired[str]
    raw_question: NotRequired[str]
    raw_question_hash: NotRequired[str]
    patient_case: NotRequired[dict[str, Any] | None]
    patient_facts: NotRequired[list[dict[str, Any]]]
    plan_extraction: NotRequired[dict[str, Any]]
    plan_elements: NotRequired[list[dict[str, Any]]]
    retired_plan_elements: NotRequired[list[dict[str, Any]]]
    review_targets: NotRequired[list[dict[str, Any]]]
    ledger: NotRequired[list[dict[str, Any]]]
    ledger_history: NotRequired[list[dict[str, Any]]]
    search_records: NotRequired[list[dict[str, Any]]]
    open_records: NotRequired[list[dict[str, Any]]]
    evidence: NotRequired[list[dict[str, Any]]]
    evidence_assessments: NotRequired[list[dict[str, Any]]]
    evidence_assessment_audits: NotRequired[list[dict[str, Any]]]
    active_evidence_ids: NotRequired[list[str]]
    cross_element_findings: NotRequired[list[dict[str, Any]]]
    gap_reports: NotRequired[list[dict[str, Any]]]
    agent_steps: NotRequired[list[dict[str, Any]]]
    final_review: NotRequired[dict[str, Any] | None]
    answer_generation: NotRequired[dict[str, Any]]
    answer_validation: NotRequired[dict[str, Any]]
    rendered_answer: NotRequired[str]
    run_status: NotRequired[str]
    completion_reason: NotRequired[str]
    knowledge_base_snapshot: NotRequired[dict[str, Any]]
    agent_config_snapshot: NotRequired[dict[str, Any]]
    usage: NotRequired[dict[str, Any]]
    warnings: NotRequired[list[str]]
    errors: NotRequired[list[dict[str, Any]]]
    consecutive_zero_yield_searches: NotRequired[int]
    search_closed_by_early_stop: NotRequired[bool]
    finalization_requested: NotRequired[bool]
    last_tool_summary: NotRequired[dict[str, Any] | None]
    pending_tool_call: NotRequired[dict[str, Any] | None]
    tool_route: NotRequired[str]


RunMode = Literal[
    "full",
    "stop_after_plan",
    "stop_after_agenda",
    "stop_after_retrieval",
    "stop_after_claims",
]
AgendaMode = Literal["none", "dynamic"]
SynthesisMode = Literal["direct_chunks", "claims"]


class ReviewQuestionDraft(StrictModel):
    question_text: str
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_patient_fact_ids: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    reason: str


class ReviewAgendaDraft(StrictModel):
    questions: list[ReviewQuestionDraft] = Field(default_factory=list)


class ReviewQuestion(ReviewQuestionDraft):
    question_id: str
    status: Literal["open", "searched", "deferred"] = "open"
    query_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class SearchSubqueryDraft(StrictModel):
    query_text: str
    linked_question_ids: list[str] = Field(default_factory=list)
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_patient_fact_ids: list[str] = Field(default_factory=list)
    search_reason: str


class SearchSubquery(SearchSubqueryDraft):
    query_id: str


class EvidenceOccurrenceV3(StrictModel):
    query_id: str
    linked_question_ids: list[str] = Field(default_factory=list)
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_patient_fact_ids: list[str] = Field(default_factory=list)
    rank: int
    score: float | None = None
    distance: float | None = None


class EvidenceCandidate(StrictModel):
    content_hash: str
    raw_text: str
    source_document: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    source_method: Literal["search", "open"] = "search"
    parent_content_hash: str | None = None
    occurrences: list[EvidenceOccurrenceV3] = Field(default_factory=list)


class EvidenceItemV3(EvidenceCandidate):
    evidence_id: str
    parent_evidence_id: str | None = None


class EvidenceSearchRecordV3(StrictModel):
    subquery: SearchSubquery
    status: Literal[
        "success",
        "success_empty",
        "technical_failed",
        "invalid_query",
        "skipped_budget",
    ]
    attempts: list[TechnicalAttempt] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    returned_count: int = 0
    duplicate_ratio: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


class EvidenceOpenRecordV3(StrictModel):
    open_id: str
    parent_evidence_id: str
    reason: str
    window_before: int
    window_after: int
    status: Literal[
        "success",
        "success_empty",
        "timeout",
        "backend_error",
        "invalid_source",
    ]
    started_at: str
    elapsed_ms: int
    attempt_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_ids: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class EvidenceSelectionAudit(StrictModel):
    selected_evidence_ids: list[str] = Field(default_factory=list)
    priority_evidence_ids: list[str] = Field(default_factory=list)
    opened_evidence_ids: list[str] = Field(default_factory=list)
    skipped_evidence_ids: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    max_evidence: int = 15
    max_tokens: int = 12000
    warnings: list[str] = Field(default_factory=list)


ClaimRoleHint = Literal[
    "supports",
    "raises_concern",
    "conditions",
    "recommends",
    "context",
]


class NumericFact(StrictModel):
    value: str
    unit: str | None = None
    context: str


class EvidenceClaimDraft(StrictModel):
    evidence_id: str
    source_span: str
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_question_ids: list[str] = Field(default_factory=list)
    statement: str
    role_hints: list[ClaimRoleHint] = Field(default_factory=list)
    stated_conditions: list[str] = Field(default_factory=list)
    stated_actions: list[str] = Field(default_factory=list)
    numeric_facts: list[NumericFact] = Field(default_factory=list)


class EvidenceClaimBatchDraft(StrictModel):
    claims: list[EvidenceClaimDraft] = Field(default_factory=list)


class EvidenceClaim(EvidenceClaimDraft):
    claim_id: str


class SourceCitationDraft(StrictModel):
    evidence_id: str
    source_span: str
    claim_id: str | None = None


class ClinicalFindingDraft(StrictModel):
    linked_element_ids: list[str] = Field(default_factory=list)
    linked_patient_fact_ids: list[str] = Field(default_factory=list)
    assessment: Literal["appropriate", "concern", "uncertain"]
    statement: str
    importance: Literal["major", "minor", "context"] = "minor"
    citations: list[SourceCitationDraft] = Field(default_factory=list)


class ClinicalFinding(ClinicalFindingDraft):
    finding_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class SupportedStatementDraft(StrictModel):
    text: str
    citations: list[SourceCitationDraft] = Field(default_factory=list)
    source_scope: Literal["retrieved_corpus", "general_review"]
    grounded_terms: list[str] = Field(default_factory=list)
    numeric_facts: list[NumericFact] = Field(default_factory=list)


class SupportedStatement(SupportedStatementDraft):
    evidence_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


Disposition = Literal[
    "continue",
    "continue_with_monitoring",
    "modify",
    "avoid",
    "uncertain",
]
EvidenceBasis = Literal[
    "direct_support",
    "mixed_evidence",
    "no_material_conflict_found",
    "insufficient",
]


class ElementReviewDraft(StrictModel):
    element_id: str
    overall_disposition: Disposition
    evidence_basis: EvidenceBasis
    summary: str
    recommendation: SupportedStatementDraft | None = None
    monitoring: SupportedStatementDraft | None = None
    missing_information: list[str] = Field(default_factory=list)


class ElementReview(ElementReviewDraft):
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    recommendation: SupportedStatement | None = None
    monitoring: SupportedStatement | None = None


class ReviewSynthesisDraft(StrictModel):
    findings: list[ClinicalFindingDraft] = Field(default_factory=list)
    element_reviews: list[ElementReviewDraft] = Field(default_factory=list)
    integrated_recommendations: list[SupportedStatementDraft] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)


class ReviewSynthesis(StrictModel):
    case_id: str
    findings: list[ClinicalFinding] = Field(default_factory=list)
    element_reviews: list[ElementReview] = Field(default_factory=list)
    integrated_recommendations: list[SupportedStatement] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)
    used_claim_ids: list[str] = Field(default_factory=list)


class LocalValidationEvent(StrictModel):
    stage: str
    object_type: str
    object_id: str | None = None
    action: Literal["kept", "dropped", "degraded", "repaired"]
    reason: str


class AgentDecisionRecordV3(StrictModel):
    step: int
    action: str
    tool_call_id: str | None = None
    started_at: str
    elapsed_ms: int
    status: Literal["success", "protocol_error", "technical_error"]
    args: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class MedicationReviewTraceV3(StrictModel):
    schema_version: str = V3_SCHEMA_VERSION
    method_version: str = V3_DEFAULT_METHOD_VERSION
    method_family: str = V3_METHOD_FAMILY
    review_run_id: str
    run_status: Literal["completed", "partial", "debug_stopped", "failed"]
    run_mode: RunMode
    agenda_mode: AgendaMode
    synthesis_mode: SynthesisMode
    effective_profile: str
    completion_reason: str
    last_completed_stage: str
    case_id: str | None = None
    patient_case: dict[str, Any] | None = None
    patient_facts: list[dict[str, Any]] = Field(default_factory=list)
    plan_extraction: dict[str, Any] = Field(default_factory=dict)
    plan_elements: list[dict[str, Any]] = Field(default_factory=list)
    review_agenda: list[dict[str, Any]] = Field(default_factory=list)
    agenda_audit: dict[str, Any] = Field(default_factory=dict)
    agent_steps: list[dict[str, Any]] = Field(default_factory=list)
    search_records: list[dict[str, Any]] = Field(default_factory=list)
    open_records: list[dict[str, Any]] = Field(default_factory=list)
    finish_retrieval: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_selection: dict[str, Any] = Field(default_factory=dict)
    evidence_claims: list[dict[str, Any]] = Field(default_factory=list)
    claim_extraction: dict[str, Any] = Field(default_factory=dict)
    review_synthesis: dict[str, Any] = Field(default_factory=dict)
    local_validation_events: list[dict[str, Any]] = Field(default_factory=list)
    final_review: dict[str, Any] | None = None
    rendered_answer_hash: str | None = None
    knowledge_base_snapshot: dict[str, Any] = Field(default_factory=dict)
    agent_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class MedicationReviewStateV3(BaseState, total=False):
    review_run_id: NotRequired[str]
    stage: NotRequired[str]
    run_mode: NotRequired[str]
    agenda_mode: NotRequired[str]
    synthesis_mode: NotRequired[str]
    raw_question: NotRequired[str]
    raw_question_hash: NotRequired[str]
    patient_case: NotRequired[dict[str, Any] | None]
    patient_facts: NotRequired[list[dict[str, Any]]]
    plan_extraction: NotRequired[dict[str, Any]]
    plan_elements: NotRequired[list[dict[str, Any]]]
    review_agenda: NotRequired[list[dict[str, Any]]]
    agenda_audit: NotRequired[dict[str, Any]]
    search_records: NotRequired[list[dict[str, Any]]]
    open_records: NotRequired[list[dict[str, Any]]]
    finish_retrieval: NotRequired[dict[str, Any]]
    evidence: NotRequired[list[dict[str, Any]]]
    evidence_selection: NotRequired[dict[str, Any]]
    selected_evidence_ids: NotRequired[list[str]]
    evidence_claims: NotRequired[list[dict[str, Any]]]
    claim_extraction: NotRequired[dict[str, Any]]
    review_synthesis: NotRequired[dict[str, Any]]
    synthesis_fatal: NotRequired[bool]
    local_validation_events: NotRequired[list[dict[str, Any]]]
    agent_steps: NotRequired[list[dict[str, Any]]]
    final_review: NotRequired[dict[str, Any] | None]
    rendered_answer: NotRequired[str]
    run_status: NotRequired[str]
    completion_reason: NotRequired[str]
    knowledge_base_snapshot: NotRequired[dict[str, Any]]
    agent_config_snapshot: NotRequired[dict[str, Any]]
    usage: NotRequired[dict[str, Any]]
    warnings: NotRequired[list[str]]
    errors: NotRequired[list[dict[str, Any]]]
    logical_step_count: NotRequired[int]
    technical_attempt_count: NotRequired[int]
    agent_model_error_count: NotRequired[int]
    consecutive_tool_error_count: NotRequired[int]
    executed_query_count: NotRequired[int]
    degraded: NotRequired[bool]
    pending_tool_call: NotRequired[dict[str, Any] | None]
    tool_route: NotRequired[str]
    last_tool_summary: NotRequired[dict[str, Any] | None]
