from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired

from pydantic import Field

from yuxi.agents.buildin.medication_review_acm_prim.models import (
    AdaptiveAuditDimension,
    AdaptiveAuditStatus,
    AdaptiveDecisionTag,
    AdaptiveInvestigationKind,
    MedicationReviewAcmAdaptiveTrace,
    MedicationReviewAcmPrimState,
)
from yuxi.agents.buildin.medication_review_lite.models import add_int
from yuxi.agents.buildin.medication_review_prim.models import StrictModel

BoundedPhase = Literal[
    "PROPOSE_INITIAL_AGENDA",
    "EXTEND_AGENDA",
    "SEARCH_ACTIVE_OBLIGATION",
    "REVIEW_ACTIVE_OBLIGATION",
    "CLOSE_ACTIVE_INVESTIGATION",
    "AUDIT_COVERAGE",
    "DRAFT_FINAL",
    "FAIL_EXPLICIT",
]
SemanticOutcome = Literal[
    "SUCCESS",
    "NO_RESULT",
    "REJECTED",
    "INVALID_ARGUMENT",
    "NEEDS_INPUT",
    "RETRYABLE_ERROR",
    "FATAL_ERROR",
]


class BoundedAgendaItemDraft(StrictModel):
    question: str = Field(min_length=1, max_length=500)
    why_it_matters: str = Field(min_length=1, max_length=500)
    distinct_scope: str = Field(min_length=1, max_length=300)
    investigation_kind: AdaptiveInvestigationKind
    evidence_obligations: list[str] = Field(min_length=1)
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    decision_tags: list[AdaptiveDecisionTag] = Field(default_factory=list)
    parent_evidence_aliases: list[str] = Field(default_factory=list)


class BoundedAuditEntry(StrictModel):
    dimension: AdaptiveAuditDimension
    status: AdaptiveAuditStatus
    investigation_aliases: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=800)


class ActionDirective(StrictModel):
    directive_id: str
    phase: BoundedPhase
    state_version: int
    state_fingerprint: str
    active_investigation_id: str | None = None
    active_obligation: str | None = None
    active_recovery_id: str | None = None
    active_investigation_kind: AdaptiveInvestigationKind | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    allowed_routes: list[Literal["global", "document"]] = Field(default_factory=list)
    bound_retrieval_scope: Literal["global", "document"] | None = None
    bound_retrieval_intent: (
        Literal[
            "source_discovery",
            "within_document_localization",
        ]
        | None
    ) = None
    bound_file_id: str | None = None
    evidence_aliases: dict[str, str] = Field(default_factory=dict)
    file_aliases: dict[str, str] = Field(default_factory=dict)
    atlas_document_aliases: dict[str, str] = Field(default_factory=dict)
    investigation_aliases: dict[str, str] = Field(default_factory=dict)
    retry_policy: dict[str, int] = Field(default_factory=dict)
    expected_output_kind: Literal["tool_call", "final_answer"] = "tool_call"


class ContextAtom(StrictModel):
    atom_id: str
    kind: Literal[
        "USER_CASE",
        "ACTION_DIRECTIVE",
        "LEDGER_SNAPSHOT",
        "TOOL_INTERACTION",
        "STATE_TRANSITION",
        "EVIDENCE_ARTIFACT",
        "EVIDENCE_RECEIPT",
        "REJECTION",
        "GENERATION_ABORT",
        "AUDIT_GAP",
        "DRAFT",
        "CITATION_CHECK",
    ]
    phase: BoundedPhase | None = None
    investigation_id: str | None = None
    obligation: str | None = None
    probe_id: str | None = None
    recovery_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    tool_call_id: str | None = None
    state_version: int = 0
    material_state_change: bool = False
    outcome: SemanticOutcome | None = None
    token_count: int = 0
    rehydration_handle: str | None = None


class ContextManifestEntry(StrictModel):
    source_id: str
    component: str
    representation: Literal["raw", "receipt", "digest", "reference"]
    retention_reason: str
    scope_keys: dict[str, str] = Field(default_factory=dict)
    token_count: int
    priority: int
    rehydration_handle: str | None = None


class ContextManifest(StrictModel):
    model_call_id: str
    directive_id: str
    phase: BoundedPhase
    entries: list[ContextManifestEntry] = Field(default_factory=list)
    excluded_counts: dict[str, int] = Field(default_factory=dict)
    component_tokens: dict[str, int] = Field(default_factory=dict)
    projected_input_tokens: int
    reserved_output_tokens: int
    projected_total_tokens: int
    configured_context_window: int
    phase_observation_target: int | None = None
    above_observation_target: bool = False
    reduction_actions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolOutcome(StrictModel):
    call_id: str
    directive_id: str
    tool_name: str
    transport_status: Literal["COMPLETED", "FAILED"]
    semantic_outcome: SemanticOutcome
    reason_code: str | None = None
    retryable_by_model: bool = False
    technical_retryable: bool = False
    state_changed: bool = False
    executed_backend: bool = False
    state_version_before: int
    state_version_after: int
    recovery_action: str | None = None
    message_for_model: str
    next_legal_actions: list[str] = Field(default_factory=list)
    investigation_id: str | None = None
    obligation: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ActionAttempt(StrictModel):
    fingerprint: str
    directive_id: str
    state_version: int
    tool_name: str
    canonical_arguments: dict[str, Any]
    semantic_outcome: SemanticOutcome
    reason_code: str | None = None
    state_changed: bool = False
    created_at: str


class BoundedObligationJudgment(StrictModel):
    judgment_id: str
    tool_call_id: str
    investigation_id: str
    obligation: str
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    state_version_after: int
    created_at: str


class GenerationAbortRecord(StrictModel):
    abort_id: str
    model_call_id: str
    directive_id: str
    phase: BoundedPhase
    reason_codes: list[str]
    output_tokens: int
    retry_number: int
    raw_output: str
    created_at: str


class CitationClaimRecord(StrictModel):
    claim_id: str
    claim_text: str
    evidence_ids: list[str] = Field(default_factory=list)


class CitationEvidenceSnapshot(StrictModel):
    evidence_id: str
    content_hash: str
    raw_text_sha256: str
    raw_text: str
    source_document: str | None = None
    file_id: str | None = None
    chunk_id: str | None = None
    chunk_index: int | str | None = None


class CitationVerificationRecord(StrictModel):
    verification_version: Literal["bounded-citation-rehydration-v1"] = "bounded-citation-rehydration-v1"
    draft_hash: str
    status: Literal["ready", "no_citations", "unknown_evidence"]
    cited_evidence_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    claims: list[CitationClaimRecord] = Field(default_factory=list)
    evidence_snapshots: list[CitationEvidenceSnapshot] = Field(default_factory=list)


def _merge_models(left, right, model_type, key: str):
    by_id = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, model_type) else model_type.model_validate(raw)
        by_id[str(getattr(value, key))] = value
    return list(by_id.values())


def merge_directives(left, right):
    return _merge_models(left, right, ActionDirective, "directive_id")


def merge_context_manifests(left, right):
    return _merge_models(left, right, ContextManifest, "model_call_id")


def merge_context_atoms(left, right):
    return _merge_models(left, right, ContextAtom, "atom_id")


def merge_tool_outcomes(left, right):
    return _merge_models(left, right, ToolOutcome, "call_id")


def merge_action_attempts(left, right):
    return _merge_models(left, right, ActionAttempt, "fingerprint")


def merge_obligation_judgments(left, right):
    return _merge_models(left, right, BoundedObligationJudgment, "judgment_id")


def merge_generation_aborts(left, right):
    return _merge_models(left, right, GenerationAbortRecord, "abort_id")


class MedicationReviewAcmBoundedState(MedicationReviewAcmPrimState, total=False):
    bounded_state_version: NotRequired[Annotated[int, add_int]]
    action_directive: NotRequired[ActionDirective]
    bounded_directives: NotRequired[Annotated[list[ActionDirective], merge_directives]]
    bounded_context_manifests: NotRequired[Annotated[list[ContextManifest], merge_context_manifests]]
    bounded_context_atoms: NotRequired[Annotated[list[ContextAtom], merge_context_atoms]]
    bounded_tool_outcomes: NotRequired[Annotated[list[ToolOutcome], merge_tool_outcomes]]
    bounded_action_attempts: NotRequired[Annotated[list[ActionAttempt], merge_action_attempts]]
    bounded_obligation_judgments: NotRequired[Annotated[list[BoundedObligationJudgment], merge_obligation_judgments]]
    bounded_generation_aborts: NotRequired[Annotated[list[GenerationAbortRecord], merge_generation_aborts]]
    bounded_citation_verification: NotRequired[CitationVerificationRecord]


class MedicationReviewAcmBoundedTrace(MedicationReviewAcmAdaptiveTrace):
    schema_version: Literal["13.0"] = "13.0"
    method_family: Literal["acm-prim-rag-v9"] = "acm-prim-rag-v9"
    controller_version: str
    context_view_version: str
    prompt_version: str
    model_context_window_tokens: int
    directives: list[ActionDirective] = Field(default_factory=list)
    context_manifests: list[ContextManifest] = Field(default_factory=list)
    context_atoms: list[ContextAtom] = Field(default_factory=list)
    tool_outcomes: list[ToolOutcome] = Field(default_factory=list)
    action_attempts: list[ActionAttempt] = Field(default_factory=list)
    obligation_judgments: list[BoundedObligationJudgment] = Field(default_factory=list)
    generation_abort_records: list[GenerationAbortRecord] = Field(default_factory=list)
    citation_verification: CitationVerificationRecord | None = None
    provider_context_window_tokens: int
    context_window_verified: bool
