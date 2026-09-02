from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired

from pydantic import Field

from yuxi.agents.buildin.medication_review_lite.models import (
    merge_snapshot,
    merge_unique_strings,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimState,
    MedicationReviewPrimTrace,
    StrictModel,
)

AtlasProfile = Literal["map", "route", "full"]
RouteViewType = Literal["case", "regimen", "plan", "modifier"]


class RankedDocument(StrictModel):
    file_id: str
    file_name: str
    document_title: str
    rank: int
    score: float
    max_similarity: float
    source_view_ids: list[str] = Field(default_factory=list)
    injected: bool = False


class RankedSection(StrictModel):
    section_id: str
    file_id: str
    file_name: str
    heading_path: list[str] = Field(default_factory=list)
    rank: int
    similarity: float


class RouteViewRecord(StrictModel):
    view_id: str
    view_type: RouteViewType
    source_ids: list[str] = Field(default_factory=list)
    text: str
    ranked_documents: list[RankedDocument] = Field(default_factory=list)


class CaseRouteRecord(StrictModel):
    atlas_snapshot_hash: str
    views: list[RouteViewRecord] = Field(default_factory=list)
    ranked_documents: list[RankedDocument] = Field(default_factory=list)
    map_sections: list[RankedSection] = Field(default_factory=list)


class OpportunityNodeMatch(StrictModel):
    node_id: str
    node_type: Literal["plan", "modifier"]
    source_span: str
    section_rank: int
    similarity: float


class RetrievalOpportunity(StrictModel):
    opportunity_id: str
    opportunity_type: Literal["plan_modifier", "multi_plan"]
    section_id: str
    file_id: str
    file_name: str
    heading_path: list[str] = Field(default_factory=list)
    focus_plan_ids: list[str] = Field(default_factory=list)
    focus_modifier_ids: list[str] = Field(default_factory=list)
    node_matches: list[OpportunityNodeMatch] = Field(default_factory=list)
    score: float


class RetrievalCandidate(StrictModel):
    chunk_key: str
    file_id: str
    file_name: str
    chunk_id: str | None = None
    chunk_index: int | None = None
    similarity: float
    global_rank: int | None = None
    local_rank: int | None = None
    within_document_rank: int | None = None
    document_route_rank: int | None = None
    fusion_score: float = 0.0
    retrieval_paths: list[Literal["global", "local"]] = Field(default_factory=list)


class RoutedRetrievalRecord(StrictModel):
    retrieval_record_id: str
    query_id: str
    opportunity_id: str | None = None
    strategy: Literal["flat", "routed"]
    case_route_documents: list[RankedDocument] = Field(default_factory=list)
    query_route_documents: list[RankedDocument] = Field(default_factory=list)
    effective_documents: list[RankedDocument] = Field(default_factory=list)
    opportunity_injected_file_id: str | None = None
    global_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    local_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    fused_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    embedding_batch_count: int = 0
    backend_search_count: int = 0
    stage_elapsed_ms: dict[str, int] = Field(default_factory=dict)
    degraded_reasons: list[str] = Field(default_factory=list)


def _merge_opportunities(
    left: list[RetrievalOpportunity] | None,
    right: list[RetrievalOpportunity] | None,
) -> list[RetrievalOpportunity]:
    by_id: dict[str, RetrievalOpportunity] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, RetrievalOpportunity) else RetrievalOpportunity.model_validate(raw)
        by_id[value.opportunity_id] = value
    return sorted(by_id.values(), key=lambda value: value.opportunity_id)


def _merge_routed_records(
    left: list[RoutedRetrievalRecord] | None,
    right: list[RoutedRetrievalRecord] | None,
) -> list[RoutedRetrievalRecord]:
    by_id: dict[str, RoutedRetrievalRecord] = {}
    for raw in [*(left or []), *(right or [])]:
        value = raw if isinstance(raw, RoutedRetrievalRecord) else RoutedRetrievalRecord.model_validate(raw)
        by_id[value.retrieval_record_id] = value
    return sorted(by_id.values(), key=lambda value: value.retrieval_record_id)


class MedicationReviewDaPrimState(MedicationReviewPrimState, total=False):
    atlas_profile: NotRequired[AtlasProfile]
    atlas_snapshot: NotRequired[Annotated[dict[str, Any], merge_snapshot]]
    case_route_record: NotRequired[CaseRouteRecord]
    retrieval_opportunities: NotRequired[Annotated[list[RetrievalOpportunity], _merge_opportunities]]
    adopted_opportunity_ids: NotRequired[Annotated[list[str], merge_unique_strings]]
    routed_retrieval_records: NotRequired[Annotated[list[RoutedRetrievalRecord], _merge_routed_records]]


class MedicationReviewDaPrimTrace(MedicationReviewPrimTrace):
    schema_version: Literal["6.0"] = "6.0"
    method_family: Literal["da-prim-rag-v1"] = "da-prim-rag-v1"
    atlas_profile: AtlasProfile
    atlas_snapshot: dict[str, Any] = Field(default_factory=dict)
    case_route_record: CaseRouteRecord
    retrieval_opportunities: list[RetrievalOpportunity] = Field(default_factory=list)
    adopted_opportunity_ids: list[str] = Field(default_factory=list)
    routed_retrieval_records: list[RoutedRetrievalRecord] = Field(default_factory=list)
