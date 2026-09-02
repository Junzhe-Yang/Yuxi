from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from .models import CorpusAtlas, SectionCard
from .router import CaseRoutingComputation, rank_sections_for_embedding
from ..models import OpportunityNodeMatch, RetrievalOpportunity

SECTION_TOP_R = 3
MAX_OPPORTUNITIES = 3
MAX_NODES_PER_OPPORTUNITY = 4
MIN_SECTION_SIMILARITY: float | None = None


def _value(raw: Any, field: str) -> str:
    if isinstance(raw, dict):
        return str(raw.get(field) or "")
    return str(getattr(raw, field, "") or "")


def _is_parent_path(left: list[str], right: list[str]) -> bool:
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return shorter == longer[: len(shorter)]


def _opportunity_id(
    *,
    section_id: str,
    plan_ids: list[str],
    modifier_ids: list[str],
) -> str:
    source = "\0".join([section_id, *sorted(plan_ids), *sorted(modifier_ids)])
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"OP-{digest.upper()}"


def build_retrieval_opportunities(
    *,
    atlas: CorpusAtlas,
    computation: CaseRoutingComputation,
    plan_anchors: list[Any],
    patient_modifiers: list[Any],
    min_similarity: float | None = MIN_SECTION_SIMILARITY,
) -> list[RetrievalOpportunity]:
    nodes: list[tuple[str, str, str]] = []
    for raw in plan_anchors:
        node_id = _value(raw, "element_id")
        source_span = _value(raw, "source_span")
        if node_id and source_span:
            nodes.append((node_id, "plan", source_span))
    for raw in patient_modifiers:
        node_id = _value(raw, "modifier_id")
        source_span = _value(raw, "source_span")
        if node_id and source_span:
            nodes.append((node_id, "modifier", source_span))

    matches_by_section: dict[str, list[OpportunityNodeMatch]] = defaultdict(list)
    sections_by_id: dict[str, SectionCard] = {section.section_id: section for section in atlas.section_cards}
    top_document_ids = {value.file_id for value in computation.record.ranked_documents[:6]}
    for node_id, node_type, source_span in nodes:
        embedding = computation.view_embeddings.get(f"VIEW-{node_id}")
        if embedding is None:
            continue
        for rank, (section, similarity) in enumerate(
            rank_sections_for_embedding(
                atlas=atlas,
                embedding=embedding,
                top_r=SECTION_TOP_R,
            ),
            start=1,
        ):
            if section.file_id not in top_document_ids:
                continue
            if min_similarity is not None and similarity < min_similarity:
                continue
            matches_by_section[section.section_id].append(
                OpportunityNodeMatch(
                    node_id=node_id,
                    node_type=node_type,
                    source_span=source_span,
                    section_rank=rank,
                    similarity=similarity,
                )
            )

    opportunities: list[RetrievalOpportunity] = []
    for section_id, matches in matches_by_section.items():
        plan_matches = [value for value in matches if value.node_type == "plan"]
        modifier_matches = [value for value in matches if value.node_type == "modifier"]
        if plan_matches and modifier_matches:
            opportunity_type = "plan_modifier"
        elif len(plan_matches) >= 2:
            opportunity_type = "multi_plan"
        else:
            continue
        ranked_matches = sorted(
            matches,
            key=lambda value: (
                value.section_rank,
                -value.similarity,
                value.node_id,
            ),
        )[:MAX_NODES_PER_OPPORTUNITY]
        node_count = len(ranked_matches)
        score = math.log1p(node_count) * node_count / sum(60 + value.section_rank for value in ranked_matches)
        section = sections_by_id[section_id]
        plan_ids = [value.node_id for value in ranked_matches if value.node_type == "plan"]
        modifier_ids = [value.node_id for value in ranked_matches if value.node_type == "modifier"]
        opportunities.append(
            RetrievalOpportunity(
                opportunity_id=_opportunity_id(
                    section_id=section_id,
                    plan_ids=plan_ids,
                    modifier_ids=modifier_ids,
                ),
                opportunity_type=opportunity_type,
                section_id=section_id,
                file_id=section.file_id,
                file_name=section.file_name,
                heading_path=section.heading_path,
                focus_plan_ids=plan_ids,
                focus_modifier_ids=modifier_ids,
                node_matches=ranked_matches,
                score=score,
            )
        )

    opportunities.sort(key=lambda value: (-value.score, value.file_id, value.section_id))
    deduplicated: list[RetrievalOpportunity] = []
    for candidate in opportunities:
        if any(
            current.file_id == candidate.file_id and _is_parent_path(current.heading_path, candidate.heading_path)
            for current in deduplicated
        ):
            continue
        deduplicated.append(candidate)
        if len(deduplicated) >= MAX_OPPORTUNITIES:
            break
    return deduplicated
