from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .models import CorpusAtlas, DocumentCard, SectionCard
from ..models import (
    CaseRouteRecord,
    RankedDocument,
    RankedSection,
    RouteViewRecord,
    RouteViewType,
)

RRF_K = 60
DOCUMENT_TOP_K = 6
MAP_SECTIONS_PER_DOCUMENT = 2


@dataclass(frozen=True)
class RouteView:
    view_id: str
    view_type: RouteViewType
    source_ids: list[str]
    text: str


@dataclass(frozen=True)
class CaseRoutingComputation:
    record: CaseRouteRecord
    view_embeddings: dict[str, list[float]]


def normalize_route_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _value(raw: Any, field: str) -> str:
    if isinstance(raw, dict):
        return str(raw.get(field) or "")
    return str(getattr(raw, field, "") or "")


def build_case_views(
    *,
    raw_case_text: str,
    plan_anchors: list[Any],
    patient_modifiers: list[Any],
) -> list[RouteView]:
    views = [
        RouteView(
            view_id="VIEW-CASE",
            view_type="case",
            source_ids=[],
            text=normalize_route_text(raw_case_text),
        )
    ]
    plan_spans = [
        normalize_route_text(_value(value, "source_span"))
        for value in plan_anchors
        if normalize_route_text(_value(value, "source_span"))
    ]
    plan_ids = [_value(value, "element_id") for value in plan_anchors if _value(value, "element_id")]
    if plan_spans:
        views.append(
            RouteView(
                view_id="VIEW-REGIMEN",
                view_type="regimen",
                source_ids=plan_ids,
                text="；".join(plan_spans),
            )
        )
    for raw in plan_anchors:
        element_id = _value(raw, "element_id")
        source_span = normalize_route_text(_value(raw, "source_span"))
        if element_id and source_span:
            views.append(
                RouteView(
                    view_id=f"VIEW-{element_id}",
                    view_type="plan",
                    source_ids=[element_id],
                    text=source_span,
                )
            )
    for raw in patient_modifiers:
        modifier_id = _value(raw, "modifier_id")
        source_span = normalize_route_text(_value(raw, "source_span"))
        if modifier_id and source_span:
            views.append(
                RouteView(
                    view_id=f"VIEW-{modifier_id}",
                    view_type="modifier",
                    source_ids=[modifier_id],
                    text=source_span,
                )
            )
    return views


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("路由向量维度不一致")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _rank_document_cards(
    atlas: CorpusAtlas,
    embedding: list[float],
) -> list[tuple[DocumentCard, float]]:
    values = [(card, cosine_similarity(embedding, card.embedding)) for card in atlas.document_cards]
    values.sort(key=lambda item: (-item[1], item[0].file_id))
    return values


def _rank_section_cards(
    cards: list[SectionCard],
    embedding: list[float],
) -> list[tuple[SectionCard, float]]:
    values = [(card, cosine_similarity(embedding, card.embedding)) for card in cards]
    values.sort(key=lambda item: (-item[1], item[0].section_id))
    return values


def _ranked_document(
    *,
    card: DocumentCard,
    rank: int,
    score: float,
    max_similarity: float,
    source_view_ids: list[str],
    injected: bool = False,
) -> RankedDocument:
    return RankedDocument(
        file_id=card.file_id,
        file_name=card.file_name,
        document_title=card.document_title,
        rank=rank,
        score=score,
        max_similarity=max_similarity,
        source_view_ids=source_view_ids,
        injected=injected,
    )


async def route_case(
    *,
    manager: Any,
    db_id: str,
    atlas: CorpusAtlas,
    raw_case_text: str,
    plan_anchors: list[Any],
    patient_modifiers: list[Any],
) -> CaseRoutingComputation:
    views = build_case_views(
        raw_case_text=raw_case_text,
        plan_anchors=plan_anchors,
        patient_modifiers=patient_modifiers,
    )
    embeddings = await manager.aembed_texts(
        db_id,
        [view.text for view in views],
    )
    if len(embeddings) != len(views):
        raise ValueError("病例路由 embedding 数量与视图数量不一致")
    view_embeddings = {view.view_id: embedding for view, embedding in zip(views, embeddings)}
    rankings: dict[str, list[tuple[DocumentCard, float]]] = {
        view.view_id: _rank_document_cards(atlas, view_embeddings[view.view_id]) for view in views
    }
    rank_maps = {
        view_id: {card.file_id: (rank, similarity) for rank, (card, similarity) in enumerate(values, start=1)}
        for view_id, values in rankings.items()
    }
    by_type = {
        view_type: [view for view in views if view.view_type == view_type]
        for view_type in ("case", "regimen", "plan", "modifier")
    }
    scores: dict[str, float] = {}
    max_similarities: dict[str, float] = {}
    source_views: dict[str, list[str]] = {}
    for card in atlas.document_cards:
        file_id = card.file_id
        score = 0.0
        similarities: list[float] = []
        contributing: list[str] = []
        for view_type in ("case", "regimen"):
            for view in by_type[view_type]:
                rank, similarity = rank_maps[view.view_id][file_id]
                score += 1.0 / (RRF_K + rank)
                similarities.append(similarity)
                if rank <= DOCUMENT_TOP_K:
                    contributing.append(view.view_id)
        for view_type in ("plan", "modifier"):
            typed_views = by_type[view_type]
            if not typed_views:
                continue
            contribution = 0.0
            for view in typed_views:
                rank, similarity = rank_maps[view.view_id][file_id]
                contribution += 1.0 / (RRF_K + rank)
                similarities.append(similarity)
                if rank <= DOCUMENT_TOP_K:
                    contributing.append(view.view_id)
            score += contribution / len(typed_views)
        scores[file_id] = score
        max_similarities[file_id] = max(similarities, default=0.0)
        source_views[file_id] = list(dict.fromkeys(contributing))

    cards_by_id = {card.file_id: card for card in atlas.document_cards}
    ordered_ids = sorted(
        cards_by_id,
        key=lambda file_id: (
            -scores[file_id],
            -max_similarities[file_id],
            file_id,
        ),
    )
    ranked_documents = [
        _ranked_document(
            card=cards_by_id[file_id],
            rank=rank,
            score=scores[file_id],
            max_similarity=max_similarities[file_id],
            source_view_ids=source_views[file_id],
        )
        for rank, file_id in enumerate(ordered_ids, start=1)
    ]
    view_records: list[RouteViewRecord] = []
    for view in views:
        values = rankings[view.view_id][:DOCUMENT_TOP_K]
        view_records.append(
            RouteViewRecord(
                view_id=view.view_id,
                view_type=view.view_type,
                source_ids=view.source_ids,
                text=view.text,
                ranked_documents=[
                    _ranked_document(
                        card=card,
                        rank=rank,
                        score=similarity,
                        max_similarity=similarity,
                        source_view_ids=[view.view_id],
                    )
                    for rank, (card, similarity) in enumerate(values, start=1)
                ],
            )
        )

    case_embedding = view_embeddings["VIEW-CASE"]
    map_sections: list[RankedSection] = []
    for document in ranked_documents[:DOCUMENT_TOP_K]:
        cards = [card for card in atlas.section_cards if card.file_id == document.file_id]
        for rank, (card, similarity) in enumerate(
            _rank_section_cards(cards, case_embedding)[:MAP_SECTIONS_PER_DOCUMENT],
            start=1,
        ):
            map_sections.append(
                RankedSection(
                    section_id=card.section_id,
                    file_id=card.file_id,
                    file_name=card.file_name,
                    heading_path=card.heading_path,
                    rank=rank,
                    similarity=similarity,
                )
            )
    return CaseRoutingComputation(
        record=CaseRouteRecord(
            atlas_snapshot_hash=atlas.snapshot_hash,
            views=view_records,
            ranked_documents=ranked_documents,
            map_sections=map_sections,
        ),
        view_embeddings=view_embeddings,
    )


def rank_sections_for_embedding(
    *,
    atlas: CorpusAtlas,
    embedding: list[float],
    top_r: int,
) -> list[tuple[SectionCard, float]]:
    return _rank_section_cards(atlas.section_cards, embedding)[:top_r]


def route_query_documents(
    *,
    atlas: CorpusAtlas,
    case_route: CaseRouteRecord,
    query_embedding: list[float],
    opportunity_file_id: str | None = None,
    top_k: int = DOCUMENT_TOP_K,
) -> tuple[list[RankedDocument], str | None]:
    query_ranking = _rank_document_cards(atlas, query_embedding)
    query_rank = {card.file_id: (rank, similarity) for rank, (card, similarity) in enumerate(query_ranking, start=1)}
    case_rank = {value.file_id: value.rank for value in case_route.ranked_documents}
    cards_by_id = {card.file_id: card for card in atlas.document_cards}
    scored = []
    for file_id, card in cards_by_id.items():
        query_position, similarity = query_rank[file_id]
        score = 1.0 / (RRF_K + case_rank[file_id]) + 1.0 / (RRF_K + query_position)
        scored.append((card, score, similarity))
    scored.sort(key=lambda item: (-item[1], -item[2], item[0].file_id))
    selected = scored[:top_k]
    injected_file_id: str | None = None
    if (
        opportunity_file_id
        and opportunity_file_id in cards_by_id
        and opportunity_file_id not in {card.file_id for card, _, _ in selected}
    ):
        opportunity_value = next(value for value in scored if value[0].file_id == opportunity_file_id)
        selected[-1] = opportunity_value
        injected_file_id = opportunity_file_id
    return (
        [
            _ranked_document(
                card=card,
                rank=rank,
                score=score,
                max_similarity=similarity,
                source_view_ids=["VIEW-CASE", "VIEW-QUERY"],
                injected=card.file_id == injected_file_id,
            )
            for rank, (card, score, similarity) in enumerate(selected, start=1)
        ],
        injected_file_id,
    )


def rank_query_document_cards(
    *,
    atlas: CorpusAtlas,
    query_embedding: list[float],
    top_k: int = DOCUMENT_TOP_K,
) -> list[RankedDocument]:
    return [
        _ranked_document(
            card=card,
            rank=rank,
            score=similarity,
            max_similarity=similarity,
            source_view_ids=["VIEW-QUERY"],
        )
        for rank, (card, similarity) in enumerate(
            _rank_document_cards(atlas, query_embedding)[:top_k],
            start=1,
        )
    ]
