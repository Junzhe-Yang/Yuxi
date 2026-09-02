from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from yuxi import knowledge_base
from yuxi.agents.buildin.medication_review_lite.models import TechnicalAttempt
from yuxi.agents.buildin.medication_review_lite.tools import _attempt_error
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrievalOutcome,
    RetrievalRequest,
    RetrievalStrategy,
)
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .corpus_atlas.models import CorpusAtlas
from .corpus_atlas.router import (
    normalize_route_text,
    rank_query_document_cards,
    route_query_documents,
)
from .models import (
    CaseRouteRecord,
    RetrievalCandidate,
    RetrievalOpportunity,
    RoutedRetrievalRecord,
)

GLOBAL_TOP_K = 10
PER_DOCUMENT_TOP_K = 2
FINAL_TOP_K = 5
RRF_K = 60


@dataclass
class _CandidateEntry:
    chunk_key: str
    chunk: dict[str, Any]
    file_id: str
    file_name: str
    chunk_id: str | None
    chunk_index: int | None
    similarity: float
    global_rank: int | None = None
    local_rank: int | None = None
    within_document_rank: int | None = None
    document_route_rank: int | None = None


def _as_case_route(value: Any) -> CaseRouteRecord:
    return value if isinstance(value, CaseRouteRecord) else CaseRouteRecord.model_validate(value)


def _opportunities(state: dict[str, Any]) -> dict[str, RetrievalOpportunity]:
    result: dict[str, RetrievalOpportunity] = {}
    for raw in state.get("retrieval_opportunities") or []:
        try:
            value = raw if isinstance(raw, RetrievalOpportunity) else RetrievalOpportunity.model_validate(raw)
        except Exception:  # noqa: BLE001 - corrupted checkpoint entry
            continue
        result[value.opportunity_id] = value
    return result


def _source_span(raw: Any, id_field: str, expected_ids: list[str]) -> list[str]:
    values = []
    for item in raw or []:
        item_id = item.get(id_field) if isinstance(item, dict) else getattr(item, id_field, None)
        if item_id not in expected_ids:
            continue
        span = item.get("source_span") if isinstance(item, dict) else getattr(item, "source_span", None)
        if span:
            values.append(str(span))
    return values


def build_document_route_query(request: RetrievalRequest) -> str:
    values = [request.query_text]
    values.extend(
        _source_span(
            request.state.get("plan_anchors"),
            "element_id",
            request.focus_plan_ids,
        )
    )
    values.extend(
        _source_span(
            request.state.get("patient_modifiers"),
            "modifier_id",
            request.focus_modifier_ids,
        )
    )
    return normalize_route_text("\n".join(values))


def _chunk_identity(
    chunk: dict[str, Any],
) -> tuple[str, str, str, str | None, int | None]:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    file_id = str(metadata.get("file_id") or chunk.get("file_id") or "")
    file_name = str(metadata.get("source") or chunk.get("source") or "")
    chunk_id_raw = metadata.get("chunk_id") or chunk.get("chunk_id")
    chunk_id = str(chunk_id_raw) if chunk_id_raw is not None else None
    chunk_index_raw = (
        metadata.get("chunk_index") if metadata.get("chunk_index") is not None else chunk.get("chunk_index")
    )
    try:
        chunk_index = int(chunk_index_raw) if chunk_index_raw is not None else None
    except (TypeError, ValueError):
        chunk_index = None
    if file_id and chunk_id:
        key = f"{file_id}:{chunk_id}"
    else:
        digest = hashlib.sha256(f"{file_id}\0{chunk.get('content') or ''}".encode("utf-8")).hexdigest()[:20]
        key = f"CONTENT:{digest}"
    return key, file_id, file_name, chunk_id, chunk_index


def _similarity(chunk: dict[str, Any]) -> float:
    for key in ("score", "distance"):
        try:
            value = chunk.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _candidate_model(entry: _CandidateEntry) -> RetrievalCandidate:
    paths = []
    if entry.global_rank is not None:
        paths.append("global")
    if entry.local_rank is not None:
        paths.append("local")
    fusion_score = 0.0
    if entry.global_rank is not None:
        fusion_score += 1.0 / (RRF_K + entry.global_rank)
    if entry.local_rank is not None:
        fusion_score += 1.0 / (RRF_K + entry.local_rank)
    return RetrievalCandidate(
        chunk_key=entry.chunk_key,
        file_id=entry.file_id,
        file_name=entry.file_name,
        chunk_id=entry.chunk_id,
        chunk_index=entry.chunk_index,
        similarity=entry.similarity,
        global_rank=entry.global_rank,
        local_rank=entry.local_rank,
        within_document_rank=entry.within_document_rank,
        document_route_rank=entry.document_route_rank,
        fusion_score=fusion_score,
        retrieval_paths=paths,
    )


async def _query_branch(
    *,
    request: RetrievalRequest,
    query_embedding: list[float],
    final_top_k: int,
    filter_file_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    errors: list[str] = []
    calls = 0
    for attempt in range(1, request.context.technical_retry_limit + 2):
        calls += 1
        try:
            async with asyncio.timeout(request.context.retrieval_timeout_seconds):
                result = await knowledge_base.aquery(
                    request.query_text,
                    request.selection.db_id,
                    search_mode="vector",
                    final_top_k=final_top_k,
                    use_reranker=False,
                    include_distances=True,
                    query_embedding=query_embedding,
                    filter_file_ids=filter_file_ids,
                    raise_on_error=True,
                )
            if not isinstance(result, list):
                raise TypeError(f"Milvus routed query 返回 {type(result).__name__}，预期 list")
            return [value for value in result if isinstance(value, dict)], calls, errors
        except Exception as exc:  # noqa: BLE001 - backend errors are traced
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
    return [], calls, errors


def make_routed_retrieval_strategy(
    *,
    opportunity_id: str | None,
) -> RetrievalStrategy:
    async def retrieve(request: RetrievalRequest) -> RetrievalOutcome:
        started_at = utc_isoformat()
        started = time.monotonic()
        context = request.context
        atlas = getattr(context, "_da_prim_atlas", None)
        if not isinstance(atlas, CorpusAtlas):
            raise RuntimeError("DA-PRIM Context 缺少已校验的 Corpus Atlas")
        case_route = _as_case_route(request.state.get("case_route_record"))
        opportunity = _opportunities(request.state).get(opportunity_id or "")
        route_text = build_document_route_query(request)
        texts = [request.query_text]
        if route_text != request.query_text:
            texts.append(route_text)

        stage_started = time.monotonic()
        embedding_attempts: list[TechnicalAttempt] = []
        embeddings: list[list[float]] | None = None
        async with getattr(context, "_prim_search_lock"):
            for attempt_number in range(1, context.technical_retry_limit + 2):
                attempt_started_at = utc_isoformat()
                attempt_started = time.monotonic()
                try:
                    async with asyncio.timeout(context.retrieval_timeout_seconds):
                        embeddings = await knowledge_base.aembed_texts(
                            request.selection.db_id,
                            texts,
                        )
                    if len(embeddings) != len(texts):
                        raise ValueError(
                            "批量 embedding 返回数量不匹配：" f"expected={len(texts)}, actual={len(embeddings)}"
                        )
                    embedding_attempts.append(
                        TechnicalAttempt(
                            attempt=attempt_number,
                            started_at=attempt_started_at,
                            elapsed_ms=round((time.monotonic() - attempt_started) * 1000),
                            status="success",
                            returned_count=len(embeddings),
                        )
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - provider adapters vary
                    embeddings = None
                    status, error_type, error_message = _attempt_error(exc)
                    embedding_attempts.append(
                        TechnicalAttempt(
                            attempt=attempt_number,
                            started_at=attempt_started_at,
                            elapsed_ms=round((time.monotonic() - attempt_started) * 1000),
                            status=status,
                            error_type=error_type,
                            error_message=error_message,
                        )
                    )
            if embeddings is None:
                last_attempt = embedding_attempts[-1]
                elapsed_ms = round((time.monotonic() - started) * 1000)
                diagnostic = RoutedRetrievalRecord(
                    retrieval_record_id=f"RR-{request.query_id.removeprefix('Q-')}",
                    query_id=request.query_id,
                    opportunity_id=(opportunity.opportunity_id if opportunity else None),
                    strategy="routed",
                    case_route_documents=case_route.ranked_documents[:6],
                    effective_documents=case_route.ranked_documents[:6],
                    embedding_batch_count=len(embedding_attempts),
                    stage_elapsed_ms={
                        "embedding": elapsed_ms,
                        "search": 0,
                        "total": elapsed_ms,
                    },
                    degraded_reasons=[
                        "embedding: "
                        f"{attempt.error_type or attempt.status}: "
                        f"{attempt.error_message or ''}".rstrip()
                        for attempt in embedding_attempts
                    ],
                )
                logger.warning(
                    "DA-PRIM routed retrieval embedding failed: query_id=%s batches=%s elapsed_ms=%s",
                    request.query_id,
                    len(embedding_attempts),
                    elapsed_ms,
                )
                return RetrievalOutcome(
                    attempts=embedding_attempts,
                    error_type=last_attempt.error_type,
                    error_message=last_attempt.error_message,
                    diagnostic_record=diagnostic,
                )
            embedding_ms = round((time.monotonic() - stage_started) * 1000)
            query_embedding = embeddings[0]
            route_embedding = embeddings[-1]
            query_route = rank_query_document_cards(
                atlas=atlas,
                query_embedding=route_embedding,
            )
            effective_documents, injected_file_id = route_query_documents(
                atlas=atlas,
                case_route=case_route,
                query_embedding=route_embedding,
                opportunity_file_id=(opportunity.file_id if opportunity else None),
            )

            search_started = time.monotonic()
            global_chunks, backend_calls, global_errors = await _query_branch(
                request=request,
                query_embedding=query_embedding,
                final_top_k=GLOBAL_TOP_K,
                filter_file_ids=None,
            )
            local_results: list[tuple[int, list[dict[str, Any]]]] = []
            degraded_reasons = [f"global: {value}" for value in global_errors]
            for document in effective_documents:
                chunks, calls, errors = await _query_branch(
                    request=request,
                    query_embedding=query_embedding,
                    final_top_k=PER_DOCUMENT_TOP_K,
                    filter_file_ids=[document.file_id],
                )
                backend_calls += calls
                local_results.append((document.rank, chunks))
                degraded_reasons.extend(f"local:{document.file_id}: {value}" for value in errors)
            search_ms = round((time.monotonic() - search_started) * 1000)

        entries: dict[str, _CandidateEntry] = {}
        for rank, chunk in enumerate(global_chunks, start=1):
            key, file_id, file_name, chunk_id, chunk_index = _chunk_identity(chunk)
            entries[key] = _CandidateEntry(
                chunk_key=key,
                chunk=chunk,
                file_id=file_id,
                file_name=file_name,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                similarity=_similarity(chunk),
                global_rank=rank,
            )
        for document_rank, chunks in local_results:
            for within_rank, chunk in enumerate(chunks, start=1):
                key, file_id, file_name, chunk_id, chunk_index = _chunk_identity(chunk)
                local_rank = (within_rank - 1) * len(effective_documents) + document_rank
                current = entries.get(key)
                if current is None:
                    entries[key] = _CandidateEntry(
                        chunk_key=key,
                        chunk=chunk,
                        file_id=file_id,
                        file_name=file_name,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        similarity=_similarity(chunk),
                        local_rank=local_rank,
                        within_document_rank=within_rank,
                        document_route_rank=document_rank,
                    )
                else:
                    current.local_rank = local_rank
                    current.within_document_rank = within_rank
                    current.document_route_rank = document_rank
                    current.similarity = max(current.similarity, _similarity(chunk))

        global_candidates = [_candidate_model(value) for value in entries.values() if value.global_rank is not None]
        global_candidates.sort(key=lambda value: value.global_rank or 10**9)
        local_candidates = [_candidate_model(value) for value in entries.values() if value.local_rank is not None]
        local_candidates.sort(key=lambda value: value.local_rank or 10**9)
        fused_candidates = [_candidate_model(value) for value in entries.values()]
        fused_candidates.sort(
            key=lambda value: (
                -len(value.retrieval_paths),
                -value.fusion_score,
                -value.similarity,
                value.global_rank or 10**9,
                value.local_rank or 10**9,
                value.chunk_key,
            )
        )
        retained_candidates = fused_candidates[:FINAL_TOP_K]
        retained_chunks = []
        for candidate in retained_candidates:
            chunk = dict(entries[candidate.chunk_key].chunk)
            metadata = dict(chunk.get("metadata") or {})
            metadata["retrieval_paths"] = candidate.retrieval_paths
            metadata["fusion_score"] = candidate.fusion_score
            chunk["metadata"] = metadata
            chunk["score"] = candidate.similarity
            retained_chunks.append(chunk)

        elapsed_ms = round((time.monotonic() - started) * 1000)
        if retained_chunks:
            attempt_status = "success"
        elif backend_calls and not degraded_reasons:
            attempt_status = "success_empty"
        else:
            attempt_status = "backend_error"
        diagnostic = RoutedRetrievalRecord(
            retrieval_record_id=f"RR-{request.query_id.removeprefix('Q-')}",
            query_id=request.query_id,
            opportunity_id=opportunity.opportunity_id if opportunity else None,
            strategy="routed",
            case_route_documents=case_route.ranked_documents[:6],
            query_route_documents=query_route,
            effective_documents=effective_documents,
            opportunity_injected_file_id=injected_file_id,
            global_candidates=global_candidates,
            local_candidates=local_candidates,
            fused_candidates=fused_candidates,
            embedding_batch_count=len(embedding_attempts),
            backend_search_count=backend_calls,
            stage_elapsed_ms={
                "embedding": embedding_ms,
                "search": search_ms,
                "total": elapsed_ms,
            },
            degraded_reasons=degraded_reasons,
        )
        error_type = None
        error_message = None
        if attempt_status == "backend_error":
            error_type = "routed_retrieval_failed"
            error_message = "；".join(degraded_reasons) or "所有检索分支均失败"
        logger.info(
            "DA-PRIM routed retrieval complete: query_id=%s status=%s "
            "embedding_batches=%s backend_searches=%s candidates=%s retained=%s "
            "degraded_branches=%s elapsed_ms=%s",
            request.query_id,
            attempt_status,
            len(embedding_attempts),
            backend_calls,
            len(entries),
            len(retained_chunks),
            len(degraded_reasons),
            elapsed_ms,
        )
        return RetrievalOutcome(
            chunks=retained_chunks,
            returned_count=len(entries),
            attempts=[
                TechnicalAttempt(
                    attempt=1,
                    started_at=started_at,
                    elapsed_ms=elapsed_ms,
                    status=attempt_status,
                    returned_count=len(entries),
                    error_type=error_type,
                    error_message=error_message,
                )
            ],
            error_type=error_type,
            error_message=error_message,
            diagnostic_record=diagnostic,
        )

    return retrieve
