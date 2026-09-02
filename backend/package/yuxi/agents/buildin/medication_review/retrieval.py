from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from yuxi import knowledge_base
from yuxi.agents.backends.knowledge_base_backend import resolve_visible_knowledge_bases_for_context
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .context import MedicationReviewContext
from .models import (
    EvidenceCandidate,
    EvidenceItem,
    EvidenceItemV3,
    EvidenceOccurrence,
    EvidenceOccurrenceV2,
    EvidenceOccurrenceV3,
    EvidenceOpenRecord,
    EvidenceSearchRecord,
    QueryBundle,
    RetrievalRecord,
    RetrievedEvidence,
    SearchIntent,
    SearchSubquery,
    TechnicalAttempt,
)

Retriever = Callable[..., Awaitable[list[dict[str, Any]]]]


class MedicationReviewConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RetrieverSelection:
    db_id: str
    retriever: Retriever
    snapshot: dict[str, Any]


def validate_context(context: MedicationReviewContext) -> None:
    selected = [str(value).strip() for value in (context.knowledges or []) if str(value).strip()]
    if len(selected) != 1:
        raise MedicationReviewConfigError("必须且只能选择一个 Milvus 知识库")
    if context.per_query_top_k != 3:
        raise MedicationReviewConfigError("relation-coverage-a1-vector-atomic-v1 固定每个查询 Top-3")
    if not 1 <= context.retrieval_concurrency <= 2:
        raise MedicationReviewConfigError("retrieval_concurrency 必须在 1–2 之间")
    if not 30 <= context.retrieval_timeout_seconds <= 900:
        raise MedicationReviewConfigError("retrieval_timeout_seconds 必须在 30–900 之间")
    if not 1 <= context.max_query_bundles <= 256:
        raise MedicationReviewConfigError("max_query_bundles 必须在 1–256 之间")


def validate_v3_context(context: MedicationReviewContext) -> None:
    selected = [str(value).strip() for value in (context.knowledges or []) if str(value).strip()]
    if len(selected) != 1:
        raise MedicationReviewConfigError("必须且只能选择一个 Milvus 知识库")
    if context.retrieval_top_k != 5:
        raise MedicationReviewConfigError("PEA-RAG v2 实验固定 retrieval_top_k=5")
    if not 30 <= context.retrieval_timeout_seconds <= 900:
        raise MedicationReviewConfigError("retrieval_timeout_seconds 必须在 30–900 之间")
    if not 1 <= context.max_search_calls <= 8:
        raise MedicationReviewConfigError("max_search_calls 必须在 1–8 之间")
    if not 0 <= context.max_open_calls <= 2:
        raise MedicationReviewConfigError("max_open_calls 必须在 0–2 之间")
    if not 1 <= context.max_agent_steps <= 64:
        raise MedicationReviewConfigError("max_agent_steps 必须在 1–64 之间")
    if not 0 <= context.technical_retry_limit <= 3:
        raise MedicationReviewConfigError("technical_retry_limit 必须在 0–3 之间")
    if not 1000 <= context.max_final_evidence_tokens <= 100000:
        raise MedicationReviewConfigError(
            "max_final_evidence_tokens 必须在 1000–100000 之间"
        )
    if not 0 <= context.plan_repair_limit <= 1:
        raise MedicationReviewConfigError("plan_repair_limit 必须在 0–1 之间")
    if not 1 <= context.max_review_questions <= 12:
        raise MedicationReviewConfigError("max_review_questions 必须在 1–12 之间")
    if not 1 <= context.max_claim_evidence <= 30:
        raise MedicationReviewConfigError("max_claim_evidence 必须在 1–30 之间")
    if not 1 <= context.max_subqueries_per_action <= 3:
        raise MedicationReviewConfigError("max_subqueries_per_action 必须在 1–3 之间")
    if context.run_mode == "stop_after_agenda" and context.agenda_mode != "dynamic":
        raise MedicationReviewConfigError(
            "stop_after_agenda 只能与 agenda_mode=dynamic 组合"
        )
    if context.run_mode == "stop_after_claims" and context.synthesis_mode != "claims":
        raise MedicationReviewConfigError(
            "stop_after_claims 只能与 synthesis_mode=claims 组合"
        )


def validate_v2_context(context: MedicationReviewContext) -> None:
    """Compatibility alias for callers migrating from the v1.4 graph."""
    validate_v3_context(context)


def _embedding_model_id(metadata: dict[str, Any]) -> str | None:
    embed_info = metadata.get("embed_info")
    if not isinstance(embed_info, dict):
        return None
    value = embed_info.get("model_id") or embed_info.get("model")
    return str(value) if value else None


def _effective_query_params(metadata: dict[str, Any], context: MedicationReviewContext) -> dict[str, Any]:
    persisted = metadata.get("query_params")
    if not isinstance(persisted, dict):
        persisted = {}
    options = persisted.get("options")
    if isinstance(options, dict):
        persisted = options
    return {
        "search_mode": "vector",
        "final_top_k": context.per_query_top_k,
        "similarity_threshold": float(persisted.get("similarity_threshold", 0.2)),
        "include_distances": True,
        "use_reranker": False,
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "use_async_embedding": True,
        "raise_on_error": True,
    }


async def resolve_milvus_retriever(context: MedicationReviewContext) -> RetrieverSelection:
    validate_context(context)
    selected_name = str(context.knowledges[0]).strip()
    visible = await resolve_visible_knowledge_bases_for_context(context)
    matches = [item for item in visible if str(item.get("name") or "").strip() == selected_name]
    if len(matches) != 1:
        raise MedicationReviewConfigError(
            f"知识库 {selected_name!r} 在当前用户可见范围内不存在或名称不唯一"
        )

    database = matches[0]
    db_id = str(database.get("db_id") or "").strip()
    kb_type = str(database.get("kb_type") or "").strip().lower()
    if not db_id:
        raise MedicationReviewConfigError("选定知识库缺少 db_id")
    if kb_type != "milvus":
        raise MedicationReviewConfigError(f"第一版只支持 Milvus 知识库，当前类型为 {kb_type or 'unknown'}")

    retriever_info = knowledge_base.get_retrievers().get(db_id)
    if not isinstance(retriever_info, dict) or not callable(retriever_info.get("retriever")):
        raise MedicationReviewConfigError(f"无法取得知识库 {db_id} 的 Retriever")
    metadata = retriever_info.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    snapshot = {
        "db_id": db_id,
        "name": str(database.get("name") or retriever_info.get("name") or ""),
        "kb_type": "milvus",
        "embedding_model": _embedding_model_id(metadata),
        "query_params": _effective_query_params(metadata, context),
    }
    return RetrieverSelection(db_id=db_id, retriever=retriever_info["retriever"], snapshot=snapshot)


async def resolve_milvus_retriever_v2(context: MedicationReviewContext) -> RetrieverSelection:
    validate_v3_context(context)
    selected_name = str(context.knowledges[0]).strip()
    visible = await resolve_visible_knowledge_bases_for_context(context)
    matches = [item for item in visible if str(item.get("name") or "").strip() == selected_name]
    if len(matches) != 1:
        raise MedicationReviewConfigError(
            f"知识库 {selected_name!r} 在当前用户可见范围内不存在或名称不唯一"
        )

    database = matches[0]
    db_id = str(database.get("db_id") or "").strip()
    kb_type = str(database.get("kb_type") or "").strip().lower()
    if not db_id:
        raise MedicationReviewConfigError("选定知识库缺少 db_id")
    if kb_type != "milvus":
        raise MedicationReviewConfigError(f"PEA-RAG 只支持 Milvus，当前类型为 {kb_type or 'unknown'}")

    retriever_info = knowledge_base.get_retrievers().get(db_id)
    if not isinstance(retriever_info, dict) or not callable(retriever_info.get("retriever")):
        raise MedicationReviewConfigError(f"无法取得知识库 {db_id} 的 Retriever")
    metadata = retriever_info.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    persisted = metadata.get("query_params")
    if not isinstance(persisted, dict):
        persisted = {}
    if isinstance(persisted.get("options"), dict):
        persisted = persisted["options"]
    snapshot = {
        "db_id": db_id,
        "name": str(database.get("name") or retriever_info.get("name") or ""),
        "kb_type": "milvus",
        "embedding_model": _embedding_model_id(metadata),
        "query_params": {
            "search_mode": "vector",
            "final_top_k": context.retrieval_top_k,
            "similarity_threshold": float(persisted.get("similarity_threshold", 0.2)),
            "include_distances": True,
            "use_reranker": False,
            "metric_type": "COSINE",
            "use_async_embedding": True,
            "raise_on_error": True,
        },
    }
    return RetrieverSelection(
        db_id=db_id,
        retriever=retriever_info["retriever"],
        snapshot=snapshot,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def _evidence_id(db_id: str, chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    raw_text = str(chunk.get("content") or "")
    identity = "\x1f".join(
        [
            db_id,
            str(metadata.get("file_id") or ""),
            str(metadata.get("chunk_id") or ""),
            str(metadata.get("chunk_index") or ""),
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _numeric_value(chunk: dict[str, Any], key: str) -> float | None:
    value = chunk.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _error_message(exc: BaseException) -> str:
    cause = exc.__cause__
    if cause:
        return f"{exc}；cause={type(cause).__name__}: {cause}"
    return str(exc)


async def retrieve_query_bundles(
    bundles: list[QueryBundle],
    context: MedicationReviewContext,
    review_run_id: str,
    case_id: str,
) -> tuple[
    list[QueryBundle],
    list[RetrievalRecord],
    list[RetrievedEvidence],
    dict[str, Any],
    dict[str, Any],
]:
    selection = await resolve_milvus_retriever(context)
    semaphore = asyncio.Semaphore(context.retrieval_concurrency)
    started = time.monotonic()

    async def retrieve_one(
        index: int,
        bundle: QueryBundle,
    ) -> tuple[QueryBundle, RetrievalRecord, list[dict[str, Any]]]:
        started_at = utc_isoformat()
        bundle_started = time.monotonic()
        if bundle.validation_status != "valid":
            record = RetrievalRecord(
                bundle_id=bundle.bundle_id,
                query_text=bundle.query_text,
                slot_ids=bundle.slot_ids,
                status="invalid_query",
                started_at=started_at,
                elapsed_ms=0,
                error_type="invalid_query",
                error_message="；".join(bundle.validation_errors),
            )
            return bundle, record, []

        logger.info(
            f"Medication review retrieval start: review_run_id={review_run_id}, case_id={case_id}, "
            f"bundle={index + 1}/{len(bundles)}, bundle_id={bundle.bundle_id}, "
            f"query_hash={hashlib.sha256(bundle.query_text.encode('utf-8')).hexdigest()[:12]}"
        )
        try:
            async with semaphore:
                async with asyncio.timeout(context.retrieval_timeout_seconds):
                    chunks = await selection.retriever(
                        bundle.query_text,
                        search_mode="vector",
                        final_top_k=3,
                        use_reranker=False,
                        include_distances=True,
                        raise_on_error=True,
                        use_async_embedding=True,
                    )
            if not isinstance(chunks, list):
                raise TypeError(f"Retriever 返回类型不是 list：{type(chunks).__name__}")
            retained = [item for item in chunks[:3] if isinstance(item, dict)]
            status = "success" if retained else "success_empty"
            record = RetrievalRecord(
                bundle_id=bundle.bundle_id,
                query_text=bundle.query_text,
                slot_ids=bundle.slot_ids,
                status=status,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - bundle_started) * 1000),
                returned_count=len(chunks),
                retained_count=len(retained),
            )
            return bundle, record, retained
        except TimeoutError as exc:
            status, error_type = "timeout", "timeout"
            error_message = _error_message(exc) or f"超过 {context.retrieval_timeout_seconds} 秒"
        except Exception as exc:  # noqa: BLE001 - isolate one failed relation query
            if type(exc).__name__ == "MilvusEmbeddingError":
                status, error_type = "embedding_error", "embedding_error"
            else:
                status, error_type = "backend_error", type(exc).__name__
            error_message = _error_message(exc)

        record = RetrievalRecord(
            bundle_id=bundle.bundle_id,
            query_text=bundle.query_text,
            slot_ids=bundle.slot_ids,
            status=status,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - bundle_started) * 1000),
            error_type=error_type,
            error_message=error_message,
        )
        return bundle, record, []

    results = await asyncio.gather(*(retrieve_one(index, bundle) for index, bundle in enumerate(bundles)))
    evidence_index: dict[str, RetrievedEvidence] = {}
    updated_bundles: list[QueryBundle] = []
    records: list[RetrievalRecord] = []

    for bundle, record, chunks in results:
        evidence_ids: list[str] = []
        for rank, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            evidence_id = _evidence_id(selection.db_id, chunk)
            evidence_ids.append(evidence_id)
            occurrence = EvidenceOccurrence(
                bundle_id=bundle.bundle_id,
                slot_ids=bundle.slot_ids,
                rank=rank,
                score=_numeric_value(chunk, "score"),
                distance=_numeric_value(chunk, "distance"),
            )
            existing = evidence_index.get(evidence_id)
            if existing:
                evidence_index[evidence_id] = existing.model_copy(
                    update={"occurrences": [*existing.occurrences, occurrence]}
                )
                continue
            evidence_index[evidence_id] = RetrievedEvidence(
                evidence_id=evidence_id,
                raw_text=str(chunk.get("content") or ""),
                source_document=str(metadata.get("source")) if metadata.get("source") is not None else None,
                file_id=str(metadata.get("file_id")) if metadata.get("file_id") is not None else None,
                chunk_id=str(metadata.get("chunk_id")) if metadata.get("chunk_id") is not None else None,
                chunk_index=metadata.get("chunk_index"),
                raw_metadata=_json_safe(metadata),
                occurrences=[occurrence],
            )
        updated_bundles.append(bundle.model_copy(update={"evidence_ids": evidence_ids}))
        records.append(
            record.model_copy(
                update={
                    "evidence_ids": evidence_ids,
                    "retained_count": len(evidence_ids) if record.status == "success" else record.retained_count,
                }
            )
        )
        logger.info(
            f"Medication review retrieval end: review_run_id={review_run_id}, case_id={case_id}, "
            f"bundle_id={bundle.bundle_id}, status={record.status}, elapsed_ms={record.elapsed_ms}"
        )

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    usage = {
        "query_count": len(bundles),
        "status_counts": status_counts,
        "unique_evidence_count": len(evidence_index),
        "retrieval_elapsed_ms": round((time.monotonic() - started) * 1000),
        "retrieval_concurrency": context.retrieval_concurrency,
    }
    return updated_bundles, records, list(evidence_index.values()), selection.snapshot, usage


def _v2_evidence_from_chunk(
    *,
    db_id: str,
    chunk: dict[str, Any],
    intent: SearchIntent,
    rank: int,
) -> EvidenceItem:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    occurrence = EvidenceOccurrenceV2(
        query_id=intent.query_id,
        target_element_ids=intent.target_element_ids,
        target_review_ids=intent.target_review_ids,
        intended_evidence_role=intent.intended_evidence_role,
        rank=rank,
        score=_numeric_value(chunk, "score"),
        distance=_numeric_value(chunk, "distance"),
    )
    return EvidenceItem(
        evidence_id=_evidence_id(db_id, chunk),
        raw_text=str(chunk.get("content") or ""),
        source_document=str(metadata.get("source")) if metadata.get("source") is not None else None,
        file_id=str(metadata.get("file_id")) if metadata.get("file_id") is not None else None,
        chunk_id=str(metadata.get("chunk_id")) if metadata.get("chunk_id") is not None else None,
        chunk_index=metadata.get("chunk_index"),
        raw_metadata=_json_safe(metadata),
        occurrences=[occurrence],
    )


async def retrieve_for_agent(
    *,
    intent: SearchIntent,
    context: MedicationReviewContext,
    review_run_id: str,
    case_id: str,
    existing_evidence_ids: set[str] | None = None,
) -> tuple[EvidenceSearchRecord, list[EvidenceItem], dict[str, Any]]:
    selection = await resolve_milvus_retriever_v2(context)
    attempts: list[TechnicalAttempt] = []
    chunks: list[dict[str, Any]] = []

    for attempt_number in range(1, context.technical_retry_limit + 2):
        started_at = utc_isoformat()
        started = time.monotonic()
        logger.info(
            f"PEA-RAG search start: review_run_id={review_run_id}, case_id={case_id}, "
            f"query_id={intent.query_id}, attempt={attempt_number}, "
            f"query_hash={hashlib.sha256(intent.query_text.encode('utf-8')).hexdigest()[:12]}"
        )
        try:
            async with asyncio.timeout(context.retrieval_timeout_seconds):
                result = await selection.retriever(
                    intent.query_text,
                    search_mode="vector",
                    final_top_k=context.retrieval_top_k,
                    use_reranker=False,
                    include_distances=True,
                    raise_on_error=True,
                    use_async_embedding=True,
                )
            if not isinstance(result, list):
                raise TypeError(f"Retriever 返回类型不是 list：{type(result).__name__}")
            chunks = [item for item in result[: context.retrieval_top_k] if isinstance(item, dict)]
            attempts.append(
                TechnicalAttempt(
                    attempt=attempt_number,
                    started_at=started_at,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    status="success" if chunks else "success_empty",
                    returned_count=len(result),
                )
            )
            break
        except TimeoutError as exc:
            status, error_type = "timeout", "timeout"
            error_message = _error_message(exc) or f"超过 {context.retrieval_timeout_seconds} 秒"
        except Exception as exc:  # noqa: BLE001 - preserve retriever failure category
            if type(exc).__name__ == "MilvusEmbeddingError":
                status, error_type = "embedding_error", "embedding_error"
            else:
                status, error_type = "backend_error", type(exc).__name__
            error_message = _error_message(exc)
        attempts.append(
            TechnicalAttempt(
                attempt=attempt_number,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                status=status,
                error_type=error_type,
                error_message=error_message,
            )
        )

    evidence = [
        _v2_evidence_from_chunk(
            db_id=selection.db_id,
            chunk=chunk,
            intent=intent,
            rank=rank,
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]
    existing = existing_evidence_ids or set()
    duplicate_count = sum(item.evidence_id in existing for item in evidence)
    duplicate_ratio = duplicate_count / len(evidence) if evidence else 0.0
    if evidence:
        record_status = "success"
    elif attempts and attempts[-1].status == "success_empty":
        record_status = "success_empty"
    else:
        record_status = "technical_failed"
    record = EvidenceSearchRecord(
        intent=intent,
        status=record_status,
        attempts=attempts,
        candidate_evidence_ids=[item.evidence_id for item in evidence],
        duplicate_ratio=duplicate_ratio,
    )
    logger.info(
        f"PEA-RAG search end: review_run_id={review_run_id}, case_id={case_id}, "
        f"query_id={intent.query_id}, status={record.status}, candidates={len(evidence)}, "
        f"duplicate_ratio={duplicate_ratio:.3f}"
    )
    return record, evidence, selection.snapshot


async def open_evidence_window(
    *,
    parent: EvidenceItem,
    context: MedicationReviewContext,
    reason: str,
    open_id: str,
    window_before: int = 1,
    window_after: int = 1,
) -> tuple[EvidenceOpenRecord, list[EvidenceItem]]:
    started_at = utc_isoformat()
    started = time.monotonic()
    if not parent.file_id or parent.chunk_index is None:
        return (
            EvidenceOpenRecord(
                open_id=open_id,
                parent_evidence_id=parent.evidence_id,
                reason=reason,
                window_before=window_before,
                window_after=window_after,
                status="invalid_source",
                started_at=started_at,
                elapsed_ms=0,
                error_type="invalid_source",
                error_message="证据缺少 file_id 或 chunk_index",
            ),
            [],
        )

    selection = await resolve_milvus_retriever_v2(context)
    try:
        target_index = int(parent.chunk_index)
    except (TypeError, ValueError):
        return (
            EvidenceOpenRecord(
                open_id=open_id,
                parent_evidence_id=parent.evidence_id,
                reason=reason,
                window_before=window_before,
                window_after=window_after,
                status="invalid_source",
                started_at=started_at,
                elapsed_ms=0,
                error_type="invalid_chunk_index",
                error_message=f"无法解析 chunk_index：{parent.chunk_index}",
            ),
            [],
        )

    try:
        async with asyncio.timeout(context.retrieval_timeout_seconds):
            content_info = await knowledge_base.get_file_content(selection.db_id, parent.file_id)
        lines = content_info.get("lines") if isinstance(content_info, dict) else None
        if not isinstance(lines, list):
            lines = []
        lower = max(target_index - window_before, 0)
        upper = target_index + window_after
        selected_lines = [
            line
            for line in lines
            if isinstance(line, dict)
            and lower <= int(line.get("chunk_order_index", -1)) <= upper
        ]
        evidence: list[EvidenceItem] = []
        for line in selected_lines:
            chunk_index = int(line.get("chunk_order_index", 0))
            chunk_id = str(line.get("id") or f"{parent.file_id}_chunk_{chunk_index}")
            chunk = {
                "content": str(line.get("content") or ""),
                "metadata": {
                    "source": parent.source_document,
                    "file_id": parent.file_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                },
            }
            evidence.append(
                EvidenceItem(
                    evidence_id=_evidence_id(selection.db_id, chunk),
                    raw_text=chunk["content"],
                    source_document=parent.source_document,
                    file_id=parent.file_id,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    raw_metadata=chunk["metadata"],
                    source_method="open",
                    parent_evidence_id=parent.evidence_id,
                )
            )
        record = EvidenceOpenRecord(
            open_id=open_id,
            parent_evidence_id=parent.evidence_id,
            reason=reason,
            window_before=window_before,
            window_after=window_after,
            status="success" if evidence else "success_empty",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            evidence_ids=[item.evidence_id for item in evidence],
        )
        return record, evidence
    except TimeoutError as exc:
        status, error_type = "timeout", "timeout"
        error_message = _error_message(exc) or f"超过 {context.retrieval_timeout_seconds} 秒"
    except Exception as exc:  # noqa: BLE001
        status, error_type = "backend_error", type(exc).__name__
        error_message = _error_message(exc)
    return (
        EvidenceOpenRecord(
            open_id=open_id,
            parent_evidence_id=parent.evidence_id,
            reason=reason,
            window_before=window_before,
            window_after=window_after,
            status=status,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            error_type=error_type,
            error_message=error_message,
        ),
        [],
    )


@dataclass(frozen=True)
class V3RetrievalResult:
    candidates: list[EvidenceCandidate]
    attempts: list[TechnicalAttempt]
    knowledge_base_snapshot: dict[str, Any]
    status: str
    returned_count: int
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class V3OpenResult:
    candidates: list[EvidenceCandidate]
    status: str
    started_at: str
    elapsed_ms: int
    attempt_count: int = 0
    error_type: str | None = None
    error_message: str | None = None


def _v3_candidate_from_chunk(
    *,
    db_id: str,
    chunk: dict[str, Any],
    subquery: SearchSubquery,
    rank: int,
) -> EvidenceCandidate:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return EvidenceCandidate(
        content_hash=_evidence_id(db_id, chunk),
        raw_text=str(chunk.get("content") or ""),
        source_document=str(metadata.get("source")) if metadata.get("source") is not None else None,
        file_id=str(metadata.get("file_id")) if metadata.get("file_id") is not None else None,
        chunk_id=str(metadata.get("chunk_id")) if metadata.get("chunk_id") is not None else None,
        chunk_index=metadata.get("chunk_index"),
        raw_metadata=_json_safe(metadata),
        occurrences=[
            EvidenceOccurrenceV3(
                query_id=subquery.query_id,
                linked_question_ids=subquery.linked_question_ids,
                linked_element_ids=subquery.linked_element_ids,
                linked_patient_fact_ids=subquery.linked_patient_fact_ids,
                rank=rank,
                score=_numeric_value(chunk, "score"),
                distance=_numeric_value(chunk, "distance"),
            )
        ],
    )


async def retrieve_subquery(
    *,
    subquery: SearchSubquery,
    context: MedicationReviewContext,
    review_run_id: str,
    case_id: str,
) -> V3RetrievalResult:
    selection: RetrieverSelection | None = None
    attempts: list[TechnicalAttempt] = []
    chunks: list[dict[str, Any]] = []
    returned_count = 0
    last_error_type: str | None = None
    last_error_message: str | None = None
    for attempt_number in range(1, context.technical_retry_limit + 2):
        started_at = utc_isoformat()
        started = time.monotonic()
        logger.info(
            "PEA-RAG v3 search start: "
            f"review_run_id={review_run_id}, case_id={case_id}, "
            f"query_id={subquery.query_id}, attempt={attempt_number}"
        )
        try:
            if selection is None:
                selection = await resolve_milvus_retriever_v2(context)
            async with asyncio.timeout(context.retrieval_timeout_seconds):
                result = await selection.retriever(
                    subquery.query_text,
                    search_mode="vector",
                    final_top_k=context.retrieval_top_k,
                    use_reranker=False,
                    include_distances=True,
                    raise_on_error=True,
                    use_async_embedding=True,
                )
            if not isinstance(result, list):
                raise TypeError(f"Retriever 返回类型不是 list：{type(result).__name__}")
            returned_count = len(result)
            chunks = [
                item for item in result[: context.retrieval_top_k]
                if isinstance(item, dict)
            ]
            attempts.append(
                TechnicalAttempt(
                    attempt=attempt_number,
                    started_at=started_at,
                    elapsed_ms=round((time.monotonic() - started) * 1000),
                    status="success" if chunks else "success_empty",
                    returned_count=returned_count,
                )
            )
            break
        except TimeoutError as exc:
            status, last_error_type = "timeout", "timeout"
            last_error_message = _error_message(exc) or (
                f"超过 {context.retrieval_timeout_seconds} 秒"
            )
        except Exception as exc:  # noqa: BLE001 - preserve backend category
            if type(exc).__name__ == "MilvusEmbeddingError":
                status, last_error_type = "embedding_error", "embedding_error"
            else:
                status, last_error_type = "backend_error", type(exc).__name__
            last_error_message = _error_message(exc)
        attempts.append(
            TechnicalAttempt(
                attempt=attempt_number,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                status=status,
                error_type=last_error_type,
                error_message=last_error_message,
            )
        )

    candidates = [
        _v3_candidate_from_chunk(
            db_id=selection.db_id,
            chunk=chunk,
            subquery=subquery,
            rank=rank,
        )
        for rank, chunk in enumerate(chunks, start=1)
    ] if selection is not None else []
    final_status = (
        "success"
        if candidates
        else "success_empty"
        if attempts and attempts[-1].status == "success_empty"
        else "technical_failed"
    )
    return V3RetrievalResult(
        candidates=candidates,
        attempts=attempts,
        knowledge_base_snapshot=selection.snapshot if selection is not None else {},
        status=final_status,
        returned_count=returned_count,
        error_type=last_error_type,
        error_message=last_error_message,
    )


async def open_evidence_window_v3(
    *,
    parent: EvidenceItemV3,
    context: MedicationReviewContext,
    window_before: int = 1,
    window_after: int = 1,
) -> V3OpenResult:
    started_at = utc_isoformat()
    started = time.monotonic()
    if not parent.file_id or parent.chunk_index is None:
        return V3OpenResult(
            candidates=[],
            status="invalid_source",
            started_at=started_at,
            elapsed_ms=0,
            attempt_count=0,
            error_type="invalid_source",
            error_message="证据缺少 file_id 或 chunk_index",
        )
    try:
        target_index = int(parent.chunk_index)
    except (TypeError, ValueError):
        return V3OpenResult(
            candidates=[],
            status="invalid_source",
            started_at=started_at,
            elapsed_ms=0,
            attempt_count=0,
            error_type="invalid_chunk_index",
            error_message=f"无法解析 chunk_index：{parent.chunk_index}",
        )
    status = "backend_error"
    error_type: str | None = None
    error_message: str | None = None
    attempt_count = 0
    selection: RetrieverSelection | None = None
    for attempt_count in range(1, context.technical_retry_limit + 2):
        try:
            if selection is None:
                selection = await resolve_milvus_retriever_v2(context)
            async with asyncio.timeout(context.retrieval_timeout_seconds):
                content_info = await knowledge_base.get_file_content(
                    selection.db_id,
                    parent.file_id,
                )
            lines = content_info.get("lines") if isinstance(content_info, dict) else []
            if not isinstance(lines, list):
                lines = []
            lower = max(target_index - window_before, 0)
            upper = target_index + window_after
            candidates: list[EvidenceCandidate] = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                try:
                    chunk_index = int(line.get("chunk_order_index", -1))
                except (TypeError, ValueError):
                    continue
                if not lower <= chunk_index <= upper:
                    continue
                chunk_id = str(line.get("id") or f"{parent.file_id}_chunk_{chunk_index}")
                chunk = {
                    "content": str(line.get("content") or ""),
                    "metadata": {
                        "source": parent.source_document,
                        "file_id": parent.file_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                    },
                }
                candidates.append(
                    EvidenceCandidate(
                        content_hash=_evidence_id(selection.db_id, chunk),
                        raw_text=chunk["content"],
                        source_document=parent.source_document,
                        file_id=parent.file_id,
                        chunk_id=chunk_id,
                        chunk_index=chunk_index,
                        raw_metadata=chunk["metadata"],
                        source_method="open",
                        parent_content_hash=parent.content_hash,
                    )
                )
            return V3OpenResult(
                candidates=candidates,
                status="success" if candidates else "success_empty",
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                attempt_count=attempt_count,
            )
        except TimeoutError as exc:
            status, error_type = "timeout", "timeout"
            error_message = _error_message(exc) or f"超过 {context.retrieval_timeout_seconds} 秒"
        except Exception as exc:  # noqa: BLE001
            status, error_type = "backend_error", type(exc).__name__
            error_message = _error_message(exc)
    return V3OpenResult(
        candidates=[],
        status=status,
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        attempt_count=attempt_count,
        error_type=error_type,
        error_message=error_message,
    )
