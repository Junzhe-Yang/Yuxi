import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import Field
from yuxi import knowledge_base
from yuxi.agents.backends.knowledge_base_backend import (
    resolve_visible_knowledge_bases_for_context,
)
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .context import (
    MedicationReviewLiteContext,
    profile_uses_query_centered_cards,
    validate_context_values,
)
from .evidence import (
    ExcerptView,
    build_query_centered_excerpt,
    content_identity_hash,
    evidence_id_from_hash,
    format_evidence_card,
    json_safe,
)
from .models import (
    EvidenceItem,
    EvidenceOccurrence,
    OpenRecord,
    SearchRecord,
    TechnicalAttempt,
)

Retriever = Callable[..., Awaitable[list[dict[str, Any]]]]
RETRIEVAL_TOP_K = 5


class MedicationReviewLiteConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RetrieverSelection:
    db_id: str
    retriever: Retriever
    snapshot: dict[str, Any]


def _embedding_model_id(metadata: dict[str, Any]) -> str | None:
    embed_info = metadata.get("embed_info")
    if not isinstance(embed_info, dict):
        return None
    value = embed_info.get("model_id") or embed_info.get("model")
    return str(value) if value else None


async def resolve_milvus_retriever(
    context: MedicationReviewLiteContext,
) -> RetrieverSelection:
    validate_context_values(context)
    cached = getattr(context, "_pat_rag_retriever_selection", None)
    if isinstance(cached, RetrieverSelection):
        return cached

    selected_name = str(context.knowledges[0]).strip()
    visible = await resolve_visible_knowledge_bases_for_context(context)
    matches = [
        item
        for item in visible
        if str(item.get("name") or "").strip() == selected_name
    ]
    if len(matches) != 1:
        raise MedicationReviewLiteConfigError(
            f"知识库 {selected_name!r} 在当前用户可见范围内不存在或名称不唯一"
        )

    database = matches[0]
    db_id = str(database.get("db_id") or "").strip()
    kb_type = str(database.get("kb_type") or "").strip().lower()
    if not db_id:
        raise MedicationReviewLiteConfigError("选定知识库缺少 db_id")
    if kb_type != "milvus":
        raise MedicationReviewLiteConfigError(
            f"PAT-RAG 第一版只支持 Milvus，当前类型为 {kb_type or 'unknown'}"
        )

    retriever_info = knowledge_base.get_retrievers().get(db_id)
    if not isinstance(retriever_info, dict) or not callable(
        retriever_info.get("retriever")
    ):
        raise MedicationReviewLiteConfigError(
            f"无法取得知识库 {db_id} 的 Retriever"
        )
    metadata = retriever_info.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    persisted = metadata.get("query_params")
    if not isinstance(persisted, dict):
        persisted = {}
    if isinstance(persisted.get("options"), dict):
        persisted = persisted["options"]

    selection = RetrieverSelection(
        db_id=db_id,
        retriever=retriever_info["retriever"],
        snapshot={
            "db_id": db_id,
            "name": str(
                database.get("name") or retriever_info.get("name") or ""
            ),
            "kb_type": "milvus",
            "embedding_model": _embedding_model_id(metadata),
            "query_params": {
                "search_mode": "vector",
                "final_top_k": RETRIEVAL_TOP_K,
                "similarity_threshold": float(
                    persisted.get("similarity_threshold", 0.2)
                ),
                "include_distances": True,
                "use_reranker": False,
                "metric_type": "COSINE",
                "use_async_embedding": True,
                "raise_on_error": True,
            },
        },
    )
    setattr(context, "_pat_rag_retriever_selection", selection)
    return selection


def ensure_runtime_resources(context: MedicationReviewLiteContext) -> None:
    if getattr(context, "_pat_rag_search_lock", None) is None:
        setattr(context, "_pat_rag_search_lock", asyncio.Lock())


def _record_id(prefix: str, tool_call_id: str | None) -> str:
    source = tool_call_id or str(uuid.uuid4())
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


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


def _attempt_error(
    exc: BaseException,
) -> tuple[str, str, str]:
    if isinstance(exc, TimeoutError):
        return "timeout", "timeout", _error_message(exc) or "检索超时"
    if type(exc).__name__ == "MilvusEmbeddingError":
        return "embedding_error", "embedding_error", _error_message(exc)
    return "backend_error", type(exc).__name__, _error_message(exc)


def _state_evidence(
    state: dict[str, Any],
) -> dict[str, EvidenceItem]:
    result: dict[str, EvidenceItem] = {}
    raw_store = state.get("evidence_store")
    if not isinstance(raw_store, dict):
        return result
    for key, value in raw_store.items():
        try:
            result[str(key)] = (
                value
                if isinstance(value, EvidenceItem)
                else EvidenceItem.model_validate(value)
            )
        except Exception:  # noqa: BLE001 - ignore corrupted prior item locally
            continue
    return result


def _known_element_ids(state: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for raw in state.get("plan_anchors") or []:
        if isinstance(raw, dict):
            value = raw.get("element_id")
        else:
            value = getattr(raw, "element_id", None)
        if value:
            result.add(str(value))
    return result


def _full_chunk_excerpt(raw_text: str) -> ExcerptView:
    return ExcerptView(
        text=raw_text.strip(),
        start=0,
        end=len(raw_text),
        fallback=False,
    )


def _item_from_chunk(
    *,
    selection: RetrieverSelection,
    chunk: dict[str, Any],
) -> EvidenceItem:
    metadata = (
        chunk.get("metadata")
        if isinstance(chunk.get("metadata"), dict)
        else {}
    )
    raw_text = str(chunk.get("content") or "")
    file_id_raw = metadata.get("file_id") or chunk.get("file_id")
    chunk_id_raw = metadata.get("chunk_id") or chunk.get("chunk_id")
    chunk_index = (
        metadata.get("chunk_index")
        if metadata.get("chunk_index") is not None
        else chunk.get("chunk_index")
    )
    file_id = str(file_id_raw) if file_id_raw is not None else None
    chunk_id = str(chunk_id_raw) if chunk_id_raw is not None else None
    content_hash = content_identity_hash(
        db_id=selection.db_id,
        raw_text=raw_text,
        file_id=file_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
    )
    return EvidenceItem(
        evidence_id=evidence_id_from_hash(content_hash),
        content_hash=content_hash,
        raw_text=raw_text,
        source_document=(
            str(metadata.get("source") or chunk.get("source"))
            if metadata.get("source") is not None
            or chunk.get("source") is not None
            else None
        ),
        file_id=file_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        raw_metadata=json_safe(metadata),
    )


async def _retrieve(
    *,
    selection: RetrieverSelection,
    context: MedicationReviewLiteContext,
    query_text: str,
) -> tuple[list[dict[str, Any]], int, list[TechnicalAttempt], str | None, str | None]:
    attempts: list[TechnicalAttempt] = []
    returned_count = 0
    last_error_type: str | None = None
    last_error_message: str | None = None
    lock = getattr(context, "_pat_rag_search_lock")

    async with lock:
        for attempt_number in range(1, context.technical_retry_limit + 2):
            started_at = utc_isoformat()
            started = time.monotonic()
            try:
                async with asyncio.timeout(context.retrieval_timeout_seconds):
                    result = await selection.retriever(
                        query_text,
                        search_mode="vector",
                        final_top_k=RETRIEVAL_TOP_K,
                        use_reranker=False,
                        include_distances=True,
                        raise_on_error=True,
                        use_async_embedding=True,
                    )
                if not isinstance(result, list):
                    raise TypeError(
                        f"Retriever 返回类型不是 list：{type(result).__name__}"
                    )
                returned_count = len(result)
                retained = [
                    item
                    for item in result[:RETRIEVAL_TOP_K]
                    if isinstance(item, dict)
                ]
                attempts.append(
                    TechnicalAttempt(
                        attempt=attempt_number,
                        started_at=started_at,
                        elapsed_ms=round(
                            (time.monotonic() - started) * 1000
                        ),
                        status="success" if retained else "success_empty",
                        returned_count=returned_count,
                    )
                )
                return (
                    retained,
                    returned_count,
                    attempts,
                    None,
                    None,
                )
            except Exception as exc:  # noqa: BLE001 - backend adapters vary
                status, last_error_type, last_error_message = _attempt_error(exc)
                attempts.append(
                    TechnicalAttempt(
                        attempt=attempt_number,
                        started_at=started_at,
                        elapsed_ms=round(
                            (time.monotonic() - started) * 1000
                        ),
                        status=status,
                        error_type=last_error_type,
                        error_message=last_error_message,
                    )
                )
    return [], returned_count, attempts, last_error_type, last_error_message


@tool
async def search_review_kb(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    focus_element_ids: list[str] | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """检索当前治疗方案审查所选的唯一 Milvus 知识库，并返回 Top-5 Evidence。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    context: MedicationReviewLiteContext = runtime.context
    ensure_runtime_resources(context)
    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    record_id = _record_id("SEARCH", tool_call_id)
    query = query_text.strip()
    purpose = reason.strip()
    focus = list(dict.fromkeys(focus_element_ids or []))
    unknown_focus = [
        value for value in focus if value not in _known_element_ids(state)
    ]
    warnings = [
        f"{record_id} 使用未知方案锚点：{value}" for value in unknown_focus
    ]
    started_at = utc_isoformat()
    started = time.monotonic()

    logger.info(
        "PAT-RAG search start: record_id=%s query_hash=%s",
        record_id,
        hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
    )
    try:
        selection = await resolve_milvus_retriever(context)
        chunks, returned_count, attempts, error_type, error_message = (
            await _retrieve(
                selection=selection,
                context=context,
                query_text=query,
            )
        )
    except Exception as exc:  # noqa: BLE001 - return failure to the Agent
        selection = None
        chunks = []
        returned_count = 0
        status, error_type, error_message = _attempt_error(exc)
        attempts = [
            TechnicalAttempt(
                attempt=1,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                status=status,
                error_type=error_type,
                error_message=error_message,
            )
        ]

    current = _state_evidence(state)
    delta: dict[str, EvidenceItem] = {}
    cards: list[str] = []
    evidence_ids: list[str] = []
    new_ids: list[str] = []
    if selection is not None:
        for rank, chunk in enumerate(chunks, start=1):
            item = _item_from_chunk(selection=selection, chunk=chunk)
            score = _numeric_value(chunk, "score")
            distance = _numeric_value(chunk, "distance")
            if profile_uses_query_centered_cards(
                context.experiment_profile
            ):
                anchor_focus: list[str] = []
                for raw in state.get("plan_anchors") or []:
                    element_id = (
                        raw.get("element_id")
                        if isinstance(raw, dict)
                        else getattr(raw, "element_id", None)
                    )
                    if element_id not in focus:
                        continue
                    label = (
                        raw.get("label")
                        if isinstance(raw, dict)
                        else getattr(raw, "label", "")
                    )
                    source_span = (
                        raw.get("source_span")
                        if isinstance(raw, dict)
                        else getattr(raw, "source_span", "")
                    )
                    anchor_focus.extend([str(label or ""), str(source_span or "")])
                excerpt = build_query_centered_excerpt(
                    raw_text=item.raw_text,
                    focus_text=" ".join(
                        [query, purpose, *anchor_focus]
                    ),
                    target_chars=context.evidence_excerpt_chars,
                )
            else:
                excerpt = _full_chunk_excerpt(item.raw_text)
            occurrence = EvidenceOccurrence(
                record_id=record_id,
                tool_call_id=tool_call_id,
                source_method="search",
                query_text=query,
                reason=purpose,
                focus_element_ids=focus,
                rank=rank,
                score=score,
                distance=distance,
                shown_excerpt=excerpt.text,
                excerpt_start=excerpt.start,
                excerpt_end=excerpt.end,
                excerpt_fallback=excerpt.fallback,
            )
            item = item.model_copy(update={"occurrences": [occurrence]})
            delta[item.evidence_id] = item
            evidence_ids.append(item.evidence_id)
            if item.evidence_id not in current:
                new_ids.append(item.evidence_id)
            cards.append(
                format_evidence_card(
                    item=item,
                    excerpt=excerpt,
                    rank=rank,
                    score=score,
                    distance=distance,
                )
            )

    if chunks:
        final_status = "success"
        tool_content = (
            f"查询：{query}\n目的：{purpose}\n\n"
            + "\n\n---\n\n".join(cards)
        )
    elif attempts and attempts[-1].status == "success_empty":
        final_status = "success_empty"
        tool_content = (
            f"查询“{query}”成功执行，但没有返回可用片段。"
            "请改写查询或使用已有证据回答。"
        )
    else:
        final_status = "technical_failed"
        tool_content = (
            f"查询“{query}”因技术问题失败："
            f"{error_type or 'unknown'}: {error_message or '无详细信息'}。"
            "你可以改写查询重试，或基于已有证据回答。"
        )

    record = SearchRecord(
        record_id=record_id,
        tool_call_id=tool_call_id,
        query_text=query,
        reason=purpose,
        focus_element_ids=focus,
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        status=final_status,
        returned_count=returned_count,
        retained_count=len(evidence_ids),
        evidence_ids=evidence_ids,
        new_evidence_ids=new_ids,
        attempts=attempts,
        error_type=error_type,
        error_message=error_message,
    )
    logger.info(
        "PAT-RAG search complete: record_id=%s status=%s retained=%s",
        record_id,
        final_status,
        len(evidence_ids),
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=tool_content,
                    tool_call_id=tool_call_id,
                    name="search_review_kb",
                )
            ],
            "evidence_store": delta,
            "search_records": [record],
            "search_count": 1,
            "technical_attempts": len(attempts),
            "knowledge_base_snapshot": (
                selection.snapshot if selection is not None else {}
            ),
            "warnings": warnings,
        }
    )


async def _open_document_window(
    *,
    selection: RetrieverSelection,
    context: MedicationReviewLiteContext,
    parent: EvidenceItem,
    window_before: int,
    window_after: int,
) -> tuple[list[dict[str, Any]], list[TechnicalAttempt], str | None, str | None]:
    attempts: list[TechnicalAttempt] = []
    last_error_type: str | None = None
    last_error_message: str | None = None
    target_index = int(parent.chunk_index)

    for attempt_number in range(1, context.technical_retry_limit + 2):
        started_at = utc_isoformat()
        started = time.monotonic()
        try:
            async with asyncio.timeout(context.retrieval_timeout_seconds):
                content_info = await knowledge_base.get_file_content(
                    selection.db_id,
                    parent.file_id,
                )
            lines = (
                content_info.get("lines")
                if isinstance(content_info, dict)
                else []
            )
            if not isinstance(lines, list):
                lines = []
            lower = max(target_index - window_before, 0)
            upper = target_index + window_after
            chunks: list[dict[str, Any]] = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                try:
                    chunk_index = int(
                        line.get("chunk_order_index", -1)
                    )
                except (TypeError, ValueError):
                    continue
                if not lower <= chunk_index <= upper:
                    continue
                chunks.append(
                    {
                        "content": str(line.get("content") or ""),
                        "metadata": {
                            "source": parent.source_document,
                            "file_id": parent.file_id,
                            "chunk_id": str(
                                line.get("id")
                                or f"{parent.file_id}_chunk_{chunk_index}"
                            ),
                            "chunk_index": chunk_index,
                        },
                    }
                )
            attempts.append(
                TechnicalAttempt(
                    attempt=attempt_number,
                    started_at=started_at,
                    elapsed_ms=round(
                        (time.monotonic() - started) * 1000
                    ),
                    status="success" if chunks else "success_empty",
                    returned_count=len(chunks),
                )
            )
            return chunks, attempts, None, None
        except Exception as exc:  # noqa: BLE001 - backend adapters vary
            status, last_error_type, last_error_message = _attempt_error(exc)
            attempts.append(
                TechnicalAttempt(
                    attempt=attempt_number,
                    started_at=started_at,
                    elapsed_ms=round(
                        (time.monotonic() - started) * 1000
                    ),
                    status=status,
                    error_type=last_error_type,
                    error_message=last_error_message,
                )
            )
    return [], attempts, last_error_type, last_error_message


@tool
async def open_review_evidence(
    evidence_id: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    window_before: Annotated[int, Field(ge=0, le=3)] = 1,
    window_after: Annotated[int, Field(ge=0, le=3)] = 1,
    runtime: ToolRuntime = None,
) -> Command:
    """按 Evidence ID 打开同一知识库文档的相邻原文片段。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("open_review_evidence 缺少 ToolRuntime")
    context: MedicationReviewLiteContext = runtime.context
    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    record_id = _record_id("OPEN", tool_call_id)
    started_at = utc_isoformat()
    started = time.monotonic()
    normalized_id = evidence_id.strip().upper()
    current = _state_evidence(state)
    parent = current.get(normalized_id)
    attempts: list[TechnicalAttempt] = []
    error_type: str | None = None
    error_message: str | None = None
    selection: RetrieverSelection | None = None
    chunks: list[dict[str, Any]] = []

    if parent is None:
        final_status = "invalid_source"
        error_type = "unknown_evidence_id"
        error_message = f"当前运行不存在 Evidence ID：{normalized_id}"
    elif not parent.file_id or parent.chunk_index is None:
        final_status = "invalid_source"
        error_type = "missing_source_location"
        error_message = "Evidence 缺少 file_id 或 chunk_index"
    else:
        try:
            selection = await resolve_milvus_retriever(context)
            (
                chunks,
                attempts,
                error_type,
                error_message,
            ) = await _open_document_window(
                selection=selection,
                context=context,
                parent=parent,
                window_before=window_before,
                window_after=window_after,
            )
            if chunks:
                final_status = "success"
            elif attempts and attempts[-1].status == "success_empty":
                final_status = "success_empty"
            else:
                final_status = "technical_failed"
        except Exception as exc:  # noqa: BLE001 - return failure to Agent
            final_status = "technical_failed"
            status, error_type, error_message = _attempt_error(exc)
            attempts = [
                TechnicalAttempt(
                    attempt=1,
                    started_at=started_at,
                    elapsed_ms=round(
                        (time.monotonic() - started) * 1000
                    ),
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                )
            ]

    delta: dict[str, EvidenceItem] = {}
    evidence_ids: list[str] = []
    new_ids: list[str] = []
    cards: list[str] = []
    if selection is not None and parent is not None:
        inherited_queries = " ".join(
            occurrence.query_text for occurrence in parent.occurrences
        )
        for rank, chunk in enumerate(chunks, start=1):
            item = _item_from_chunk(selection=selection, chunk=chunk)
            if profile_uses_query_centered_cards(
                context.experiment_profile
            ):
                excerpt = build_query_centered_excerpt(
                    raw_text=item.raw_text,
                    focus_text=f"{reason} {inherited_queries}",
                    target_chars=context.evidence_excerpt_chars,
                )
            else:
                excerpt = _full_chunk_excerpt(item.raw_text)
            occurrence = EvidenceOccurrence(
                record_id=record_id,
                tool_call_id=tool_call_id,
                source_method="open",
                query_text=inherited_queries,
                reason=reason.strip(),
                shown_excerpt=excerpt.text,
                excerpt_start=excerpt.start,
                excerpt_end=excerpt.end,
                excerpt_fallback=excerpt.fallback,
                parent_evidence_id=normalized_id,
                rank=rank,
            )
            item = item.model_copy(update={"occurrences": [occurrence]})
            delta[item.evidence_id] = item
            evidence_ids.append(item.evidence_id)
            if item.evidence_id not in current:
                new_ids.append(item.evidence_id)
            cards.append(
                format_evidence_card(
                    item=item,
                    excerpt=excerpt,
                    rank=rank,
                    score=None,
                    distance=None,
                )
            )

    if final_status == "success":
        tool_content = (
            f"已打开 [{normalized_id}] 的相邻原文：\n\n"
            + "\n\n---\n\n".join(cards)
        )
    elif final_status == "success_empty":
        tool_content = (
            f"[{normalized_id}] 的相邻原文窗口为空。"
            "请使用已有证据回答或执行其它检索。"
        )
    else:
        tool_content = (
            f"无法打开 [{normalized_id}]："
            f"{error_type or 'unknown'}: {error_message or '无详细信息'}。"
            "该错误不妨碍你使用已有证据回答。"
        )

    record = OpenRecord(
        record_id=record_id,
        tool_call_id=tool_call_id,
        parent_evidence_id=normalized_id,
        reason=reason.strip(),
        window_before=window_before,
        window_after=window_after,
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        status=final_status,
        evidence_ids=evidence_ids,
        new_evidence_ids=new_ids,
        attempts=attempts,
        error_type=error_type,
        error_message=error_message,
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=tool_content,
                    tool_call_id=tool_call_id,
                    name="open_review_evidence",
                )
            ],
            "evidence_store": delta,
            "open_records": [record],
            "open_count": 1,
            "technical_attempts": len(attempts),
            "knowledge_base_snapshot": (
                selection.snapshot if selection is not None else {}
            ),
        }
    )
