from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import Field

from yuxi import knowledge_base
from yuxi.agents.backends.knowledge_base_backend import (
    resolve_visible_knowledge_bases_for_context,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    build_query_centered_excerpt,
    format_evidence_card,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    EvidenceItem,
    EvidenceOccurrence,
    OpenRecord,
    TechnicalAttempt,
)
from yuxi.agents.buildin.medication_review_lite.tools import (
    _attempt_error,
    _item_from_chunk,
    _numeric_value,
    _open_document_window,
    _record_id,
    _state_evidence,
)
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .context import MedicationReviewPrimContext, validate_context_values
from .models import (
    DeferredKnowledgeCall,
    ExperimentProfile,
    InvestigationItem,
    InvestigationStatus,
    QueryRecord,
    RetrievalScope,
)

Retriever = Callable[..., Awaitable[list[dict[str, Any]]]]
RetrievalStrategy = Callable[["RetrievalRequest"], Awaitable["RetrievalOutcome"]]
RETRIEVAL_TOP_K = 10
KNOWLEDGE_TOOL_NAMES = {"search_review_kb", "open_review_evidence"}


class MedicationReviewPrimConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RetrieverSelection:
    db_id: str
    retriever: Retriever
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class RetrievalRequest:
    selection: RetrieverSelection
    context: MedicationReviewPrimContext
    query_id: str
    query_text: str
    state: dict[str, Any]
    retrieval_scope: RetrievalScope = "global"
    file_id: str | None = None
    focus_plan_ids: list[str] = field(default_factory=list)
    focus_modifier_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalOutcome:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    returned_count: int = 0
    attempts: list[TechnicalAttempt] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    diagnostic_record: Any | None = None
    diagnostic_state_key: str = "routed_retrieval_records"


def _retrieval_depth(context: MedicationReviewPrimContext) -> tuple[int, int]:
    """Return backend fetch depth and Agent-visible depth.

    The private context attributes are set only by experimental wrappers.  The
    default remains the historical PRIM Top-10 behavior.
    """
    fetch_k = int(getattr(context, "_prim_retrieval_fetch_k", RETRIEVAL_TOP_K))
    visible_k = int(getattr(context, "_prim_retrieval_visible_k", fetch_k))
    if fetch_k < 1 or visible_k < 1 or visible_k > fetch_k:
        raise MedicationReviewPrimConfigError(
            "检索深度配置无效：必须满足 1 <= visible_k <= fetch_k"
        )
    return fetch_k, visible_k


def _diagnostic_candidate(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    content = str(
        chunk.get("content")
        or chunk.get("text")
        or metadata.get("content")
        or ""
    )
    raw_chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id")
    raw_chunk_index = (
        metadata.get("chunk_index")
        if metadata.get("chunk_index") is not None
        else chunk.get("chunk_index")
    )
    return {
        "rank": rank,
        "file_id": str(metadata.get("file_id") or chunk.get("file_id") or "")
        or None,
        "source_document": str(
            metadata.get("source")
            or metadata.get("file_name")
            or chunk.get("source")
            or ""
        )
        or None,
        "chunk_id": (
            raw_chunk_id
            if isinstance(raw_chunk_id, (str, int))
            else str(raw_chunk_id)
            if raw_chunk_id is not None
            else None
        ),
        "chunk_index": (
            raw_chunk_index
            if isinstance(raw_chunk_index, (str, int))
            else str(raw_chunk_index)
            if raw_chunk_index is not None
            else None
        ),
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "score": _numeric_value(chunk, "score"),
        "distance": _numeric_value(chunk, "distance"),
    }


def _embedding_model_id(metadata: dict[str, Any]) -> str | None:
    embed_info = metadata.get("embed_info")
    if not isinstance(embed_info, dict):
        return None
    value = embed_info.get("model_id") or embed_info.get("model")
    return str(value) if value else None


def _file_ids_from_database_info(database_info: Any) -> set[str]:
    if not isinstance(database_info, dict):
        return set()
    files = database_info.get("files")
    if isinstance(files, dict):
        values = [*files.keys()]
        values.extend(
            value.get("file_id") or value.get("id")
            for value in files.values()
            if isinstance(value, dict)
        )
    elif isinstance(files, list):
        values = [
            value.get("file_id") or value.get("id")
            for value in files
            if isinstance(value, dict)
        ]
    else:
        values = []
    return {str(value).strip() for value in values if str(value or "").strip()}


async def resolve_milvus_retriever(
    context: MedicationReviewPrimContext,
) -> RetrieverSelection:
    validate_context_values(context)
    fetch_k, visible_k = _retrieval_depth(context)
    cached = getattr(context, "_prim_retriever_selection", None)
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
        raise MedicationReviewPrimConfigError(
            f"知识库 {selected_name!r} 在当前用户可见范围内不存在或名称不唯一"
        )

    database = matches[0]
    db_id = str(database.get("db_id") or "").strip()
    kb_type = str(database.get("kb_type") or "").strip().lower()
    if not db_id:
        raise MedicationReviewPrimConfigError("选定知识库缺少 db_id")
    if kb_type != "milvus":
        raise MedicationReviewPrimConfigError(
            f"PRIM-RAG 只支持 Milvus，当前类型为 {kb_type or 'unknown'}"
        )

    retriever_info = knowledge_base.get_retrievers().get(db_id)
    if not isinstance(retriever_info, dict) or not callable(
        retriever_info.get("retriever")
    ):
        raise MedicationReviewPrimConfigError(
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
                "final_top_k": fetch_k,
                "similarity_threshold": float(
                    persisted.get("similarity_threshold", 0.2)
                ),
                "include_distances": True,
                "use_reranker": False,
                "metric_type": "COSINE",
                "use_async_embedding": True,
                "raise_on_error": True,
                "supports_document_filter": True,
            },
        },
    )
    if visible_k != fetch_k:
        selection.snapshot["query_params"]["agent_visible_top_k"] = visible_k
    setattr(context, "_prim_retriever_selection", selection)
    try:
        database_info = await knowledge_base.get_database_info(db_id)
        setattr(
            context,
            "_prim_allowed_file_ids",
            _file_ids_from_database_info(database_info),
        )
    except Exception as exc:  # noqa: BLE001 - global search remains usable
        logger.warning(
            "PRIM-RAG could not cache database file ids: db_id=%s error=%s",
            db_id,
            exc,
        )
        setattr(context, "_prim_allowed_file_ids", None)
    return selection


def ensure_runtime_resources(context: MedicationReviewPrimContext) -> None:
    if getattr(context, "_prim_search_lock", None) is None:
        setattr(context, "_prim_search_lock", asyncio.Lock())


def _stable_id(prefix: str, source: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


def _known_ids(state: dict[str, Any]) -> tuple[set[str], set[str]]:
    plan_ids = {
        str(
            raw.get("element_id")
            if isinstance(raw, dict)
            else getattr(raw, "element_id", "")
        )
        for raw in state.get("plan_anchors") or []
    }
    modifier_ids = {
        str(
            raw.get("modifier_id")
            if isinstance(raw, dict)
            else getattr(raw, "modifier_id", "")
        )
        for raw in state.get("patient_modifiers") or []
    }
    return plan_ids - {""}, modifier_ids - {""}


def _investigation_map(
    state: dict[str, Any],
) -> dict[str, InvestigationItem]:
    result: dict[str, InvestigationItem] = {}
    for raw in state.get("investigations") or []:
        try:
            value = (
                raw
                if isinstance(raw, InvestigationItem)
                else InvestigationItem.model_validate(raw)
            )
        except Exception:  # noqa: BLE001 - ignore malformed prior state
            continue
        result[value.investigation_id] = value
    return result


def _tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    return calls if isinstance(calls, list) else []


def _knowledge_call_ids_for_current_turn(
    state: dict[str, Any],
    current_tool_call_id: str,
) -> list[str]:
    for message in reversed(state.get("messages") or []):
        calls = _tool_calls(message)
        if not calls:
            continue
        call_ids = {
            str(call.get("id") or call.get("tool_call_id") or "")
            for call in calls
            if isinstance(call, dict)
        }
        if current_tool_call_id not in call_ids:
            continue
        return [
            str(call.get("id") or call.get("tool_call_id") or "")
            for call in calls
            if isinstance(call, dict)
            and str(call.get("name") or "") in KNOWLEDGE_TOOL_NAMES
            and str(call.get("id") or call.get("tool_call_id") or "")
        ]
    return []


def _deferred_command(
    *,
    tool_call_id: str,
    tool_name: str,
    reason: str,
) -> Command:
    if reason == "same_model_turn":
        content = (
            "本轮已有一个知识库读取正在执行，因此本次调用未执行。"
            "请先阅读该结果，再在下一轮决定是否继续检索或打开原文。"
        )
    else:
        content = (
            "该知识库工具的执行预算已经用完，本次调用未执行。"
            "请使用已有证据完成审查，并明确证据不足的部分。"
        )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            ],
            "deferred_knowledge_calls": [
                DeferredKnowledgeCall(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    reason=reason,
                    created_at=utc_isoformat(),
                )
            ],
        }
    )


def _knowledge_call_guard(
    *,
    runtime: ToolRuntime,
    tool_name: str,
    counter_key: str,
    limit: int,
) -> Command | None:
    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    turn_call_ids = _knowledge_call_ids_for_current_turn(
        state,
        tool_call_id,
    )
    if turn_call_ids and tool_call_id != turn_call_ids[0]:
        return _deferred_command(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason="same_model_turn",
        )
    if int(state.get(counter_key) or 0) >= limit:
        return _deferred_command(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason="budget_exhausted",
        )
    return None


async def _known_database_file_ids(
    *,
    context: MedicationReviewPrimContext,
    selection: RetrieverSelection,
    state: dict[str, Any],
    requested_file_id: str | None = None,
) -> set[str]:
    known = {
        value.file_id
        for value in _state_evidence(state).values()
        if value.file_id
    }
    # A file_id returned by this run's retriever is already grounded in the
    # selected database. Do not make document-local follow-up depend on a
    # second metadata lookup that may be temporarily unavailable.
    if requested_file_id and requested_file_id in known:
        return known
    cached = getattr(context, "_prim_allowed_file_ids", None)
    if isinstance(cached, set):
        return known | {str(value) for value in cached}
    try:
        database_info = await knowledge_base.get_database_info(selection.db_id)
    except Exception as exc:  # noqa: BLE001
        raise MedicationReviewPrimConfigError(
            "无法读取知识库文件目录，暂不能执行文档内检索"
        ) from exc
    allowed = _file_ids_from_database_info(database_info)
    setattr(context, "_prim_allowed_file_ids", allowed)
    return known | allowed


async def _retrieve(request: RetrievalRequest) -> RetrievalOutcome:
    selection = request.selection
    context = request.context
    fetch_k, visible_k = _retrieval_depth(context)
    attempts: list[TechnicalAttempt] = []
    returned_count = 0
    last_error_type: str | None = None
    last_error_message: str | None = None

    retriever_kwargs: dict[str, Any] = {
        "search_mode": "vector",
        "final_top_k": fetch_k,
        "use_reranker": False,
        "include_distances": True,
        "raise_on_error": True,
        "use_async_embedding": True,
    }
    if request.retrieval_scope == "document":
        if not request.file_id:
            raise ValueError("文档内检索必须提供 file_id")
        allowed = await _known_database_file_ids(
            context=context,
            selection=selection,
            state=request.state,
            requested_file_id=request.file_id,
        )
        if request.file_id not in allowed:
            raise ValueError(
                f"file_id {request.file_id!r} 不属于当前知识库，"
                "也不是本轮检索返回的文档"
            )
        retriever_kwargs["filter_file_ids"] = [request.file_id]

    async with getattr(context, "_prim_search_lock"):
        for attempt_number in range(1, context.technical_retry_limit + 2):
            started_at = utc_isoformat()
            started = time.monotonic()
            try:
                async with asyncio.timeout(context.retrieval_timeout_seconds):
                    result = await selection.retriever(
                        request.query_text,
                        **retriever_kwargs,
                    )
                if not isinstance(result, list):
                    raise TypeError(
                        "Retriever 返回类型不是 list："
                        f"{type(result).__name__}"
                    )
                returned_count = len(result)
                fetched = [
                    item
                    for item in result[:fetch_k]
                    if isinstance(item, dict)
                ]
                retained = fetched[:visible_k]
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
                diagnostic_record = None
                diagnostic_state_key = "routed_retrieval_records"
                configured_key = getattr(
                    context,
                    "_prim_retrieval_diagnostic_state_key",
                    None,
                )
                if isinstance(configured_key, str) and configured_key.strip():
                    diagnostic_state_key = configured_key.strip()
                    diagnostic_record = {
                        "record_id": f"VR-{request.query_id}",
                        "query_id": request.query_id,
                        "fetch_k": fetch_k,
                        "visible_k": visible_k,
                        "returned_count": returned_count,
                        "candidates": [
                            _diagnostic_candidate(item, rank)
                            for rank, item in enumerate(fetched, start=1)
                        ],
                    }
                return RetrievalOutcome(
                    chunks=retained,
                    returned_count=returned_count,
                    attempts=attempts,
                    diagnostic_record=diagnostic_record,
                    diagnostic_state_key=diagnostic_state_key,
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
    return RetrievalOutcome(
        returned_count=returned_count,
        attempts=attempts,
        error_type=last_error_type,
        error_message=last_error_message,
    )


def _focus_text(
    state: dict[str, Any],
    *,
    query: str,
    reason: str,
    investigation_question: str | None,
    focus_plan_ids: list[str],
    focus_modifier_ids: list[str],
) -> str:
    values = [query, reason, investigation_question or ""]
    for raw in state.get("plan_anchors") or []:
        element_id = (
            raw.get("element_id")
            if isinstance(raw, dict)
            else getattr(raw, "element_id", None)
        )
        if element_id not in focus_plan_ids:
            continue
        values.extend(
            [
                str(
                    raw.get("label")
                    if isinstance(raw, dict)
                    else getattr(raw, "label", "")
                ),
                str(
                    raw.get("source_span")
                    if isinstance(raw, dict)
                    else getattr(raw, "source_span", "")
                ),
            ]
        )
    for raw in state.get("patient_modifiers") or []:
        modifier_id = (
            raw.get("modifier_id")
            if isinstance(raw, dict)
            else getattr(raw, "modifier_id", None)
        )
        if modifier_id in focus_modifier_ids:
            values.append(
                str(
                    raw.get("source_span")
                    if isinstance(raw, dict)
                    else getattr(raw, "source_span", "")
                )
            )
    return " ".join(value for value in values if value)


def _prepare_investigation(
    state: dict[str, Any],
    *,
    tool_call_id: str,
    requested_investigation_id: str | None,
    question: str | None,
    focus_plan_ids: list[str],
    focus_modifier_ids: list[str],
    atlas_companion_ids: list[str],
    suggested_file_ids: list[str],
    origin: str,
    warnings: list[str],
) -> InvestigationItem | None:
    requested_id = (requested_investigation_id or "").strip().upper()
    normalized_question = (question or "").strip()
    existing = _investigation_map(state).get(requested_id)
    now = utc_isoformat()
    if existing is not None:
        if normalized_question and normalized_question != existing.question:
            warnings.append(
                f"{existing.investigation_id} 已存在；本次沿用原调查问题"
            )
        return existing.model_copy(
            update={
                "focus_plan_ids": list(
                    dict.fromkeys(
                        [*existing.focus_plan_ids, *focus_plan_ids]
                    )
                ),
                "focus_modifier_ids": list(
                    dict.fromkeys(
                        [
                            *existing.focus_modifier_ids,
                            *focus_modifier_ids,
                        ]
                    )
                ),
                "atlas_companion_ids": list(
                    dict.fromkeys(
                        [
                            *existing.atlas_companion_ids,
                            *atlas_companion_ids,
                        ]
                    )
                ),
                "candidate_file_ids": list(
                    dict.fromkeys(
                        [
                            *existing.candidate_file_ids,
                            *suggested_file_ids,
                        ]
                    )
                ),
                "status": "open",
                "updated_at": now,
            }
        )

    if requested_id:
        warnings.append(
            f"未知 investigation_id：{requested_id}；"
            "如提供了 question，将创建新的调查"
        )
    if not normalized_question:
        return None
    return InvestigationItem(
        investigation_id=_stable_id("INV", tool_call_id),
        question=normalized_question,
        origin="atlas" if origin == "atlas" else "agent",
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        atlas_companion_ids=atlas_companion_ids,
        candidate_file_ids=suggested_file_ids,
        status="open",
        created_at=now,
        updated_at=now,
        warnings=warnings,
    )


async def _search_review_kb_impl(
    *,
    query_text: str,
    reason: str,
    runtime: ToolRuntime,
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    question: str | None = None,
    investigation_id: str | None = None,
    atlas_companion_ids: list[str] | None = None,
    atlas_suggested_file_ids: list[str] | None = None,
    investigation_origin: str = "agent",
    retrieval_strategy: RetrievalStrategy | None = None,
    # Historical DA-PRIM callers use these names. They map to the new
    # investigation protocol but are not exposed by the PRIM-v2 tool schema.
    relation_question: str | None = None,
    relation_id: str | None = None,
) -> Command:
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    context: MedicationReviewPrimContext = runtime.context
    ensure_runtime_resources(context)
    guard = _knowledge_call_guard(
        runtime=runtime,
        tool_name="search_review_kb",
        counter_key="search_count",
        limit=context.max_search_calls,
    )
    if guard is not None:
        return guard

    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    query_id = _stable_id("Q", tool_call_id)
    query = query_text.strip()
    purpose = reason.strip()
    scope: RetrievalScope = retrieval_scope
    normalized_file_id = (file_id or "").strip() or None
    if scope == "global" and normalized_file_id:
        normalized_file_id = None

    requested_plan = list(dict.fromkeys(focus_plan_ids or []))
    requested_modifier = list(dict.fromkeys(focus_modifier_ids or []))
    known_plan, known_modifier = _known_ids(state)
    invalid_focus = [
        *[value for value in requested_plan if value not in known_plan],
        *[value for value in requested_modifier if value not in known_modifier],
    ]
    focus_plan = [value for value in requested_plan if value in known_plan]
    focus_modifier = [
        value for value in requested_modifier if value in known_modifier
    ]
    warnings = [
        f"{query_id} 忽略未知病例节点：{value}"
        for value in invalid_focus
    ]
    if retrieval_scope == "global" and file_id:
        warnings.append(
            f"{query_id} 的 global 检索忽略了多余的 file_id"
        )
    investigation = _prepare_investigation(
        state,
        tool_call_id=tool_call_id,
        requested_investigation_id=investigation_id or relation_id,
        question=question or relation_question,
        focus_plan_ids=focus_plan,
        focus_modifier_ids=focus_modifier,
        atlas_companion_ids=list(dict.fromkeys(atlas_companion_ids or [])),
        suggested_file_ids=list(
            dict.fromkeys(atlas_suggested_file_ids or [])
        ),
        origin=investigation_origin,
        warnings=warnings,
    )
    started_at = utc_isoformat()
    started = time.monotonic()

    logger.info(
        "PRIM-RAG search start: query_id=%s investigation_id=%s "
        "scope=%s file_id=%s query_hash=%s",
        query_id,
        investigation.investigation_id if investigation else None,
        scope,
        normalized_file_id,
        hashlib.sha256(query.encode("utf-8")).hexdigest()[:12],
    )
    outcome: RetrievalOutcome | None = None
    try:
        selection = await resolve_milvus_retriever(context)
        outcome = await (retrieval_strategy or _retrieve)(
            RetrievalRequest(
                selection=selection,
                context=context,
                query_id=query_id,
                query_text=query,
                state=state,
                retrieval_scope=scope,
                file_id=normalized_file_id,
                focus_plan_ids=focus_plan,
                focus_modifier_ids=focus_modifier,
            )
        )
        chunks = outcome.chunks
        returned_count = outcome.returned_count
        attempts = outcome.attempts
        error_type = outcome.error_type
        error_message = outcome.error_message
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
    candidate_file_ids: list[str] = []
    if selection is not None:
        focus_text = _focus_text(
            state,
            query=query,
            reason=purpose,
            investigation_question=(
                investigation.question if investigation else None
            ),
            focus_plan_ids=focus_plan,
            focus_modifier_ids=focus_modifier,
        )
        for rank, chunk in enumerate(chunks, start=1):
            item = _item_from_chunk(selection=selection, chunk=chunk)
            excerpt = build_query_centered_excerpt(
                raw_text=item.raw_text,
                focus_text=focus_text,
                target_chars=context.evidence_excerpt_chars,
            )
            occurrence = EvidenceOccurrence(
                record_id=query_id,
                tool_call_id=tool_call_id,
                source_method="search",
                query_text=query,
                reason=purpose,
                focus_element_ids=focus_plan,
                rank=rank,
                score=_numeric_value(chunk, "score"),
                distance=_numeric_value(chunk, "distance"),
                shown_excerpt=excerpt.text,
                excerpt_start=excerpt.start,
                excerpt_end=excerpt.end,
                excerpt_fallback=excerpt.fallback,
            )
            item = item.model_copy(update={"occurrences": [occurrence]})
            delta[item.evidence_id] = item
            evidence_ids.append(item.evidence_id)
            if item.file_id:
                candidate_file_ids.append(item.file_id)
            if item.evidence_id not in current:
                new_ids.append(item.evidence_id)
            cards.append(
                format_evidence_card(
                    item=item,
                    excerpt=excerpt,
                    rank=rank,
                    score=occurrence.score,
                    distance=occurrence.distance,
                    include_file_id=True,
                )
            )

    scope_label = (
        f"文档内（file_id={normalized_file_id}）"
        if scope == "document"
        else "全库"
    )
    if chunks:
        final_status = "success"
        tool_content = (
            f"检索范围：{scope_label}\n查询：{query}\n目的：{purpose}\n\n"
            + "\n\n---\n\n".join(cards)
        )
    elif attempts and attempts[-1].status == "success_empty":
        final_status = "success_empty"
        tool_content = (
            f"{scope_label}查询“{query}”成功执行，但没有返回可用片段。"
            "可根据缺口改写短查询、切换全库/文档内范围，或使用已有证据回答。"
        )
    else:
        final_status = "technical_failed"
        tool_content = (
            f"{scope_label}查询“{query}”因技术问题失败："
            f"{error_type or 'unknown'}: {error_message or '无详细信息'}。"
            "可改写查询重试，或基于已有证据回答。"
        )

    investigation_delta: list[InvestigationItem] = []
    if investigation is not None:
        investigation = investigation.model_copy(
            update={
                "status": "open",
                "query_ids": list(
                    dict.fromkeys([*investigation.query_ids, query_id])
                ),
                "candidate_evidence_ids": list(
                    dict.fromkeys(
                        [
                            *investigation.candidate_evidence_ids,
                            *evidence_ids,
                        ]
                    )
                ),
                "candidate_file_ids": list(
                    dict.fromkeys(
                        [
                            *investigation.candidate_file_ids,
                            *candidate_file_ids,
                        ]
                    )
                ),
                "updated_at": utc_isoformat(),
                "warnings": list(
                    dict.fromkeys([*investigation.warnings, *warnings])
                ),
            }
        )
        investigation_delta.append(investigation)
        tool_content = (
            f"调查 [{investigation.investigation_id}]："
            f"{investigation.question}\n{tool_content}\n\n"
            "这些片段只是候选证据。请判断是否真正回答了调查问题；"
            "必要时继续检索或打开原文，随后调用 update_investigation。"
        )

    record = QueryRecord(
        query_id=query_id,
        tool_call_id=tool_call_id,
        investigation_id=(
            investigation.investigation_id if investigation else None
        ),
        query_text=query,
        reason=purpose,
        retrieval_scope=scope,
        file_id=normalized_file_id,
        focus_plan_ids=focus_plan,
        focus_modifier_ids=focus_modifier,
        atlas_companion_ids=list(dict.fromkeys(atlas_companion_ids or [])),
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        status=final_status,
        returned_count=returned_count,
        retained_count=len(evidence_ids),
        evidence_ids=evidence_ids,
        new_evidence_ids=new_ids,
        attempts=attempts,
        invalid_focus_ids=invalid_focus,
        error_type=error_type,
        error_message=error_message,
    )
    logger.info(
        "PRIM-RAG search complete: query_id=%s status=%s retained=%s",
        query_id,
        final_status,
        len(evidence_ids),
    )
    update: dict[str, Any] = {
        "messages": [
            ToolMessage(
                content=tool_content,
                tool_call_id=tool_call_id,
                name="search_review_kb",
            )
        ],
        "evidence_store": delta,
        "query_records": [record],
        "investigations": investigation_delta,
        "search_count": 1,
        "technical_attempts": len(attempts),
        "knowledge_base_snapshot": (
            selection.snapshot if selection is not None else {}
        ),
        "warnings": warnings,
    }
    if outcome is not None and outcome.diagnostic_record is not None:
        update[outcome.diagnostic_state_key] = [outcome.diagnostic_record]
    return Command(update=update)


@tool("search_review_kb")
async def search_review_kb_b1(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """用一个短查询检索 Milvus Top-10；可全库检索或按已知 file_id 限定文档。"""
    return await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        runtime=runtime,
    )


@tool("search_review_kb")
async def search_review_kb_m1(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    focus_plan_ids: list[str] | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """围绕方案要素用短查询检索 Milvus Top-10，可限定到一个已知文档。"""
    return await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        focus_plan_ids=focus_plan_ids,
        runtime=runtime,
    )


@tool("search_review_kb")
async def search_review_kb_m2(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """围绕方案与患者事实用短查询检索 Milvus Top-10，可限定到一个已知文档。"""
    return await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        runtime=runtime,
    )


@tool("search_review_kb")
async def search_review_kb_investigation(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    question: Annotated[str, Field(min_length=1, max_length=500)] | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    investigation_id: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """创建或继续一个证据问题，并用短查询检索全库或指定文档的 Top-10。"""
    return await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        question=question,
        investigation_id=investigation_id,
        runtime=runtime,
    )


# Import compatibility for historical tests and DA-PRIM. The public schema and
# implementation are PRIM-v2 InvestigationItem semantics.
search_review_kb_relation = search_review_kb_investigation


def search_tool_for_profile(profile: ExperimentProfile) -> BaseTool:
    if profile == "b1":
        return search_review_kb_b1
    if profile == "m1":
        return search_review_kb_m1
    if profile == "m2":
        return search_review_kb_m2
    return search_review_kb_investigation


@tool
async def update_investigation(
    investigation_id: Annotated[str, Field(min_length=1)],
    status: InvestigationStatus,
    selected_evidence_ids: list[str] | None = None,
    working_note: Annotated[str, Field(max_length=800)] = "",
    runtime: ToolRuntime = None,
) -> Command:
    """根据已读证据更新调查状态；answered 时必须选择真实候选 Evidence ID。"""
    if runtime is None:
        raise RuntimeError("update_investigation 缺少 ToolRuntime")
    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    normalized_id = investigation_id.strip().upper()
    investigation = _investigation_map(state).get(normalized_id)
    if investigation is None:
        message = f"未知 Investigation ID：{normalized_id}，本次未更新。"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=message,
                        tool_call_id=tool_call_id,
                        name="update_investigation",
                    )
                ],
                "warnings": [message],
            }
        )

    selected = [
        value.strip().upper()
        for value in dict.fromkeys(
            selected_evidence_ids
            if selected_evidence_ids is not None
            else investigation.selected_evidence_ids
        )
        if value.strip()
    ]
    known_evidence = set(_state_evidence(state))
    invalid = [value for value in selected if value not in known_evidence]
    non_candidate = [
        value
        for value in selected
        if value not in investigation.candidate_evidence_ids
    ]
    errors: list[str] = []
    if invalid:
        errors.append("未知 Evidence ID：" + "、".join(invalid))
    if non_candidate:
        errors.append(
            "Evidence 不属于该调查的候选集合：" + "、".join(non_candidate)
        )
    if status == "answered" and not selected:
        errors.append("answered 状态必须至少选择一个候选 Evidence ID")
    if errors:
        message = "调查状态未更新：" + "；".join(errors)
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=message,
                        tool_call_id=tool_call_id,
                        name="update_investigation",
                    )
                ],
                "warnings": [f"{normalized_id} {message}"],
            }
        )

    updated = investigation.model_copy(
        update={
            "status": status,
            "selected_evidence_ids": selected,
            "working_note": working_note.strip(),
            "updated_at": utc_isoformat(),
        }
    )
    content = (
        f"调查 [{normalized_id}] 已更新为 {status}。"
        f"采用证据：{'、'.join(selected) if selected else '无'}。"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name="update_investigation",
                )
            ],
            "investigations": [updated],
        }
    )


@tool
async def open_review_evidence(
    evidence_id: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    investigation_id: str | None = None,
    window_before: Annotated[int, Field(ge=0, le=3)] = 1,
    window_after: Annotated[int, Field(ge=0, le=3)] = 1,
    runtime: ToolRuntime = None,
) -> Command:
    """按 Evidence ID 打开同一 Milvus 文档的相邻原文，并可关联到一个调查。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("open_review_evidence 缺少 ToolRuntime")
    context: MedicationReviewPrimContext = runtime.context
    guard = _knowledge_call_guard(
        runtime=runtime,
        tool_name="open_review_evidence",
        counter_key="open_count",
        limit=context.max_open_calls,
    )
    if guard is not None:
        return guard

    state = runtime.state if isinstance(runtime.state, dict) else {}
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    record_id = _record_id("OPEN", tool_call_id)
    started_at = utc_isoformat()
    started = time.monotonic()
    normalized_id = evidence_id.strip().upper()
    normalized_investigation_id = (
        (investigation_id or "").strip().upper() or None
    )
    investigations = _investigation_map(state)
    investigation = investigations.get(normalized_investigation_id or "")
    current = _state_evidence(state)
    parent = current.get(normalized_id)
    attempts: list[TechnicalAttempt] = []
    error_type: str | None = None
    error_message: str | None = None
    selection: RetrieverSelection | None = None
    chunks: list[dict[str, Any]] = []

    if normalized_investigation_id and investigation is None:
        final_status = "invalid_source"
        error_type = "unknown_investigation_id"
        error_message = (
            f"当前运行不存在 Investigation ID："
            f"{normalized_investigation_id}"
        )
    elif parent is None:
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
            chunks, attempts, error_type, error_message = (
                await _open_document_window(
                    selection=selection,
                    context=context,
                    parent=parent,
                    window_before=window_before,
                    window_after=window_after,
                )
            )
            if chunks:
                final_status = "success"
            elif attempts and attempts[-1].status == "success_empty":
                final_status = "success_empty"
            else:
                final_status = "technical_failed"
        except Exception as exc:  # noqa: BLE001 - return failure to Agent
            final_status = "technical_failed"
            attempt_status, error_type, error_message = _attempt_error(exc)
            attempts = [
                TechnicalAttempt(
                    attempt=1,
                    started_at=started_at,
                    elapsed_ms=round(
                        (time.monotonic() - started) * 1000
                    ),
                    status=attempt_status,
                    error_type=error_type,
                    error_message=error_message,
                )
            ]

    delta: dict[str, EvidenceItem] = {}
    evidence_ids: list[str] = []
    new_ids: list[str] = []
    candidate_file_ids: list[str] = []
    cards: list[str] = []
    if selection is not None and parent is not None:
        inherited_queries = " ".join(
            occurrence.query_text for occurrence in parent.occurrences
        )
        for rank, chunk in enumerate(chunks, start=1):
            item = _item_from_chunk(selection=selection, chunk=chunk)
            excerpt = build_query_centered_excerpt(
                raw_text=item.raw_text,
                focus_text=f"{reason} {inherited_queries}",
                target_chars=context.evidence_excerpt_chars,
            )
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
            if item.file_id:
                candidate_file_ids.append(item.file_id)
            if item.evidence_id not in current:
                new_ids.append(item.evidence_id)
            cards.append(
                format_evidence_card(
                    item=item,
                    excerpt=excerpt,
                    rank=rank,
                    score=None,
                    distance=None,
                    include_file_id=True,
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
            "该错误不妨碍使用已有证据回答。"
        )

    investigation_delta: list[InvestigationItem] = []
    if investigation is not None:
        investigation_delta.append(
            investigation.model_copy(
                update={
                    "status": "open",
                    "candidate_evidence_ids": list(
                        dict.fromkeys(
                            [
                                *investigation.candidate_evidence_ids,
                                *evidence_ids,
                            ]
                        )
                    ),
                    "candidate_file_ids": list(
                        dict.fromkeys(
                            [
                                *investigation.candidate_file_ids,
                                *candidate_file_ids,
                            ]
                        )
                    ),
                    "updated_at": utc_isoformat(),
                }
            )
        )
        tool_content += (
            f"\n\n结果已关联到调查 [{investigation.investigation_id}]；"
            "阅读后请调用 update_investigation。"
        )

    record = OpenRecord(
        record_id=record_id,
        tool_call_id=tool_call_id,
        parent_evidence_id=normalized_id,
        investigation_id=(
            investigation.investigation_id if investigation else None
        ),
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
            "investigations": investigation_delta,
            "open_records": [record],
            "open_count": 1,
            "technical_attempts": len(attempts),
            "knowledge_base_snapshot": (
                selection.snapshot if selection is not None else {}
            ),
        }
    )


@tool
async def coverage_reflection(
    runtime: ToolRuntime = None,
) -> Command:
    """内部软反思信号；该工具不会暴露给模型自主选择。"""
    if runtime is None:
        raise RuntimeError("coverage_reflection 缺少 ToolRuntime")
    state = runtime.state if isinstance(runtime.state, dict) else {}
    report = state.get("reflection_report")
    if hasattr(report, "open_investigation_ids_before"):
        open_ids = list(report.open_investigation_ids_before)
        plan_ids = list(report.uninvestigated_plan_ids_before)
    elif isinstance(report, dict):
        open_ids = list(report.get("open_investigation_ids_before") or [])
        plan_ids = list(report.get("uninvestigated_plan_ids_before") or [])
    else:
        open_ids = []
        plan_ids = []
    parts = [
        "请进行最后一次软反思。它不是临床结论校验，也不要求机械补齐。"
    ]
    if open_ids:
        parts.append("尚未关闭的调查：" + "、".join(open_ids) + "。")
    if plan_ids:
        parts.append(
            "尚未关联任何调查的方案要素：" + "、".join(plan_ids) + "。"
        )
    parts.append(
        "请判断这些缺口是否值得在剩余预算内继续检索；若不值得或证据不足，"
        "保留第一版有效内容并直接给出完整最终回答。"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="".join(parts),
                    tool_call_id=str(runtime.tool_call_id or uuid.uuid4()),
                    name="coverage_reflection",
                )
            ]
        }
    )
