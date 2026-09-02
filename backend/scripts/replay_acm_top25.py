"""Replay historical Agent search queries as Milvus vector Top-25.

The script never calls an LLM.  Gold annotations are not read or used during
retrieval.  Its JSONL output can be evaluated later with the existing document
or chunk retrieval evaluators.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Top25ReplayError(ValueError):
    pass


def _runtime_dependencies() -> tuple[Any, Any]:
    from yuxi import knowledge_base
    from yuxi.storage.postgres.manager import pg_manager

    return pg_manager, knowledge_base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-id")
    target.add_argument("--knowledge-name")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    return parser


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSON array, a single JSON object, or JSONL records."""
    with path.open("r", encoding="utf-8-sig") as stream:
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False
        array_mode: bool | None = None
        need_separator = False
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = stream.read(64 * 1024)
                buffer += chunk
                eof = not chunk
                continue
            if array_mode is None:
                if not buffer and eof:
                    return
                array_mode = buffer.startswith("[")
                if array_mode:
                    buffer = buffer[1:]
                continue
            if array_mode:
                buffer = buffer.lstrip()
                if need_separator:
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        need_separator = False
                        continue
                    if buffer.startswith("]"):
                        return
                    if not eof:
                        chunk = stream.read(64 * 1024)
                        buffer += chunk
                        eof = not chunk
                        continue
                    raise Top25ReplayError(f"{path} 的 JSON 数组缺少分隔符")
                if buffer.startswith("]"):
                    return
            if not buffer and eof:
                return
            try:
                value, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if not eof:
                    chunk = stream.read(64 * 1024)
                    buffer += chunk
                    eof = not chunk
                    continue
                raise Top25ReplayError(f"{path} 包含无效 JSON：{exc}") from exc
            if not isinstance(value, dict):
                raise Top25ReplayError(f"{path} 的记录必须是 JSON object")
            yield value
            buffer = buffer[offset:]
            need_separator = bool(array_mode)


def _query_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("medication_review_trace")
    sources = []
    if isinstance(trace, dict):
        sources = trace.get("query_records") or trace.get("search_records") or []
    if not sources:
        sources = record.get("search_records") or []
    result: list[dict[str, Any]] = []
    if isinstance(sources, list):
        for index, value in enumerate(sources, start=1):
            if not isinstance(value, dict):
                continue
            query = str(value.get("query_text") or "").strip()
            if not query:
                continue
            result.append(
                {
                    "query_id": str(value.get("query_id") or f"Q-{index:03d}"),
                    "query_text": query,
                    "reason": str(value.get("reason") or "历史查询 Top-25 回放"),
                    "retrieval_scope": str(
                        value.get("retrieval_scope") or "global"
                    ),
                    "file_id": value.get("file_id"),
                    "investigation_id": value.get("investigation_id"),
                }
            )
        if result:
            return result

    calls = record.get("retrieval_calls")
    if not isinstance(calls, list):
        return []
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("tool_name") or call.get("name") or "")
        if tool_name != "search_review_kb":
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        query = str(args.get("query_text") or call.get("query_text") or "").strip()
        if not query:
            continue
        result.append(
            {
                "query_id": str(call.get("query_id") or f"Q-{index:03d}"),
                "query_text": query,
                "reason": str(args.get("reason") or call.get("reason") or "历史查询 Top-25 回放"),
                "retrieval_scope": str(args.get("retrieval_scope") or "global"),
                "file_id": args.get("file_id"),
                "investigation_id": args.get("investigation_id"),
            }
        )
    return result


def _candidate(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    content = str(chunk.get("content") or chunk.get("text") or "")
    raw_chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id")
    raw_chunk_index = (
        metadata.get("chunk_index")
        if metadata.get("chunk_index") is not None
        else chunk.get("chunk_index")
    )

    def scalar(value: Any) -> str | int | float | None:
        if value is None or isinstance(value, (str, int, float)):
            return value
        return str(value)

    return {
        "rank": rank,
        "file_id": str(metadata.get("file_id") or chunk.get("file_id") or "") or None,
        "source_document": str(
            metadata.get("source")
            or metadata.get("file_name")
            or chunk.get("source")
            or ""
        )
        or None,
        "chunk_id": scalar(raw_chunk_id),
        "chunk_index": scalar(raw_chunk_index),
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "score": scalar(chunk.get("score")),
        "distance": scalar(chunk.get("distance")),
    }


async def _resolve_db_id(args: argparse.Namespace, manager: Any) -> str:
    response = await manager.get_databases()
    databases = response.get("databases") if isinstance(response, dict) else []
    matches = [
        value
        for value in databases or []
        if isinstance(value, dict)
        and str(value.get("kb_type") or "").casefold() == "milvus"
        and (
            str(value.get("db_id") or "").strip() == str(args.db_id).strip()
            if args.db_id
            else str(value.get("name") or "").strip()
            == str(args.knowledge_name).strip()
        )
    ]
    if len(matches) != 1:
        target = args.db_id or args.knowledge_name
        raise Top25ReplayError(
            f"Milvus 知识库 {target!r} 不存在或不唯一"
        )
    return str(matches[0].get("db_id") or "").strip()


async def _replay_query(
    *,
    manager: Any,
    db_id: str,
    query: dict[str, Any],
    timeout_seconds: int,
    technical_retry_limit: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, technical_retry_limit + 2):
        try:
            async with asyncio.timeout(timeout_seconds):
                embeddings = await manager.aembed_texts(
                    db_id,
                    [query["query_text"]],
                )
                if len(embeddings) != 1:
                    raise Top25ReplayError("embedding 返回数量不为 1")
                kwargs: dict[str, Any] = {
                    "search_mode": "vector",
                    "final_top_k": 25,
                    "use_reranker": False,
                    "include_distances": True,
                    "query_embedding": embeddings[0],
                    "raise_on_error": True,
                }
                if query["retrieval_scope"] == "document":
                    file_id = str(query.get("file_id") or "").strip()
                    if not file_id:
                        raise Top25ReplayError("文档内历史查询缺少 file_id")
                    kwargs["filter_file_ids"] = [file_id]
                chunks = await manager.aquery(
                    query["query_text"],
                    db_id,
                    **kwargs,
                )
            if not isinstance(chunks, list):
                raise Top25ReplayError("Milvus 返回值不是 list")
            candidates = [
                _candidate(value, rank)
                for rank, value in enumerate(chunks[:25], start=1)
                if isinstance(value, dict)
            ]
            return {
                "tool_name": "search_review_kb",
                "query_id": query["query_id"],
                "args": {
                    "query_text": query["query_text"],
                    "reason": query["reason"],
                    "retrieval_scope": query["retrieval_scope"],
                    "file_id": query.get("file_id"),
                    "investigation_id": query.get("investigation_id"),
                },
                "status": "success" if candidates else "success_empty",
                "attempts": attempt,
                "retrieved_items": candidates,
            }
        except Exception as exc:  # noqa: BLE001 - audit every backend failure
            last_error = exc
    return {
        "tool_name": "search_review_kb",
        "query_id": query["query_id"],
        "args": {
            "query_text": query["query_text"],
            "reason": query["reason"],
            "retrieval_scope": query["retrieval_scope"],
            "file_id": query.get("file_id"),
            "investigation_id": query.get("investigation_id"),
        },
        "status": "technical_failed",
        "attempts": technical_retry_limit + 1,
        "error_type": type(last_error).__name__ if last_error else "unknown",
        "error_message": str(last_error or "unknown error"),
        "retrieved_items": [],
    }


async def main_async(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise Top25ReplayError("--limit 必须大于 0")
    if args.timeout_seconds < 1:
        raise Top25ReplayError("--timeout-seconds 必须大于 0")
    if not 0 <= args.technical_retry_limit <= 3:
        raise Top25ReplayError("--technical-retry-limit 必须在 0–3 之间")

    pg_manager, manager = _runtime_dependencies()
    pg_manager.initialize()
    try:
        db_id = await _resolve_db_id(args, manager)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        case_count = 0
        query_count = 0
        failed_query_count = 0
        with args.output.open("w", encoding="utf-8") as output:
            for ordinal, source in enumerate(iter_records(args.records)):
                if args.limit is not None and case_count >= args.limit:
                    break
                queries = _query_records(source)
                if not queries:
                    continue
                calls = []
                row_index = source.get("row_index", ordinal)
                for query_index, query in enumerate(queries, start=1):
                    print(
                        "Top-25 replay start: "
                        f"row={row_index} query={query_index}/{len(queries)} "
                        f"scope={query['retrieval_scope']} "
                        f"query_id={query['query_id']}",
                        flush=True,
                    )
                    call = await _replay_query(
                        manager=manager,
                        db_id=db_id,
                        query=query,
                        timeout_seconds=args.timeout_seconds,
                        technical_retry_limit=args.technical_retry_limit,
                    )
                    print(
                        "Top-25 replay complete: "
                        f"row={row_index} query={query_index}/{len(queries)} "
                        f"status={call['status']} "
                        f"candidates={len(call['retrieved_items'])} "
                        f"attempts={call['attempts']}",
                        flush=True,
                    )
                    calls.append(call)
                    query_count += 1
                    failed_query_count += int(
                        call["status"] == "technical_failed"
                    )
                input_record = source.get("input_record")
                if not isinstance(input_record, dict):
                    input_record = {}
                record = {
                    "schema_version": "acm-top25-replay-1.0",
                    "row_index": row_index,
                    "case_id": source.get("case_id")
                    or input_record.get("case_id"),
                    "question": source.get("question")
                    or input_record.get("question"),
                    "result_status": (
                        "success"
                        if all(
                            value["status"] != "technical_failed"
                            for value in calls
                        )
                        else "partial"
                    ),
                    "retrieval_calls": calls,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                case_count += 1

        summary = {
            "schema_version": "acm-top25-replay-1.0",
            "db_id": db_id,
            "source_records": str(args.records),
            "output": str(args.output),
            "case_count": case_count,
            "query_count": query_count,
            "failed_query_count": failed_query_count,
            "top_k": 25,
            "llm_calls": 0,
        }
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        await pg_manager.close()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
