from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import unicodedata
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    from yuxi.agents import load_chat_model
    from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas import (
        AtlasStore,
        CorpusAtlasBuilder,
    )
    from yuxi.agents.buildin.medication_review_acm_prim.selector import (
        select_companion_cues,
    )
    from yuxi.storage.postgres.manager import pg_manager

    return (
        pg_manager,
        load_chat_model,
        AtlasStore,
        CorpusAtlasBuilder,
        select_companion_cues,
    )


DOCUMENT_EXTENSIONS = (".pdf", ".md", ".txt", ".doc", ".docx", ".html", ".htm")
REFERENCE_CITATION_RE = re.compile(r"【\s*依据\s*[:：]\s*(.*?)】", flags=re.DOTALL)
TRAILING_CHUNK_LABEL_RE = re.compile(r"#\s*\d+\s*$")
SOURCE_WRAPPER_PREFIX_RE = re.compile(r"^(?:【[^】]+】)+")
GENERATED_SOURCE_SUFFIX_RE = re.compile(
    r"(?:\.pdf)?_by_[^/]+$",
    flags=re.IGNORECASE,
)
CHUNK_SUFFIX_RE = re.compile(
    r"(?:[_\-\s]+(?:chunk|segment|part)[_\-\s]*\d+)$",
    flags=re.IGNORECASE,
)
JSON_READ_CHUNK_SIZE = 64 * 1024


class ReplayError(ValueError):
    pass


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """逐条读取 JSON 数组、JSONL 或连续 JSON 对象，不加载整个文件。"""
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise ReplayError(f"无法读取 {path}：{exc}") from exc

    with handle:
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False
        mode: str | None = None
        array_item_count = 0
        array_needs_separator = False
        record_index = 0

        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = handle.read(JSON_READ_CHUNK_SIZE)
                buffer += chunk
                eof = not chunk
                continue
            if not buffer and eof and mode == "sequence":
                return

            if mode is None:
                if not buffer and eof:
                    return
                if buffer.startswith("["):
                    mode = "array"
                    buffer = buffer[1:]
                else:
                    mode = "sequence"
                continue

            if mode == "array":
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    chunk = handle.read(JSON_READ_CHUNK_SIZE)
                    buffer += chunk
                    eof = not chunk
                    continue
                if not buffer and eof:
                    raise ReplayError(f"JSON 数组未闭合：{path}")
                if array_needs_separator:
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        array_needs_separator = False
                        continue
                    if buffer.startswith("]"):
                        trailing = buffer[1:] + handle.read()
                        if trailing.strip():
                            raise ReplayError(f"JSON 数组结尾后存在多余内容：{path}")
                        return
                    raise ReplayError(
                        f"JSON 数组第 {record_index - 1} 项后缺少逗号或右括号"
                    )
                if buffer.startswith("]"):
                    if array_item_count:
                        raise ReplayError(f"JSON 数组末尾存在多余逗号：{path}")
                    trailing = buffer[1:] + handle.read()
                    if trailing.strip():
                        raise ReplayError(f"JSON 数组结尾后存在多余内容：{path}")
                    return

            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if not eof:
                    chunk = handle.read(JSON_READ_CHUNK_SIZE)
                    buffer += chunk
                    eof = not chunk
                    continue
                raise ReplayError(
                    f"第 {record_index} 项附近 JSON 格式错误：{exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ReplayError(f"第 {record_index} 项必须是 JSON 对象")
            yield value
            record_index += 1
            if mode == "array":
                array_item_count += 1
                array_needs_separator = True
            buffer = buffer[end:]


def _trace(record: dict[str, Any]) -> dict[str, Any] | None:
    raw = record.get("medication_review_trace")
    if isinstance(raw, dict):
        return raw
    history = record.get("history")
    if isinstance(history, list):
        for message in reversed(history):
            if not isinstance(message, dict):
                continue
            metadata = message.get("extra_metadata")
            if not isinstance(metadata, dict):
                continue
            additional = metadata.get("additional_kwargs")
            if not isinstance(additional, dict):
                continue
            raw = additional.get("medication_review_trace")
            if isinstance(raw, dict):
                return raw
    method_family = str(record.get("method_family") or "")
    if method_family not in {"prim-rag-v1", "prim-rag-v2"}:
        return None
    queries = record.get("search_records")
    evidence = record.get("retrieved_evidence")
    if not isinstance(queries, list) or not isinstance(evidence, list):
        return None
    is_v2 = method_family == "prim-rag-v2"
    return {
        "schema_version": "8.0" if is_v2 else "5.0",
        "method_family": method_family,
        "plan_anchors": record.get("plan_anchors") or [],
        "patient_modifiers": record.get("patient_modifiers") or [],
        "query_records": queries,
        "relation_investigations": record.get("relation_investigations") or [],
        "investigations": record.get("investigations") or [],
        "evidence_store": evidence,
    }


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        metadata = message.get("extra_metadata")
        calls = metadata.get("tool_calls") if isinstance(metadata, dict) else []
    if not isinstance(calls, list):
        return set()
    return {
        str(call.get("id") or call.get("tool_call_id") or "").strip()
        for call in calls
        if isinstance(call, dict)
        and str(call.get("id") or call.get("tool_call_id") or "").strip()
    }


def _first_batch_queries(
    record: dict[str, Any],
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Recover the first tool-call batch without using later Agent state."""
    eligible = [
        value
        for value in queries
        if value.get("status") in {"success", "success_empty"}
    ]
    if not eligible:
        return [], "missing_eligible_query"
    history = record.get("history")
    if isinstance(history, list):
        for message in history:
            if not isinstance(message, dict):
                continue
            call_ids = _tool_call_ids(message)
            matched = [
                value
                for value in queries
                if str(value.get("tool_call_id") or "").strip()
                in call_ids
            ]
            if any(
                value.get("status") in {"success", "success_empty"}
                for value in matched
            ):
                return matched, "history_tool_call_batch"

    # Exported records omit message boundaries. Queries launched by one AI
    # action overlap in time, while a later action can only start after all
    # prior tools have returned. This reconstructs the largest defensible
    # first batch and reports that it is an approximation.
    first = eligible[0]
    first_started_at = str(first.get("started_at") or "")
    first_elapsed_ms = first.get("elapsed_ms")
    try:
        first_start = datetime.fromisoformat(
            first_started_at.replace("Z", "+00:00")
        )
        if first_start.tzinfo is None:
            first_start = first_start.replace(tzinfo=UTC)
        first_end = first_start + timedelta(
            milliseconds=max(float(first_elapsed_ms or 0), 0.0)
        )
        batch = []
        for value in queries:
            started_at = str(value.get("started_at") or "")
            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            elapsed_ms = value.get("elapsed_ms")
            ended = started + timedelta(
                milliseconds=max(float(elapsed_ms or 0), 0.0)
            )
            if started <= first_end and ended >= first_start:
                batch.append(value)
        if batch:
            return batch, "time_overlap_approximation"
    except (TypeError, ValueError):
        pass
    return [first], "first_query_only_approximation"


def _first_query_batch_records(
    record: dict[str, Any],
    trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], str] | None:
    raw_queries = [
        value
        for value in trace.get("query_records") or []
        if isinstance(value, dict)
    ]
    indexed_queries = list(enumerate(raw_queries))

    def order_key(item: tuple[int, dict[str, Any]]) -> tuple[bool, float, int]:
        index, value = item
        try:
            started = datetime.fromisoformat(
                str(value.get("started_at") or "").replace("Z", "+00:00")
            )
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            return False, started.timestamp(), index
        except (TypeError, ValueError):
            return True, 0.0, index

    queries = [value for _, value in sorted(indexed_queries, key=order_key)]
    if not any(
        value.get("status") in {"success", "success_empty"}
        for value in queries
    ):
        return None
    return _first_batch_queries(record, queries)


def _first_query_state(record: dict[str, Any]) -> dict[str, Any] | None:
    trace = _trace(record)
    if not isinstance(trace, dict) or trace.get("schema_version") not in {
        "5.0",
        "8.0",
    }:
        return None
    reconstructed = _first_query_batch_records(record, trace)
    if reconstructed is None:
        return None
    first_batch, reconstruction_mode = reconstructed
    trigger_query = next(
        (
            value
            for value in first_batch
            if value.get("status") in {"success", "success_empty"}
        ),
        None,
    )
    if trigger_query is None:
        return None
    trigger_query_id = str(trigger_query.get("query_id") or "").strip()
    if not trigger_query_id:
        return None
    first_query_ids = {
        str(value.get("query_id") or "").strip()
        for value in first_batch
        if str(value.get("query_id") or "").strip()
    }

    evidence_store: dict[str, Any] = {}
    for raw_item in trace.get("evidence_store") or []:
        if not isinstance(raw_item, dict) or not raw_item.get("evidence_id"):
            continue
        occurrences = [
            occurrence
            for occurrence in raw_item.get("occurrences") or []
            if isinstance(occurrence, dict)
            and str(occurrence.get("record_id") or "") in first_query_ids
        ]
        if occurrences:
            evidence_store[str(raw_item["evidence_id"])] = {
                **raw_item,
                "occurrences": occurrences,
            }

    relations = []
    legacy_investigation_ids: dict[str, str] = {}
    for raw_relation in trace.get("relation_investigations") or []:
        if not isinstance(raw_relation, dict):
            continue
        relation_id = str(raw_relation.get("relation_id") or "").strip()
        relation_queries = [
            value
            for value in first_batch
            if str(value.get("relation_id") or "").strip() == relation_id
        ]
        if not relation_id or not relation_queries:
            continue
        legacy_investigation_ids[relation_id] = f"INV-LEGACY-{relation_id}"
        query_ids = [
            str(value.get("query_id") or "").strip()
            for value in relation_queries
            if str(value.get("query_id") or "").strip()
        ]
        focus_plan_ids = list(
            dict.fromkeys(
                str(focus_id)
                for value in relation_queries
                for focus_id in value.get("focus_plan_ids") or []
                if str(focus_id)
            )
        )
        focus_modifier_ids = list(
            dict.fromkeys(
                str(focus_id)
                for value in relation_queries
                for focus_id in value.get("focus_modifier_ids") or []
                if str(focus_id)
            )
        )
        evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for value in relation_queries
                for evidence_id in value.get("evidence_ids") or []
                if str(evidence_id) in evidence_store
            )
        )
        new_evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for value in relation_queries
                for evidence_id in value.get("new_evidence_ids") or []
                if str(evidence_id) in evidence_store
            )
        )
        query_statuses = [
            (
                "technical_failed"
                if value.get("status") == "technical_failed"
                else "evidence_returned"
                if value.get("evidence_ids")
                else "empty"
            )
            for value in relation_queries
        ]
        status_set = set(query_statuses)
        retrieval_status = (
            query_statuses[0]
            if len(status_set) == 1
            else "evidence_returned"
            if status_set <= {"evidence_returned", "empty"}
            else "mixed"
        )
        relations.append(
            {
                "relation_id": relation_id,
                "relation_question": str(
                    raw_relation.get("relation_question") or ""
                ),
                "focus_plan_ids": focus_plan_ids,
                "focus_modifier_ids": focus_modifier_ids,
                "created_at": str(raw_relation.get("created_at") or ""),
                "query_ids": query_ids,
                "evidence_ids": evidence_ids,
                "new_evidence_ids": new_evidence_ids,
                "retrieval_status": retrieval_status,
                # The final Trace merges warnings across rounds and cannot
                # attribute them to the trigger batch. Omitting them is the
                # only non-leaking reconstruction.
                "warnings": [],
            }
        )

    allowed_query_fields = {
        "query_id",
        "tool_call_id",
        "investigation_id",
        "query_text",
        "reason",
        "retrieval_scope",
        "file_id",
        "focus_plan_ids",
        "focus_modifier_ids",
        "atlas_companion_ids",
        "started_at",
        "elapsed_ms",
        "status",
        "returned_count",
        "retained_count",
        "evidence_ids",
        "new_evidence_ids",
        "attempts",
        "invalid_focus_ids",
        "error_type",
        "error_message",
    }
    normalized_queries = []
    for raw_query in first_batch:
        normalized = {
            key: value
            for key, value in raw_query.items()
            if key in allowed_query_fields
        }
        relation_id = str(raw_query.get("relation_id") or "").strip()
        if not normalized.get("investigation_id") and relation_id:
            normalized["investigation_id"] = legacy_investigation_ids.get(
                relation_id,
                f"INV-LEGACY-{relation_id}",
            )
        normalized.setdefault("retrieval_scope", "global")
        normalized.setdefault("file_id", None)
        normalized.setdefault("focus_plan_ids", [])
        normalized.setdefault("focus_modifier_ids", [])
        normalized.setdefault("atlas_companion_ids", [])
        normalized.setdefault("returned_count", len(normalized.get("evidence_ids") or []))
        normalized.setdefault("retained_count", len(normalized.get("evidence_ids") or []))
        normalized.setdefault("new_evidence_ids", [])
        normalized.setdefault("attempts", [])
        normalized.setdefault("invalid_focus_ids", [])
        normalized.setdefault("error_type", None)
        normalized.setdefault("error_message", None)
        normalized_queries.append(normalized)

    investigation_sources = {
        str(value.get("investigation_id") or ""): value
        for value in trace.get("investigations") or []
        if isinstance(value, dict) and value.get("investigation_id")
    }
    for relation in relations:
        relation_id = str(relation.get("relation_id") or "")
        investigation_id = legacy_investigation_ids.get(relation_id)
        if investigation_id:
            investigation_sources[investigation_id] = {
                "investigation_id": investigation_id,
                "question": relation.get("relation_question") or "",
                "origin": "agent",
                "created_at": relation.get("created_at") or "",
            }

    evidence_file_ids = {
        evidence_id: str(value.get("file_id") or "")
        for evidence_id, value in evidence_store.items()
        if isinstance(value, dict)
    }
    investigations = []
    grouped_queries: dict[str, list[dict[str, Any]]] = {}
    for query in normalized_queries:
        investigation_id = str(query.get("investigation_id") or "").strip()
        if investigation_id:
            grouped_queries.setdefault(investigation_id, []).append(query)
    for investigation_id, queries in grouped_queries.items():
        source = investigation_sources.get(investigation_id, {})
        evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for query in queries
                for evidence_id in query.get("evidence_ids") or []
                if str(evidence_id) in evidence_store
            )
        )
        query_ids = [
            str(query.get("query_id") or "")
            for query in queries
            if query.get("query_id")
        ]
        focus_plan_ids = list(
            dict.fromkeys(
                str(value)
                for query in queries
                for value in query.get("focus_plan_ids") or []
                if str(value)
            )
        )
        focus_modifier_ids = list(
            dict.fromkeys(
                str(value)
                for query in queries
                for value in query.get("focus_modifier_ids") or []
                if str(value)
            )
        )
        atlas_companion_ids = list(
            dict.fromkeys(
                str(value)
                for query in queries
                for value in query.get("atlas_companion_ids") or []
                if str(value)
            )
        )
        created_at = str(
            source.get("created_at")
            or next(
                (query.get("started_at") for query in queries if query.get("started_at")),
                "",
            )
        )
        investigations.append(
            {
                "investigation_id": investigation_id,
                "question": str(
                    source.get("question")
                    or next(
                        (
                            query.get("reason") or query.get("query_text")
                            for query in queries
                        ),
                        "",
                    )
                ),
                "origin": (
                    "atlas" if source.get("origin") == "atlas" else "agent"
                ),
                "focus_plan_ids": focus_plan_ids,
                "focus_modifier_ids": focus_modifier_ids,
                "atlas_companion_ids": atlas_companion_ids,
                # Only state observable immediately after the trigger batch is
                # replayed. Final status, note and selected evidence are future
                # Agent decisions and must not leak into Selector input.
                "status": "open",
                "query_ids": query_ids,
                "candidate_evidence_ids": evidence_ids,
                "selected_evidence_ids": [],
                "candidate_file_ids": list(
                    dict.fromkeys(
                        evidence_file_ids[evidence_id]
                        for evidence_id in evidence_ids
                        if evidence_file_ids.get(evidence_id)
                    )
                ),
                "working_note": "",
                "created_at": created_at,
                "updated_at": created_at,
                "warnings": [],
            }
        )

    question = str(record.get("question") or trace.get("raw_case_text") or "")
    if not question:
        input_record = record.get("input_record")
        if isinstance(input_record, dict):
            question = str(input_record.get("question") or "")
    return {
        "raw_case_text": question,
        "plan_anchors": trace.get("plan_anchors") or [],
        "patient_modifiers": trace.get("patient_modifiers") or [],
        # Recreate the first AI tool-call batch as well as the query records.
        # The online selector uses this boundary to avoid seeing later searches;
        # without it, an offline replay would incorrectly keep only the first
        # query from an originally parallel first batch.
        "messages": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "id": str(value.get("tool_call_id") or ""),
                        "name": "search_review_kb",
                        "args": {},
                    }
                    for value in first_batch
                    if str(value.get("tool_call_id") or "").strip()
                ],
            }
        ],
        "query_records": normalized_queries,
        "trigger_query_id": trigger_query_id,
        "relation_investigations": relations,
        "investigations": investigations,
        "evidence_store": evidence_store,
        "reconstruction_mode": reconstruction_mode,
    }


def _canonical_document(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = SOURCE_WRAPPER_PREFIX_RE.sub("", text).casefold()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\s+", "", text)
    for _ in range(2):
        for suffix in DOCUMENT_EXTENSIONS:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        text = GENERATED_SOURCE_SUFFIX_RE.sub("", text)
    text = TRAILING_CHUNK_LABEL_RE.sub("", text)
    return CHUNK_SUFFIX_RE.sub("", text)


def _matches_document(document: Any, gold: Any) -> bool:
    document_key = _canonical_document(document)
    gold_key = _canonical_document(gold)
    return bool(
        document_key
        and gold_key
        and (document_key == gold_key or gold_key in document_key)
    )


def _gold_documents(record: dict[str, Any]) -> list[str]:
    input_record = record.get("input_record")
    sources = [input_record, record] if isinstance(input_record, dict) else [record]
    for source in sources:
        for field in ("documents", "must_retrieve_documents", "reference_documents"):
            values = source.get(field)
            if isinstance(values, str) and values.strip():
                return [values.strip()]
            if isinstance(values, list):
                result = [str(value).strip() for value in values if str(value).strip()]
                if result:
                    return result
    reference = next(
        (
            str(source.get("reference") or "")
            for source in sources
            if source.get("reference")
        ),
        "",
    )
    result: list[str] = []
    for citation in REFERENCE_CITATION_RE.findall(reference):
        if "·" not in citation:
            continue
        document = TRAILING_CHUNK_LABEL_RE.sub(
            "", citation.split("·", 1)[0].strip()
        ).strip()
        if document and _canonical_document(document) not in {
            _canonical_document(value) for value in result
        }:
            result.append(document)
    return result


def _normalized_question(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip()


def _record_case_id(record: dict[str, Any]) -> str:
    input_record = record.get("input_record")
    for source in (
        record,
        input_record if isinstance(input_record, dict) else {},
    ):
        value = str(source.get("case_id") or "").strip()
        if value:
            return value
    return ""


def _record_question(record: dict[str, Any]) -> str:
    input_record = record.get("input_record")
    for source in (
        record,
        input_record if isinstance(input_record, dict) else {},
    ):
        value = str(source.get("question") or "").strip()
        if value:
            return value
    return ""


class GoldIndex:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.by_case_id = self._unique_map(_record_case_id)
        self.by_question = self._unique_map(
            lambda record: _normalized_question(_record_question(record))
        )

    def _unique_map(self, key_function: Any) -> dict[str, int]:
        values: dict[str, int] = {}
        duplicates: set[str] = set()
        for index, record in enumerate(self.records):
            key = key_function(record)
            if not key:
                continue
            if key in values:
                duplicates.add(key)
            else:
                values[key] = index
        for key in duplicates:
            values.pop(key, None)
        return values

    def match(
        self,
        record: dict[str, Any],
        fallback_position: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        case_id = _record_case_id(record)
        if case_id and case_id in self.by_case_id:
            return self.records[self.by_case_id[case_id]], "case_id"
        question = _normalized_question(_record_question(record))
        if question and question in self.by_question:
            return self.records[self.by_question[question]], "question"
        row_index = record.get("row_index")
        if isinstance(row_index, int) and 0 <= row_index < len(self.records):
            return self.records[row_index], "row_index"
        if (
            not case_id
            and not question
            and 0 <= fallback_position < len(self.records)
        ):
            return self.records[fallback_position], "position"
        return None, None


def _actual_documents(record: dict[str, Any]) -> list[str]:
    trace = _trace(record) or {}
    return list(
        dict.fromkeys(
            str(item.get("source_document") or item.get("file_id") or "").strip()
            for item in trace.get("evidence_store") or []
            if isinstance(item, dict)
            and str(item.get("source_document") or item.get("file_id") or "").strip()
        )
    )


def _trigger_documents(record: dict[str, Any]) -> list[str]:
    trace = _trace(record) or {}
    reconstructed = _first_query_batch_records(record, trace)
    if reconstructed is None:
        return []
    first_batch, _ = reconstructed
    query_ids = {
        str(value.get("query_id") or "")
        for value in first_batch
        if str(value.get("query_id") or "")
    }
    return list(
        dict.fromkeys(
            str(item.get("source_document") or item.get("file_id") or "").strip()
            for item in trace.get("evidence_store") or []
            if isinstance(item, dict)
            and str(item.get("source_document") or item.get("file_id") or "").strip()
            and any(
                isinstance(occurrence, dict)
                and str(occurrence.get("record_id") or "") in query_ids
                for occurrence in item.get("occurrences") or []
            )
        )
    )


def _percentile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _usage_total(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    for key in ("total_tokens", "total_token_count"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    input_tokens = next(
        (
            usage.get(key)
            for key in ("input_tokens", "prompt_tokens", "input_token_count")
            if isinstance(usage.get(key), (int, float))
        ),
        None,
    )
    output_tokens = next(
        (
            usage.get(key)
            for key in (
                "output_tokens",
                "completion_tokens",
                "output_token_count",
            )
            if isinstance(usage.get(key), (int, float))
        ),
        None,
    )
    if input_tokens is None and output_tokens is None:
        return None
    return int(input_tokens or 0) + int(output_tokens or 0)


def _selected_atlas_sources(atlas: Any, cue_ids: list[str]) -> list[dict[str, Any]]:
    index = atlas.cue_index()
    result = []
    for cue_id in dict.fromkeys(cue_ids):
        card_and_cue = index.get(cue_id)
        if card_and_cue is None:
            continue
        card, cue = card_and_cue
        result.append(
            {
                "atlas_cue_id": cue_id,
                "cue_text": cue.cue_text,
                "document_id": card.doc_id,
                "document_title": card.title,
                "source_chunk_ids": cue.source_chunk_ids,
            }
        )
    return result


def _residual_metrics(
    *,
    record: dict[str, Any],
    suggested_documents: list[str],
) -> dict[str, Any]:
    gold = _gold_documents(record)
    trigger = _trigger_documents(record)
    final = _actual_documents(record)
    trigger_gap = [
        value
        for value in gold
        if not any(_matches_document(document, value) for document in trigger)
    ]
    trigger_hits = [
        value
        for value in trigger_gap
        if any(
            _matches_document(document, value)
            for document in suggested_documents
        )
    ]
    residual = [
        value
        for value in gold
        if not any(_matches_document(document, value) for document in final)
    ]
    hits = [
        value
        for value in residual
        if any(
            _matches_document(document, value)
            for document in suggested_documents
        )
    ]
    return {
        "gold_documents": gold,
        "prim_trigger_documents": trigger,
        "prim_final_documents": final,
        "prim_retrieved_documents": final,
        "trigger_point_missing_gold_documents": trigger_gap,
        "companion_trigger_point_hits": trigger_hits,
        "trigger_point_document_hit_count": len(trigger_hits),
        "trigger_point_document_recall": (
            len(trigger_hits) / len(trigger_gap) if trigger_gap else None
        ),
        "residual_gold_documents": residual,
        "companion_residual_hits": hits,
        "residual_document_hit_count": len(hits),
        "residual_document_recall": (
            len(hits) / len(residual) if residual else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在历史 PRIM Trace 的首个有效查询时点回放 Atlas 伴随线索选择器。"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gold",
        type=Path,
        help=(
            "可选的病例与标答文件。仅在选择器完成后用于残余文档统计，"
            "不会进入选择器输入。"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="汇总指标 JSON；默认写到 <output>.summary.json。",
    )
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


async def main_async(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise ReplayError("--limit 必须大于 0")
    if args.input.resolve() == args.output.resolve():
        raise ReplayError("--input 与 --output 不能相同")
    if args.gold is not None and args.gold.resolve() == args.output.resolve():
        raise ReplayError("--gold 与 --output 不能相同")
    summary_output = args.summary_output or args.output.with_suffix(
        f"{args.output.suffix}.summary.json"
    )
    if summary_output.resolve() in {
        args.input.resolve(),
        args.output.resolve(),
    }:
        raise ReplayError("--summary-output 不能与输入或逐例输出相同")
    (
        pg_manager,
        load_chat_model,
        AtlasStore,
        CorpusAtlasBuilder,
        select_companion_cues,
    ) = _runtime_dependencies()
    pg_manager.initialize()
    try:
        gold_index = (
            GoldIndex(list(_iter_records(args.gold)))
            if args.gold is not None
            else None
        )
        store = AtlasStore()
        atlas = store.load_current(str(args.db_id).strip())
        builder = CorpusAtlasBuilder(
            model_name=atlas.builder_model,
            store=store,
        )
        await builder.validate_runtime(atlas)
        model = load_chat_model(str(args.model).strip())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
        processed = selected = skipped = failed = 0
        cue_counts: list[int] = []
        elapsed_values: list[int] = []
        token_values: list[int] = []
        residual_case_count = 0
        residual_gold_count = 0
        residual_hit_count = 0
        trigger_gap_case_count = 0
        trigger_gap_gold_count = 0
        trigger_gap_hit_count = 0
        gold_matched_count = 0
        gold_unmatched_count = 0
        unique_selected_atlas_cues: set[str] = set()
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(_iter_records(args.input)):
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1
                gold_record, gold_match_by = (
                    gold_index.match(record, index)
                    if gold_index is not None
                    else (None, None)
                )
                if gold_index is not None:
                    gold_matched_count += int(gold_record is not None)
                    gold_unmatched_count += int(gold_record is None)
                reconstructed = _first_query_state(record)
                if reconstructed is None:
                    skipped += 1
                    output = {
                        "row_index": record.get("row_index", index),
                        "case_id": (record.get("input_record") or {}).get("case_id")
                        if isinstance(record.get("input_record"), dict)
                        else record.get("case_id"),
                        "question": record.get("question"),
                        "status": "skipped",
                        "reason": "缺少 PRIM Trace 5.0/8.0 或首个有效 QueryRecord",
                        "gold_match_by": gold_match_by,
                    }
                else:
                    state = reconstructed
                    selection = await select_companion_cues(
                        model=model,
                        state=state,
                        atlas=atlas,
                        trigger_type="after_query",
                        created_after_query_id=state["trigger_query_id"],
                        technical_retry_limit=args.technical_retry_limit,
                    )
                    selector_status = selection.selector_audit.status
                    selector_valid = selector_status != "failed"
                    if selector_valid:
                        selected += int(bool(selection.companion_cues))
                        cue_counts.append(len(selection.companion_cues))
                        elapsed_values.append(
                            selection.selector_audit.elapsed_ms
                        )
                        usage_tokens = _usage_total(
                            selection.selector_audit.usage
                        )
                        if usage_tokens is not None:
                            token_values.append(usage_tokens)
                    else:
                        failed += 1
                    selected_cue_ids = list(
                        dict.fromkeys(
                            cue_id
                            for cue in selection.companion_cues
                            for cue_id in cue.atlas_cue_ids
                        )
                    )
                    unique_selected_atlas_cues.update(selected_cue_ids)
                    suggested_ids = list(
                        dict.fromkeys(
                            doc_id
                            for cue in selection.companion_cues
                            for doc_id in cue.suggested_doc_ids
                        )
                    )
                    atlas_docs = {
                        card.doc_id: card for card in atlas.document_cards
                    }
                    suggested_titles = [
                        (
                            atlas_docs[doc_id].title
                            if doc_id in atlas_docs
                            else doc_id
                        )
                        for doc_id in suggested_ids
                    ]
                    suggested_file_names = [
                        (
                            atlas_docs[doc_id].file_name
                            if doc_id in atlas_docs
                            else doc_id
                        )
                        for doc_id in suggested_ids
                    ]
                    document_metrics = (
                        _residual_metrics(
                            record=(
                                {**record, "input_record": gold_record}
                                if gold_record is not None
                                else record
                            ),
                            suggested_documents=list(
                                dict.fromkeys(
                                    [*suggested_titles, *suggested_file_names]
                                )
                            ),
                        )
                        if selector_valid
                        else None
                    )
                    if (
                        document_metrics is not None
                        and document_metrics["residual_gold_documents"]
                    ):
                        residual_case_count += 1
                        residual_gold_count += len(
                            document_metrics["residual_gold_documents"]
                        )
                        residual_hit_count += int(
                            document_metrics["residual_document_hit_count"]
                        )
                    if (
                        document_metrics is not None
                        and document_metrics[
                            "trigger_point_missing_gold_documents"
                        ]
                    ):
                        trigger_gap_case_count += 1
                        trigger_gap_gold_count += len(
                            document_metrics[
                                "trigger_point_missing_gold_documents"
                            ]
                        )
                        trigger_gap_hit_count += int(
                            document_metrics[
                                "trigger_point_document_hit_count"
                            ]
                        )
                    output = {
                        "row_index": record.get("row_index", index),
                        "case_id": (record.get("input_record") or {}).get("case_id")
                        if isinstance(record.get("input_record"), dict)
                        else record.get("case_id"),
                        "question": record.get("question"),
                        "status": selector_status,
                        "gold_match_by": gold_match_by,
                        "atlas_snapshot_hash": atlas.snapshot_hash,
                        "trigger_query": next(
                            value
                            for value in state["query_records"]
                            if value.get("query_id")
                            == state["trigger_query_id"]
                        ),
                        "reconstruction_mode": state["reconstruction_mode"],
                        "trigger_relations": state["relation_investigations"],
                        "trigger_investigations": state["investigations"],
                        "companion_selection": selection.model_dump(mode="json"),
                        "suggested_document_ids": suggested_ids,
                        "suggested_documents": suggested_titles,
                        "suggested_document_file_names": suggested_file_names,
                        "selected_atlas_sources": _selected_atlas_sources(
                            atlas,
                            selected_cue_ids,
                        ),
                        "atlas_total_cue_count": sum(
                            len(card.topic_cues) for card in atlas.document_cards
                        ),
                        "atlas_compression": 1
                        - len(
                            set(selected_cue_ids)
                        )
                        / max(
                            sum(len(card.topic_cues) for card in atlas.document_cards),
                            1,
                        ),
                        "document_metrics": document_metrics,
                    }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[{processed}] status={output['status']} "
                    f"row={output.get('row_index')}",
                    flush=True,
                )
        temporary.replace(args.output)
        atlas_total_cues = sum(
            len(card.topic_cues) for card in atlas.document_cards
        )
        valid_selector_count = len(cue_counts)
        summary = {
            "processed": processed,
            "selector_runs": valid_selector_count + failed,
            "valid_selector_runs": valid_selector_count,
            "with_companion_cues": selected,
            "selector_empty": sum(value == 0 for value in cue_counts),
            "selector_empty_rate": (
                sum(value == 0 for value in cue_counts)
                / valid_selector_count
                if valid_selector_count
                else None
            ),
            "skipped": skipped,
            "selector_failed": failed,
            "gold_alignment": {
                "provided": gold_index is not None,
                "matched": gold_matched_count,
                "unmatched": gold_unmatched_count,
            },
            "cue_count": {
                "mean": (
                    sum(cue_counts) / len(cue_counts) if cue_counts else None
                ),
                "p50": _percentile(cue_counts, 0.5),
                "p90": _percentile(cue_counts, 0.9),
                "max": max(cue_counts) if cue_counts else None,
            },
            "selector_elapsed_ms": {
                "mean": (
                    sum(elapsed_values) / len(elapsed_values)
                    if elapsed_values
                    else None
                ),
                "p50": _percentile(elapsed_values, 0.5),
                "p90": _percentile(elapsed_values, 0.9),
            },
            "selector_tokens": {
                "mean": (
                    sum(token_values) / len(token_values)
                    if token_values
                    else None
                ),
                "p50": _percentile(token_values, 0.5),
                "p90": _percentile(token_values, 0.9),
                "reported_case_count": len(token_values),
            },
            "residual_document": {
                "case_count": residual_case_count,
                "gold_count": residual_gold_count,
                "hit_count": residual_hit_count,
                "micro_recall": (
                    residual_hit_count / residual_gold_count
                    if residual_gold_count
                    else None
                ),
            },
            "trigger_point_document": {
                "case_count": trigger_gap_case_count,
                "gold_count": trigger_gap_gold_count,
                "hit_count": trigger_gap_hit_count,
                "micro_recall": (
                    trigger_gap_hit_count / trigger_gap_gold_count
                    if trigger_gap_gold_count
                    else None
                ),
            },
            "atlas": {
                "snapshot_hash": atlas.snapshot_hash,
                "total_cue_count": atlas_total_cues,
                "selected_unique_cue_count": len(
                    unique_selected_atlas_cues
                ),
                "selected_cue_fraction": (
                    len(unique_selected_atlas_cues) / atlas_total_cues
                    if atlas_total_cues
                    else None
                ),
            },
            "output": str(args.output),
            "summary_output": str(summary_output),
        }
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_temporary = summary_output.with_suffix(
            f"{summary_output.suffix}.tmp"
        )
        summary_temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_temporary.replace(summary_output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if failed == 0 else 2
    finally:
        await pg_manager.close()


def main() -> int:
    try:
        return asyncio.run(main_async(build_parser().parse_args()))
    except (OSError, ReplayError, ValueError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
