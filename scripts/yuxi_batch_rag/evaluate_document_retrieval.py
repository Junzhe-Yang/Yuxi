"""Offline document-level retrieval evaluation for Yuxi batch results.

The evaluator does not call Yuxi or any model. Gold annotations are treated as
required-document positives only: a retrieved document that is not annotated
is unjudged, not irrelevant.

For agentic retrieval, ranked metrics use the order in which unique documents
are first discovered by successful search calls. Evidence-window expansion is
reported separately as an unranked final evidence pool. Chunk IDs and chunk
contents are intentionally not evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


DEFAULT_K_VALUES = (1, 3, 5, 10, 20, 50)
SUCCESS_STATUSES = {
    "success",
    "success_empty",
    "succeeded",
    "completed",
    "done",
}
FAILURE_STATUSES = {
    "failed",
    "technical_failed",
    "error",
    "cancelled",
    "canceled",
    "interrupted",
    "incomplete",
}
DOCUMENT_EXTENSIONS = (".pdf", ".md", ".txt", ".doc", ".docx", ".html", ".htm")
JSON_READ_CHUNK_SIZE = 64 * 1024
CHUNK_SUFFIX_RE = re.compile(
    r"(?:[_\-\s]+(?:chunk|segment|part)[_\-\s]*\d+)$",
    flags=re.IGNORECASE,
)
TRAILING_CHUNK_LABEL_RE = re.compile(r"#\s*\d+\s*$")
SOURCE_WRAPPER_PREFIX_RE = re.compile(r"^(?:【[^】]+】)+")
GENERATED_SOURCE_SUFFIX_RE = re.compile(
    r"(?:\.pdf)?_by_[^/]+$",
    flags=re.IGNORECASE,
)
REFERENCE_CITATION_RE = re.compile(
    r"【\s*依据\s*[:：]\s*(.*?)】",
    flags=re.DOTALL,
)


class EvaluationError(ValueError):
    """Raised when evaluation input is invalid."""


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_more(handle: Any, buffer: str) -> tuple[str, bool]:
    chunk = handle.read(JSON_READ_CHUNK_SIZE)
    return buffer + chunk, not chunk


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON-array, JSONL, or single-object input without loading it all."""
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        raise EvaluationError(f"Cannot read input file '{path}': {exc}") from exc

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
                buffer, eof = _read_more(handle, buffer)
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
                    buffer, eof = _read_more(handle, buffer)
                    continue
                if not buffer and eof:
                    raise EvaluationError(f"Unterminated JSON array in '{path}'")

                if array_needs_separator:
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        array_needs_separator = False
                        continue
                    elif buffer.startswith("]"):
                        trailing = buffer[1:] + handle.read()
                        if trailing.strip():
                            raise EvaluationError(
                                f"Unexpected content after JSON array in '{path}'"
                            )
                        return
                    else:
                        raise EvaluationError(
                            f"Expected ',' or ']' after item {record_index - 1} "
                            f"in '{path}'"
                        )
                elif buffer.startswith("]"):
                    if array_item_count:
                        raise EvaluationError(
                            f"Trailing comma in JSON array '{path}'"
                        )
                    trailing = buffer[1:] + handle.read()
                    if trailing.strip():
                        raise EvaluationError(
                            f"Unexpected content after JSON array in '{path}'"
                        )
                    return

            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if not eof:
                    buffer, eof = _read_more(handle, buffer)
                    continue
                raise EvaluationError(
                    f"Invalid JSON in '{path}' near item {record_index}: {exc.msg}"
                ) from exc

            if not isinstance(value, dict):
                raise EvaluationError(
                    f"{path} item {record_index} must contain a JSON object"
                )
            yield value
            record_index += 1
            if mode == "array":
                array_item_count += 1
                array_needs_separator = True
            buffer = buffer[end:]


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load records for small inputs and compatibility with existing callers."""
    return list(iter_records(path))


def load_aliases(path: Path | None) -> dict[str, list[str]]:
    """Load a mapping of canonical gold document name to aliases."""
    if path is None:
        return {}

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationError(f"Cannot read aliases file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Aliases file '{path}' is invalid JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise EvaluationError("Aliases file must contain a JSON object")

    aliases: dict[str, list[str]] = {}
    for gold_name, alias_values in value.items():
        if not isinstance(gold_name, str) or not gold_name.strip():
            raise EvaluationError("Aliases file contains an empty gold document name")
        if isinstance(alias_values, str):
            alias_values = [alias_values]
        if not isinstance(alias_values, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in alias_values
        ):
            raise EvaluationError(
                f"Aliases for '{gold_name}' must be a string list"
            )
        aliases[gold_name.strip()] = [alias.strip() for alias in alias_values]
    return aliases


def _strip_document_extension(text: str) -> str:
    changed = True
    while changed:
        changed = False
        for extension in DOCUMENT_EXTENSIONS:
            if text.endswith(extension):
                text = text[: -len(extension)]
                changed = True
                break
    return text


def canonicalize_document_name(value: Any) -> str:
    """Normalize a source/document name without applying fuzzy matching."""
    text = _string(value)
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = SOURCE_WRAPPER_PREFIX_RE.sub("", text)
    text = text.casefold()
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("【", "[").replace("】", "]")
    text = re.sub(r"\s+", "", text)
    text = _strip_document_extension(text)
    text = GENERATED_SOURCE_SUFFIX_RE.sub("", text)
    text = _strip_document_extension(text)
    text = TRAILING_CHUNK_LABEL_RE.sub("", text)
    text = CHUNK_SUFFIX_RE.sub("", text)
    return text


def extract_reference_documents(reference: Any) -> list[str]:
    """Extract unique document titles from ``【依据：文档 · ...】`` citations."""
    text = _string(reference)
    if not text:
        return []

    documents: list[str] = []
    seen: set[str] = set()
    for citation in REFERENCE_CITATION_RE.findall(text):
        if "·" not in citation:
            continue
        document = citation.split("·", 1)[0].strip()
        document = TRAILING_CHUNK_LABEL_RE.sub("", document).strip()
        canonical = canonicalize_document_name(document)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        documents.append(document)
    return documents


def _gold_documents(record: dict[str, Any]) -> list[Any]:
    for field in ("documents", "must_retrieve_documents"):
        values = record.get(field)
        if isinstance(values, str) and values.strip():
            return [values]
        if isinstance(values, list) and values:
            return values
    return extract_reference_documents(record.get("reference"))


def _gold_document_entries(
    gold_documents: Iterable[Any],
    aliases: dict[str, list[str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    aliases_by_canonical = {
        canonicalize_document_name(name): values
        for name, values in aliases.items()
    }
    for value in gold_documents:
        raw = _string(value)
        canonical = canonicalize_document_name(raw)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        targets = {canonical}
        for alias in aliases.get(raw, aliases_by_canonical.get(canonical, [])):
            alias_canonical = canonicalize_document_name(alias)
            if alias_canonical:
                targets.add(alias_canonical)
        entries.append(
            {
                "raw": raw,
                "canonical": canonical,
                "targets": targets,
            }
        )
    return entries


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    raw_metadata = item.get("raw_metadata")
    if isinstance(raw_metadata, dict):
        merged.update(raw_metadata)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        merged.update(metadata)
    return merged


def _first_string(
    item: dict[str, Any],
    metadata: dict[str, Any],
    fields: tuple[str, ...],
) -> tuple[str, str]:
    for field in fields:
        value = _string(item.get(field))
        if value:
            return value, field
        value = _string(metadata.get(field))
        if value:
            return value, f"metadata.{field}"
    return "", ""


def extract_item_identity(item: Any) -> dict[str, Any]:
    """Extract document identity and lightweight provenance from one evidence item."""
    if not isinstance(item, dict):
        return {
            "raw_item": item,
            "evidence_id": "",
            "chunk_id": "",
            "file_id": "",
            "document_raw": "",
            "document_canonical": "",
            "source_field": "",
            "score": None,
            "document_key": None,
            "item_key": None,
            "unstable_item_key": True,
            "call_index": None,
            "local_rank": None,
            "source_method": "",
        }

    metadata = _metadata(item)
    evidence_id, _ = _first_string(item, metadata, ("evidence_id",))
    file_id, file_id_field = _first_string(item, metadata, ("file_id",))
    chunk_id, _ = _first_string(item, metadata, ("chunk_id",))
    source, source_field = _first_string(
        item,
        metadata,
        (
            "source_document",
            "source",
            "file_name",
            "filename",
            "filepath",
            "file_path",
            "parsed_path",
        ),
    )
    document_raw = source or file_id
    document_canonical = canonicalize_document_name(document_raw)
    content, _ = _first_string(item, metadata, ("raw_text", "content"))

    if file_id:
        document_key = f"id:{file_id}"
    elif document_canonical:
        document_key = f"name:{document_canonical}"
    else:
        document_key = None

    if evidence_id:
        item_key = f"evidence:{evidence_id}"
        unstable_item_key = False
    elif chunk_id:
        item_key = f"chunk:{file_id}:{chunk_id}" if file_id else f"chunk:{chunk_id}"
        unstable_item_key = False
    elif content:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        item_key = f"content:{document_key}:{content_hash}"
        unstable_item_key = True
    else:
        item_key = None
        unstable_item_key = True

    score = None
    for score_field in ("score", "hybrid_score", "distance", "bm25_score"):
        if isinstance(item.get(score_field), (int, float)):
            score = item[score_field]
            break

    call_index = item.get("_evaluation_call_index")
    local_rank = item.get("_evaluation_local_rank")
    source_method = _string(item.get("_evaluation_source_method"))
    if not source_method:
        occurrences = item.get("occurrences")
        if isinstance(occurrences, list):
            source_methods = [
                _string(occurrence.get("source_method"))
                for occurrence in occurrences
                if isinstance(occurrence, dict)
            ]
            source_method = next((value for value in source_methods if value), "")

    return {
        "raw_item": item,
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "file_id": file_id,
        "file_id_field": file_id_field,
        "document_raw": document_raw,
        "document_canonical": document_canonical,
        "source_field": source_field or (file_id_field if file_id else ""),
        "score": score,
        "document_key": document_key,
        "item_key": item_key,
        "unstable_item_key": unstable_item_key,
        "call_index": call_index if isinstance(call_index, int) else None,
        "local_rank": local_rank if isinstance(local_rank, int) else None,
        "source_method": source_method,
    }


def match_document(
    document_raw: str,
    gold_entries: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Match one source name to gold documents using exact/substring rules."""
    document_canonical = canonicalize_document_name(document_raw)
    if not document_canonical:
        return "missing_source", []

    matches = []
    for entry in gold_entries:
        if any(
            document_canonical == target or target in document_canonical
            for target in entry["targets"]
        ):
            matches.append(entry)

    if len(matches) == 1:
        return "matched", [matches[0]["raw"]]
    if len(matches) > 1:
        return "ambiguous", [entry["raw"] for entry in matches]
    return "unmatched", []


def _item_key(identity: dict[str, Any], fallback_index: int) -> str:
    item_key = identity.get("item_key")
    if isinstance(item_key, str) and item_key:
        return item_key
    return f"position:{fallback_index}"


def _rank_items(
    items: Iterable[Any],
    gold_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    ranked: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    duplicate_count = 0

    for source_index, item in enumerate(items):
        identity = extract_item_identity(item)
        item_key = _item_key(identity, source_index)
        if item_key in seen_items:
            duplicate_count += 1
            continue
        seen_items.add(item_key)

        match_status, matched_gold_documents = match_document(
            identity["document_raw"], gold_entries
        )
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "evidence_id": identity["evidence_id"],
                "chunk_id": identity["chunk_id"],
                "file_id": identity["file_id"],
                "document_raw": identity["document_raw"],
                "document_canonical": identity["document_canonical"],
                "document_key": identity["document_key"],
                "source_field": identity["source_field"],
                "document_match_status": match_status,
                "matched_gold_documents": matched_gold_documents,
                "score": identity["score"],
                "unstable_item_key": identity["unstable_item_key"],
                "call_index": identity["call_index"],
                "local_rank": identity["local_rank"],
                "source_method": identity["source_method"],
            }
        )

    return ranked, duplicate_count


def _rank_documents(ranked_items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    document_by_key: dict[str, dict[str, Any]] = {}
    for item in ranked_items:
        document_key = item.get("document_key")
        if not isinstance(document_key, str) or not document_key:
            continue
        existing = document_by_key.get(document_key)
        if existing is not None:
            existing["evidence_item_count"] += 1
            continue
        document = {
            "rank": len(documents) + 1,
            "document_key": document_key,
            "file_id": item.get("file_id"),
            "document_raw": item.get("document_raw"),
            "document_canonical": item.get("document_canonical"),
            "source_field": item.get("source_field"),
            "document_match_status": item.get("document_match_status"),
            "matched_gold_documents": item.get("matched_gold_documents", []),
            "first_evidence_id": item.get("evidence_id"),
            "first_chunk_id": item.get("chunk_id"),
            "first_item_rank": item.get("rank"),
            "first_call_index": item.get("call_index"),
            "first_local_rank": item.get("local_rank"),
            "first_source_method": item.get("source_method"),
            "first_score": item.get("score"),
            "evidence_item_count": 1,
        }
        documents.append(document)
        document_by_key[document_key] = document
    return documents


def _metrics_for_documents(
    documents: list[dict[str, Any]],
    gold_entries: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] | None,
) -> dict[str, float | int | None]:
    gold_keys = {entry["canonical"] for entry in gold_entries}

    def values_for(document_subset: list[dict[str, Any]]) -> dict[str, float | int]:
        matched_gold_keys = {
            canonicalize_document_name(gold)
            for document in document_subset
            if document["document_match_status"] == "matched"
            for gold in document["matched_gold_documents"]
        }
        matched_gold_keys &= gold_keys
        matched_document_count = sum(
            document["document_match_status"] == "matched"
            for document in document_subset
        )
        unmatched_document_count = sum(
            document["document_match_status"] == "unmatched"
            for document in document_subset
        )
        ambiguous_document_count = sum(
            document["document_match_status"] == "ambiguous"
            for document in document_subset
        )
        document_count = len(document_subset)
        recall = len(matched_gold_keys) / len(gold_keys)
        return {
            "required_document_recall": recall,
            "required_document_hit": int(bool(matched_gold_keys)),
            "all_required_documents_hit": int(matched_gold_keys == gold_keys),
            "matched_required_document_count": len(matched_gold_keys),
            "missing_required_document_count": len(gold_keys - matched_gold_keys),
            "retrieved_document_count": document_count,
            "annotated_required_document_fraction": (
                matched_document_count / document_count if document_count else 0.0
            ),
            "unjudged_document_count": unmatched_document_count,
            "unjudged_document_rate": (
                unmatched_document_count / document_count if document_count else 0.0
            ),
            "ambiguous_document_count": ambiguous_document_count,
            "ambiguous_document_rate": (
                ambiguous_document_count / document_count if document_count else 0.0
            ),
        }

    all_values = values_for(documents)
    metrics: dict[str, float | int | None] = {
        "gold_document_count": len(gold_keys),
        **{f"{name}_all": value for name, value in all_values.items()},
    }

    if k_values is None:
        return metrics

    first_required_rank = next(
        (
            document["rank"]
            for document in documents
            if document["document_match_status"] == "matched"
        ),
        None,
    )
    metrics["first_required_document_rank"] = first_required_rank
    metrics["document_mrr"] = (
        1.0 / first_required_rank if first_required_rank else 0.0
    )
    for k in k_values:
        for name, value in values_for(documents[:k]).items():
            metrics[f"{name}@{k}"] = value
    return metrics


def _evaluate_items(
    items: Iterable[Any],
    gold_documents: Iterable[Any],
    *,
    aliases: dict[str, list[str]],
    k_values: tuple[int, ...] | None,
) -> dict[str, Any]:
    gold_entries = _gold_document_entries(gold_documents, aliases)
    if not gold_entries:
        raise EvaluationError("At least one gold document is required")

    raw_items = list(items)
    ranked_items, duplicate_count = _rank_items(raw_items, gold_entries)
    documents = _rank_documents(ranked_items)
    return {
        "raw_item_count": len(raw_items),
        "retrieved_item_count": len(ranked_items),
        "duplicate_item_count": duplicate_count,
        "unidentified_item_count": sum(
            not item.get("document_key") for item in ranked_items
        ),
        "retrieved_document_count": len(documents),
        "metrics": _metrics_for_documents(
            documents,
            gold_entries,
            k_values=k_values,
        ),
        "documents": documents,
        "items": ranked_items,
    }


def evaluate_ranked_items(
    items: Iterable[Any],
    gold_documents: Iterable[Any],
    *,
    aliases: dict[str, list[str]] | None = None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Calculate ranked metrics over unique documents, not chunks."""
    normalized_k_values = tuple(sorted(set(k_values)))
    if not normalized_k_values or any(k <= 0 for k in normalized_k_values):
        raise EvaluationError("k_values must contain positive integers")
    return _evaluate_items(
        items,
        gold_documents,
        aliases=aliases or {},
        k_values=normalized_k_values,
    )


def evaluate_document_set(
    items: Iterable[Any],
    gold_documents: Iterable[Any],
    *,
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Calculate unranked document metrics for the final evidence pool."""
    return _evaluate_items(
        items,
        gold_documents,
        aliases=aliases or {},
        k_values=None,
    )


def _retrieved_items_from_result(result: Any) -> list[Any]:
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for field in ("retrieved_items", "chunks", "evidence_store"):
            if isinstance(result.get(field), list):
                return result[field]
    return []


def retrieved_items_from_call(call: dict[str, Any]) -> list[Any]:
    """Support exported calls, raw parsed calls, and JSON-string results."""
    for field in ("retrieved_items", "retrieval_result", "result_parsed"):
        if field in call:
            items = _retrieved_items_from_result(call.get(field))
            if items:
                return items

    for field in ("retrieval_result_raw", "result_raw"):
        raw_result = call.get(field)
        if not isinstance(raw_result, str) or not raw_result.strip():
            continue
        try:
            items = _retrieved_items_from_result(json.loads(raw_result))
        except json.JSONDecodeError:
            continue
        if items:
            return items
    return []


def is_successful_call(call: dict[str, Any]) -> bool:
    status = _string(call.get("status")).casefold()
    if status in FAILURE_STATUSES:
        return False
    if status in SUCCESS_STATUSES:
        return True
    return any(
        field in call
        and (
            isinstance(call[field], list)
            or isinstance(call[field], dict)
            or (isinstance(call[field], str) and call[field].strip())
        )
        for field in (
            "retrieved_items",
            "retrieval_result",
            "result_parsed",
            "retrieval_result_raw",
            "result_raw",
        )
    )


def successful_retrieval_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = record.get("retrieval_calls")
    if not isinstance(calls, list):
        return []

    successful: list[dict[str, Any]] = []
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, dict) or not is_successful_call(call):
            continue
        call_copy = dict(call)
        call_copy["_evaluation_call_index"] = (
            call.get("call_index") if isinstance(call.get("call_index"), int) else index
        )
        call_copy["_evaluation_items"] = retrieved_items_from_call(call)
        successful.append(call_copy)
    return successful


def _is_open_call(call: dict[str, Any]) -> bool:
    tool_name = _string(call.get("tool_name")).casefold()
    return tool_name.startswith("open_") or tool_name.startswith("open-")


def _items_from_calls(calls: Iterable[dict[str, Any]]) -> list[Any]:
    items: list[Any] = []
    for call in calls:
        call_index = call["_evaluation_call_index"]
        source_method = "open" if _is_open_call(call) else "search"
        for position, item in enumerate(call["_evaluation_items"], start=1):
            if not isinstance(item, dict):
                items.append(item)
                continue
            decorated = dict(item)
            decorated["_evaluation_call_index"] = call_index
            item_rank = item.get("rank")
            decorated["_evaluation_local_rank"] = (
                item_rank if isinstance(item_rank, int) else position
            )
            decorated["_evaluation_source_method"] = source_method
            items.append(decorated)
    return items


def _query_summary(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    return {
        "call_index": call["_evaluation_call_index"],
        "query_text": _string(call.get("query_text"))
        or _string(args.get("query_text")),
        "kb_name": _string(call.get("kb_name")) or _string(args.get("kb_name")),
        "returned_item_count": len(call["_evaluation_items"]),
    }


def _normalize_question(value: Any) -> str:
    text = _string(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def evaluate_record(
    gold_record: dict[str, Any],
    result_record: dict[str, Any] | None,
    *,
    row_index: int,
    variant: str,
    aliases: dict[str, list[str]] | None = None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate one gold record against one batch result record."""
    aliases = aliases or {}
    normalized_k_values = tuple(sorted(set(k_values)))
    gold_documents = _gold_documents(gold_record)
    base = {
        "gid": gold_record.get("gid"),
        "case_id": gold_record.get("case_id"),
        "doc_no": gold_record.get("doc_no"),
        "subset": gold_record.get("subset"),
        "row_index": row_index,
        "variant": variant,
        "question": _string(gold_record.get("question")),
        "gold_documents": gold_documents,
        "status": "ok",
        "warnings": [],
    }

    if not gold_documents:
        base["status"] = "annotation_missing"
        base["warnings"].append("no_gold_documents")
        return base

    if result_record is None:
        base["status"] = "missing_result"
        base["warnings"].append("no_result_record")
        return base

    gold_question = _normalize_question(gold_record.get("question"))
    result_question = _normalize_question(result_record.get("question"))
    if gold_question and result_question and gold_question != result_question:
        base["status"] = "alignment_error"
        base["warnings"].append("question_mismatch")
        return base
    if gold_question and not result_question:
        base["warnings"].append("result_question_missing")

    result_status = _string(result_record.get("result_status")).casefold()
    if result_status and result_status not in SUCCESS_STATUSES:
        base["status"] = "result_failed"
        base["warnings"].append(f"result_status:{result_status}")
        return base

    successful_calls = successful_retrieval_calls(result_record)
    search_calls = [call for call in successful_calls if not _is_open_call(call)]
    search_items = _items_from_calls(search_calls)

    final_evidence = result_record.get("retrieved_evidence")
    if isinstance(final_evidence, list):
        final_items = final_evidence
    else:
        final_items = _items_from_calls(successful_calls)

    search_union = evaluate_ranked_items(
        search_items,
        gold_documents,
        aliases=aliases,
        k_values=normalized_k_values,
    )
    final_evidence_pool = evaluate_document_set(
        final_items,
        gold_documents,
        aliases=aliases,
    )
    search_union.pop("items", None)
    final_evidence_pool.pop("items", None)

    base["search_union"] = {
        "call_count": len(search_calls),
        "queries": [_query_summary(call) for call in search_calls],
        **search_union,
    }
    base["final_evidence_pool"] = final_evidence_pool

    if not search_calls:
        base["warnings"].append("no_successful_search_call")
    if not search_items and not final_items:
        base["status"] = "not_retrieved"
        base["warnings"].append("no_retrieved_evidence")
    elif not search_items:
        base["warnings"].append("search_union_empty")
    if not final_items:
        base["warnings"].append("final_evidence_pool_empty")
    return base


def _unique_map(
    pairs: Iterable[tuple[str, int]],
    *,
    description: str,
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    duplicates: set[str] = set()
    for value, index in pairs:
        if not value:
            continue
        if value in mapping:
            duplicates.add(value)
        else:
            mapping[value] = index
    for duplicate in duplicates:
        mapping.pop(duplicate, None)
    if duplicates and description == "case_id":
        raise EvaluationError(f"Duplicate gold case_id: {sorted(duplicates)[0]!r}")
    return mapping


def _align_result_record(
    result_record: dict[str, Any],
    *,
    result_position: int,
    gold_records: list[dict[str, Any]],
    gold_case_ids: dict[str, int],
    gold_questions: dict[str, int],
) -> int | None:
    case_id = _string(result_record.get("case_id"))
    if case_id and case_id in gold_case_ids:
        return gold_case_ids[case_id]

    question = _normalize_question(result_record.get("question"))
    if question and question in gold_questions:
        return gold_questions[question]

    row_index = result_record.get("row_index")
    if isinstance(row_index, int):
        return row_index if 0 <= row_index < len(gold_records) else None

    if not case_id and not question and 0 <= result_position < len(gold_records):
        return result_position
    return None


def _scope_summary(
    details: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    metric_values: defaultdict[str, list[float]] = defaultdict(list)
    matched_total = 0
    gold_total = 0
    for detail in details:
        metrics = detail.get(scope, {}).get("metrics", {})
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values[metric].append(float(value))
        matched_total += int(metrics.get("matched_required_document_count_all", 0))
        gold_total += int(metrics.get("gold_document_count", 0))

    macro_metrics = {
        metric: sum(values) / len(values)
        for metric, values in sorted(metric_values.items())
        if values
    }
    return {
        "macro_metrics": macro_metrics,
        "micro_required_document_recall_all": (
            matched_total / gold_total if gold_total else 0.0
        ),
        "matched_required_document_total": matched_total,
        "gold_document_total": gold_total,
    }


def _summarize_core(
    details: list[dict[str, Any]],
    *,
    condition: str,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    valid_details = [
        detail
        for detail in details
        if detail["status"] in {"ok", "not_retrieved"}
    ]
    warning_counts: defaultdict[str, int] = defaultdict(int)
    for detail in details:
        for warning in detail.get("warnings", []):
            warning_counts[warning] += 1

    return {
        "condition": condition,
        "k_values": list(k_values),
        "total_case_count": len(details),
        "valid_case_count": len(valid_details),
        "status_counts": {
            status: sum(1 for detail in details if detail["status"] == status)
            for status in sorted({detail["status"] for detail in details})
        },
        "search_union": _scope_summary(valid_details, "search_union"),
        "final_evidence_pool": _scope_summary(
            valid_details,
            "final_evidence_pool",
        ),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def summarize_details(
    details: list[dict[str, Any]],
    *,
    condition: str,
    k_values: Iterable[int],
) -> dict[str, Any]:
    k_values_tuple = tuple(sorted(set(k_values)))
    summary = _summarize_core(
        details,
        condition=condition,
        k_values=k_values_tuple,
    )
    subsets = sorted(
        {
            _string(detail.get("subset"))
            for detail in details
            if _string(detail.get("subset"))
        }
    )
    summary["subsets"] = {
        subset: _summarize_core(
            [detail for detail in details if _string(detail.get("subset")) == subset],
            condition=f"{condition}:{subset}",
            k_values=k_values_tuple,
        )
        for subset in subsets
    }
    return summary


def evaluate_dataset(
    gold_records: list[dict[str, Any]],
    result_records: Iterable[dict[str, Any]],
    *,
    condition: str,
    aliases: dict[str, list[str]] | None = None,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one result dataset and return ordered case details and summary."""
    k_values_tuple = tuple(sorted(set(k_values)))
    gold_case_ids = _unique_map(
        (
            (_string(record.get("case_id")), index)
            for index, record in enumerate(gold_records)
        ),
        description="case_id",
    )
    gold_questions = _unique_map(
        (
            (_normalize_question(record.get("question")), index)
            for index, record in enumerate(gold_records)
        ),
        description="question",
    )

    detail_by_gold_index: dict[int, dict[str, Any]] = {}
    unmatched_result_count = 0
    for result_position, result_record in enumerate(result_records):
        gold_index = _align_result_record(
            result_record,
            result_position=result_position,
            gold_records=gold_records,
            gold_case_ids=gold_case_ids,
            gold_questions=gold_questions,
        )
        if gold_index is None:
            unmatched_result_count += 1
            continue
        if gold_index in detail_by_gold_index:
            raise EvaluationError(
                f"Duplicate result record for gold row_index={gold_index}"
            )
        detail_by_gold_index[gold_index] = evaluate_record(
            gold_records[gold_index],
            result_record,
            row_index=gold_index,
            variant=condition,
            aliases=aliases,
            k_values=k_values_tuple,
        )

    details = [
        detail_by_gold_index.get(index)
        or evaluate_record(
            gold_record,
            None,
            row_index=index,
            variant=condition,
            aliases=aliases,
            k_values=k_values_tuple,
        )
        for index, gold_record in enumerate(gold_records)
    ]
    summary = summarize_details(
        details,
        condition=condition,
        k_values=k_values_tuple,
    )
    summary["unmatched_result_record_count"] = unmatched_result_count
    return details, summary


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "condition"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise EvaluationError(f"Cannot write '{path}': {exc}") from exc


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise EvaluationError(f"Cannot write '{path}': {exc}") from exc


def _condition_for_record(record: dict[str, Any], path: Path) -> str:
    return _string(record.get("variant")) or path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate offline document-level retrieval metrics from Yuxi "
            "batch results. No Yuxi or LLM connection is required."
        )
    )
    parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        help="Gold JSON/JSONL with documents or reference citations",
    )
    parser.add_argument(
        "--results",
        required=True,
        nargs="+",
        type=Path,
        help="One or more exported JSON-array or batch JSONL result files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for detail JSONL and summary JSON",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        help="Optional JSON object mapping gold document names to aliases",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Positive unique-document cutoffs (default: 1 3 5 10 20 50)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        k_values = tuple(sorted(set(args.k)))
        if not k_values or any(k <= 0 for k in k_values):
            raise EvaluationError("--k must contain positive integers")

        gold_records = load_records(args.gold)
        aliases = load_aliases(args.aliases)
        summaries: list[dict[str, Any]] = []
        used_filenames: set[str] = set()

        for result_path in args.results:
            result_iterator = iter_records(result_path)
            first_record = next(result_iterator, None)
            condition = (
                _condition_for_record(first_record, result_path)
                if first_record is not None
                else result_path.stem
            )
            records: Iterable[dict[str, Any]] = (
                itertools.chain([first_record], result_iterator)
                if first_record is not None
                else ()
            )

            base_filename = safe_filename(condition)
            filename = base_filename
            suffix = 2
            while filename in used_filenames:
                filename = f"{base_filename}_{suffix}"
                suffix += 1
            used_filenames.add(filename)

            details, summary = evaluate_dataset(
                gold_records,
                records,
                condition=condition,
                aliases=aliases,
                k_values=k_values,
            )
            write_jsonl(args.output_dir / f"{filename}_detail.jsonl", details)
            write_json(args.output_dir / f"{filename}_summary.json", summary)
            summaries.append(summary)
            print(
                f"Evaluated {condition}: {summary['valid_case_count']}/"
                f"{summary['total_case_count']} valid cases"
            )

        write_json(
            args.output_dir / "summary.json",
            {
                "gold_path": str(args.gold),
                "result_paths": [str(path) for path in args.results],
                "k_values": list(k_values),
                "metric_unit": "unique_document",
                "chunk_metrics_enabled": False,
                "conditions": summaries,
            },
        )
        return 0
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
