"""Export answers and RAG evidence calls from a Yuxi batch JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class ExportError(ValueError):
    """Raised when the source result file cannot be exported safely."""


SEARCH_TOOL_NAMES = {
    "query_kb",
    "search_evidence",
    "search_review_kb",
    "search_active_obligation",
}
DOCUMENT_OPEN_TOOL_NAMES = {
    "open_kb_document",
    "open_evidence_source",
    "open_review_evidence",
    "open_active_evidence",
}


def _retrieved_items(result: Any) -> list[Any]:
    """Return the directly rankable retrieval items for vector or LightRAG output."""
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and isinstance(result.get("chunks"), list):
        return result["chunks"]
    return []


def _call_items(result: Any, tool_name: str) -> list[Any]:
    items = _retrieved_items(result)
    if items or tool_name != "open_kb_document" or not isinstance(result, dict):
        return items

    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        return []

    metadata = {
        key: result.get(key)
        for key in (
            "resource_id",
            "file_id",
            "start_line",
            "end_line",
            "total_lines",
            "offset",
            "window_size",
            "has_more_before",
            "has_more_after",
            "next_offset",
        )
        if result.get(key) is not None
    }
    return [{"content": content, "metadata": metadata}]


def _source_method(tool_name: Any) -> str:
    return "open" if str(tool_name or "") in DOCUMENT_OPEN_TOOL_NAMES else "search"


def _compact_call(call: Any, call_index: int) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ExportError(f"retrieval_calls[{call_index}] must be a JSON object")

    args = call.get("args") if isinstance(call.get("args"), dict) else {}
    result = call.get("result_parsed")
    tool_name = str(call.get("tool_name") or "query_kb")
    items = _call_items(result, tool_name)
    return {
        "call_index": call_index + 1,
        "source_method": _source_method(tool_name),
        "tool_name": tool_name,
        "message_id": call.get("message_id"),
        "tool_call_id": call.get("tool_call_id"),
        "kb_name": args.get("kb_name"),
        "query_text": args.get("query_text"),
        "args": args,
        "status": call.get("status"),
        "error_message": call.get("error_message"),
        "retrieved_item_count": len(items),
        "retrieved_items": items,
        "retrieval_result": result,
        "retrieval_result_raw": call.get("result_raw"),
    }


def _trace_v2_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "2.0":
        return []
    records = trace.get("search_records")
    evidence_values = trace.get("evidence")
    if not isinstance(records, list) or not isinstance(evidence_values, list):
        return []
    evidence = {
        item.get("evidence_id"): item for item in evidence_values if isinstance(item, dict) and item.get("evidence_id")
    }
    assessments = {
        item.get("evidence_id"): item
        for item in trace.get("evidence_assessments") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    calls: list[dict[str, Any]] = []
    for call_index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            continue
        intent = record.get("intent")
        if not isinstance(intent, dict):
            intent = {}
        candidate_ids = record.get("candidate_evidence_ids")
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        active_id_list = list(record.get("active_evidence_ids") or [])
        active_ids = set(active_id_list)
        items = []
        for evidence_id in candidate_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            occurrences = item.get("occurrences")
            rank = None
            if isinstance(occurrences, list):
                occurrence = next(
                    (
                        value
                        for value in occurrences
                        if isinstance(value, dict) and value.get("query_id") == intent.get("query_id")
                    ),
                    None,
                )
                if occurrence:
                    rank = occurrence.get("rank")
            items.append(
                {
                    **item,
                    "rank": rank,
                    "assessment": assessments.get(evidence_id),
                    "active": evidence_id in active_ids,
                }
            )
        items.sort(
            key=lambda item: (
                item.get("rank") is None,
                item.get("rank") or 0,
                str(item.get("evidence_id") or ""),
            )
        )
        calls.append(
            {
                "call_index": call_index,
                "tool_name": "search_evidence",
                "message_id": None,
                "tool_call_id": None,
                "kb_name": (trace.get("knowledge_base_snapshot") or {}).get("name"),
                "query_text": intent.get("query_text"),
                "args": {
                    "query_text": intent.get("query_text"),
                    "target_element_ids": intent.get("target_element_ids") or [],
                    "target_review_ids": intent.get("target_review_ids") or [],
                    "evidence_role": intent.get("intended_evidence_role"),
                    "search_reason": intent.get("search_reason"),
                },
                "status": record.get("status"),
                "error_message": None,
                "retrieved_item_count": len(items),
                "retrieved_items": items,
                "retrieval_result": {
                    "candidate_evidence_ids": candidate_ids,
                    "active_evidence_ids": active_id_list,
                    "new_requirements_closed": (record.get("new_requirements_closed") or []),
                },
                "retrieval_result_raw": None,
            }
        )
    return calls


def _trace_v3_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "3.0":
        return []
    records = trace.get("search_records")
    evidence_values = trace.get("evidence")
    if not isinstance(records, list) or not isinstance(evidence_values, list):
        return []
    evidence = {
        item.get("evidence_id"): item for item in evidence_values if isinstance(item, dict) and item.get("evidence_id")
    }
    selected_ids = set((trace.get("evidence_selection") or {}).get("selected_evidence_ids") or [])
    claims_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for claim in trace.get("evidence_claims") or []:
        if not isinstance(claim, dict) or not claim.get("evidence_id"):
            continue
        claims_by_evidence.setdefault(str(claim["evidence_id"]), []).append(claim)

    calls: list[dict[str, Any]] = []
    for call_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        subquery = record.get("subquery")
        if not isinstance(subquery, dict):
            subquery = {}
        evidence_ids = record.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        items: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            rank = None
            for occurrence in item.get("occurrences") or []:
                if isinstance(occurrence, dict) and occurrence.get("query_id") == subquery.get("query_id"):
                    rank = occurrence.get("rank")
                    break
            items.append(
                {
                    **item,
                    "rank": rank,
                    "selected": evidence_id in selected_ids,
                    "claims": claims_by_evidence.get(str(evidence_id), []),
                }
            )
        items.sort(
            key=lambda item: (
                item.get("rank") is None,
                item.get("rank") or 0,
                str(item.get("evidence_id") or ""),
            )
        )
        calls.append(
            {
                "call_index": call_index,
                "tool_name": "search_evidence",
                "message_id": None,
                "tool_call_id": None,
                "kb_name": (trace.get("knowledge_base_snapshot") or {}).get("name"),
                "query_text": subquery.get("query_text"),
                "args": {
                    "query_text": subquery.get("query_text"),
                    "query_id": subquery.get("query_id"),
                    "linked_question_ids": subquery.get("linked_question_ids") or [],
                    "linked_element_ids": subquery.get("linked_element_ids") or [],
                    "linked_patient_fact_ids": (subquery.get("linked_patient_fact_ids") or []),
                    "search_reason": subquery.get("search_reason"),
                },
                "status": record.get("status"),
                "error_message": record.get("error_message"),
                "retrieved_item_count": len(items),
                "retrieved_items": items,
                "retrieval_result": {
                    "evidence_ids": evidence_ids,
                    "new_evidence_ids": record.get("new_evidence_ids") or [],
                    "duplicate_ratio": record.get("duplicate_ratio"),
                },
                "retrieval_result_raw": None,
            }
        )
    return calls


def _trace_v4_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "4.0":
        return []
    evidence_values = trace.get("evidence_store")
    if not isinstance(evidence_values, list):
        return []
    evidence = {
        str(item.get("evidence_id")): item
        for item in evidence_values
        if isinstance(item, dict) and item.get("evidence_id")
    }
    cited_ids = set(trace.get("cited_evidence_ids") or [])
    knowledge_name = (trace.get("knowledge_base_snapshot") or {}).get("name")
    pending: list[tuple[str, dict[str, Any]]] = []

    for record in trace.get("search_records") or []:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("record_id") or "")
        evidence_ids = [str(value) for value in record.get("evidence_ids") or []]
        items: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            occurrence = next(
                (
                    value
                    for value in item.get("occurrences") or []
                    if isinstance(value, dict) and value.get("record_id") == record_id
                ),
                {},
            )
            items.append(
                {
                    **item,
                    "rank": occurrence.get("rank"),
                    "score": occurrence.get("score"),
                    "distance": occurrence.get("distance"),
                    "shown_excerpt": occurrence.get("shown_excerpt"),
                    "excerpt_start": occurrence.get("excerpt_start"),
                    "excerpt_end": occurrence.get("excerpt_end"),
                    "excerpt_fallback": occurrence.get("excerpt_fallback"),
                    "cited_in_answer": evidence_id in cited_ids,
                }
            )
        pending.append(
            (
                str(record.get("started_at") or ""),
                {
                    "tool_name": "search_review_kb",
                    "message_id": None,
                    "tool_call_id": record.get("tool_call_id"),
                    "kb_name": knowledge_name,
                    "query_text": record.get("query_text"),
                    "args": {
                        "query_text": record.get("query_text"),
                        "reason": record.get("reason"),
                        "focus_element_ids": (record.get("focus_element_ids") or []),
                    },
                    "status": record.get("status"),
                    "error_message": record.get("error_message"),
                    "retrieved_item_count": len(items),
                    "retrieved_items": items,
                    "retrieval_result": {
                        "evidence_ids": evidence_ids,
                        "new_evidence_ids": (record.get("new_evidence_ids") or []),
                        "attempts": record.get("attempts") or [],
                    },
                    "retrieval_result_raw": None,
                },
            )
        )

    for record in trace.get("open_records") or []:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("record_id") or "")
        evidence_ids = [str(value) for value in record.get("evidence_ids") or []]
        items: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            occurrence = next(
                (
                    value
                    for value in item.get("occurrences") or []
                    if isinstance(value, dict) and value.get("record_id") == record_id
                ),
                {},
            )
            items.append(
                {
                    **item,
                    "rank": occurrence.get("rank"),
                    "shown_excerpt": occurrence.get("shown_excerpt"),
                    "excerpt_start": occurrence.get("excerpt_start"),
                    "excerpt_end": occurrence.get("excerpt_end"),
                    "excerpt_fallback": occurrence.get("excerpt_fallback"),
                    "cited_in_answer": evidence_id in cited_ids,
                }
            )
        pending.append(
            (
                str(record.get("started_at") or ""),
                {
                    "tool_name": "open_review_evidence",
                    "message_id": None,
                    "tool_call_id": record.get("tool_call_id"),
                    "kb_name": knowledge_name,
                    "query_text": None,
                    "args": {
                        "evidence_id": record.get("parent_evidence_id"),
                        "reason": record.get("reason"),
                        "window_before": record.get("window_before"),
                        "window_after": record.get("window_after"),
                    },
                    "status": record.get("status"),
                    "error_message": record.get("error_message"),
                    "retrieved_item_count": len(items),
                    "retrieved_items": items,
                    "retrieval_result": {
                        "evidence_ids": evidence_ids,
                        "new_evidence_ids": (record.get("new_evidence_ids") or []),
                        "attempts": record.get("attempts") or [],
                    },
                    "retrieval_result_raw": None,
                },
            )
        )

    calls: list[dict[str, Any]] = []
    for call_index, (_started_at, call) in enumerate(
        sorted(
            pending,
            key=lambda value: (
                value[0],
                str(value[1].get("tool_call_id") or ""),
            ),
        ),
        start=1,
    ):
        calls.append({"call_index": call_index, **call})
    return calls


def _trace_v5_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "5.0":
        return []
    adapted = {
        **trace,
        "schema_version": "4.0",
        "search_records": [
            {
                **record,
                "record_id": record.get("query_id"),
                "focus_element_ids": record.get("focus_plan_ids") or [],
            }
            for record in trace.get("query_records") or []
            if isinstance(record, dict)
        ],
    }
    calls = _trace_v4_calls(adapted)
    query_by_call = {
        str(record.get("tool_call_id") or ""): record
        for record in trace.get("query_records") or []
        if isinstance(record, dict)
    }
    for call in calls:
        record = query_by_call.get(str(call.get("tool_call_id") or ""))
        if record is None or call.get("tool_name") != "search_review_kb":
            continue
        call["args"] = {
            "query_text": record.get("query_text"),
            "reason": record.get("reason"),
            "focus_plan_ids": record.get("focus_plan_ids") or [],
            "focus_modifier_ids": record.get("focus_modifier_ids") or [],
            "relation_id": record.get("relation_id"),
        }
        call["query_id"] = record.get("query_id")
        call["relation_id"] = record.get("relation_id")
    return calls


def _trace_v6_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "6.0":
        return []
    adapted = {**trace, "schema_version": "5.0"}
    calls = _trace_v5_calls(adapted)
    routed_by_query = {
        str(record.get("query_id") or ""): record
        for record in trace.get("routed_retrieval_records") or []
        if isinstance(record, dict) and record.get("query_id")
    }
    for call in calls:
        tool_name = call.get("tool_name")
        if tool_name == "open_review_evidence":
            open_record = next(
                (
                    record
                    for record in trace.get("open_records") or []
                    if isinstance(record, dict)
                    and str(record.get("tool_call_id") or "")
                    == str(call.get("tool_call_id") or "")
                ),
                None,
            )
            if open_record is not None:
                args = dict(call.get("args") or {})
                args["investigation_id"] = open_record.get(
                    "investigation_id"
                )
                call["args"] = args
                call["investigation_id"] = open_record.get(
                    "investigation_id"
                )
            continue
        if tool_name != "search_review_kb":
            continue
        routed = routed_by_query.get(str(call.get("query_id") or ""))
        if routed is None:
            continue
        call["routed_retrieval"] = routed
        args = dict(call.get("args") or {})
        args.pop("relation_id", None)
        args["opportunity_id"] = routed.get("opportunity_id")
        call["args"] = args
        call.pop("relation_id", None)
    return calls


def _trace_v7_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "7.0":
        return []
    adapted = {**trace, "schema_version": "5.0"}
    calls = _trace_v5_calls(adapted)
    query_by_call = {
        str(record.get("tool_call_id") or ""): record
        for record in trace.get("query_records") or []
        if isinstance(record, dict)
    }
    for call in calls:
        if call.get("tool_name") != "search_review_kb":
            continue
        record = query_by_call.get(str(call.get("tool_call_id") or ""))
        if record is None:
            continue
        args = dict(call.get("args") or {})
        args["atlas_companion_ids"] = (
            record.get("atlas_companion_ids") or []
        )
        call["args"] = args
        call["atlas_companion_ids"] = (
            record.get("atlas_companion_ids") or []
        )
        call["invalid_atlas_companion_ids"] = (
            record.get("invalid_atlas_companion_ids") or []
        )
        call["rejected_atlas_companion_ids"] = (
            record.get("rejected_atlas_companion_ids") or []
        )
    return calls


def _trace_v8_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "8.0":
        return []
    adapted = {**trace, "schema_version": "5.0"}
    calls = _trace_v5_calls(adapted)
    query_by_call = {
        str(record.get("tool_call_id") or ""): record
        for record in trace.get("query_records") or []
        if isinstance(record, dict)
    }
    for call in calls:
        if call.get("tool_name") != "search_review_kb":
            continue
        record = query_by_call.get(str(call.get("tool_call_id") or ""))
        if record is None:
            continue
        args = dict(call.get("args") or {})
        args.update(
            {
                "retrieval_scope": record.get("retrieval_scope", "global"),
                "file_id": record.get("file_id"),
                "investigation_id": record.get("investigation_id"),
            }
        )
        call["args"] = args
        call["query_id"] = record.get("query_id")
        call["investigation_id"] = record.get("investigation_id")
        call["retrieval_scope"] = record.get("retrieval_scope", "global")
        call["file_id"] = record.get("file_id")
    return calls


def _trace_v9_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "9.0":
        return []
    adapted = {**trace, "schema_version": "8.0"}
    calls = _trace_v8_calls(adapted)
    query_by_call = {
        str(record.get("tool_call_id") or ""): record
        for record in trace.get("query_records") or []
        if isinstance(record, dict)
    }
    for call in calls:
        if call.get("tool_name") != "search_review_kb":
            continue
        record = query_by_call.get(str(call.get("tool_call_id") or ""))
        if record is None:
            continue
        atlas_ids = record.get("atlas_companion_ids") or []
        args = dict(call.get("args") or {})
        args["atlas_companion_ids"] = atlas_ids
        call["args"] = args
        call["atlas_companion_ids"] = atlas_ids
        call["invalid_atlas_companion_ids"] = (
            record.get("invalid_atlas_companion_ids") or []
        )
        call["rejected_atlas_companion_ids"] = (
            record.get("rejected_atlas_companion_ids") or []
        )
    return calls


def _trace_v10_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "10.0":
        return []
    adapted = {**trace, "schema_version": "8.0"}
    return _trace_v8_calls(adapted)


def _trace_v11_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "11.0":
        return []
    adapted = {**trace, "schema_version": "8.0"}
    return _trace_v8_calls(adapted)


def _trace_v12_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "12.0":
        return []
    adapted = {**trace, "schema_version": "8.0"}
    calls = _trace_v8_calls(adapted)
    probe_by_query = {
        str(record.get("query_id") or ""): record
        for record in trace.get("probe_records") or []
        if isinstance(record, dict)
    }
    for call in calls:
        if call.get("tool_name") != "search_review_kb":
            continue
        probe = probe_by_query.get(str(call.get("query_id") or ""))
        if probe is None:
            continue
        args = dict(call.get("args") or {})
        args["retrieval_intent"] = probe.get("retrieval_intent")
        args["uncovered_aspect"] = probe.get("uncovered_aspect")
        call["args"] = args
        call["retrieval_intent"] = probe.get("retrieval_intent")
        call["uncovered_aspect"] = probe.get("uncovered_aspect")
        call["route_key"] = probe.get("route_key")
        call["redundant"] = bool(probe.get("redundant"))
    return calls


def _trace_v13_calls(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, dict) or trace.get("schema_version") != "13.0":
        return []
    calls = _trace_v12_calls({**trace, "schema_version": "12.0"})
    outcomes = {
        str(value.get("call_id") or ""): value
        for value in trace.get("tool_outcomes") or []
        if isinstance(value, dict) and value.get("call_id")
    }
    for call in calls:
        outcome = outcomes.get(str(call.get("tool_call_id") or ""))
        if outcome is None:
            continue
        bounded_name = str(outcome.get("tool_name") or "")
        if bounded_name in SEARCH_TOOL_NAMES | DOCUMENT_OPEN_TOOL_NAMES:
            call["tool_name"] = bounded_name
            call["source_method"] = _source_method(bounded_name)
        call["transport_status"] = outcome.get("transport_status")
        call["semantic_outcome"] = outcome.get("semantic_outcome")
        call["reason_code"] = outcome.get("reason_code")
        call["state_changed"] = bool(outcome.get("state_changed"))
    return calls


def _compact_record(source: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ExportError(f"Line {line_number} must contain a JSON object")

    question = source.get("question")
    response = source.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ExportError(f"Line {line_number} has no non-empty string 'question' field")
    if not isinstance(response, str):
        raise ExportError(f"Line {line_number} has no string 'answer' field")

    calls = source.get("retrieval_calls", [])
    if not isinstance(calls, list):
        raise ExportError(f"Line {line_number} field 'retrieval_calls' must be a list")

    document_open_calls = source.get("document_open_calls", [])
    if not isinstance(document_open_calls, list):
        raise ExportError(
            f"Line {line_number} field 'document_open_calls' must be a list"
        )

    all_tool_calls = source.get("all_tool_calls")
    if all_tool_calls is not None and not isinstance(all_tool_calls, list):
        raise ExportError(
            f"Line {line_number} field 'all_tool_calls' must be a list"
        )
    rag_source_calls = (
        [
            call
            for call in all_tool_calls
            if isinstance(call, dict)
            and str(call.get("tool_name") or "")
            in SEARCH_TOOL_NAMES | DOCUMENT_OPEN_TOOL_NAMES
        ]
        if isinstance(all_tool_calls, list)
        else []
    )
    if not rag_source_calls:
        rag_source_calls = [*calls, *document_open_calls]
    compact_calls = [
        _compact_call(call, call_index)
        for call_index, call in enumerate(rag_source_calls)
    ]
    trace = source.get("medication_review_trace")
    trace_calls = (
        _trace_v13_calls(trace)
        or _trace_v12_calls(trace)
        or _trace_v11_calls(trace)
        or _trace_v10_calls(trace)
        or _trace_v9_calls(trace)
        or _trace_v8_calls(trace)
        or _trace_v7_calls(trace)
        or _trace_v6_calls(trace)
        or _trace_v5_calls(trace)
        or _trace_v4_calls(trace)
        or _trace_v3_calls(trace)
        or _trace_v2_calls(trace)
    )
    if trace_calls:
        compact_calls = trace_calls
        existing_call_ids = {
            str(call.get("tool_call_id") or "")
            for call in compact_calls
            if call.get("tool_call_id")
        }
        trace_open_counts: dict[str, int] = {}
        for call in compact_calls:
            tool_name = str(call.get("tool_name") or "")
            if tool_name in DOCUMENT_OPEN_TOOL_NAMES:
                trace_open_counts[tool_name] = trace_open_counts.get(tool_name, 0) + 1
        source_open_counts: dict[str, int] = {}
        for call in rag_source_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or "")
            tool_call_id = str(call.get("tool_call_id") or "")
            if tool_name not in DOCUMENT_OPEN_TOOL_NAMES:
                continue
            source_open_counts[tool_name] = source_open_counts.get(tool_name, 0) + 1
            if tool_call_id and tool_call_id in existing_call_ids:
                continue
            if source_open_counts[tool_name] <= trace_open_counts.get(tool_name, 0):
                continue
            compact_calls.append(_compact_call(call, len(compact_calls)))
    for call_index, call in enumerate(compact_calls, start=1):
        call["call_index"] = call_index
        call["source_method"] = _source_method(call.get("tool_name"))

    search_calls = [
        call for call in compact_calls if call["source_method"] == "search"
    ]
    exported_open_calls = [
        call for call in compact_calls if call["source_method"] == "open"
    ]
    searched_evidence = [
        item for call in search_calls for item in call.get("retrieved_items") or []
    ]
    opened_evidence = [
        item
        for call in exported_open_calls
        for item in call.get("retrieved_items") or []
    ]
    result = {
        "question": question,
        "response": response,
        "retrieval_status": source.get("retrieval_status", "unknown"),
        "retrieval_calls": compact_calls,
        "search_calls": search_calls,
        "document_open_calls": exported_open_calls,
        "searched_evidence": searched_evidence,
        "opened_evidence": opened_evidence,
    }

    # These fields make it possible to join vector and LightRAG exports and to
    # calculate document-level recall without reopening the original JSONL.
    for field in (
        "job_key",
        "batch_id",
        "row_index",
        "variant",
        "attempt",
        "result_status",
        "run_status",
        "trace_schema_version",
        "run_mode",
        "agenda_mode",
        "synthesis_mode",
        "effective_profile",
        "method_family",
        "method_version",
        "experiment_profile",
        "atlas_profile",
        "atlas_snapshot_hash",
        "acm_protocol",
        "adaptive_coverage_status",
        "bounded_model_call_count",
        "bounded_generation_abort_count",
        "bounded_citation_verification_status",
        "bounded_max_projected_total_tokens",
        "v7_experiment_arm",
        "v7_retrieval_depth",
        "v7_contract_status",
    ):
        if field in source:
            result[field] = source[field]

    input_record = source.get("input_record")
    if isinstance(input_record, dict):
        if "case_id" in input_record:
            result["case_id"] = input_record["case_id"]
        if "documents" in input_record:
            result["reference_documents"] = input_record["documents"]

    if isinstance(trace, dict) and trace.get("schema_version") in {
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10.0",
        "11.0",
        "12.0",
        "13.0",
    }:
        result["method_family"] = trace.get("method_family")
        result["method_version"] = trace.get("method_version")
        result["experiment_profile"] = trace.get(
            "experiment_profile",
            trace.get("requested_profile"),
        )
        result["review_status"] = trace.get("run_status")
        result["plan_anchors"] = trace.get("plan_anchors") or []
        result["patient_modifiers"] = trace.get("patient_modifiers") or []
        result["search_records"] = trace.get("search_records") or trace.get("query_records") or []
        result["relation_investigations"] = trace.get("relation_investigations") or []
        result["investigations"] = trace.get("investigations") or []
        result["deferred_knowledge_calls"] = (
            trace.get("deferred_knowledge_calls") or []
        )
        result["open_records"] = trace.get("open_records") or []
        result["retrieved_evidence"] = trace.get("evidence_store") or []
        result["coverage_report"] = trace.get("coverage_report") or {}
        result["reflection_report"] = trace.get("reflection_report") or {}
        if trace.get("schema_version") == "6.0":
            atlas_snapshot = trace.get("atlas_snapshot") or {}
            result["atlas_profile"] = trace.get("atlas_profile")
            result["atlas_snapshot"] = atlas_snapshot
            result["atlas_snapshot_hash"] = atlas_snapshot.get("snapshot_hash")
            result["case_route_record"] = trace.get("case_route_record") or {}
            result["retrieval_opportunities"] = trace.get("retrieval_opportunities") or []
            result["adopted_opportunity_ids"] = trace.get("adopted_opportunity_ids") or []
            result["routed_retrieval_records"] = trace.get("routed_retrieval_records") or []
        if trace.get("schema_version") in {"7.0", "9.0"}:
            atlas_snapshot = trace.get("atlas_snapshot") or {}
            selection = trace.get("companion_selection") or {}
            selector = (
                selection.get("selector_audit")
                if isinstance(selection, dict)
                and isinstance(selection.get("selector_audit"), dict)
                else {}
            )
            result["atlas_snapshot"] = atlas_snapshot
            result["atlas_snapshot_hash"] = atlas_snapshot.get(
                "snapshot_hash"
            )
            result["companion_selector"] = selector
            result["companion_selection"] = selection
            result["companion_cues"] = (
                selection.get("companion_cues") or []
                if isinstance(selection, dict)
                else []
            )
            result["companion_adoption_events"] = (
                trace.get("companion_adoption_events") or []
            )
            result["remaining_companion_ids"] = (
                trace.get("remaining_companion_ids") or []
            )
            reflection = trace.get("reflection_report") or {}
            result["companion_reflection"] = {
                "triggered": (
                    reflection.get("triggered")
                    if isinstance(reflection, dict)
                    else False
                ),
                "trigger_reason": (
                    reflection.get("trigger_reason")
                    if isinstance(reflection, dict)
                    else None
                ),
                "cue_ids": trace.get("companion_cues_at_reflection") or [],
            }
        if trace.get("schema_version") in {"10.0", "11.0", "12.0", "13.0"}:
            atlas_snapshot = trace.get("atlas_snapshot") or {}
            result["atlas_snapshot"] = atlas_snapshot
            result["atlas_snapshot_hash"] = atlas_snapshot.get(
                "snapshot_hash"
            )
            result["atlas_document_open_records"] = (
                trace.get("atlas_document_open_records") or []
            )
        if trace.get("schema_version") == "11.0":
            result["v7_experiment_arm"] = trace.get("experiment_arm")
            result["v7_retrieval_depth"] = trace.get("retrieval_depth")
            result["v7_contract_report"] = trace.get("contract_report") or {}
            result["v7_investigation_agenda"] = (
                trace.get("investigation_agenda") or {}
            )
            result["v7_probe_records"] = trace.get("probe_records") or []
            result["v7_retrieval_records"] = (
                trace.get("retrieval_records") or []
            )
            result["v7_checkpoint_records"] = (
                trace.get("checkpoint_records") or []
            )
        if trace.get("schema_version") in {"12.0", "13.0"}:
            result["acm_protocol"] = trace.get("protocol")
            result["v7_retrieval_depth"] = trace.get("retrieval_depth")
            result["adaptive_coverage_report"] = (
                trace.get("adaptive_coverage_report") or {}
            )
            result["adaptive_coverage_status"] = result[
                "adaptive_coverage_report"
            ].get("status")
            result["adaptive_investigation_agenda"] = (
                trace.get("investigation_agenda") or {}
            )
            result["adaptive_investigation_meta"] = (
                trace.get("investigation_meta") or []
            )
            result["adaptive_probe_records"] = trace.get("probe_records") or []
            result["adaptive_recovery_requirements"] = (
                trace.get("recovery_requirements") or []
            )
            result["adaptive_gap_assessments"] = (
                trace.get("gap_assessments") or []
            )
            result["adaptive_retrieval_records"] = (
                trace.get("retrieval_records") or []
            )
            result["adaptive_checkpoint_records"] = (
                trace.get("checkpoint_records") or []
            )
        if trace.get("schema_version") == "13.0":
            result["bounded_directives"] = trace.get("directives") or []
            result["bounded_context_manifests"] = (
                trace.get("context_manifests") or []
            )
            result["bounded_context_atoms"] = trace.get("context_atoms") or []
            result["bounded_tool_outcomes"] = trace.get("tool_outcomes") or []
            result["bounded_action_attempts"] = trace.get("action_attempts") or []
            result["bounded_obligation_judgments"] = (
                trace.get("obligation_judgments") or []
            )
            result["bounded_generation_abort_records"] = (
                trace.get("generation_abort_records") or []
            )
            result["bounded_controller_version"] = trace.get(
                "controller_version"
            )
            result["bounded_context_view_version"] = trace.get(
                "context_view_version"
            )
            result["bounded_model_context_window_tokens"] = trace.get(
                "model_context_window_tokens"
            )
            result["bounded_provider_context_window_tokens"] = trace.get(
                "provider_context_window_tokens"
            )
            result["bounded_context_window_verified"] = trace.get(
                "context_window_verified"
            )
            result["bounded_citation_verification"] = (
                trace.get("citation_verification") or {}
            )

    if isinstance(trace, dict) and trace.get("schema_version") in {"2.0", "3.0"}:
        result["open_records"] = trace.get("open_records") or []
        result["retrieved_evidence"] = trace.get("evidence") or []

    if "retrieved_evidence" not in result:
        result["retrieved_evidence"] = [
            *searched_evidence,
            *opened_evidence,
        ]

    return result


def export_rag_records(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Export a JSON list and return (record_count, call_count, item_count)."""
    if input_path.resolve() == output_path.resolve():
        raise ExportError("--input and --output cannot point to the same file")

    records: list[dict[str, Any]] = []
    call_count = 0
    item_count = 0

    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExportError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc

                record = _compact_record(source, line_number)
                records.append(record)
                calls = record["retrieval_calls"]
                call_count += len(calls)
                item_count += sum(call["retrieved_item_count"] for call in calls)
    except OSError as exc:
        raise ExportError(f"Cannot read input file '{input_path}': {exc}") from exc

    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ExportError(f"Cannot write output file '{output_path}': {exc}") from exc

    return len(records), call_count, item_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Yuxi batch result JSONL file into a JSON list containing "
            "answers, search results, and original-document open results."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record_count, call_count, item_count = export_rag_records(args.input, args.output)
    except ExportError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {record_count} records, {call_count} retrieval calls, "
        f"{item_count} rankable items to: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
