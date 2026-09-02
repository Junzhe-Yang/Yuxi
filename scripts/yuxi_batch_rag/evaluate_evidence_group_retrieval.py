# -*- coding: utf-8 -*-
"""Evaluate V2 evidence-group retrieval with direct Yuxi chunk IDs.

The V2 gold format stores claims with one or more acceptable evidence groups.
Groups under the same claim are alternatives (OR), while every chunk inside one
group is required together (AND).  Gold chunk IDs are compared directly with
the IDs exported from Yuxi; no manual-to-automatic mapping or fuzzy quote
alignment is performed.

Both chunk-level and document-level claim coverage are reported.  Search calls
are evaluated at a fixed per-call cutoff, while evidence obtained by opening a
document is only included in the separate final-evidence-pool view.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


try:  # Direct CLI execution finds the sibling module on sys.path.
    import evaluate_document_retrieval as document_eval
except ImportError:  # Import-by-path tests do not add the script directory.
    _DOCUMENT_EVAL_PATH = Path(__file__).with_name("evaluate_document_retrieval.py")
    _DOCUMENT_EVAL_SPEC = importlib.util.spec_from_file_location(
        "yuxi_document_retrieval_for_evidence_groups",
        _DOCUMENT_EVAL_PATH,
    )
    if _DOCUMENT_EVAL_SPEC is None or _DOCUMENT_EVAL_SPEC.loader is None:
        raise
    document_eval = importlib.util.module_from_spec(_DOCUMENT_EVAL_SPEC)
    _DOCUMENT_EVAL_SPEC.loader.exec_module(document_eval)


SCOPES = ("all", "core", "supporting")
MODES = ("as_delivered", "snapshot_resolvable")
UNITS = ("chunk", "document")
POOLS = ("search", "final_evidence_pool")
EPSILON = 1e-12


class EvidenceGroupEvaluationError(ValueError):
    """Raised when an input cannot be evaluated without guessing."""


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise EvidenceGroupEvaluationError(f"Cannot read '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGroupEvaluationError(f"Invalid JSON in '{path}': {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceGroupEvaluationError(f"Cannot hash '{path}': {exc}") from exc
    return digest.hexdigest()


def _normalize_question(value: Any) -> str:
    text = _string(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(text.split())


def load_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    """Load a frozen JSONL chunk snapshot and fail on duplicate IDs."""
    snapshot: dict[str, dict[str, Any]] = {}
    for row_index, record in enumerate(document_eval.iter_records(path)):
        chunk_id = _string(record.get("chunk_id"))
        file_id = _string(record.get("file_id"))
        if not chunk_id or not file_id:
            raise EvidenceGroupEvaluationError(
                f"Snapshot row {row_index} lacks chunk_id or file_id"
            )
        if chunk_id in snapshot:
            raise EvidenceGroupEvaluationError(f"Duplicate snapshot chunk_id: {chunk_id}")
        snapshot[chunk_id] = {
            "chunk_id": chunk_id,
            "file_id": file_id,
            "filename": _string(record.get("filename"))
            or _string(record.get("source")),
            "content_sha256": _string(record.get("content_sha256")),
        }
    if not snapshot:
        raise EvidenceGroupEvaluationError("Snapshot contains no chunks")
    return snapshot


def _gold_group(
    raw_group: dict[str, Any],
    *,
    claim_id: str,
    group_index: int,
    snapshot: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_chunks = raw_group.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise EvidenceGroupEvaluationError(
            f"Claim {claim_id} group {group_index} contains no chunks"
        )

    chunk_ids: list[str] = []
    document_ids: list[str] = []
    seen_chunks: set[str] = set()
    seen_documents: set[str] = set()
    for chunk_index, raw_chunk in enumerate(raw_chunks):
        if not isinstance(raw_chunk, dict):
            raise EvidenceGroupEvaluationError(
                f"Claim {claim_id} group {group_index} chunk {chunk_index} is not an object"
            )
        chunk_id = _string(raw_chunk.get("chunk_id"))
        if not chunk_id:
            raise EvidenceGroupEvaluationError(
                f"Claim {claim_id} group {group_index} chunk {chunk_index} lacks chunk_id"
            )
        file_id = _string(raw_chunk.get("file_id"))
        if not file_id and chunk_id in snapshot:
            file_id = snapshot[chunk_id]["file_id"]
        if not file_id:
            raise EvidenceGroupEvaluationError(
                f"Gold chunk {chunk_id} lacks a file_id"
            )
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            chunk_ids.append(chunk_id)
        if file_id not in seen_documents:
            seen_documents.add(file_id)
            document_ids.append(file_id)

    missing_chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in snapshot]
    snapshot_file_ids = {record["file_id"] for record in snapshot.values()}
    missing_document_ids = [
        file_id for file_id in document_ids if file_id not in snapshot_file_ids
    ]
    return {
        "group_id": _string(raw_group.get("group_id"))
        or f"{claim_id}-group-{group_index + 1}",
        "chunk_ids": chunk_ids,
        "document_ids": document_ids,
        "support": _string(raw_group.get("support")),
        "origin": _string(raw_group.get("origin")),
        "approval": _string(raw_group.get("approval")),
        "missing_snapshot_chunk_ids": missing_chunk_ids,
        "missing_snapshot_document_ids": missing_document_ids,
        "chunk_resolvable": not missing_chunk_ids,
        "document_resolvable": not missing_document_ids,
    }


def load_gold_cases(
    path: Path,
    *,
    snapshot: Mapping[str, dict[str, Any]],
    gids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """Load selected cases from the V2 case package."""
    payload = _read_json(path)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise EvidenceGroupEvaluationError("V2 gold must contain a top-level cases list")

    selected_gids = set(gids or [])
    cases: list[dict[str, Any]] = []
    seen_gids: set[int] = set()
    seen_questions: set[str] = set()
    for case_index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise EvidenceGroupEvaluationError(f"Gold case {case_index} is not an object")
        gid = raw_case.get("gid")
        if not isinstance(gid, int):
            raise EvidenceGroupEvaluationError(f"Gold case {case_index} lacks integer gid")
        if selected_gids and gid not in selected_gids:
            continue
        if gid in seen_gids:
            raise EvidenceGroupEvaluationError(f"Duplicate selected gid: {gid}")
        question = _string(raw_case.get("question"))
        normalized_question = _normalize_question(question)
        if not normalized_question:
            raise EvidenceGroupEvaluationError(f"Gold gid={gid} has no question")
        if normalized_question in seen_questions:
            raise EvidenceGroupEvaluationError(
                f"Duplicate selected gold question at gid={gid}"
            )

        raw_claims = raw_case.get("claims")
        if not isinstance(raw_claims, list):
            raise EvidenceGroupEvaluationError(f"Gold gid={gid} has no claims list")
        claims: list[dict[str, Any]] = []
        seen_claim_ids: set[str] = set()
        for claim_index, raw_claim in enumerate(raw_claims):
            if not isinstance(raw_claim, dict):
                raise EvidenceGroupEvaluationError(
                    f"Gold gid={gid} claim {claim_index} is not an object"
                )
            raw_groups = raw_claim.get("evidence_groups")
            if not isinstance(raw_groups, list) or not raw_groups:
                continue
            claim_id = _string(raw_claim.get("claim_id"))
            level = _string(raw_claim.get("claim_level"))
            if not claim_id or level not in {"core", "supporting"}:
                raise EvidenceGroupEvaluationError(
                    f"Gold gid={gid} claim {claim_index} has invalid id or level"
                )
            if claim_id in seen_claim_ids:
                raise EvidenceGroupEvaluationError(
                    f"Gold gid={gid} contains duplicate claim_id {claim_id}"
                )
            seen_claim_ids.add(claim_id)
            groups = [
                _gold_group(
                    group,
                    claim_id=claim_id,
                    group_index=group_index,
                    snapshot=snapshot,
                )
                for group_index, group in enumerate(raw_groups)
                if isinstance(group, dict)
            ]
            if len(groups) != len(raw_groups):
                raise EvidenceGroupEvaluationError(
                    f"Gold gid={gid} claim {claim_id} contains a non-object group"
                )
            claims.append(
                {
                    "claim_id": claim_id,
                    "claim_text": _string(raw_claim.get("claim_text")),
                    "claim_level": level,
                    "groups": groups,
                }
            )

        seen_gids.add(gid)
        seen_questions.add(normalized_question)
        cases.append(
            {
                "gid": gid,
                "subset": _string(raw_case.get("subset")) or "unknown",
                "doc_no": _string(raw_case.get("doc_no")),
                "question": question,
                "normalized_question": normalized_question,
                "claims": claims,
                "all_claim_count": len(raw_claims),
                "claims_without_evidence_groups": len(raw_claims) - len(claims),
            }
        )

    if selected_gids:
        missing_gids = sorted(selected_gids - seen_gids)
        if missing_gids:
            raise EvidenceGroupEvaluationError(
                f"Selected gid(s) missing from gold: {missing_gids}"
            )
    if not cases:
        raise EvidenceGroupEvaluationError("No gold cases selected")
    return cases


def gold_audit(
    cases: Sequence[dict[str, Any]],
    *,
    snapshot: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    groups = [
        group
        for case in cases
        for claim in case["claims"]
        for group in claim["groups"]
    ]
    chunk_ids = [chunk_id for group in groups for chunk_id in group["chunk_ids"]]
    missing_occurrences = [
        {
            "gid": case["gid"],
            "claim_id": claim["claim_id"],
            "claim_level": claim["claim_level"],
            "group_id": group["group_id"],
            "chunk_id": chunk_id,
        }
        for case in cases
        for claim in case["claims"]
        for group in claim["groups"]
        for chunk_id in group["missing_snapshot_chunk_ids"]
    ]
    unresolvable_claims = [
        {
            "gid": case["gid"],
            "claim_id": claim["claim_id"],
            "claim_level": claim["claim_level"],
        }
        for case in cases
        for claim in case["claims"]
        if not any(group["chunk_resolvable"] for group in claim["groups"])
    ]
    return {
        "selected_case_count": len(cases),
        "selected_gids": [case["gid"] for case in cases],
        "all_claim_count": sum(case["all_claim_count"] for case in cases),
        "evaluable_claim_count_as_delivered": sum(len(case["claims"]) for case in cases),
        "core_claim_count_as_delivered": sum(
            claim["claim_level"] == "core"
            for case in cases
            for claim in case["claims"]
        ),
        "supporting_claim_count_as_delivered": sum(
            claim["claim_level"] == "supporting"
            for case in cases
            for claim in case["claims"]
        ),
        "claims_without_evidence_groups": sum(
            case["claims_without_evidence_groups"] for case in cases
        ),
        "evidence_group_count": len(groups),
        "chunk_id_occurrence_count": len(chunk_ids),
        "unique_gold_chunk_id_count": len(set(chunk_ids)),
        "snapshot_chunk_count": len(snapshot),
        "missing_snapshot_chunk_id_occurrence_count": len(missing_occurrences),
        "missing_snapshot_chunk_ids": sorted(
            {row["chunk_id"] for row in missing_occurrences}
        ),
        "missing_snapshot_occurrences": missing_occurrences,
        "chunk_unresolvable_claim_count": len(unresolvable_claims),
        "chunk_unresolvable_claims": unresolvable_claims,
        "support_labels": dict(
            sorted(Counter(group["support"] or "missing" for group in groups).items())
        ),
        "approval_labels": dict(
            sorted(Counter(group["approval"] or "missing" for group in groups).items())
        ),
    }


def _item_identity(
    item: Any,
    snapshot: Mapping[str, dict[str, Any]],
) -> tuple[str, str]:
    identity = document_eval.extract_item_identity(item)
    chunk_id = _string(identity.get("chunk_id"))
    file_id = _string(identity.get("file_id"))
    if not file_id and chunk_id in snapshot:
        file_id = snapshot[chunk_id]["file_id"]
    return chunk_id, file_id


def extract_search_calls(
    record: dict[str, Any],
    *,
    snapshot: Mapping[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in document_eval.successful_retrieval_calls(record):
        if document_eval._is_open_call(call):
            continue
        items: list[dict[str, Any]] = []
        for local_rank, raw_item in enumerate(
            call.get("_evaluation_items", [])[:top_k], start=1
        ):
            chunk_id, file_id = _item_identity(raw_item, snapshot)
            items.append(
                {
                    "local_rank": local_rank,
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "known_snapshot_chunk": chunk_id in snapshot if chunk_id else False,
                }
            )
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        calls.append(
            {
                "call_position": len(calls) + 1,
                "call_index": call.get("_evaluation_call_index"),
                "tool_name": _string(call.get("tool_name")),
                "query_text": _string(call.get("query_text"))
                or _string(args.get("query_text")),
                "returned_item_count_before_top_k": len(
                    call.get("_evaluation_items", [])
                ),
                "items": items,
            }
        )
    return calls


def extract_final_evidence_pool(
    record: dict[str, Any],
    *,
    snapshot: Mapping[str, dict[str, Any]],
) -> dict[str, set[str]]:
    raw_items = record.get("retrieved_evidence")
    if not isinstance(raw_items, list):
        raw_items = [
            item
            for call in document_eval.successful_retrieval_calls(record)
            for item in call.get("_evaluation_items", [])
        ]
    chunk_ids: set[str] = set()
    document_ids: set[str] = set()
    for item in raw_items:
        chunk_id, file_id = _item_identity(item, snapshot)
        if chunk_id:
            chunk_ids.add(chunk_id)
        if file_id:
            document_ids.add(file_id)
    return {"chunk": chunk_ids, "document": document_ids}


def _claims_for_scope(
    case: dict[str, Any],
    *,
    unit: str,
    mode: str,
    scope: str,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    resolvable_key = f"{unit}_resolvable"
    for claim in case["claims"]:
        if scope != "all" and claim["claim_level"] != scope:
            continue
        groups = claim["groups"]
        if mode == "snapshot_resolvable":
            groups = [group for group in groups if group[resolvable_key]]
        if not groups:
            continue
        claims.append({**claim, "groups": groups})
    return claims


def _targets(group: dict[str, Any], unit: str) -> set[str]:
    return set(group[f"{unit}_ids"])


def _retrieved_item_field(unit: str) -> str:
    return "chunk_id" if unit == "chunk" else "file_id"


def _claim_is_covered(
    claim: dict[str, Any],
    retrieved: set[str],
    *,
    unit: str,
) -> bool:
    return any(_targets(group, unit) <= retrieved for group in claim["groups"])


def _best_rank_by_target(
    calls: Sequence[dict[str, Any]],
    *,
    unit: str,
) -> dict[str, int]:
    field = _retrieved_item_field(unit)
    best: dict[str, int] = {}
    for call in calls:
        for item in call["items"]:
            target = _string(item.get(field))
            if not target:
                continue
            rank = int(item["local_rank"])
            best[target] = min(rank, best.get(target, sys.maxsize))
    return best


def _claim_best_rank(
    claim: dict[str, Any],
    best_rank_by_target: Mapping[str, int],
    *,
    unit: str,
) -> int | None:
    group_ranks: list[int] = []
    for group in claim["groups"]:
        targets = _targets(group, unit)
        if not targets or not targets <= set(best_rank_by_target):
            continue
        group_ranks.append(max(best_rank_by_target[target] for target in targets))
    return min(group_ranks) if group_ranks else None


def _scope_metrics(
    claims: Sequence[dict[str, Any]],
    calls: Sequence[dict[str, Any]],
    *,
    unit: str,
) -> dict[str, Any]:
    if not claims:
        return {"applicable": False, "gold_claim_count": 0}

    field = _retrieved_item_field(unit)
    cumulative: set[str] = set()
    trajectory: list[float] = []
    productive_call_count = 0
    first_hit_call: int | None = None
    full_coverage_call: int | None = None
    previous_covered = 0
    for call_position, call in enumerate(calls, start=1):
        cumulative.update(
            _string(item.get(field)) for item in call["items"] if item.get(field)
        )
        covered = sum(
            _claim_is_covered(claim, cumulative, unit=unit) for claim in claims
        )
        if covered > previous_covered:
            productive_call_count += 1
            if first_hit_call is None:
                first_hit_call = call_position
        if covered == len(claims) and full_coverage_call is None:
            full_coverage_call = call_position
        previous_covered = covered
        trajectory.append(covered / len(claims))

    best_rank_by_target = _best_rank_by_target(calls, unit=unit)
    best_rank_by_claim = {
        claim["claim_id"]: _claim_best_rank(
            claim, best_rank_by_target, unit=unit
        )
        for claim in claims
    }
    covered_claim_count = sum(rank is not None for rank in best_rank_by_claim.values())
    reciprocal_gains = [
        1.0 / rank if rank is not None else 0.0
        for rank in best_rank_by_claim.values()
    ]
    ndcg_gains = [
        1.0 / math.log2(rank + 1) if rank is not None else 0.0
        for rank in best_rank_by_claim.values()
    ]
    call_count = len(calls)
    return {
        "applicable": True,
        "gold_claim_count": len(claims),
        "gold_group_count": sum(len(claim["groups"]) for claim in claims),
        "covered_claim_count": covered_claim_count,
        "claim_coverage": covered_claim_count / len(claims),
        "hit": int(covered_claim_count > 0),
        "complete_coverage": int(covered_claim_count == len(claims)),
        "claim_mrr_at_k": statistics.fmean(reciprocal_gains),
        "claim_ndcg_at_k": statistics.fmean(ndcg_gains),
        "best_local_rank_by_claim": best_rank_by_claim,
        "coverage_trajectory": trajectory,
        "autonomous_coverage_auc": (
            statistics.fmean(trajectory) if trajectory else 0.0
        ),
        "productive_call_count": productive_call_count,
        "productive_call_rate": (
            productive_call_count / call_count if call_count else 0.0
        ),
        "calls_to_first_hit": first_hit_call,
        "calls_to_full_coverage": full_coverage_call,
    }


def _final_scope_metrics(
    claims: Sequence[dict[str, Any]],
    retrieved: set[str],
    *,
    unit: str,
) -> dict[str, Any]:
    if not claims:
        return {"applicable": False, "gold_claim_count": 0}
    coverage = {
        claim["claim_id"]: _claim_is_covered(claim, retrieved, unit=unit)
        for claim in claims
    }
    covered = sum(coverage.values())
    return {
        "applicable": True,
        "gold_claim_count": len(claims),
        "gold_group_count": sum(len(claim["groups"]) for claim in claims),
        "covered_claim_count": covered,
        "claim_coverage": covered / len(claims),
        "hit": int(covered > 0),
        "complete_coverage": int(covered == len(claims)),
        "covered_by_claim": coverage,
    }


def evaluate_case(
    case: dict[str, Any],
    record: dict[str, Any],
    *,
    system: str,
    snapshot: Mapping[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    calls = extract_search_calls(record, snapshot=snapshot, top_k=top_k)
    final_pool = extract_final_evidence_pool(record, snapshot=snapshot)
    source_result_status = _string(record.get("result_status")).casefold() or "missing"
    warnings = []
    if source_result_status not in document_eval.SUCCESS_STATUSES:
        warnings.append(f"source_result_status:{source_result_status}")
    if not calls:
        warnings.append("no_successful_search_call")
    all_items = [item for call in calls for item in call["items"]]
    identified_chunk_ids = [item["chunk_id"] for item in all_items if item["chunk_id"]]
    unknown_chunk_ids = sorted(
        {
            chunk_id
            for chunk_id in identified_chunk_ids
            if chunk_id not in snapshot
        }
    )
    metrics: dict[str, Any] = {}
    for unit in UNITS:
        metrics[unit] = {}
        for mode in MODES:
            search_scopes: dict[str, Any] = {}
            final_scopes: dict[str, Any] = {}
            for scope in SCOPES:
                claims = _claims_for_scope(
                    case, unit=unit, mode=mode, scope=scope
                )
                search_scopes[scope] = _scope_metrics(claims, calls, unit=unit)
                final_scopes[scope] = _final_scope_metrics(
                    claims, final_pool[unit], unit=unit
                )
            metrics[unit][mode] = {
                "search": search_scopes,
                "final_evidence_pool": final_scopes,
            }

    unique_chunk_ids = set(identified_chunk_ids)
    return {
        "system": system,
        "status": "retrieval_trace_evaluable" if calls else "retrieval_trace_empty",
        "source_result_status": source_result_status,
        "source_retrieval_status": _string(record.get("retrieval_status"))
        or "missing",
        "warnings": warnings,
        "gid": case["gid"],
        "subset": case["subset"],
        "doc_no": case["doc_no"],
        "question": case["question"],
        "result_row_index": record.get("row_index"),
        "top_k_per_search_call": top_k,
        "retrieval_efficiency": {
            "search_call_count": len(calls),
            "returned_item_count_at_k": len(all_items),
            "identified_chunk_position_count": len(identified_chunk_ids),
            "unique_retrieved_chunk_count": len(unique_chunk_ids),
            "duplicate_chunk_position_rate": (
                1.0 - len(unique_chunk_ids) / len(identified_chunk_ids)
                if identified_chunk_ids
                else 0.0
            ),
            "unknown_snapshot_chunk_ids": unknown_chunk_ids,
        },
        "calls": calls,
        "final_evidence_pool": {
            "unique_chunk_count": len(final_pool["chunk"]),
            "unique_document_count": len(final_pool["document"]),
        },
        "metrics": metrics,
    }


def _metric_at(
    detail: dict[str, Any],
    *,
    unit: str,
    mode: str,
    pool: str,
    scope: str,
) -> dict[str, Any]:
    return detail["metrics"][unit][mode][pool][scope]


def _aggregate_scope(
    details: Sequence[dict[str, Any]],
    *,
    unit: str,
    mode: str,
    pool: str,
    scope: str,
) -> dict[str, Any]:
    rows = [
        _metric_at(
            detail, unit=unit, mode=mode, pool=pool, scope=scope
        )
        for detail in details
    ]
    rows = [row for row in rows if row.get("applicable")]
    if not rows:
        return {"applicable_case_count": 0}
    gold_count = sum(int(row["gold_claim_count"]) for row in rows)
    covered_count = sum(int(row["covered_claim_count"]) for row in rows)
    aggregate: dict[str, Any] = {
        "applicable_case_count": len(rows),
        "gold_claim_count": gold_count,
        "covered_claim_count": covered_count,
        "micro_claim_coverage": covered_count / gold_count if gold_count else None,
        "macro_claim_coverage": statistics.fmean(
            float(row["claim_coverage"]) for row in rows
        ),
        "hit_rate": statistics.fmean(float(row["hit"]) for row in rows),
        "complete_coverage_rate": statistics.fmean(
            float(row["complete_coverage"]) for row in rows
        ),
    }
    if pool == "search":
        for metric in (
            "claim_mrr_at_k",
            "claim_ndcg_at_k",
            "autonomous_coverage_auc",
            "productive_call_rate",
        ):
            aggregate[f"macro_{metric}"] = statistics.fmean(
                float(row[metric]) for row in rows
            )
    return aggregate


def summarize_system(
    system: str,
    details: Sequence[dict[str, Any]],
    *,
    selected_cases: Sequence[dict[str, Any]],
    scanned_record_count: int,
    unmatched_record_count: int,
) -> dict[str, Any]:
    matched_gids = {detail["gid"] for detail in details}
    summary: dict[str, Any] = {
        "system": system,
        "scanned_record_count": scanned_record_count,
        "matched_case_count": len(details),
        "matched_gids": [detail["gid"] for detail in details],
        "missing_result_gids": [
            case["gid"] for case in selected_cases if case["gid"] not in matched_gids
        ],
        "unmatched_result_record_count": unmatched_record_count,
        "source_result_status_counts": dict(
            sorted(Counter(detail["source_result_status"] for detail in details).items())
        ),
        "retrieval_trace_status_counts": dict(
            sorted(Counter(detail["status"] for detail in details).items())
        ),
        "retrieval_efficiency": {
            "search_call_count_total": sum(
                detail["retrieval_efficiency"]["search_call_count"]
                for detail in details
            ),
            "mean_search_call_count": (
                statistics.fmean(
                    detail["retrieval_efficiency"]["search_call_count"]
                    for detail in details
                )
                if details
                else None
            ),
            "returned_item_count_at_k_total": sum(
                detail["retrieval_efficiency"]["returned_item_count_at_k"]
                for detail in details
            ),
            "mean_duplicate_chunk_position_rate": (
                statistics.fmean(
                    detail["retrieval_efficiency"]["duplicate_chunk_position_rate"]
                    for detail in details
                )
                if details
                else None
            ),
            "unknown_snapshot_chunk_ids": sorted(
                {
                    chunk_id
                    for detail in details
                    for chunk_id in detail["retrieval_efficiency"][
                        "unknown_snapshot_chunk_ids"
                    ]
                }
            ),
        },
        "metrics": {},
    }
    for unit in UNITS:
        summary["metrics"][unit] = {}
        for mode in MODES:
            summary["metrics"][unit][mode] = {}
            for pool in POOLS:
                summary["metrics"][unit][mode][pool] = {
                    scope: _aggregate_scope(
                        details,
                        unit=unit,
                        mode=mode,
                        pool=pool,
                        scope=scope,
                    )
                    for scope in SCOPES
                }
    return summary


def evaluate_system(
    system: str,
    path: Path,
    *,
    cases: Sequence[dict[str, Any]],
    snapshot: Mapping[str, dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases_by_question = {case["normalized_question"]: case for case in cases}
    details_by_gid: dict[int, dict[str, Any]] = {}
    scanned = 0
    unmatched = 0
    for record in document_eval.iter_records(path):
        scanned += 1
        normalized_question = _normalize_question(record.get("question"))
        case = cases_by_question.get(normalized_question)
        if case is None:
            unmatched += 1
            continue
        if case["gid"] in details_by_gid:
            raise EvidenceGroupEvaluationError(
                f"{system} contains duplicate result for gid={case['gid']}"
            )
        details_by_gid[case["gid"]] = evaluate_case(
            case,
            record,
            system=system,
            snapshot=snapshot,
            top_k=top_k,
        )
    details = [
        details_by_gid[case["gid"]]
        for case in cases
        if case["gid"] in details_by_gid
    ]
    return details, summarize_system(
        system,
        details,
        selected_cases=cases,
        scanned_record_count=scanned,
        unmatched_record_count=unmatched,
    )


def _paired_values(
    primary: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    accessor: Any,
) -> dict[str, Any]:
    primary_by_gid = {detail["gid"]: detail for detail in primary}
    baseline_by_gid = {detail["gid"]: detail for detail in baseline}
    gids = sorted(set(primary_by_gid) & set(baseline_by_gid))
    pairs = [
        (
            float(accessor(primary_by_gid[gid])),
            float(accessor(baseline_by_gid[gid])),
        )
        for gid in gids
    ]
    differences = [left - right for left, right in pairs]
    return {
        "pair_count": len(pairs),
        "primary_mean": statistics.fmean(left for left, _ in pairs) if pairs else None,
        "baseline_mean": statistics.fmean(right for _, right in pairs) if pairs else None,
        "primary_minus_baseline": statistics.fmean(differences) if pairs else None,
        "primary_wins": sum(value > EPSILON for value in differences),
        "ties": sum(abs(value) <= EPSILON for value in differences),
        "primary_losses": sum(value < -EPSILON for value in differences),
        "per_gid": [
            {
                "gid": gid,
                "primary": pairs[index][0],
                "baseline": pairs[index][1],
                "difference": differences[index],
            }
            for index, gid in enumerate(gids)
        ],
    }


def paired_comparison(
    primary_name: str,
    primary: Sequence[dict[str, Any]],
    baseline_name: str,
    baseline: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    primary_by_gid = {detail["gid"]: detail for detail in primary}
    baseline_by_gid = {detail["gid"]: detail for detail in baseline}
    paired_gids = sorted(set(primary_by_gid) & set(baseline_by_gid))
    comparison: dict[str, Any] = {
        "primary_system": primary_name,
        "baseline_system": baseline_name,
        "paired_gids": paired_gids,
        "pair_count": len(paired_gids),
        "inference_policy": "descriptive_only_no_significance_claim",
        "metrics": {},
        "retrieval_efficiency": {},
    }
    for unit in UNITS:
        comparison["metrics"][unit] = {}
        for mode in MODES:
            comparison["metrics"][unit][mode] = {}
            for pool in POOLS:
                comparison["metrics"][unit][mode][pool] = {}
                for scope in SCOPES:
                    metrics = ["claim_coverage"]
                    if pool == "search":
                        metrics.extend(
                            [
                                "claim_mrr_at_k",
                                "claim_ndcg_at_k",
                                "autonomous_coverage_auc",
                                "productive_call_rate",
                            ]
                        )
                    comparison["metrics"][unit][mode][pool][scope] = {
                        metric: _paired_values(
                            primary,
                            baseline,
                            lambda detail, metric=metric, unit=unit, mode=mode,
                            pool=pool, scope=scope: _metric_at(
                                detail,
                                unit=unit,
                                mode=mode,
                                pool=pool,
                                scope=scope,
                            )[metric],
                        )
                        for metric in metrics
                        if all(
                            _metric_at(
                                detail,
                                unit=unit,
                                mode=mode,
                                pool=pool,
                                scope=scope,
                            ).get("applicable")
                            for detail in [*primary, *baseline]
                            if detail["gid"] in paired_gids
                        )
                    }
    for metric in (
        "search_call_count",
        "returned_item_count_at_k",
        "duplicate_chunk_position_rate",
        "unique_retrieved_chunk_count",
    ):
        comparison["retrieval_efficiency"][metric] = _paired_values(
            primary,
            baseline,
            lambda detail, metric=metric: detail["retrieval_efficiency"][metric],
        )
    return comparison


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(
    path: Path,
    system_summaries: Mapping[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for system, summary in system_summaries.items():
        for unit in UNITS:
            for mode in MODES:
                for pool in POOLS:
                    for scope in SCOPES:
                        metrics = summary["metrics"][unit][mode][pool][scope]
                        rows.append(
                            {
                                "system": system,
                                "unit": unit,
                                "mode": mode,
                                "pool": pool,
                                "scope": scope,
                                **metrics,
                            }
                        )
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentage(value: Any) -> str:
    return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else "—"


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    details: Mapping[str, Sequence[dict[str, Any]]],
) -> None:
    audit = summary["gold_audit"]
    lines = [
        "# V2 直接块 ID 检索评估",
        "",
        "本报告直接比较 V2 金标 `chunk_id` 与 RAG 返回的 `chunk_id`，不使用旧的块映射 JSON，也不做 quote 模糊映射。",
        "",
        "## 输入与金标审计",
        "",
        f"- 评估病例：{audit['selected_case_count']} 例，gid={audit['selected_gids']}。",
        f"- 有证据组命题：{audit['evaluable_claim_count_as_delivered']} 条，其中 core {audit['core_claim_count_as_delivered']} 条。",
        f"- 证据组：{audit['evidence_group_count']} 个；块 ID 引用：{audit['chunk_id_occurrence_count']} 个。",
        f"- 冻结快照：{audit['snapshot_chunk_count']} 块。",
        "- 快照缺失金标 ID："
        f"{audit['missing_snapshot_chunk_id_occurrence_count']} 个，影响 "
        f"{audit['chunk_unresolvable_claim_count']} 条命题。",
        "",
    ]
    if audit["missing_snapshot_occurrences"]:
        lines.extend(
            [
                "这些 ID 不做补映射；`as_delivered` 结果按原样计分，`snapshot_resolvable` 结果排除没有完整可解析证据组的命题：",
                "",
                "| gid | claim | 层级 | group | 缺失 chunk_id |",
                "|---:|---|---|---|---|",
            ]
        )
        for row in audit["missing_snapshot_occurrences"]:
            lines.append(
                f"| {row['gid']} | {row['claim_id']} | {row['claim_level']} | {row['group_id']} | `{row['chunk_id']}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## 系统汇总",
            "",
            "下表主视图为搜索阶段每次调用 "
            f"Top-{summary['protocol']['per_search_call_cutoff']} 的命题覆盖，按病例宏平均。"
            "块级同时报告原样金标和快照可解析敏感性；文档级按 evidence group "
            "的 file_id 直接计算。",
            "",
            "顶层 `result_status` 仅作为来源运行状态审计；只要存在成功的搜索调用，其检索轨迹仍独立计分。",
            "各系统匹配病例数不同时，系统汇总不可直接横比；跨系统结论以共同问题的配对比较为准。",
            "",
            "| 系统 | 匹配病例 | 来源 failed | 块 All（原样） | 块 All（可解析） | 块 Core（原样） | 文档 All | 平均搜索调用 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system, system_summary in summary["systems"].items():
        metrics = system_summary["metrics"]
        strict_all = metrics["chunk"]["as_delivered"]["search"]["all"]
        resolvable_all = metrics["chunk"]["snapshot_resolvable"]["search"]["all"]
        strict_core = metrics["chunk"]["as_delivered"]["search"]["core"]
        document_all = metrics["document"]["as_delivered"]["search"]["all"]
        lines.append(
            "| "
            + " | ".join(
                [
                    system,
                    str(system_summary["matched_case_count"]),
                    str(system_summary["source_result_status_counts"].get("failed", 0)),
                    _percentage(strict_all.get("macro_claim_coverage")),
                    _percentage(resolvable_all.get("macro_claim_coverage")),
                    _percentage(strict_core.get("macro_claim_coverage")),
                    _percentage(document_all.get("macro_claim_coverage")),
                    f"{system_summary['retrieval_efficiency']['mean_search_call_count']:.2f}",
                ]
            )
            + " |"
        )
    lines.append("")

    primary_name = summary.get("paired_comparison", {}).get("primary_system")
    if primary_name and primary_name in details:
        lines.extend(
            [
                f"## {primary_name} 逐例结果",
                "",
                "| gid | 子集 | 块 All 原样 | 块 All 可解析 | 块 Core 原样 | 最终池块 All | 文档 All | 搜索调用 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for detail in details[primary_name]:
            metrics = detail["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(detail["gid"]),
                        detail["subset"],
                        _percentage(
                            metrics["chunk"]["as_delivered"]["search"]["all"].get(
                                "claim_coverage"
                            )
                        ),
                        _percentage(
                            metrics["chunk"]["snapshot_resolvable"]["search"][
                                "all"
                            ].get("claim_coverage")
                        ),
                        _percentage(
                            metrics["chunk"]["as_delivered"]["search"]["core"].get(
                                "claim_coverage"
                            )
                        ),
                        _percentage(
                            metrics["chunk"]["as_delivered"]["final_evidence_pool"][
                                "all"
                            ].get("claim_coverage")
                        ),
                        _percentage(
                            metrics["document"]["as_delivered"]["search"]["all"].get(
                                "claim_coverage"
                            )
                        ),
                        str(detail["retrieval_efficiency"]["search_call_count"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    paired = summary.get("paired_comparison")
    if paired:
        lines.extend(
            [
                f"## {paired['pair_count']} 例配对比较",
                "",
                f"配对 gid={paired['paired_gids']}。样本量只有 {paired['pair_count']}，以下仅作描述性比较。",
                "",
                f"| 指标 | {paired['primary_system']} | {paired['baseline_system']} | 差值 | 胜/平/负 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        paired_rows = [
            (
                "块 All 覆盖（原样）",
                paired["metrics"]["chunk"]["as_delivered"]["search"]["all"][
                    "claim_coverage"
                ],
            ),
            (
                "块 Core 覆盖（原样）",
                paired["metrics"]["chunk"]["as_delivered"]["search"]["core"][
                    "claim_coverage"
                ],
            ),
            (
                f"块 All MRR@{summary['protocol']['per_search_call_cutoff']}（原样）",
                paired["metrics"]["chunk"]["as_delivered"]["search"]["all"][
                    "claim_mrr_at_k"
                ],
            ),
            (
                "文档 All 覆盖",
                paired["metrics"]["document"]["as_delivered"]["search"]["all"][
                    "claim_coverage"
                ],
            ),
        ]
        for label, values in paired_rows:
            result_counts = (
                f"{values['primary_wins']}/{values['ties']}/"
                f"{values['primary_losses']}"
            )
            lines.append(
                f"| {label} | {_percentage(values['primary_mean'])} | "
                f"{_percentage(values['baseline_mean'])} | "
                f"{_percentage(values['primary_minus_baseline'])} | "
                f"{result_counts} |"
            )
        lines.append("")

        paired_all = paired_rows[0][1]
        paired_core = paired_rows[1][1]
        all_by_gid = {row["gid"]: row for row in paired_all["per_gid"]}
        core_by_gid = {row["gid"]: row for row in paired_core["per_gid"]}
        calls_by_gid = {
            row["gid"]: row
            for row in paired["retrieval_efficiency"]["search_call_count"][
                "per_gid"
            ]
        }
        lines.extend(
            [
                "### 逐例配对",
                "",
                f"| gid | {paired['primary_system']} 块 All | "
                f"{paired['baseline_system']} 块 All | All 差值 | Core 差值 | "
                "搜索调用（前者/后者） |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for gid in paired["paired_gids"]:
            all_values = all_by_gid[gid]
            core_values = core_by_gid[gid]
            call_values = calls_by_gid[gid]
            lines.append(
                f"| {gid} | {_percentage(all_values['primary'])} | "
                f"{_percentage(all_values['baseline'])} | "
                f"{_percentage(all_values['difference'])} | "
                f"{_percentage(core_values['difference'])} | "
                f"{call_values['primary']:.0f}/{call_values['baseline']:.0f} |"
            )
        lines.append("")

        coverage_auc = paired["metrics"]["chunk"]["as_delivered"]["search"][
            "all"
        ]["autonomous_coverage_auc"]
        productive_rate = paired["metrics"]["chunk"]["as_delivered"][
            "search"
        ]["all"]["productive_call_rate"]
        efficiency = paired["retrieval_efficiency"]
        lines.extend(
            [
                "### 检索进展与成本",
                "",
                "MRR 取跨调用最佳局部排名，不惩罚额外调用；因此同时查看覆盖 AUC、"
                "有效调用率、调用数和重复块位置率。",
                "",
                f"| 指标 | {paired['primary_system']} | "
                f"{paired['baseline_system']} | 差值 |",
                "|---|---:|---:|---:|",
                "| 块 All 覆盖 AUC（原样） | "
                f"{_percentage(coverage_auc['primary_mean'])} | "
                f"{_percentage(coverage_auc['baseline_mean'])} | "
                f"{_percentage(coverage_auc['primary_minus_baseline'])} |",
                "| 有效搜索调用率（原样） | "
                f"{_percentage(productive_rate['primary_mean'])} | "
                f"{_percentage(productive_rate['baseline_mean'])} | "
                f"{_percentage(productive_rate['primary_minus_baseline'])} |",
                "| 平均搜索调用数 | "
                f"{efficiency['search_call_count']['primary_mean']:.2f} | "
                f"{efficiency['search_call_count']['baseline_mean']:.2f} | "
                f"{efficiency['search_call_count']['primary_minus_baseline']:+.2f} |",
                "| 平均 Top-K 返回位置数 | "
                f"{efficiency['returned_item_count_at_k']['primary_mean']:.2f} | "
                f"{efficiency['returned_item_count_at_k']['baseline_mean']:.2f} | "
                f"{efficiency['returned_item_count_at_k']['primary_minus_baseline']:+.2f} |",
                "| 重复块位置率 | "
                f"{_percentage(efficiency['duplicate_chunk_position_rate']['primary_mean'])} | "
                f"{_percentage(efficiency['duplicate_chunk_position_rate']['baseline_mean'])} | "
                f"{_percentage(efficiency['duplicate_chunk_position_rate']['primary_minus_baseline'])} |",
                "",
            ]
        )

    lines.extend(
        [
            "## 解释边界",
            "",
            "- 命中 evidence group 代表检索覆盖了金标证据义务，不等价于最终答案正确使用了证据。",
            "- 未进入金标的检索块是 unjudged，因此本报告不计算 precision/F1。",
            "- 没有共同问题的病例只进入各系统绝对结果，不进入配对比较。",
            "- 来源记录顶层失败不等于检索调用失败；本报告保留来源状态，并仅从成功的搜索调用提取检索结果。",
            "- `snapshot_resolvable` 不是块映射，而是对交付中不存在于冻结快照的直接 ID 做透明的敏感性分析。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("--result must use non-empty NAME=PATH")
    return name, Path(raw_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--result",
        required=True,
        action="append",
        type=_parse_result,
        help="Repeatable system input in NAME=PATH form",
    )
    parser.add_argument("--gids", required=True, nargs="+", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pair", nargs=2, metavar=("PRIMARY", "BASELINE"))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run_evaluation(
    *,
    gold_path: Path,
    snapshot_path: Path,
    result_paths: Sequence[tuple[str, Path]],
    gids: Sequence[int],
    top_k: int,
    output_dir: Path,
    pair: tuple[str, str] | None = None,
) -> dict[str, Any]:
    if top_k <= 0:
        raise EvidenceGroupEvaluationError("top_k must be positive")
    names = [name for name, _ in result_paths]
    if len(names) != len(set(names)):
        raise EvidenceGroupEvaluationError("Result system names must be unique")
    if pair and (pair[0] not in names or pair[1] not in names):
        raise EvidenceGroupEvaluationError("--pair names must exist in --result")

    snapshot = load_snapshot(snapshot_path)
    cases = load_gold_cases(gold_path, snapshot=snapshot, gids=gids)
    audit = gold_audit(cases, snapshot=snapshot)
    details_by_system: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, result_path in result_paths:
        details, system_summary = evaluate_system(
            name,
            result_path,
            cases=cases,
            snapshot=snapshot,
            top_k=top_k,
        )
        details_by_system[name] = details
        summaries[name] = system_summary

    paired = None
    if pair:
        paired = paired_comparison(
            pair[0],
            details_by_system[pair[0]],
            pair[1],
            details_by_system[pair[1]],
        )

    summary = {
        "schema_version": "1.0",
        "protocol": {
            "gold_format": "v2_claim_evidence_groups",
            "chunk_alignment": "direct_chunk_id_no_mapping_no_fuzzy_quote_match",
            "group_semantics": "OR_between_groups_AND_within_group",
            "per_search_call_cutoff": top_k,
            "search_calls": "successful_non_open_calls_only",
            "final_evidence_pool": "retrieved_evidence_including_opened_evidence",
            "source_result_status_policy": (
                "audit_only; successful retrieval calls are scored independently"
            ),
            "unannotated_retrieval_items": "unjudged",
            "aggregation": {
                "report_primary": "macro_mean_over_cases",
                "micro_claim_coverage": (
                    "covered claims summed over cases / gold claims summed over cases"
                ),
            },
            "metric_definitions": {
                "claim_coverage": (
                    "covered claims / evaluable claims; a claim is covered when any "
                    "complete evidence group is retrieved"
                ),
                "claim_mrr_at_k": (
                    "mean over claims of 1 / best local rank across search calls; "
                    "for a multi-target group use its worst target rank; missing=0"
                ),
                "claim_ndcg_at_k": (
                    "mean over claims of 1/log2(best local rank+1); missing=0"
                ),
                "autonomous_coverage_auc": (
                    "mean cumulative claim coverage after each successful search call"
                ),
                "productive_call_rate": (
                    "search calls that add at least one newly covered claim / search calls"
                ),
            },
        },
        "inputs": {
            "gold": {"path": str(gold_path), "sha256": _sha256(gold_path)},
            "snapshot": {
                "path": str(snapshot_path),
                "sha256": _sha256(snapshot_path),
            },
            "results": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in result_paths
            },
        },
        "gold_audit": audit,
        "systems": summaries,
        "paired_comparison": paired,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in details_by_system.items():
        safe_name = document_eval.safe_filename(name)
        _write_jsonl(output_dir / f"{safe_name}_detail.jsonl", rows)
    _write_json(output_dir / "summary.json", summary)
    _write_summary_csv(output_dir / "summary.csv", summaries)
    write_report(
        output_dir / "REPORT.md",
        summary=summary,
        details=details_by_system,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = run_evaluation(
            gold_path=args.gold,
            snapshot_path=args.snapshot,
            result_paths=args.result,
            gids=args.gids,
            top_k=args.top_k,
            output_dir=args.output_dir,
            pair=tuple(args.pair) if args.pair else None,
        )
        print(
            "Evaluated "
            + ", ".join(
                f"{name}={value['matched_case_count']}"
                for name, value in summary["systems"].items()
            )
        )
        return 0
    except EvidenceGroupEvaluationError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
