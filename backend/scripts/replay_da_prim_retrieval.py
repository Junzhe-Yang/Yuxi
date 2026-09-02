"""Replay historical PRIM search queries through DA-PRIM retrieval.

This script does not invoke an LLM.  It rebuilds the case route from exported
PlanAnchor/PatientModifier records, runs the current routed retrieval strategy,
and compares its candidate pools with a flat vector Top-22 pool.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from yuxi import knowledge_base
from yuxi.agents.buildin.medication_review_da_prim.context import (
    MedicationReviewDaPrimContext,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas import (
    AtlasStore,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.router import (
    route_case,
)
from yuxi.agents.buildin.medication_review_da_prim.models import (
    RoutedRetrievalRecord,
)
from yuxi.agents.buildin.medication_review_da_prim.routed_retrieval import (
    make_routed_retrieval_strategy,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrievalRequest,
    RetrieverSelection,
    ensure_runtime_resources,
)

FLAT_POOL_TOP_K = 22
DOCUMENT_SUFFIX_RE = re.compile(r"\.(md|markdown|pdf|docx?|txt)$")
REFERENCE_DOCUMENT_RE = re.compile(r"【\s*依据\s*[：:]\s*([^】\n]+?)\s*·")
NON_WORD_RE = re.compile(r"[^0-9a-z\u3400-\u9fff]+")


class ReplayError(ValueError):
    pass


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSON object, JSON array, or JSONL file."""
    with path.open("r", encoding="utf-8-sig") as handle:
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False
        mode: str | None = None
        needs_separator = False
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                chunk = handle.read(64 * 1024)
                buffer += chunk
                eof = not chunk
                continue
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
                if needs_separator:
                    if buffer.startswith(","):
                        buffer = buffer[1:]
                        needs_separator = False
                        continue
                    if buffer.startswith("]"):
                        return
                    if not eof:
                        chunk = handle.read(64 * 1024)
                        buffer += chunk
                        eof = not chunk
                        continue
                    raise ReplayError(f"{path} 的 JSON 数组缺少分隔符")
                if buffer.startswith("]"):
                    return
            if not buffer and eof:
                return
            try:
                value, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if not eof:
                    chunk = handle.read(64 * 1024)
                    buffer += chunk
                    eof = not chunk
                    continue
                raise ReplayError(f"{path} 包含无效 JSON：{exc}") from exc
            if not isinstance(value, dict):
                raise ReplayError(f"{path} 的记录必须是 JSON object")
            yield value
            buffer = buffer[offset:]
            needs_separator = mode == "array"


def normalize_document_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = DOCUMENT_SUFFIX_RE.sub("", text)
    return NON_WORD_RE.sub("", text)


def document_matches(gold: str, candidate: str) -> bool:
    left = normalize_document_name(gold)
    right = normalize_document_name(candidate)
    if not left or not right:
        return False
    if left == right:
        return True
    return len(left) >= 5 and len(right) >= 5 and (left in right or right in left)


def gold_documents(record: dict[str, Any]) -> list[str]:
    for field in ("documents", "must_retrieve_documents", "reference_documents"):
        raw = record.get(field)
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        if isinstance(raw, list):
            values = [str(value).strip() for value in raw if str(value).strip()]
            if values:
                return list(dict.fromkeys(values))
    reference = str(record.get("reference") or record.get("answer") or "")
    return list(
        dict.fromkeys(
            match.group(1).strip() for match in REFERENCE_DOCUMENT_RE.finditer(reference) if match.group(1).strip()
        )
    )


def _search_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = record.get("retrieval_calls")
    if not isinstance(calls, list):
        calls = []
    result = []
    for call in calls:
        if not isinstance(call, dict) or call.get("tool_name") != "search_review_kb":
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        query = str(args.get("query_text") or call.get("query_text") or "").strip()
        if not query:
            continue
        result.append({**call, "_query_text": query, "_args": args})
    return result


def _chunk_key(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    file_id = str(metadata.get("file_id") or chunk.get("file_id") or "")
    chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id")
    if chunk_id is not None:
        return f"{file_id}:{chunk_id}"
    digest = hashlib.sha256(f"{file_id}\0{chunk.get('content') or ''}".encode("utf-8")).hexdigest()[:20]
    return f"CONTENT:{digest}"


def _flat_candidate(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "chunk_key": _chunk_key(chunk),
        "file_id": str(metadata.get("file_id") or chunk.get("file_id") or ""),
        "file_name": str(metadata.get("source") or chunk.get("source") or ""),
        "chunk_id": metadata.get("chunk_id") or chunk.get("chunk_id"),
        "chunk_index": (
            metadata.get("chunk_index") if metadata.get("chunk_index") is not None else chunk.get("chunk_index")
        ),
        "rank": rank,
        "similarity": chunk.get("score", chunk.get("distance")),
    }


def _matched_documents(gold: list[str], candidates: list[str]) -> list[str]:
    return [value for value in gold if any(document_matches(value, candidate) for candidate in candidates)]


async def _flat_top_22(db_id: str, query: str, timeout: int) -> list[dict[str, Any]]:
    async with asyncio.timeout(timeout):
        embeddings = await knowledge_base.aembed_texts(db_id, [query])
        if len(embeddings) != 1:
            raise ReplayError("Flat Top-22 embedding 返回数量不为 1")
        chunks = await knowledge_base.aquery(
            query,
            db_id,
            search_mode="vector",
            final_top_k=FLAT_POOL_TOP_K,
            use_reranker=False,
            include_distances=True,
            query_embedding=embeddings[0],
            raise_on_error=True,
        )
    if not isinstance(chunks, list):
        raise ReplayError("Flat Top-22 检索返回值不是 list")
    return [_flat_candidate(chunk, rank) for rank, chunk in enumerate(chunks, start=1) if isinstance(chunk, dict)]


def _document_names(candidates: list[dict[str, Any]]) -> list[str]:
    values = []
    seen: set[str] = set()
    for candidate in candidates:
        name = str(candidate.get("file_name") or candidate.get("file_id") or "")
        normalized = normalize_document_name(name)
        if name and normalized not in seen:
            values.append(name)
            seen.add(normalized)
    return values


def _metric_block(gold: list[str], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    names = _document_names(candidates)
    matched = _matched_documents(gold, names)
    return {
        "candidate_documents": names,
        "matched_gold_documents": matched,
        "gold_document_recall": len(matched) / len(gold) if gold else None,
    }


def _exclusive_path_gold_documents(
    gold: list[str],
    candidates: list[dict[str, Any]],
    *,
    required_path: str,
    excluded_path: str,
) -> list[str]:
    required_names = _document_names(
        [value for value in candidates if required_path in value.get("retrieval_paths", [])]
    )
    excluded_names = _document_names(
        [value for value in candidates if excluded_path in value.get("retrieval_paths", [])]
    )
    return [
        value
        for value in gold
        if any(document_matches(value, candidate) for candidate in required_names)
        and not any(document_matches(value, candidate) for candidate in excluded_names)
    ]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-timeout-seconds", type=int, default=600)
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    atlas = AtlasStore().load_current(args.db_id)
    await CorpusAtlasBuilder().validate_current(atlas)
    gold_records = list(iter_records(args.gold)) if args.gold else []
    context = MedicationReviewDaPrimContext(
        knowledges=[atlas.knowledge_name],
        atlas_profile="route",
        retrieval_timeout_seconds=args.retrieval_timeout_seconds,
        technical_retry_limit=args.technical_retry_limit,
    )
    ensure_runtime_resources(context)
    setattr(context, "_da_prim_atlas", atlas)
    selection = RetrieverSelection(
        db_id=args.db_id,
        retriever=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        snapshot={"db_id": args.db_id, "name": atlas.knowledge_name},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "da_prim_replay_details.jsonl"
    processed_cases = 0
    processed_queries = 0
    failed_queries = 0
    adopted_opportunities = 0
    displayed_opportunity_ids: set[tuple[int, str]] = set()
    adopted_opportunity_ids: set[tuple[int, str]] = set()
    injected_opportunities = 0
    injected_gold_hits = 0
    case_metric_rows: list[dict[str, Any]] = []
    with details_path.open("w", encoding="utf-8") as output:
        for ordinal, record in enumerate(iter_records(args.records)):
            if args.limit is not None and processed_cases >= args.limit:
                break
            calls = _search_calls(record)
            if not calls:
                continue
            row_index_raw = record.get("row_index")
            row_index = int(row_index_raw) if row_index_raw is not None else ordinal
            gold = gold_documents(gold_records[row_index]) if 0 <= row_index < len(gold_records) else []
            plan_anchors = list(record.get("plan_anchors") or [])
            patient_modifiers = list(record.get("patient_modifiers") or [])
            route = await route_case(
                manager=knowledge_base,
                db_id=args.db_id,
                atlas=atlas,
                raw_case_text=str(record.get("question") or ""),
                plan_anchors=plan_anchors,
                patient_modifiers=patient_modifiers,
            )
            opportunities = list(record.get("retrieval_opportunities") or [])
            displayed_opportunity_ids.update(
                (row_index, str(value.get("opportunity_id")))
                for value in opportunities
                if isinstance(value, dict) and value.get("opportunity_id")
            )
            state = {
                "plan_anchors": plan_anchors,
                "patient_modifiers": patient_modifiers,
                "case_route_record": route.record,
                "retrieval_opportunities": opportunities,
            }
            case_flat: list[dict[str, Any]] = []
            case_flat_top_5: list[dict[str, Any]] = []
            case_union: list[dict[str, Any]] = []
            case_fused: list[dict[str, Any]] = []
            for call_number, call in enumerate(calls, start=1):
                processed_queries += 1
                query = call["_query_text"]
                call_args = call["_args"]
                query_id = str(call.get("query_id") or f"Q-{call_number:03d}")
                opportunity_id = call_args.get("opportunity_id")
                strategy = make_routed_retrieval_strategy(
                    opportunity_id=(str(opportunity_id) if opportunity_id else None)
                )
                outcome = await strategy(
                    RetrievalRequest(
                        selection=selection,
                        context=context,
                        query_id=query_id,
                        query_text=query,
                        state=state,
                        focus_plan_ids=list(call_args.get("focus_plan_ids") or []),
                        focus_modifier_ids=list(call_args.get("focus_modifier_ids") or []),
                    )
                )
                diagnostic = outcome.diagnostic_record
                routed = diagnostic if isinstance(diagnostic, RoutedRetrievalRecord) else None
                try:
                    flat = await _flat_top_22(
                        args.db_id,
                        query,
                        args.retrieval_timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001 - recorded per query
                    flat = []
                    flat_error = f"{type(exc).__name__}: {exc}"
                else:
                    flat_error = None
                if routed is None:
                    failed_queries += 1
                    union: list[dict[str, Any]] = []
                    fused: list[dict[str, Any]] = []
                    routed_json = None
                else:
                    union = [value.model_dump(mode="json") for value in routed.fused_candidates]
                    fused = union[:5]
                    routed_json = routed.model_dump(mode="json")
                    failed_queries += int(not fused and bool(outcome.error_type))
                    adopted_opportunities += int(routed.opportunity_id is not None)
                    if routed.opportunity_id is not None:
                        adopted_opportunity_ids.add((row_index, routed.opportunity_id))
                    if routed.opportunity_injected_file_id is not None:
                        injected_opportunities += 1
                        injected = next(
                            (
                                value.file_name
                                for value in routed.effective_documents
                                if value.file_id == routed.opportunity_injected_file_id
                            ),
                            routed.opportunity_injected_file_id,
                        )
                        injected_gold_hits += int(any(document_matches(value, injected) for value in gold))
                case_flat.extend(flat)
                case_flat_top_5.extend(flat[:5])
                case_union.extend(union)
                case_fused.extend(fused)
                flat_metrics = _metric_block(gold, flat)
                union_metrics = _metric_block(gold, union)
                fused_metrics = _metric_block(gold, fused)
                routed_unique_gold_documents = _exclusive_path_gold_documents(
                    gold,
                    union,
                    required_path="local",
                    excluded_path="global",
                )
                global_escape_gold_documents = _exclusive_path_gold_documents(
                    gold,
                    union,
                    required_path="global",
                    excluded_path="local",
                )
                detail = {
                    "row_index": row_index,
                    "query_id": query_id,
                    "query_text": query,
                    "opportunity_id": opportunity_id,
                    "flat_top_22": flat_metrics,
                    "flat_top_5": _metric_block(gold, flat[:5]),
                    "da_union": union_metrics,
                    "da_fused_top_5": fused_metrics,
                    "routed_unique_gold_documents": routed_unique_gold_documents,
                    "da_over_flat_unique_gold_documents": [
                        value
                        for value in union_metrics["matched_gold_documents"]
                        if value not in flat_metrics["matched_gold_documents"]
                    ],
                    "routed_unique_gain": (len(routed_unique_gold_documents) / len(gold) if gold else None),
                    "global_escape_gold_documents": global_escape_gold_documents,
                    "global_escape_rate": (
                        len(global_escape_gold_documents) / len(union_metrics["matched_gold_documents"])
                        if union_metrics["matched_gold_documents"]
                        else None
                    ),
                    "flat_error": flat_error,
                    "routed_error_type": outcome.error_type,
                    "routed_error_message": outcome.error_message,
                    "routed_retrieval": routed_json,
                }
                output.write(json.dumps(detail, ensure_ascii=False) + "\n")
            case_metrics = {
                "row_index": row_index,
                "gold_documents": gold,
                "flat_top_22": _metric_block(gold, case_flat),
                "flat_top_5": _metric_block(gold, case_flat_top_5),
                "da_union": _metric_block(gold, case_union),
                "da_fused_top_5": _metric_block(gold, case_fused),
            }
            case_metrics["routed_unique_gold_documents"] = _exclusive_path_gold_documents(
                gold,
                case_union,
                required_path="local",
                excluded_path="global",
            )
            case_metrics["global_escape_gold_documents"] = _exclusive_path_gold_documents(
                gold,
                case_union,
                required_path="global",
                excluded_path="local",
            )
            case_metrics["da_over_flat_unique_gold_documents"] = [
                value
                for value in case_metrics["da_union"]["matched_gold_documents"]
                if value not in case_metrics["flat_top_22"]["matched_gold_documents"]
            ]
            case_metric_rows.append(case_metrics)
            processed_cases += 1

    valid_gold_cases = [row for row in case_metric_rows if row["gold_documents"]]
    gold_document_count = sum(len(row["gold_documents"]) for row in valid_gold_cases)
    recalled_gold_document_count = sum(len(row["da_union"]["matched_gold_documents"]) for row in valid_gold_cases)
    routed_unique_gold_count = sum(len(row["routed_unique_gold_documents"]) for row in valid_gold_cases)
    global_escape_gold_count = sum(len(row["global_escape_gold_documents"]) for row in valid_gold_cases)
    summary = {
        "schema_version": "1.0",
        "db_id": args.db_id,
        "atlas_snapshot_hash": atlas.snapshot_hash,
        "processed_cases": processed_cases,
        "processed_queries": processed_queries,
        "failed_queries": failed_queries,
        "adopted_opportunity_queries": adopted_opportunities,
        "displayed_opportunities": len(displayed_opportunity_ids),
        "adopted_unique_opportunities": len(adopted_opportunity_ids),
        "opportunity_adoption_rate": (
            len(adopted_opportunity_ids) / len(displayed_opportunity_ids) if displayed_opportunity_ids else None
        ),
        "opportunity_injected_queries": injected_opportunities,
        "opportunity_injected_gold_hits": injected_gold_hits,
        "routed_unique_gain": (routed_unique_gold_count / gold_document_count if gold_document_count else None),
        "global_escape_rate": (
            global_escape_gold_count / recalled_gold_document_count if recalled_gold_document_count else None
        ),
        "gold_cases": len(valid_gold_cases),
        "gold_document_count": gold_document_count,
        "macro_recall": {
            key: (
                sum(row[key]["gold_document_recall"] for row in valid_gold_cases) / len(valid_gold_cases)
                if valid_gold_cases
                else None
            )
            for key in (
                "flat_top_5",
                "flat_top_22",
                "da_union",
                "da_fused_top_5",
            )
        },
        "routed_unique_gold_documents": routed_unique_gold_count,
        "global_escape_gold_documents": global_escape_gold_count,
        "da_over_flat_unique_gold_documents": sum(
            len(row["da_over_flat_unique_gold_documents"]) for row in valid_gold_cases
        ),
        "details_file": str(details_path),
    }
    summary_path = args.output_dir / "da_prim_replay_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        return asyncio.run(_main_async(_arguments()))
    except (OSError, ReplayError, ValueError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
