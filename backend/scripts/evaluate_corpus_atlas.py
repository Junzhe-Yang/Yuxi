from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas import (
    AtlasStore,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.opportunities import (
    build_retrieval_opportunities,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.router import (
    route_case,
)
from yuxi import knowledge_base

REFERENCE_DOCUMENT_RE = re.compile(r"【依据[：:]\s*([^】\n]+?)(?:\s*[·•]\s*|】)")
NON_WORD_RE = re.compile(r"[^0-9a-z\u3400-\u9fff]+")
ROUTER_K_VALUES = (1, 3, 6)


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = ""
        while not first:
            character = handle.read(1)
            if not character:
                return
            if not character.isspace():
                first = character
        handle.seek(0)
        if first != "[":
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} 不是 JSON object")
                yield value
            return

        decoder = json.JSONDecoder()
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof and len(buffer) < 1024 * 1024:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    if eof:
                        return
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"{path} 不是 JSON array")
                buffer = buffer[1:]
                started = True
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            try:
                value, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{path} 的数组元素不是 JSON object")
            yield value
            buffer = buffer[offset:]


def normalize_document_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = re.sub(r"\.(md|markdown|pdf|docx?|txt)$", "", text)
    return NON_WORD_RE.sub("", text)


def document_matches(gold: str, candidate: str) -> bool:
    normalized_gold = normalize_document_name(gold)
    normalized_candidate = normalize_document_name(candidate)
    if not normalized_gold or not normalized_candidate:
        return False
    if normalized_gold == normalized_candidate:
        return True
    return (
        len(normalized_gold) >= 5
        and len(normalized_candidate) >= 5
        and (normalized_gold in normalized_candidate or normalized_candidate in normalized_gold)
    )


def gold_documents(record: dict[str, Any]) -> list[str]:
    for key in ("reference_documents", "documents"):
        raw = record.get(key)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用已有 PlanAnchor/PatientModifier 评价 Corpus Atlas 文档路由。")
    parser.add_argument("--db-id", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--opportunity-min-similarity",
        type=float,
        help="仅用于 P0 机会数量/得分校准；省略表示不设最低 cosine。",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    if args.opportunity_min_similarity is not None and not -1.0 <= args.opportunity_min_similarity <= 1.0:
        raise ValueError("--opportunity-min-similarity 必须在 [-1, 1] 范围内")
    atlas = AtlasStore().load_current(args.db_id)
    await CorpusAtlasBuilder().validate_current(atlas)
    gold_values = list(iter_json_records(args.gold))
    details = []
    micro_hits = {value: 0 for value in ROUTER_K_VALUES}
    micro_total = 0
    macro_recall_sum = {value: 0.0 for value in ROUTER_K_VALUES}
    complete_count = 0
    valid_count = 0
    opportunity_count = 0
    cases_with_opportunity = 0
    for ordinal, record in enumerate(iter_json_records(args.records)):
        if args.limit is not None and ordinal >= args.limit:
            break
        row_index_raw = record.get("row_index")
        row_index = int(row_index_raw) if row_index_raw is not None else ordinal
        if row_index < 0 or row_index >= len(gold_values):
            raise ValueError(f"records row_index 越界：{row_index}")
        gold = gold_documents(gold_values[row_index])
        if not gold:
            continue
        computation = await route_case(
            manager=knowledge_base,
            db_id=args.db_id,
            atlas=atlas,
            raw_case_text=str(record.get("question") or ""),
            plan_anchors=list(record.get("plan_anchors") or []),
            patient_modifiers=list(record.get("patient_modifiers") or []),
        )
        top_documents = computation.record.ranked_documents[: max(ROUTER_K_VALUES)]
        opportunities = build_retrieval_opportunities(
            atlas=atlas,
            computation=computation,
            plan_anchors=list(record.get("plan_anchors") or []),
            patient_modifiers=list(record.get("patient_modifiers") or []),
            min_similarity=args.opportunity_min_similarity,
        )
        opportunity_count += len(opportunities)
        cases_with_opportunity += int(bool(opportunities))
        matched_by_k: dict[int, list[str]] = {}
        for k in ROUTER_K_VALUES:
            candidates = [value.file_name for value in top_documents[:k]]
            matched_by_k[k] = [
                value for value in gold if any(document_matches(value, candidate) for candidate in candidates)
            ]
            micro_hits[k] += len(matched_by_k[k])
            macro_recall_sum[k] += len(matched_by_k[k]) / len(gold)
        matched = matched_by_k[6]
        micro_total += len(gold)
        complete_count += int(len(matched) == len(gold))
        valid_count += 1
        details.append(
            {
                "row_index": row_index,
                "question": record.get("question"),
                "gold_documents": gold,
                "matched_documents": matched,
                "missed_documents": [value for value in gold if value not in matched],
                "router_recall_at_1": len(matched_by_k[1]) / len(gold),
                "router_recall_at_3": len(matched_by_k[3]) / len(gold),
                "router_recall_at_6": len(matched_by_k[6]) / len(gold),
                "ranked_documents": [value.model_dump(mode="json") for value in top_documents],
                "retrieval_opportunities": [value.model_dump(mode="json") for value in opportunities],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "atlas_router_details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "1.0",
        "db_id": args.db_id,
        "atlas_snapshot_hash": atlas.snapshot_hash,
        "valid_case_count": valid_count,
        "gold_document_count": micro_total,
        **{
            f"router_micro_recall_at_{k}": (micro_hits[k] / micro_total if micro_total else None)
            for k in ROUTER_K_VALUES
        },
        **{
            f"router_macro_recall_at_{k}": (macro_recall_sum[k] / valid_count if valid_count else None)
            for k in ROUTER_K_VALUES
        },
        "router_all_at_6": (complete_count / valid_count if valid_count else None),
        "opportunity_min_similarity": args.opportunity_min_similarity,
        "opportunity_count": opportunity_count,
        "cases_with_opportunity": cases_with_opportunity,
        "case_opportunity_rate": (cases_with_opportunity / valid_count if valid_count else None),
        "mean_opportunities_per_case": (opportunity_count / valid_count if valid_count else None),
        "details_file": str(details_path),
    }
    summary_path = args.output_dir / "atlas_router_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
