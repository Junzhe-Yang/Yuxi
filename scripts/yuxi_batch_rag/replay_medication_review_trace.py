"""Replay the PEA-RAG post-retrieval pipeline from Trace 2.0 or 3.0."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from yuxi.agents import load_chat_model
from yuxi.agents.buildin.medication_review.claim_extraction import extract_claims
from yuxi.agents.buildin.medication_review.evidence_board import (
    merge_evidence,
    select_evidence,
)
from yuxi.agents.buildin.medication_review.models import (
    EvidenceCandidate,
    EvidenceClaim,
    EvidenceItemV3,
    EvidenceOccurrenceV3,
    PatientCase,
    ReviewQuestion,
    TreatmentPlanElement,
)
from yuxi.agents.buildin.medication_review.prompt import PEA_RAG_DEFAULT_SYSTEM_PROMPT
from yuxi.agents.buildin.medication_review.rendering import render_review_v3
from yuxi.agents.buildin.medication_review.review_synthesis import synthesize_review


class ReplayError(ValueError):
    pass


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"第 {line_number} 行不是有效 JSON：{exc}") from exc
            if not isinstance(item, dict):
                raise ReplayError(f"第 {line_number} 行必须是 JSON 对象")
            records.append(item)
        return records
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ReplayError("输入必须是 JSON 对象、JSON 对象列表或 JSONL")


def _find_trace(record: dict[str, Any]) -> dict[str, Any] | None:
    trace = record.get("medication_review_trace")
    if isinstance(trace, dict):
        return trace
    for value in record.values():
        if isinstance(value, dict):
            found = _find_trace(value)
            if found is not None:
                return found
        elif isinstance(value, list):
            for item in reversed(value):
                if isinstance(item, dict):
                    found = _find_trace(item)
                    if found is not None:
                        return found
    if record.get("schema_version") in {"2.0", "3.0"} and "plan_elements" in record:
        return record
    return None


def _find_question(record: dict[str, Any]) -> str | None:
    for key in ("question", "raw_question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    history = record.get("history")
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            if item.get("type") not in {"human", "user"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content
    for value in record.values():
        if isinstance(value, dict):
            found = _find_question(value)
            if found is not None:
                return found
        elif isinstance(value, list):
            for item in reversed(value):
                if isinstance(item, dict):
                    found = _find_question(item)
                    if found is not None:
                        return found
    return None


def _content_hash(item: dict[str, Any]) -> str:
    value = item.get("content_hash") or item.get("evidence_id")
    if value:
        return str(value)
    return hashlib.sha256(
        str(item.get("raw_text") or "").encode("utf-8")
    ).hexdigest()


def _adapt_evidence(trace: dict[str, Any]) -> list[EvidenceItemV3]:
    if trace.get("schema_version") == "3.0":
        return [
            EvidenceItemV3.model_validate(item)
            for item in trace.get("evidence") or []
            if isinstance(item, dict)
        ]
    candidates: list[EvidenceCandidate] = []
    for item in trace.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        occurrences: list[EvidenceOccurrenceV3] = []
        for occurrence in item.get("occurrences") or []:
            if not isinstance(occurrence, dict):
                continue
            occurrences.append(
                EvidenceOccurrenceV3(
                    query_id=str(
                        occurrence.get("query_id")
                        or occurrence.get("bundle_id")
                        or "Q000"
                    ),
                    linked_element_ids=list(
                        occurrence.get("target_element_ids") or []
                    ),
                    rank=int(occurrence.get("rank") or 1),
                    score=occurrence.get("score"),
                    distance=occurrence.get("distance"),
                )
            )
        candidates.append(
            EvidenceCandidate(
                content_hash=_content_hash(item),
                raw_text=str(item.get("raw_text") or ""),
                source_document=item.get("source_document"),
                file_id=item.get("file_id"),
                chunk_id=item.get("chunk_id"),
                chunk_index=item.get("chunk_index"),
                raw_metadata=item.get("raw_metadata") or {},
                source_method=item.get("source_method") or "search",
                occurrences=occurrences,
            )
        )
    return merge_evidence(existing=[], candidates=candidates).evidence


async def _replay_one(
    *,
    record: dict[str, Any],
    synthesis_mode: str,
    model_override: str | None,
    system_prompt: str,
    diagnostic_trace: bool,
    max_evidence: int,
    max_tokens: int,
    technical_retry_limit: int,
) -> dict[str, Any]:
    trace = _find_trace(record)
    if trace is None:
        raise ReplayError("记录中没有 medication_review_trace")
    question = _find_question(record)
    if question is None:
        raise ReplayError("Trace 回放需要记录中的 question/raw_question 原文")
    patient_case = PatientCase.model_validate(trace.get("patient_case"))
    patient_facts = list(trace.get("patient_facts") or [])
    elements = [
        TreatmentPlanElement.model_validate(item)
        for item in trace.get("plan_elements") or []
    ]
    questions = [
        ReviewQuestion.model_validate(item)
        for item in trace.get("review_agenda") or []
    ]
    evidence = _adapt_evidence(trace)
    existing_selection = trace.get("evidence_selection")
    available_ids = {item.evidence_id for item in evidence}
    if isinstance(existing_selection, dict) and existing_selection.get(
        "selected_evidence_ids"
    ) and set(existing_selection["selected_evidence_ids"]) <= available_ids:
        selected_ids = list(existing_selection["selected_evidence_ids"])
        selection = existing_selection
    else:
        audit = select_evidence(
            evidence=evidence,
            priority_evidence_ids=[],
            max_evidence=max_evidence,
            max_tokens=max_tokens,
        )
        selected_ids = audit.selected_evidence_ids
        selection = audit.model_dump(mode="json")
    selected_set = set(selected_ids)
    selected = [item for item in evidence if item.evidence_id in selected_set]
    model_id = model_override or (
        (trace.get("agent_config_snapshot") or {}).get("model")
    )
    if not isinstance(model_id, str) or not model_id.strip():
        raise ReplayError("未提供 --model，Trace 中也没有 agent_config_snapshot.model")
    model = load_chat_model(model_id)

    claims: list[EvidenceClaim] = []
    claim_audit: dict[str, Any] = {}
    warnings: list[str] = []
    degraded = False
    if synthesis_mode == "claims":
        claim_result = await extract_claims(
            model=model,
            evidence=selected,
            elements=elements,
            questions=questions,
            technical_retry_limit=technical_retry_limit,
            retain_raw_output=diagnostic_trace,
        )
        claims = claim_result.claims
        claim_audit = claim_result.audit
        warnings.extend(claim_result.warnings)
        degraded = claim_result.degraded

    synthesis = await synthesize_review(
        model=model,
        raw_case_text=question,
        patient_case=patient_case,
        patient_facts=patient_facts,
        elements=elements,
        questions=questions,
        evidence=selected,
        claims=claims,
        unresolved_questions=[],
        synthesis_mode=synthesis_mode,  # type: ignore[arg-type]
        system_prompt=system_prompt,
        technical_retry_limit=technical_retry_limit,
        retain_raw_output=diagnostic_trace,
    )
    warnings.extend(synthesis.warnings)
    answer = render_review_v3(
        review=synthesis.review,
        plan_elements=elements,
        evidence_items=evidence,
        claims=claims,
    )
    return {
        "source_schema_version": trace.get("schema_version"),
        "question": question,
        "synthesis_mode": synthesis_mode,
        "model": model_id,
        "run_status": "partial" if degraded or synthesis.degraded else "completed",
        "evidence_selection": selection,
        "evidence_claims": [item.model_dump(mode="json") for item in claims],
        "claim_extraction": claim_audit,
        "review_synthesis": synthesis.audit,
        "local_validation_events": [
            item.model_dump(mode="json") for item in synthesis.validation_events
        ],
        "final_review": synthesis.review.model_dump(mode="json"),
        "answer": answer,
        "warnings": warnings,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--synthesis-mode",
        required=True,
        choices=["direct_chunks", "claims"],
    )
    parser.add_argument("--model")
    parser.add_argument("--system-prompt", default=PEA_RAG_DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--diagnostic-trace", action="store_true")
    parser.add_argument("--max-evidence", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    records = _load_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        result = await _replay_one(
            record=record,
            synthesis_mode=args.synthesis_mode,
            model_override=args.model,
            system_prompt=args.system_prompt,
            diagnostic_trace=args.diagnostic_trace,
            max_evidence=args.max_evidence,
            max_tokens=args.max_tokens,
            technical_retry_limit=args.technical_retry_limit,
        )
        output.append(result)
        stem = f"{index:04d}-{args.synthesis_mode}"
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"{stem}.md").write_text(
            result["answer"] + "\n",
            encoding="utf-8",
        )
    (args.output_dir / f"replay-{args.synthesis_mode}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Replayed {len(output)} record(s) into {args.output_dir}")
    return 0


def main() -> int:
    args = _arguments()
    try:
        return asyncio.run(_main_async(args))
    except (OSError, ReplayError, ValueError) as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
