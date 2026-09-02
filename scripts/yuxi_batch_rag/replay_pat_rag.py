"""Replay V2 retrieval evidence with the thin PAT-RAG answer path (R0-R4)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from yuxi.agents import load_chat_model
from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    extract_plan_anchors,
    merge_usage,
    message_text,
    response_usage,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    build_evidence_section,
    build_plan_section,
    extract_evidence_ids,
    extract_item_element_ids,
    insert_coverage_patch,
    strip_program_owned_sections,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    EvidenceItem,
    PlanAnchor,
)
from yuxi.agents.buildin.medication_review_lite.prompt import (
    DEFAULT_REVIEW_SYSTEM_PROMPT,
    build_coverage_patch_prompt,
)


class ReplayError(ValueError):
    pass


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError(
                    f"第 {line_number} 行不是有效 JSON：{exc}"
                ) from exc
            if not isinstance(item, dict):
                raise ReplayError(f"第 {line_number} 行必须是 JSON 对象")
            records.append(item)
        return records
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(
        isinstance(item, dict) for item in value
    ):
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
    if record.get("schema_version") in {"2.0", "3.0"}:
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
    return None


def _find_old_answer(record: dict[str, Any]) -> str:
    for key in ("answer", "response"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    history = record.get("history")
    if isinstance(history, list):
        for item in reversed(history):
            if (
                isinstance(item, dict)
                and item.get("type") == "ai"
                and isinstance(item.get("content"), str)
            ):
                return item["content"]
    return ""


def _adapt_evidence(trace: dict[str, Any]) -> list[EvidenceItem]:
    raw_values = trace.get("evidence")
    if not isinstance(raw_values, list):
        raw_values = trace.get("evidence_store")
    if not isinstance(raw_values, list):
        return []
    result: list[EvidenceItem] = []
    for index, raw in enumerate(raw_values, start=1):
        if not isinstance(raw, dict):
            continue
        raw_text = str(raw.get("raw_text") or "")
        content_hash = str(
            raw.get("content_hash")
            or hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        )
        evidence_id = str(raw.get("evidence_id") or f"EV{index:03d}")
        result.append(
            EvidenceItem(
                evidence_id=evidence_id,
                content_hash=content_hash,
                raw_text=raw_text,
                source_document=raw.get("source_document"),
                file_id=raw.get("file_id"),
                chunk_id=raw.get("chunk_id"),
                chunk_index=raw.get("chunk_index"),
                raw_metadata=raw.get("raw_metadata") or {},
            )
        )
    return result


def _selected_evidence(
    trace: dict[str, Any],
    evidence: list[EvidenceItem],
    limit: int,
) -> list[EvidenceItem]:
    selection = trace.get("evidence_selection")
    selected_ids = (
        selection.get("selected_evidence_ids")
        if isinstance(selection, dict)
        else None
    )
    if isinstance(selected_ids, list) and selected_ids:
        selected_set = {str(value) for value in selected_ids}
        selected = [
            item for item in evidence if item.evidence_id in selected_set
        ]
        if selected:
            return selected[:limit]
    return evidence[:limit]


def _evidence_context(evidence: list[EvidenceItem]) -> str:
    cards: list[str] = []
    for item in evidence:
        source = item.source_document or "<未知来源>"
        position = (
            f"chunk {item.chunk_index}"
            if item.chunk_index is not None
            else "chunk <未知>"
        )
        cards.append(
            f"[{item.evidence_id}]\n"
            f"来源：{source}\n"
            f"位置：{position}\n"
            f"内容：\n{item.raw_text}"
        )
    return "\n\n---\n\n".join(cards)


def _direct_prompt(
    *,
    system_prompt: str,
    question: str,
    anchors: list[PlanAnchor],
    evidence: list[EvidenceItem],
) -> tuple[str, str]:
    if anchors:
        anchor_text = "\n".join(
            f"- [{item.element_id}] {item.label}；原文：{item.source_span}"
            for item in anchors
        )
        item_rule = (
            "第②部分必须为每个 PE ID 生成且仅生成一个"
            "“■ 【PE ID】方案要素”逐项标题。"
        )
    else:
        anchor_text = "本组不提供方案锚点，请直接从原病例识别完整方案。"
        item_rule = "第②部分应为每个明确方案要素分别生成逐项标题。"
    system = f"""{system_prompt}

你正在执行 PAT-RAG 检索后回放。Evidence 已经给出，不得调用工具。
同时审查合理、不合理、需调整和证据不足项；不合理或需调整时，
在 Evidence 允许的范围内给出替代或修正建议。只能引用给出的 Evidence ID。

只生成：
②【逐项判断】
③【正面判断汇总】
④【负面判断汇总】
⑤【综合建议】

{item_rule}
不要生成第①和第⑥部分，不要输出 JSON，不要复制 source span。"""
    user = f"""原始病例：
{question}

方案锚点：
{anchor_text}

可用 Evidence：
{_evidence_context(evidence)}"""
    return system, user


async def _invoke(
    *,
    model: Any,
    messages: list[Any],
    technical_retry_limit: int,
) -> Any:
    error: BaseException | None = None
    for _attempt in range(technical_retry_limit + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider adapters vary
            error = exc
    assert error is not None
    raise error


async def _run_group(
    *,
    group: str,
    record: dict[str, Any],
    trace: dict[str, Any],
    question: str,
    model: Any,
    model_id: str,
    system_prompt: str,
    anchors: list[PlanAnchor],
    anchor_audit: dict[str, Any],
    evidence: list[EvidenceItem],
    selected: list[EvidenceItem],
    technical_retry_limit: int,
) -> dict[str, Any]:
    if group == "r0":
        return {
            "group": "r0",
            "source_schema_version": trace.get("schema_version"),
            "question": question,
            "model": model_id,
            "answer": _find_old_answer(record),
            "model_calls": 0,
            "warnings": [],
        }

    use_anchors = group in {"r2", "r3", "r4"}
    use_patch = group == "r3"
    group_anchors = anchors if use_anchors else []
    group_evidence = evidence if group == "r4" else selected
    system, user = _direct_prompt(
        system_prompt=system_prompt,
        question=question,
        anchors=group_anchors,
        evidence=group_evidence,
    )
    response = await _invoke(
        model=model,
        messages=[
            SystemMessage(content=system),
            HumanMessage(content=user),
        ],
        technical_retry_limit=technical_retry_limit,
    )
    usage = response_usage(response)
    body = strip_program_owned_sections(message_text(response))
    item_ids, section_degraded = extract_item_element_ids(body)
    expected_ids = [item.element_id for item in group_anchors]
    missing_before = [
        value for value in expected_ids if value not in set(item_ids)
    ]
    patch_attempted = False
    patch_succeeded = False
    if use_patch and missing_before:
        patch_attempted = True
        missing_set = set(missing_before)
        missing_anchors = [
            item for item in group_anchors if item.element_id in missing_set
        ]
        patch_response = await _invoke(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "只补写遗漏方案要素的逐项判断块，不得重写已有内容。"
                    )
                ),
                HumanMessage(
                    content=build_coverage_patch_prompt(
                        missing_anchors=missing_anchors,
                        existing_answer=body,
                    )
                ),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        usage = merge_usage(usage, response_usage(patch_response))
        patch = strip_program_owned_sections(message_text(patch_response))
        patch_ids, _ = extract_item_element_ids(patch)
        forbidden_sections = (
            "③【正面判断汇总】",
            "④【负面判断汇总】",
            "⑤【综合建议】",
            "⑥【依据清单】",
        )
        if (
            set(patch_ids) == missing_set
            and len(patch_ids) == len(missing_set)
            and not any(value in patch for value in forbidden_sections)
        ):
            body = insert_coverage_patch(body, patch)
            patch_succeeded = True

    final_item_ids, final_degraded = extract_item_element_ids(body)
    missing_after = [
        value for value in expected_ids if value not in set(final_item_ids)
    ]
    store = {item.evidence_id.upper(): item for item in group_evidence}
    cited = extract_evidence_ids(body)
    unknown = [value for value in cited if value not in store]
    answer = "\n\n".join(
        [
            build_plan_section(group_anchors),
            body,
            build_evidence_section(
                cited_evidence_ids=cited,
                unknown_evidence_ids=unknown,
                evidence_store=store,
            ),
        ]
    )
    warnings: list[str] = []
    if missing_after:
        warnings.append("仍遗漏：" + "、".join(missing_after))
    if unknown:
        warnings.append("未知 Evidence：" + "、".join(unknown))
    if section_degraded or final_degraded:
        warnings.append("六段结构解析降级")
    return {
        "group": group,
        "source_schema_version": trace.get("schema_version"),
        "question": question,
        "model": model_id,
        "prompt": {
            "system": system,
            "user": user,
            "sha256": hashlib.sha256(
                f"{system}\n\n{user}".encode()
            ).hexdigest(),
        },
        "anchor_extraction": anchor_audit if use_anchors else None,
        "plan_anchors": [
            item.model_dump(mode="json") for item in group_anchors
        ],
        "evidence_ids": [item.evidence_id for item in group_evidence],
        "coverage": {
            "missing_before_patch": missing_before,
            "missing_after_patch": missing_after,
            "patch_attempted": patch_attempted,
            "patch_succeeded": patch_succeeded,
            "unknown_evidence_ids": unknown,
        },
        "usage": usage,
        "answer": answer,
        "model_calls": 2 if patch_attempted else 1,
        "warnings": warnings,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--groups",
        default="r0,r1,r2,r3,r4",
        help="逗号分隔：r0,r1,r2,r3,r4",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_REVIEW_SYSTEM_PROMPT,
    )
    parser.add_argument("--max-evidence", type=int, default=15)
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    groups = [
        value.strip().lower()
        for value in args.groups.split(",")
        if value.strip()
    ]
    if not groups or any(
        value not in {"r0", "r1", "r2", "r3", "r4"}
        for value in groups
    ):
        raise ReplayError("--groups 只能包含 r0,r1,r2,r3,r4")
    records = _load_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        trace = _find_trace(record)
        if trace is None:
            raise ReplayError(
                f"第 {index} 条记录中没有 medication_review_trace"
            )
        question = _find_question(record)
        if question is None:
            raise ReplayError(f"第 {index} 条记录中没有原始 question")
        needs_model = any(group != "r0" for group in groups)
        needs_anchors = any(group in {"r2", "r3", "r4"} for group in groups)
        model_id = args.model or (
            (trace.get("agent_config_snapshot") or {}).get("model")
        )
        if needs_model and (
            not isinstance(model_id, str) or not model_id.strip()
        ):
            raise ReplayError(
                f"第 {index} 条记录未提供模型，请使用 --model"
            )
        model = load_chat_model(model_id) if needs_model else None
        evidence = _adapt_evidence(trace)
        selected = _selected_evidence(
            trace,
            evidence,
            args.max_evidence,
        )
        anchors: list[PlanAnchor] = []
        anchor_audit: dict[str, Any] = {}
        if needs_anchors:
            anchors, anchor_audit_model = await extract_plan_anchors(
                model=model,
                raw_text=question,
                technical_retry_limit=args.technical_retry_limit,
            )
            anchor_audit = anchor_audit_model.model_dump(mode="json")
        for group in groups:
            result = await _run_group(
                group=group,
                record=record,
                trace=trace,
                question=question,
                model=model,
                model_id=model_id,
                system_prompt=args.system_prompt,
                anchors=anchors,
                anchor_audit=anchor_audit,
                evidence=evidence,
                selected=selected,
                technical_retry_limit=args.technical_retry_limit,
            )
            output.append(result)
            stem = f"{index:04d}-{group}"
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (args.output_dir / f"{stem}.md").write_text(
                result["answer"] + "\n",
                encoding="utf-8",
            )
    (args.output_dir / "replay-pat-rag.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Replayed {len(output)} result(s) into {args.output_dir}")
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
