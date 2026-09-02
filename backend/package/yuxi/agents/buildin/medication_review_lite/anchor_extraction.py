from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from yuxi.utils.datetime_utils import utc_isoformat

from .models import (
    AnchorExtractionAudit,
    PlanAnchor,
    PlanAnchorDraft,
    PlanAnchorEnvelope,
)
from .prompt import ANCHOR_EXTRACTION_SYSTEM_PROMPT


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return dict(usage)
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(value, dict):
            return dict(value)
    return {}


def merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, int | float) and isinstance(
            merged.get(key), int | float
        ):
            merged[key] += value
        elif key not in merged:
            merged[key] = value
    return merged


def _parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            count=1,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\s*```$", "", value, count=1)
    start = value.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    parsed, end = json.JSONDecoder().raw_decode(value[start:])
    if value[start + end :].strip():
        raise ValueError("JSON 对象后存在额外文本")
    if not isinstance(parsed, dict):
        raise ValueError("结构化输出必须是 JSON 对象")
    return parsed


def _schema_instructions() -> str:
    schema = json.dumps(
        PlanAnchorEnvelope.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\n只返回一个符合下方 JSON Schema 的 JSON 对象。"
        "不要使用 Markdown 代码围栏，不要附加解释或其它文本。"
        "字段名、必填字段和枚举值必须严格遵守 Schema。\n\n"
        f"目标 JSON Schema：\n{schema}"
    )


def _is_auth_error(exc: BaseException) -> bool:
    value = f"{type(exc).__name__} {exc}".casefold()
    return any(
        marker in value
        for marker in (
            "authentication",
            "unauthorized",
            "permission",
            "invalid api key",
            "invalid_api_key",
            "status code: 401",
            "status_code=401",
        )
    )


async def _invoke_model(
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
            if _is_auth_error(exc):
                break
    assert error is not None
    raise error


def _normalized_whitespace_with_map(
    text: str,
) -> tuple[str, list[int], list[int]]:
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    in_whitespace = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_whitespace:
                normalized.append(" ")
                starts.append(index)
                ends.append(index + 1)
                in_whitespace = True
            else:
                ends[-1] = index + 1
            continue
        in_whitespace = False
        normalized.append(char)
        starts.append(index)
        ends.append(index + 1)
    return "".join(normalized), starts, ends


def _find_unclaimed_exact(
    raw_text: str,
    source_span: str,
    claimed: set[tuple[int, int]],
) -> tuple[int, int] | None:
    offset = 0
    while True:
        start = raw_text.find(source_span, offset)
        if start < 0:
            return None
        result = (start, start + len(source_span))
        if result not in claimed:
            return result
        offset = start + 1


def _find_unclaimed_whitespace_normalized(
    raw_text: str,
    source_span: str,
    claimed: set[tuple[int, int]],
) -> tuple[int, int] | None:
    normalized_raw, starts, ends = _normalized_whitespace_with_map(raw_text)
    normalized_span, _, _ = _normalized_whitespace_with_map(source_span)
    if not normalized_span:
        return None
    offset = 0
    while True:
        start = normalized_raw.find(normalized_span, offset)
        if start < 0:
            return None
        end_index = start + len(normalized_span) - 1
        result = (starts[start], ends[end_index])
        if result not in claimed:
            return result
        offset = start + 1


def validate_anchor_drafts(
    raw_text: str,
    drafts: list[PlanAnchorDraft],
) -> tuple[list[PlanAnchor], list[dict[str, Any]]]:
    claimed: set[tuple[int, int]] = set()
    seen_drafts: set[tuple[str, str]] = set()
    located: list[tuple[int, int, PlanAnchorDraft]] = []
    dropped: list[dict[str, Any]] = []

    for draft in drafts:
        source_span = draft.source_span.strip()
        if not source_span:
            dropped.append(
                {
                    "source_span": draft.source_span,
                    "label": draft.label,
                    "kind": draft.kind,
                    "reason": "empty_source_span",
                }
            )
            continue
        normalized_span, _, _ = _normalized_whitespace_with_map(source_span)
        draft_key = (normalized_span, draft.kind)
        if draft_key in seen_drafts:
            continue
        seen_drafts.add(draft_key)
        result = _find_unclaimed_exact(raw_text, source_span, claimed)
        if result is None:
            result = _find_unclaimed_whitespace_normalized(
                raw_text,
                source_span,
                claimed,
            )
        if result is None:
            dropped.append(
                {
                    "source_span": draft.source_span,
                    "label": draft.label,
                    "kind": draft.kind,
                    "reason": "source_span_not_found",
                }
            )
            continue
        start, end = result
        claimed.add(result)
        located.append(
            (
                start,
                end,
                draft.model_copy(
                    update={
                        "source_span": raw_text[start:end],
                        "label": draft.label.strip()[:160]
                        or raw_text[start:end][:160],
                    }
                ),
            )
        )
    anchors: list[PlanAnchor] = []
    for index, (start, end, draft) in enumerate(
        sorted(located, key=lambda value: (value[0], value[1], value[2].kind)),
        start=1,
    ):
        anchors.append(
            PlanAnchor(
                element_id=f"PE{index:03d}",
                source_span=raw_text[start:end],
                source_start=start,
                source_end=end,
                label=draft.label,
                kind=draft.kind,
            )
        )
    return anchors, dropped


async def extract_plan_anchors(
    *,
    model: Any,
    raw_text: str,
    technical_retry_limit: int,
) -> tuple[list[PlanAnchor], AnchorExtractionAudit]:
    started_at = utc_isoformat()
    started = time.monotonic()
    raw_output: str | None = None
    repair_raw_output: str | None = None
    errors: list[str] = []
    usage: dict[str, Any] = {}
    user_prompt = (
        "请从以下病例原文中提取明确写出的治疗方案要素。\n\n"
        f"病例原文：\n{raw_text}"
    )

    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=ANCHOR_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt + _schema_instructions()),
            ],
            technical_retry_limit=technical_retry_limit,
        )
    except Exception as exc:  # noqa: BLE001 - provider boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        return [], AnchorExtractionAudit(
            status="failed",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            validation_errors=errors,
            usage=usage,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    raw_output = message_text(response)
    usage = merge_usage(usage, response_usage(response))
    try:
        envelope = PlanAnchorEnvelope.model_validate(
            _parse_json_object(raw_output)
        )
        anchors, dropped = validate_anchor_drafts(raw_text, envelope.anchors)
        status = "success" if anchors else "no_valid_anchor"
        return anchors, AnchorExtractionAudit(
            status=status,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output,
            dropped_drafts=dropped,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - JSON/schema boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        first_error = exc

    repair_prompt = (
        "上一次输出未通过 JSON 或 Schema 校验。只修复格式和字段，"
        "不得改变原病例、增加方案要素或改写 source_span。\n\n"
        f"校验错误：\n{errors[-1]}\n\n"
        f"上一次输出：\n{raw_output or '<无输出>'}\n\n"
        f"原任务：\n{user_prompt}"
    )
    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=ANCHOR_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=repair_prompt + _schema_instructions()),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        repair_raw_output = message_text(response)
        usage = merge_usage(usage, response_usage(response))
        envelope = PlanAnchorEnvelope.model_validate(
            _parse_json_object(repair_raw_output)
        )
        anchors, dropped = validate_anchor_drafts(raw_text, envelope.anchors)
        status = "repaired" if anchors else "no_valid_anchor"
        return anchors, AnchorExtractionAudit(
            status=status,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output,
            repair_raw_output=repair_raw_output,
            validation_errors=errors,
            dropped_drafts=dropped,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - schema/provider boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        first_error = exc

    return [], AnchorExtractionAudit(
        status="failed",
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        raw_output=raw_output,
        repair_raw_output=repair_raw_output,
        validation_errors=errors,
        usage=usage,
        error_type=type(first_error).__name__,
        error_message=str(first_error),
    )


def disabled_anchor_audit() -> AnchorExtractionAudit:
    return AnchorExtractionAudit(
        status="disabled",
        started_at=utc_isoformat(),
    )
