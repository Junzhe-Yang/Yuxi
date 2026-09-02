from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    _find_unclaimed_exact,
    _find_unclaimed_whitespace_normalized,
    _invoke_model,
    _normalized_whitespace_with_map,
    _parse_json_object,
    merge_usage,
    message_text,
    response_usage,
)
from yuxi.utils.datetime_utils import utc_isoformat

from .models import (
    ModifierExtractionAudit,
    PatientModifier,
    PatientModifierDraft,
    PatientModifierEnvelope,
)
from .prompt import MODIFIER_EXTRACTION_SYSTEM_PROMPT


def _schema_instructions() -> str:
    schema = json.dumps(
        PatientModifierEnvelope.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\n只返回一个符合下方 JSON Schema 的 JSON 对象。"
        "不要使用 Markdown 代码围栏，不要附加解释或其它文本。"
        "字段名和必填字段必须严格遵守 Schema。\n\n"
        f"目标 JSON Schema：\n{schema}"
    )


def validate_modifier_drafts(
    raw_text: str,
    drafts: list[PatientModifierDraft],
) -> tuple[list[PatientModifier], list[dict[str, Any]]]:
    claimed: set[tuple[int, int]] = set()
    seen: set[str] = set()
    located: list[tuple[int, int, str]] = []
    dropped: list[dict[str, Any]] = []

    for draft in drafts:
        source_span = draft.source_span.strip()
        if not source_span:
            dropped.append(
                {
                    "source_span": draft.source_span,
                    "reason": "empty_source_span",
                }
            )
            continue
        normalized_span, _, _ = _normalized_whitespace_with_map(source_span)
        if normalized_span in seen:
            continue
        seen.add(normalized_span)
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
                    "reason": "source_span_not_found",
                }
            )
            continue
        start, end = result
        claimed.add(result)
        located.append((start, end, raw_text[start:end]))

    modifiers = [
        PatientModifier(
            modifier_id=f"PM{index:03d}",
            source_span=source_span,
            source_start=start,
            source_end=end,
        )
        for index, (start, end, source_span) in enumerate(
            sorted(located, key=lambda value: (value[0], value[1])),
            start=1,
        )
    ]
    return modifiers, dropped


async def extract_patient_modifiers(
    *,
    model: Any,
    raw_text: str,
    technical_retry_limit: int,
) -> tuple[list[PatientModifier], ModifierExtractionAudit]:
    started_at = utc_isoformat()
    started = time.monotonic()
    raw_output: str | None = None
    repair_raw_output: str | None = None
    errors: list[str] = []
    usage: dict[str, Any] = {}
    user_prompt = (
        "请从以下病例原文中提取可能影响当前治疗方案审查的明确患者事实。\n\n"
        f"病例原文：\n{raw_text}"
    )

    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=MODIFIER_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt + _schema_instructions()),
            ],
            technical_retry_limit=technical_retry_limit,
        )
    except Exception as exc:  # noqa: BLE001 - provider boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        return [], ModifierExtractionAudit(
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
        envelope = PatientModifierEnvelope.model_validate(
            _parse_json_object(raw_output)
        )
        modifiers, dropped = validate_modifier_drafts(
            raw_text,
            envelope.modifiers,
        )
        return modifiers, ModifierExtractionAudit(
            status="success" if modifiers else "no_valid_modifier",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output,
            dropped_drafts=dropped,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - JSON/schema boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        first_error: BaseException = exc

    repair_prompt = (
        "上一次输出未通过 JSON 或 Schema 校验。只修复格式和字段，"
        "不得增加病例事实、诊断患者或改写 source_span。\n\n"
        f"校验错误：\n{errors[-1]}\n\n"
        f"上一次输出：\n{raw_output or '<无输出>'}\n\n"
        f"原任务：\n{user_prompt}"
    )
    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=MODIFIER_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=repair_prompt + _schema_instructions()),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        repair_raw_output = message_text(response)
        usage = merge_usage(usage, response_usage(response))
        envelope = PatientModifierEnvelope.model_validate(
            _parse_json_object(repair_raw_output)
        )
        modifiers, dropped = validate_modifier_drafts(
            raw_text,
            envelope.modifiers,
        )
        return modifiers, ModifierExtractionAudit(
            status="repaired" if modifiers else "no_valid_modifier",
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

    return [], ModifierExtractionAudit(
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


def disabled_modifier_audit() -> ModifierExtractionAudit:
    return ModifierExtractionAudit(
        status="disabled",
        started_at=utc_isoformat(),
    )
