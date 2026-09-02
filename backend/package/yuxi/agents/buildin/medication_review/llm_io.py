from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from yuxi.utils.datetime_utils import utc_isoformat

from .models import StructuredInvocationAudit

OutputModel = TypeVar("OutputModel", bound=BaseModel)


class StructuredOutputError(ValueError):
    def __init__(self, message: str, audit: StructuredInvocationAudit):
        super().__init__(message)
        self.audit = audit


class ProviderInvocationError(RuntimeError):
    pass


@dataclass(frozen=True)
class StructuredInvocationResult:
    value: BaseModel
    audit: StructuredInvocationAudit


def _message_text(message: Any) -> str:
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


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    start = text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    parsed, end = json.JSONDecoder().raw_decode(text[start:])
    trailing = text[start + end :].strip()
    if trailing:
        raise ValueError("JSON 对象后存在额外文本")
    if not isinstance(parsed, dict):
        raise ValueError("结构化输出必须是 JSON 对象")
    return parsed


def _usage_from_response(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return dict(usage)
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(token_usage, dict):
            return dict(token_usage)
    return {}


def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, int | float) and isinstance(result.get(key), int | float):
            result[key] += value
        elif key not in result:
            result[key] = value
    return result


def _schema_instructions(output_model: type[BaseModel]) -> str:
    schema = json.dumps(
        output_model.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\n只返回一个符合下方 JSON Schema 的 JSON 对象。"
        "不要使用 Markdown 代码围栏，不要附加解释或其它文本。"
        "字段名、嵌套层级、必填字段和枚举值必须严格遵守 Schema；"
        "不要输出 Schema 中不存在的字段。\n\n"
        f"目标 JSON Schema：\n{schema}"
    )


def _is_auth_error(exc: BaseException) -> bool:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return any(
        marker in name or marker in message
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


def _is_schema_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ),
    )


async def _provider_call(
    *,
    model: Any,
    messages: list[Any],
    technical_retry_limit: int,
) -> Any:
    last_error: BaseException | None = None
    for _attempt in range(technical_retry_limit + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider adapters expose heterogeneous errors
            last_error = exc
            if _is_auth_error(exc):
                break
    assert last_error is not None
    raise ProviderInvocationError(str(last_error)) from last_error


async def invoke_json_schema(
    *,
    model: Any,
    stage: str,
    system_prompt: str,
    user_prompt: str,
    output_model: type[OutputModel],
    repair_limit: int = 1,
    technical_retry_limit: int = 0,
    retain_raw_output: bool = True,
) -> StructuredInvocationResult:
    started_at = utc_isoformat()
    started = time.monotonic()
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = []
    usage: dict[str, Any] = {}

    try:
        response = await _provider_call(
            model=model,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt + _schema_instructions(output_model)),
            ],
            technical_retry_limit=technical_retry_limit,
        )
    except ProviderInvocationError as exc:
        audit = StructuredInvocationAudit(
            stage=stage,
            schema_name=output_model.__name__,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            failure_kind="provider",
            error_type=type(exc.__cause__ or exc).__name__,
            error_message=str(exc),
        )
        raise StructuredOutputError(f"{stage} 模型调用失败：{exc}", audit) from exc

    raw_output = _message_text(response)
    usage = _merge_usage(usage, _usage_from_response(response))
    try:
        value = output_model.model_validate(_parse_json_object(raw_output))
        audit = StructuredInvocationAudit(
            stage=stage,
            schema_name=output_model.__name__,
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output if retain_raw_output else None,
            usage=usage,
        )
        return StructuredInvocationResult(value=value, audit=audit)
    except Exception as exc:  # noqa: BLE001 - parse and pydantic validation only
        if not _is_schema_error(exc):
            raise
        validation_errors.append(f"{type(exc).__name__}: {exc}")
        first_error: BaseException = exc

    if repair_limit > 0:
        repair_prompt = (
            "上一次结构化输出未通过 JSON 或 Schema 校验。只修复格式和字段，"
            "不得改变任务事实、增加来源外内容或省略原任务要求。\n\n"
            f"校验错误：\n{validation_errors[-1]}\n\n"
            f"上一次输出：\n{raw_output or '<无输出>'}\n\n"
            f"原任务：\n{user_prompt}"
        )
        try:
            response = await _provider_call(
                model=model,
                messages=[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=repair_prompt + _schema_instructions(output_model)),
                ],
                technical_retry_limit=technical_retry_limit,
            )
            repair_raw_output = _message_text(response)
            usage = _merge_usage(usage, _usage_from_response(response))
            value = output_model.model_validate(_parse_json_object(repair_raw_output))
            audit = StructuredInvocationAudit(
                stage=stage,
                schema_name=output_model.__name__,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                raw_output=raw_output if retain_raw_output else None,
                repair_raw_output=repair_raw_output if retain_raw_output else None,
                validation_errors=validation_errors,
                repaired=True,
                usage=usage,
            )
            return StructuredInvocationResult(value=value, audit=audit)
        except ProviderInvocationError as exc:
            validation_errors.append(f"{type(exc).__name__}: {exc}")
            first_error = exc
        except Exception as exc:  # noqa: BLE001 - second parse/schema boundary
            validation_errors.append(f"{type(exc).__name__}: {exc}")
            first_error = exc

    audit = StructuredInvocationAudit(
        stage=stage,
        schema_name=output_model.__name__,
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        raw_output=raw_output if retain_raw_output else None,
        repair_raw_output=repair_raw_output if retain_raw_output else None,
        validation_errors=validation_errors,
        repaired=repair_raw_output is not None,
        usage=usage,
        failure_kind=(
            "provider"
            if isinstance(first_error, ProviderInvocationError)
            else "structured"
        ),
        error_type=type(first_error).__name__,
        error_message=str(first_error),
    )
    raise StructuredOutputError(f"{stage} 结构化输出失败：{first_error}", audit) from first_error
