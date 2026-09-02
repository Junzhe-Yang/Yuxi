"""Export lossless messages and a readable event timeline from Yuxi sessions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


class ExportError(ValueError):
    """Raised when a batch result or conversation export cannot be read safely."""


THINK_PATTERN = re.compile(r"<think>(.*?)</think>|<think>(.*)$", re.DOTALL)


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _try_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _additional_kwargs(message: dict[str, Any]) -> dict[str, Any]:
    candidates = [message.get("additional_kwargs")]
    extra = message.get("extra_metadata")
    if isinstance(extra, dict):
        candidates.append(extra.get("additional_kwargs"))
        raw_message = extra.get("raw_message")
        if isinstance(raw_message, dict):
            candidates.append(raw_message.get("additional_kwargs"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("reasoning_content") is not None:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _assistant_body(message: dict[str, Any]) -> tuple[str, str, str]:
    raw_content = _content_to_text(message.get("content"))
    reasoning = _content_to_text(
        _additional_kwargs(message).get("reasoning_content")
    ).strip()
    visible_content = raw_content
    if raw_content:
        matches = list(THINK_PATTERN.finditer(raw_content))
        if matches:
            embedded_reasoning = "\n\n".join(
                (match.group(1) or match.group(2) or "").strip()
                for match in matches
                if (match.group(1) or match.group(2) or "").strip()
            )
            if not reasoning:
                reasoning = embedded_reasoning
            visible_content = THINK_PATTERN.sub("", raw_content).strip()
    return visible_content, reasoning, raw_content


def _message_tool_calls(message: dict[str, Any]) -> list[Any]:
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        return calls
    extra = message.get("extra_metadata")
    if isinstance(extra, dict) and isinstance(extra.get("tool_calls"), list):
        return extra["tool_calls"]
    return []


def _message_type(message: dict[str, Any]) -> str:
    return str(message.get("type") or message.get("role") or "unknown").lower()


def _tool_result(call: dict[str, Any]) -> tuple[bool, Any]:
    if "result" in call:
        return True, call.get("result")
    result = call.get("tool_call_result")
    if isinstance(result, dict) and "content" in result:
        return True, result.get("content")
    if result is not None:
        return True, result
    return False, None


def _normalize_tool_call(
    call: dict[str, Any],
    *,
    message_index: int,
    message_id: Any,
) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    tool_name = str(call.get("name") or function.get("name") or "unknown")
    args = call.get("args")
    if args is None:
        args = function.get("arguments") or {}
    has_result, result = _tool_result(call)
    return {
        "message_index": message_index,
        "message_id": message_id,
        "tool_call_id": call.get("id") or call.get("tool_call_id"),
        "tool_name": tool_name,
        "args": args,
        "status": call.get("status"),
        "error": call.get("error") or call.get("error_message"),
        "has_result": has_result,
        "result": result,
        "result_parsed": _try_parse_json(result),
        "raw_call": call,
    }


def _messages_from_source(source: dict[str, Any], source_index: int) -> list[dict[str, Any]]:
    messages = source.get("history")
    if not isinstance(messages, list):
        messages = source.get("messages")
    if not isinstance(messages, list):
        raise ExportError(
            f"Record {source_index} has neither a 'history' nor a 'messages' list"
        )
    if any(not isinstance(message, dict) for message in messages):
        raise ExportError(f"Record {source_index} contains a non-object message")
    return messages


def _compact_session(source: Any, source_index: int) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ExportError(f"Record {source_index} must be a JSON object")
    messages = _messages_from_source(source, source_index)

    final_message_index: int | None = None
    final_answer = ""
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if _message_type(message) not in {"ai", "assistant"}:
            continue
        visible_content, _reasoning, _raw = _assistant_body(message)
        if visible_content.strip():
            final_message_index = message_index
            final_answer = visible_content
            break
    if not final_answer and isinstance(source.get("answer"), str):
        final_answer = source["answer"]

    question = source.get("question")
    if not isinstance(question, str) or not question.strip():
        question = next(
            (
                _content_to_text(message.get("content"))
                for message in messages
                if _message_type(message) in {"human", "user"}
                and _content_to_text(message.get("content")).strip()
            ),
            "",
        )

    timeline: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    def append_event(event: dict[str, Any]) -> None:
        timeline.append({"event_index": len(timeline) + 1, **event})

    for message_index, message in enumerate(messages):
        message_type = _message_type(message)
        common = {
            "message_index": message_index,
            "message_id": message.get("id"),
            "created_at": message.get("created_at"),
        }
        if message_type in {"ai", "assistant"}:
            visible_content, reasoning, raw_content = _assistant_body(message)
            calls = _message_tool_calls(message)
            if reasoning:
                append_event(
                    {
                        **common,
                        "event_type": "assistant_reasoning",
                        "content": reasoning,
                    }
                )
            if visible_content:
                event_type = (
                    "assistant_final"
                    if message_index == final_message_index
                    else "assistant_intermediate"
                    if calls
                    else "assistant_message"
                )
                append_event(
                    {
                        **common,
                        "event_type": event_type,
                        "content": visible_content,
                        "raw_content": raw_content,
                    }
                )
            for call in calls:
                if not isinstance(call, dict):
                    continue
                normalized = _normalize_tool_call(
                    call,
                    message_index=message_index,
                    message_id=message.get("id"),
                )
                tool_calls.append(normalized)
                append_event(
                    {
                        **common,
                        "event_type": "tool_call",
                        "tool_call_id": normalized["tool_call_id"],
                        "tool_name": normalized["tool_name"],
                        "args": normalized["args"],
                    }
                )
                if normalized["has_result"] or normalized["status"] or normalized["error"]:
                    append_event(
                        {
                            **common,
                            "event_type": "tool_result",
                            "tool_call_id": normalized["tool_call_id"],
                            "tool_name": normalized["tool_name"],
                            "status": normalized["status"],
                            "error": normalized["error"],
                            "content": normalized["result"],
                            "content_parsed": normalized["result_parsed"],
                        }
                    )
            continue

        event_type = {
            "human": "user_message",
            "user": "user_message",
            "system": "system_message",
            "tool": "tool_message",
        }.get(message_type, "message")
        append_event(
            {
                **common,
                "event_type": event_type,
                "role": message_type,
                "content": message.get("content"),
            }
        )

    result = {
        "schema_version": "1.0",
        "source_index": source_index,
        "question": question,
        "final_answer": final_answer,
        "message_count": len(messages),
        "tool_call_count": len(tool_calls),
        "timeline": timeline,
        "tool_calls": tool_calls,
        "messages": messages,
    }
    for field in (
        "job_key",
        "batch_id",
        "row_index",
        "variant",
        "attempt",
        "thread_id",
        "run_id",
        "request_id",
        "run_status",
        "result_status",
        "title",
        "agent_id",
        "agent_config_id",
        "created_at",
        "updated_at",
    ):
        if field in source:
            result[field] = source[field]
    return result


def _read_records(input_path: Path) -> list[dict[str, Any]]:
    try:
        if input_path.suffix.lower() == ".jsonl":
            records: list[dict[str, Any]] = []
            with input_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ExportError(
                            f"Invalid JSON on line {line_number}: {exc.msg}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ExportError(
                            f"Line {line_number} must contain a JSON object"
                        )
                    records.append(value)
            return records

        value = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExportError(f"Cannot read input file '{input_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"Invalid JSON: {exc.msg}") from exc

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ExportError("Input JSON must be one session object or a list of session objects")


def export_session_records(
    input_path: Path,
    output_path: Path,
) -> tuple[int, int, int]:
    """Export sessions and return (session_count, message_count, tool_call_count)."""
    if input_path.resolve() == output_path.resolve():
        raise ExportError("--input and --output cannot point to the same file")

    sources = _read_records(input_path)
    records = [
        _compact_session(source, source_index)
        for source_index, source in enumerate(sources, start=1)
    ]
    message_count = sum(record["message_count"] for record in records)
    tool_call_count = sum(record["tool_call_count"] for record in records)

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

    return len(records), message_count, tool_call_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete Yuxi session messages plus a normalized timeline of "
            "assistant reasoning, intermediate text, tool calls/results, and answers."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Batch result JSONL or one/list of conversation JSON objects",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        session_count, message_count, tool_call_count = export_session_records(
            args.input,
            args.output,
        )
    except ExportError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Exported {session_count} sessions, {message_count} messages, and "
        f"{tool_call_count} tool calls to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
