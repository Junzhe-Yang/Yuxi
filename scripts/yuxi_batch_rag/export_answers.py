"""Export question/response pairs from a Yuxi batch result JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class ExportError(ValueError):
    """Raised when the source result file cannot be exported safely."""


def export_answers(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Export records and return (record_count, empty_response_count)."""
    if input_path.resolve() == output_path.resolve():
        raise ExportError("--input and --output cannot point to the same file")

    records: list[dict[str, str]] = []
    empty_responses = 0

    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    source = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExportError(
                        f"Invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(source, dict):
                    raise ExportError(
                        f"Line {line_number} must contain a JSON object"
                    )

                question = source.get("question")
                response = source.get("answer")
                if not isinstance(question, str) or not question.strip():
                    raise ExportError(
                        f"Line {line_number} has no non-empty string 'question' field"
                    )
                if not isinstance(response, str):
                    raise ExportError(
                        f"Line {line_number} has no string 'answer' field"
                    )
                trace = source.get("medication_review_trace")
                if (
                    isinstance(trace, dict)
                    and trace.get("schema_version") == "3.0"
                    and trace.get("run_status") == "debug_stopped"
                ):
                    raise ExportError(
                        f"Line {line_number} is a debug_stopped diagnostic record "
                        "and cannot be exported as a formal answer"
                    )

                records.append({"question": question, "response": response})
                if not response.strip():
                    empty_responses += 1
    except OSError as exc:
        raise ExportError(f"Cannot read input file '{input_path}': {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExportError(f"Cannot write output file '{output_path}': {exc}") from exc

    return len(records), empty_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Yuxi batch result JSONL file into a JSON list containing "
            "only question and response fields."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        record_count, empty_response_count = export_answers(
            args.input, args.output
        )
    except ExportError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Exported {record_count} records to: {args.output}")
    if empty_response_count:
        print(f"Warning: {empty_response_count} responses are empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
