import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "yuxi_batch_rag"
    / "export_answers.py"
)
SPEC = importlib.util.spec_from_file_location("yuxi_batch_export", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class YuxiBatchExportTests(unittest.TestCase):
    def test_exports_only_question_and_response_in_source_order(self):
        source_records = [
            {
                "question": "问题一",
                "answer": "回答一",
                "all_tool_calls": [{"name": "query_kb"}],
            },
            {
                "question": "问题二",
                "answer": "回答二",
                "retrieval_calls": [{"result_raw": "检索内容"}],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "answers.json"
            input_path.write_text(
                "\n".join(
                    json.dumps(record, ensure_ascii=False)
                    for record in source_records
                ),
                encoding="utf-8",
            )

            count, empty_count = MODULE.export_answers(input_path, output_path)

            self.assertEqual(count, 2)
            self.assertEqual(empty_count, 0)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [
                    {"question": "问题一", "response": "回答一"},
                    {"question": "问题二", "response": "回答二"},
                ],
            )

    def test_reports_invalid_json_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "answers.json"
            input_path.write_text(
                '{"question":"问题一","answer":"回答一"}\nnot-json\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.ExportError, "line 2"):
                MODULE.export_answers(input_path, output_path)

    def test_rejects_record_without_answer_field(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "answers.json"
            input_path.write_text('{"question":"问题一"}\n', encoding="utf-8")

            with self.assertRaisesRegex(MODULE.ExportError, "'answer'"):
                MODULE.export_answers(input_path, output_path)
