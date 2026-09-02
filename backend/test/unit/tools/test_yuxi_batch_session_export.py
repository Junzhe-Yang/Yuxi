import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "yuxi_batch_rag"
    / "export_session_records.py"
)
SPEC = importlib.util.spec_from_file_location(
    "yuxi_batch_session_export",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class YuxiBatchSessionExportTests(unittest.TestCase):
    def test_exports_reasoning_tool_activity_and_final_answer_losslessly(self):
        messages = [
            {
                "id": 1,
                "type": "human",
                "created_at": "2026-01-01T00:00:00Z",
                "content": "病例问题",
            },
            {
                "id": 2,
                "type": "ai",
                "created_at": "2026-01-01T00:00:01Z",
                "content": "<think>先核对剂量证据</think>准备检索。",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "search_review_kb",
                        "args": {"query_text": "剂量 调整"},
                        "status": "success",
                        "result": '{"evidence_ids":["EV-1"]}',
                    }
                ],
            },
            {
                "id": 3,
                "type": "ai",
                "created_at": "2026-01-01T00:00:02Z",
                "content": "最终回答",
                "extra_metadata": {
                    "additional_kwargs": {"reasoning_content": "综合现有证据"}
                },
            },
        ]
        source = {
            "job_key": "full:000001",
            "row_index": 1,
            "variant": "full",
            "thread_id": "thread-1",
            "question": "病例问题",
            "answer": "最终回答",
            "history": messages,
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "sessions.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_session_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 3, 1))
        self.assertEqual(exported["final_answer"], "最终回答")
        self.assertEqual(exported["messages"], messages)
        self.assertEqual(exported["tool_calls"][0]["tool_name"], "search_review_kb")
        self.assertEqual(
            exported["tool_calls"][0]["result_parsed"]["evidence_ids"],
            ["EV-1"],
        )
        self.assertEqual(
            [event["event_type"] for event in exported["timeline"]],
            [
                "user_message",
                "assistant_reasoning",
                "assistant_intermediate",
                "tool_call",
                "tool_result",
                "assistant_reasoning",
                "assistant_final",
            ],
        )

    def test_accepts_single_conversation_dump_json(self):
        source = {
            "thread_id": "thread-2",
            "title": "导出的会话",
            "agent_id": "MedicationReviewAcmPrimAgent",
            "messages": [
                {"id": 1, "role": "human", "content": "用户输入"},
                {
                    "id": 2,
                    "role": "ai",
                    "content": "会话回答",
                    "tool_calls": [],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "conversation.json"
            output_path = Path(directory) / "session.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False),
                encoding="utf-8",
            )

            counts = MODULE.export_session_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 2, 0))
        self.assertEqual(exported["question"], "用户输入")
        self.assertEqual(exported["final_answer"], "会话回答")
        self.assertEqual(exported["thread_id"], "thread-2")

    def test_reports_invalid_jsonl_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "sessions.json"
            input_path.write_text(
                '{"history": []}\nnot-json\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.ExportError, "line 2"):
                MODULE.export_session_records(input_path, output_path)


if __name__ == "__main__":
    unittest.main()
