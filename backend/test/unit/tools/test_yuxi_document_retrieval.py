import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "yuxi_batch_rag"
    / "evaluate_document_retrieval.py"
)
SPEC = importlib.util.spec_from_file_location("yuxi_document_retrieval", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def chunk(source: str, chunk_id: str, content: str = "content") -> dict:
    return {
        "content": content,
        "metadata": {
            "source": source,
            "file_id": source.split("_")[0],
            "chunk_id": chunk_id,
        },
    }


def prim_evidence(
    source: str,
    file_id: str,
    chunk_id: str,
    *,
    rank: int = 1,
) -> dict:
    return {
        "evidence_id": f"EV-{chunk_id}",
        "raw_text": "evidence text",
        "source_document": source,
        "file_id": file_id,
        "chunk_id": chunk_id,
        "rank": rank,
        "raw_metadata": {
            "source": source,
            "file_id": file_id,
            "chunk_id": chunk_id,
        },
    }


class YuxiDocumentRetrievalTests(unittest.TestCase):
    def test_document_name_normalization_matches_prefix_suffix_and_path(self):
        gold = ["Elderly TB Consensus (2023)"]
        retrieved = [
            chunk(
                "/home/kb/kb_Elderly TB Consensus （2023）_chunk_001.pdf",
                "file-1_chunk-1",
            )
        ]

        result = MODULE.evaluate_ranked_items(retrieved, gold, k_values=[1, 5])

        self.assertEqual(
            result["items"][0]["document_match_status"],
            "matched",
        )
        self.assertEqual(result["metrics"]["required_document_hit@1"], 1)
        self.assertEqual(result["metrics"]["required_document_recall@5"], 1.0)

    def test_multiple_gold_documents_use_recall_and_first_rank(self):
        retrieved = [
            chunk("unrelated.txt", "unrelated-1"),
            chunk("doc-a.pdf", "a-1"),
            chunk("doc-b.md", "b-1"),
        ]

        result = MODULE.evaluate_ranked_items(
            retrieved,
            ["doc-a", "doc-b"],
            k_values=[1, 2, 3],
        )

        self.assertEqual(result["metrics"]["required_document_recall@1"], 0.0)
        self.assertEqual(result["metrics"]["required_document_recall@2"], 0.5)
        self.assertEqual(result["metrics"]["required_document_recall@3"], 1.0)
        self.assertEqual(result["metrics"]["first_required_document_rank"], 2)
        self.assertEqual(result["metrics"]["document_mrr"], 0.5)

    def test_duplicate_chunks_are_removed_and_distinct_chunks_share_one_document(self):
        retrieved = [
            chunk("doc-a.pdf", "a-1"),
            chunk("doc-a.pdf", "a-1"),
            chunk("other.pdf", "other-1"),
            chunk("other.pdf", "other-2"),
        ]

        result = MODULE.evaluate_ranked_items(
            retrieved,
            ["doc-a"],
            k_values=[4],
        )

        self.assertEqual(result["raw_item_count"], 4)
        self.assertEqual(result["retrieved_item_count"], 3)
        self.assertEqual(result["duplicate_item_count"], 1)
        self.assertEqual(result["retrieved_document_count"], 2)
        self.assertEqual(result["documents"][1]["evidence_item_count"], 2)
        self.assertNotIn("required_document_chunk_ratio@4", result["metrics"])

    def test_annotated_fraction_is_diagnostic_and_unjudged_is_not_negative_label(self):
        retrieved = [
            chunk("doc-a.pdf", "a-1"),
            chunk("other.pdf", "other-1"),
            chunk("other.pdf", "other-2"),
        ]

        result = MODULE.evaluate_ranked_items(
            retrieved,
            ["doc-a"],
            k_values=[3],
        )

        self.assertEqual(
            result["metrics"]["annotated_required_document_fraction@3"],
            0.5,
        )
        self.assertEqual(result["metrics"]["unjudged_document_rate@3"], 0.5)
        self.assertEqual(result["metrics"]["ambiguous_document_rate@3"], 0.0)

    def test_aliases_can_resolve_a_non_substring_document_name(self):
        retrieved = [chunk("consensus_2023.pdf", "c-1")]
        aliases = {"Elderly TB Consensus": ["consensus_2023"]}

        result = MODULE.evaluate_ranked_items(
            retrieved,
            ["Elderly TB Consensus"],
            aliases=aliases,
            k_values=[1],
        )

        self.assertEqual(result["metrics"]["required_document_hit@1"], 1)
        self.assertEqual(result["items"][0]["matched_gold_documents"], ["Elderly TB Consensus"])

    def test_search_union_and_final_evidence_pool_are_reported_separately(self):
        gold = {"case_id": "case-1", "question": "question", "documents": ["doc-b"]}
        result = {
            "case_id": "case-1",
            "result_status": "success",
            "retrieval_calls": [
                {
                    "call_index": 1,
                    "status": "success",
                    "args": {"kb_name": "kb", "query_text": "question"},
                    "result_parsed": [chunk("doc-a.pdf", "a-1")],
                },
                {
                    "call_index": 2,
                    "status": "success",
                    "args": {"kb_name": "kb", "query_text": "question reformulated"},
                    "result_parsed": [chunk("doc-b.pdf", "b-1")],
                },
            ],
        }

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="lightrag",
            k_values=[1, 2],
        )

        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_hit@1"],
            0,
        )
        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_hit@2"],
            1,
        )
        self.assertEqual(evaluated["search_union"]["call_count"], 2)
        self.assertEqual(
            evaluated["final_evidence_pool"]["metrics"][
                "required_document_hit_all"
            ],
            1,
        )

    def test_lightrag_result_parsed_chunks_are_supported(self):
        gold = {"case_id": "case-1", "question": "question", "documents": ["doc-a"]}
        result = {
            "case_id": "case-1",
            "result_status": "success",
            "retrieval_calls": [
                {
                    "status": "success",
                    "result_parsed": {
                        "entities": [{"entity_name": "entity"}],
                        "relationships": [],
                        "chunks": [chunk("doc-a.pdf", "a-1")],
                    },
                }
            ],
        }

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="lightrag",
            k_values=[1],
        )

        self.assertEqual(evaluated["search_union"]["retrieved_item_count"], 1)
        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_hit@1"],
            1,
        )

    def test_successful_call_with_no_items_counts_as_not_retrieved(self):
        gold = {"case_id": "case-1", "question": "question", "documents": ["doc-a"]}
        result = {
            "case_id": "case-1",
            "result_status": "success",
            "retrieval_calls": [
                {"status": "success", "retrieved_items": []},
            ],
        }

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="vector",
            k_values=[1],
        )

        self.assertEqual(evaluated["status"], "not_retrieved")
        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_hit@1"],
            0,
        )

    def test_missing_query_call_is_not_retrieved_and_is_counted_in_summary(self):
        gold = {"case_id": "case-1", "question": "question", "documents": ["doc-a"]}
        result = {"case_id": "case-1", "result_status": "success", "retrieval_calls": []}

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="vector",
            k_values=[1],
        )
        summary = MODULE.summarize_details(
            [evaluated], condition="vector", k_values=[1]
        )

        self.assertEqual(evaluated["status"], "not_retrieved")
        self.assertEqual(summary["valid_case_count"], 1)
        self.assertEqual(summary["status_counts"]["not_retrieved"], 1)
        self.assertEqual(summary["warning_counts"]["no_successful_search_call"], 1)
        self.assertEqual(summary["warning_counts"]["no_retrieved_evidence"], 1)

    def test_failed_result_is_excluded_from_valid_metric_denominator(self):
        gold_records = [
            {"case_id": "case-1", "question": "q1", "documents": ["doc-a"]},
            {"case_id": "case-2", "question": "q2", "documents": ["doc-a"]},
        ]
        result_records = [
            {
                "case_id": "case-1",
                "result_status": "failed",
                "retrieval_calls": [],
            },
            {
                "case_id": "case-2",
                "result_status": "success",
                "retrieval_calls": [
                    {"status": "success", "result_parsed": [chunk("doc-a.pdf", "a-1")]}
                ],
            },
        ]

        details, summary = MODULE.evaluate_dataset(
            gold_records,
            result_records,
            condition="vector",
            k_values=[1],
        )

        self.assertEqual([detail["status"] for detail in details], ["result_failed", "ok"])
        self.assertEqual(summary["valid_case_count"], 1)
        self.assertEqual(
            summary["search_union"]["macro_metrics"]["required_document_hit@1"],
            1.0,
        )

    def test_load_records_accepts_single_json_object_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            single_path = directory_path / "single.jsonl"
            single_path.write_text(
                json.dumps({"case_id": "case-1"}, ensure_ascii=False),
                encoding="utf-8",
            )
            jsonl_path = directory_path / "records.jsonl"
            jsonl_path.write_text(
                json.dumps({"case_id": "case-1"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"case_id": "case-2"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(len(MODULE.load_records(single_path)), 1)
            self.assertEqual(len(MODULE.load_records(jsonl_path)), 2)

    def test_json_array_streaming_handles_a_comma_at_the_read_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.json"
            empty_record = json.dumps(
                {"payload": ""},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            padding_size = (
                MODULE.JSON_READ_CHUNK_SIZE - len("[") - len(empty_record) - len(",")
            )
            first_record = {"payload": "x" * padding_size}
            first_json = json.dumps(
                first_record,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            self.assertEqual(
                len("[") + len(first_json) + len(","),
                MODULE.JSON_READ_CHUNK_SIZE,
            )
            path.write_text(
                f"[{first_json},"
                + json.dumps({"payload": "second"}, separators=(",", ":"))
                + "]",
                encoding="utf-8",
            )

            records = list(MODULE.iter_records(path))

            self.assertEqual(len(records), 2)
            self.assertEqual(records[1]["payload"], "second")

    def test_cli_writes_detail_and_summary_files(self):
        gold_records = [
            {
                "case_id": "case-1",
                "question": "question",
                "documents": ["doc-a"],
            }
        ]
        result_records = [
            {
                "case_id": "case-1",
                "variant": "vector",
                "result_status": "success",
                "retrieval_calls": [
                    {
                        "status": "success",
                        "retrieved_items": [chunk("doc-a.pdf", "a-1")],
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            gold_path = directory_path / "gold.json"
            results_path = directory_path / "results.json"
            output_dir = directory_path / "evaluation"
            gold_path.write_text(json.dumps(gold_records, ensure_ascii=False), encoding="utf-8")
            results_path.write_text(
                json.dumps(result_records, ensure_ascii=False), encoding="utf-8"
            )

            argv = [
                "evaluate_document_retrieval.py",
                "--gold",
                str(gold_path),
                "--results",
                str(results_path),
                "--output-dir",
                str(output_dir),
                "--k",
                "1",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(MODULE.main(), 0)

            self.assertTrue((output_dir / "vector_detail.jsonl").exists())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["conditions"][0]["condition"], "vector")
            self.assertEqual(
                summary["conditions"][0]["search_union"]["macro_metrics"][
                    "required_document_hit@1"
                ],
                1.0,
            )

    def test_reference_citations_extract_unique_documents_and_ignore_chunk_labels(self):
        reference = (
            "【依据：Beers标准2023译文#48 · 表7 · #48】"
            "【依据：Beers标准2023译文 · 表3 · #34】"
            "【依据：老年肺结核诊断与治疗专家共识（2023版） · 治疗原则 · #16】"
        )

        documents = MODULE.extract_reference_documents(reference)

        self.assertEqual(
            documents,
            [
                "Beers标准2023译文",
                "老年肺结核诊断与治疗专家共识（2023版）",
            ],
        )

    def test_prim_rag_source_document_matches_gold_reference(self):
        gold = {
            "question": "question",
            "reference": (
                "【依据：中国老年高血压管理指南（2023） · 3.3 联合应用 · #45】"
            ),
        }
        evidence = prim_evidence(
            "【用药助手】中国老年高血压管理指南（2023）"
            ".pdf_by_PaddleOCR-VL-1.6.md",
            "file_68fb16",
            "file_68fb16_chunk_45",
        )
        result = {
            "row_index": 0,
            "question": "question",
            "result_status": "success",
            "retrieval_calls": [
                {
                    "call_index": 1,
                    "tool_name": "search_review_kb",
                    "status": "success",
                    "retrieved_items": [evidence],
                }
            ],
            "retrieved_evidence": [evidence],
        }

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="full",
            k_values=[1],
        )

        self.assertEqual(evaluated["status"], "ok")
        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_recall@1"],
            1.0,
        )
        self.assertEqual(
            evaluated["search_union"]["documents"][0]["source_field"],
            "source_document",
        )

    def test_rank_cutoffs_count_unique_documents_in_first_discovery_order(self):
        retrieved = [
            chunk("doc-a.pdf", "a-1"),
            chunk("doc-a.pdf", "a-2"),
            chunk("doc-b.pdf", "b-1"),
        ]

        result = MODULE.evaluate_ranked_items(
            retrieved,
            ["doc-a", "doc-b"],
            k_values=[1, 2],
        )

        self.assertEqual(result["retrieved_document_count"], 2)
        self.assertEqual(result["metrics"]["required_document_recall@1"], 0.5)
        self.assertEqual(result["metrics"]["required_document_recall@2"], 1.0)

    def test_open_evidence_is_excluded_from_search_rank_but_in_final_pool(self):
        search_evidence = prim_evidence("doc-a.pdf", "file-a", "a-1")
        opened_evidence = prim_evidence("doc-b.pdf", "file-b", "b-1")
        opened_evidence["occurrences"] = [{"source_method": "open"}]
        gold = {
            "question": "question",
            "documents": ["doc-b"],
        }
        result = {
            "row_index": 0,
            "question": "question",
            "result_status": "success",
            "retrieval_calls": [
                {
                    "call_index": 1,
                    "tool_name": "search_review_kb",
                    "status": "success",
                    "retrieved_items": [search_evidence],
                },
                {
                    "call_index": 2,
                    "tool_name": "open_review_evidence",
                    "status": "success",
                    "retrieved_items": [opened_evidence],
                },
            ],
            "retrieved_evidence": [search_evidence, opened_evidence],
        }

        evaluated = MODULE.evaluate_record(
            gold,
            result,
            row_index=0,
            variant="full",
            k_values=[1],
        )

        self.assertEqual(
            evaluated["search_union"]["metrics"]["required_document_recall_all"],
            0.0,
        )
        self.assertEqual(
            evaluated["final_evidence_pool"]["metrics"][
                "required_document_recall_all"
            ],
            1.0,
        )

    def test_success_empty_is_a_completed_search_attempt(self):
        calls = MODULE.successful_retrieval_calls(
            {
                "retrieval_calls": [
                    {
                        "tool_name": "search_review_kb",
                        "status": "success_empty",
                        "retrieved_items": [],
                    },
                    {
                        "tool_name": "search_review_kb",
                        "status": "technical_failed",
                        "retrieved_items": [],
                    },
                ]
            }
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "success_empty")

    def test_row_index_alignment_rejects_a_different_question(self):
        gold_records = [
            {"question": "gold question", "documents": ["doc-a"]},
        ]
        result_records = [
            {
                "row_index": 0,
                "question": "different question",
                "result_status": "success",
                "retrieval_calls": [],
            }
        ]

        details, summary = MODULE.evaluate_dataset(
            gold_records,
            result_records,
            condition="vector",
            k_values=[1],
        )

        self.assertEqual(details[0]["status"], "alignment_error")
        self.assertEqual(summary["valid_case_count"], 0)
        self.assertEqual(summary["warning_counts"]["question_mismatch"], 1)

    def test_question_alignment_wins_over_sample_local_row_index(self):
        gold_records = [
            {"question": "q0", "documents": ["doc-a"]},
            {"question": "q1", "documents": ["doc-b"]},
        ]
        result_records = [
            {
                "row_index": 0,
                "question": "q1",
                "result_status": "success",
                "retrieval_calls": [
                    {
                        "status": "success",
                        "retrieved_items": [chunk("doc-b.pdf", "b-1")],
                    }
                ],
            }
        ]

        details, summary = MODULE.evaluate_dataset(
            gold_records,
            result_records,
            condition="sample",
            k_values=[1],
        )

        self.assertEqual(details[0]["status"], "missing_result")
        self.assertEqual(details[1]["status"], "ok")
        self.assertEqual(summary["valid_case_count"], 1)
