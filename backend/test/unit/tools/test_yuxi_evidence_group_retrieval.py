import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "yuxi_batch_rag"
    / "evaluate_evidence_group_retrieval.py"
)
SPEC = importlib.util.spec_from_file_location(
    "yuxi_evidence_group_retrieval", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def snapshot_record(chunk_id: str, file_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "file_id": file_id,
        "filename": f"{file_id}.md",
        "content": f"content for {chunk_id}",
    }


def evidence(chunk_id: str, file_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "file_id": file_id,
        "raw_metadata": {"chunk_id": chunk_id, "file_id": file_id},
    }


def group(group_id: str, *chunks: tuple[str, str]) -> dict:
    return {
        "group_id": group_id,
        "chunks": [
            {"chunk_id": chunk_id, "file_id": file_id}
            for chunk_id, file_id in chunks
        ],
    }


class YuxiEvidenceGroupRetrievalTests(unittest.TestCase):
    def test_or_groups_and_group_chunks_are_evaluated_with_direct_ids(self):
        snapshot = {
            record["chunk_id"]: record
            for record in (
                snapshot_record("c1", "f1"),
                snapshot_record("c2", "f1"),
                snapshot_record("c3", "f2"),
            )
        }
        case = {
            "gid": 7,
            "subset": "test",
            "doc_no": "doc-7",
            "question": "question",
            "claims": [
                {
                    "claim_id": "C1",
                    "claim_level": "core",
                    "groups": [
                        {
                            "group_id": "C1-G1",
                            "chunk_ids": ["c1", "c2"],
                            "document_ids": ["f1"],
                            "chunk_resolvable": True,
                            "document_resolvable": True,
                        },
                        {
                            "group_id": "C1-G2",
                            "chunk_ids": ["c3"],
                            "document_ids": ["f2"],
                            "chunk_resolvable": True,
                            "document_resolvable": True,
                        },
                    ],
                },
                {
                    "claim_id": "C2",
                    "claim_level": "supporting",
                    "groups": [
                        {
                            "group_id": "C2-G1",
                            "chunk_ids": ["c1", "c2"],
                            "document_ids": ["f1"],
                            "chunk_resolvable": True,
                            "document_resolvable": True,
                        }
                    ],
                },
            ],
        }
        record = {
            "question": "question",
            "result_status": "failed",
            "retrieval_calls": [
                {
                    "tool_name": "search_review_kb",
                    "status": "success",
                    "retrieved_items": [evidence("c3", "f2"), evidence("c1", "f1")],
                },
                {
                    "tool_name": "open_review_evidence",
                    "status": "success",
                    "retrieved_items": [evidence("c2", "f1")],
                },
            ],
            "retrieved_evidence": [
                evidence("c3", "f2"),
                evidence("c1", "f1"),
                evidence("c2", "f1"),
            ],
        }

        detail = MODULE.evaluate_case(
            case,
            record,
            system="system",
            snapshot=snapshot,
            top_k=10,
        )

        search = detail["metrics"]["chunk"]["as_delivered"]["search"]
        final_pool = detail["metrics"]["chunk"]["as_delivered"][
            "final_evidence_pool"
        ]
        self.assertEqual(search["core"]["claim_coverage"], 1.0)
        self.assertEqual(search["supporting"]["claim_coverage"], 0.0)
        self.assertEqual(search["all"]["claim_coverage"], 0.5)
        self.assertEqual(final_pool["all"]["claim_coverage"], 1.0)
        self.assertEqual(
            detail["metrics"]["document"]["as_delivered"]["search"]["all"][
                "claim_coverage"
            ],
            1.0,
        )
        self.assertEqual(detail["retrieval_efficiency"]["search_call_count"], 1)
        self.assertEqual(detail["source_result_status"], "failed")
        self.assertEqual(detail["status"], "retrieval_trace_evaluable")
        self.assertIn("source_result_status:failed", detail["warnings"])

    def test_snapshot_resolvable_mode_excludes_only_unresolvable_claims(self):
        snapshot = {"present": snapshot_record("present", "file-present")}
        raw_group = group("missing-group", ("missing", "file-missing"))
        parsed_group = MODULE._gold_group(
            raw_group,
            claim_id="C-missing",
            group_index=0,
            snapshot=snapshot,
        )
        case = {
            "claims": [
                {
                    "claim_id": "C-missing",
                    "claim_level": "core",
                    "groups": [parsed_group],
                }
            ]
        }

        strict = MODULE._claims_for_scope(
            case, unit="chunk", mode="as_delivered", scope="all"
        )
        resolvable = MODULE._claims_for_scope(
            case, unit="chunk", mode="snapshot_resolvable", scope="all"
        )

        self.assertEqual(len(strict), 1)
        self.assertEqual(resolvable, [])
        self.assertEqual(parsed_group["missing_snapshot_chunk_ids"], ["missing"])

    def test_run_evaluation_writes_reproducible_outputs_and_paired_result(self):
        gold = {
            "cases": [
                {
                    "gid": 11,
                    "subset": "test",
                    "doc_no": "doc-11",
                    "question": "same question",
                    "claims": [
                        {
                            "claim_id": "C11",
                            "claim_level": "core",
                            "evidence_groups": [group("C11-G1", ("c1", "f1"))],
                        }
                    ],
                }
            ]
        }
        result = [
            {
                "row_index": 0,
                "question": "same question",
                "result_status": "success",
                "retrieval_calls": [
                    {
                        "tool_name": "search_review_kb",
                        "status": "success",
                        "retrieved_items": [evidence("c1", "f1")],
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = root / "chunks.jsonl"
            gold_path = root / "gold.json"
            primary_path = root / "primary.json"
            baseline_path = root / "baseline.json"
            output_dir = root / "evaluation"
            snapshot_path.write_text(
                json.dumps(snapshot_record("c1", "f1")) + "\n",
                encoding="utf-8",
            )
            gold_path.write_text(json.dumps(gold), encoding="utf-8")
            primary_path.write_text(json.dumps(result), encoding="utf-8")
            baseline_path.write_text(json.dumps(result), encoding="utf-8")

            summary = MODULE.run_evaluation(
                gold_path=gold_path,
                snapshot_path=snapshot_path,
                result_paths=[
                    ("primary", primary_path),
                    ("baseline", baseline_path),
                ],
                gids=[11],
                top_k=10,
                output_dir=output_dir,
                pair=("primary", "baseline"),
            )

            self.assertEqual(summary["systems"]["primary"]["matched_gids"], [11])
            self.assertEqual(summary["paired_comparison"]["pair_count"], 1)
            self.assertEqual(
                summary["paired_comparison"]["metrics"]["chunk"]["as_delivered"]
                ["search"]["all"]["claim_coverage"]["ties"],
                1,
            )
            for filename in (
                "primary_detail.jsonl",
                "baseline_detail.jsonl",
                "summary.json",
                "summary.csv",
                "REPORT.md",
            ):
                self.assertTrue((output_dir / filename).exists())
