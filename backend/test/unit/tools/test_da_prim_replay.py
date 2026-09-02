from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[4] / "backend" / "scripts" / "replay_da_prim_retrieval.py"
SPEC = importlib.util.spec_from_file_location("da_prim_replay", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_iter_records_streams_json_array_and_jsonl(tmp_path) -> None:
    values = [{"row_index": 0}, {"row_index": 1}]
    array_path = tmp_path / "records.json"
    array_path.write_text(json.dumps(values), encoding="utf-8")
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(value) for value in values),
        encoding="utf-8",
    )

    assert list(MODULE.iter_records(array_path)) == values
    assert list(MODULE.iter_records(jsonl_path)) == values


def test_gold_documents_and_prefix_suffix_matching() -> None:
    record = {"reference": ("【依据：老年肺结核诊断与治疗专家共识（2023版） · 治疗原则 · #16】")}

    gold = MODULE.gold_documents(record)
    metrics = MODULE._metric_block(
        gold,
        [
            {
                "file_id": "file-1",
                "file_name": "kb_老年肺结核诊断与治疗专家共识(2023版).md",
            }
        ],
    )

    assert gold == ["老年肺结核诊断与治疗专家共识（2023版）"]
    assert metrics["matched_gold_documents"] == gold
    assert metrics["gold_document_recall"] == 1.0


def test_exclusive_path_metrics_distinguish_routed_gain_and_global_escape() -> None:
    gold = ["局部指南", "全局指南", "双路指南"]
    candidates = [
        {"file_name": "局部指南.md", "retrieval_paths": ["local"]},
        {"file_name": "全局指南.md", "retrieval_paths": ["global"]},
        {"file_name": "双路指南.md", "retrieval_paths": ["global", "local"]},
    ]

    assert MODULE._exclusive_path_gold_documents(
        gold,
        candidates,
        required_path="local",
        excluded_path="global",
    ) == ["局部指南"]
    assert MODULE._exclusive_path_gold_documents(
        gold,
        candidates,
        required_path="global",
        excluded_path="local",
    ) == ["全局指南"]
