from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import replay_atlas_companion as replay


def _query(
    query_id: str,
    tool_call_id: str,
    started_at: str,
    elapsed_ms: int,
    status: str = "success",
) -> dict:
    return {
        "query_id": query_id,
        "tool_call_id": tool_call_id,
        "relation_id": "RI-1",
        "query_text": query_id,
        "reason": "test",
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
        "status": status,
        "evidence_ids": (
            [f"EV-{query_id}"]
            if status in {"success", "success_empty"}
            else []
        ),
    }


def _record() -> dict:
    queries = [
        _query("Q-1", "call-1", "2026-01-01T00:00:00Z", 1_000),
        _query("Q-2", "call-2", "2026-01-01T00:00:00.100000Z", 900),
        _query("Q-3", "call-3", "2026-01-01T00:00:02Z", 500),
    ]
    evidence = [
        {
            "evidence_id": f"EV-{query['query_id']}",
            "file_id": f"file-{index}",
            "source_document": f"文档{index}.md",
            "raw_text": "evidence",
            "occurrences": [
                {
                    "record_id": query["query_id"],
                    "tool_call_id": query["tool_call_id"],
                }
            ],
        }
        for index, query in enumerate(queries, start=1)
    ]
    return {
        "question": "病例",
        "method_family": "prim-rag-v1",
        "search_records": queries,
        "retrieved_evidence": evidence,
        "relation_investigations": [
            {
                "relation_id": "RI-1",
                "relation_question": "关系",
                "created_at": "2026-01-01T00:00:00Z",
                "query_ids": ["Q-1", "Q-2", "Q-3"],
                "evidence_ids": ["EV-Q-1", "EV-Q-2", "EV-Q-3"],
                "new_evidence_ids": ["EV-Q-1", "EV-Q-2", "EV-Q-3"],
            }
        ],
    }


def test_iter_records_streams_a_json_array_with_large_items(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    records = [{"id": 1, "text": "x" * 70_000}, {"id": 2}]
    path.write_text(json.dumps(records), encoding="utf-8")

    assert list(replay._iter_records(path)) == records


def test_exported_records_are_adapted_before_history_check() -> None:
    trace = replay._trace(_record())

    assert trace is not None
    assert trace["schema_version"] == "5.0"
    assert [value["query_id"] for value in trace["query_records"]] == [
        "Q-1",
        "Q-2",
        "Q-3",
    ]


def test_first_query_state_uses_history_tool_call_batch() -> None:
    record = _record()
    record["medication_review_trace"] = replay._trace(record)
    record["medication_review_trace"]["query_records"][0][
        "focus_plan_ids"
    ] = ["PE-FIRST"]
    record["medication_review_trace"]["query_records"][2][
        "focus_plan_ids"
    ] = ["PE-LATER"]
    relation = record["medication_review_trace"]["relation_investigations"][0]
    relation["focus_plan_ids"] = ["PE-FIRST", "PE-LATER"]
    relation["retrieval_status"] = "mixed"
    relation["warnings"] = ["后续轮次产生的警告"]
    record["history"] = [
        {
            "type": "ai",
            "tool_calls": [{"id": "call-1"}, {"id": "call-2"}],
        },
        {"type": "ai", "tool_calls": [{"id": "call-3"}]},
    ]

    state = replay._first_query_state(record)

    assert state is not None
    assert state["reconstruction_mode"] == "history_tool_call_batch"
    assert [value["query_id"] for value in state["query_records"]] == [
        "Q-1",
        "Q-2",
    ]
    assert [
        value["id"] for value in state["messages"][0]["tool_calls"]
    ] == ["call-1", "call-2"]
    assert set(state["evidence_store"]) == {"EV-Q-1", "EV-Q-2"}
    reconstructed_relation = state["relation_investigations"][0]
    assert reconstructed_relation["query_ids"] == ["Q-1", "Q-2"]
    assert reconstructed_relation["focus_plan_ids"] == ["PE-FIRST"]
    assert reconstructed_relation["retrieval_status"] == "evidence_returned"
    assert reconstructed_relation["warnings"] == []


def test_first_query_state_keeps_failed_peer_in_trigger_batch() -> None:
    record = _record()
    failed = _query(
        "Q-FAILED",
        "call-failed",
        "2025-12-31T23:59:59.900000Z",
        500,
        status="technical_failed",
    )
    record["medication_review_trace"] = replay._trace(record)
    record["medication_review_trace"]["query_records"].insert(0, failed)
    record["history"] = [
        {
            "type": "ai",
            "tool_calls": [{"id": "call-failed"}, {"id": "call-1"}],
        },
        {"type": "ai", "tool_calls": [{"id": "call-2"}]},
    ]

    state = replay._first_query_state(record)

    assert state is not None
    assert state["trigger_query_id"] == "Q-1"
    assert [value["query_id"] for value in state["query_records"]] == [
        "Q-FAILED",
        "Q-1",
    ]


def test_first_query_state_marks_time_overlap_as_approximation() -> None:
    state = replay._first_query_state(_record())

    assert state is not None
    assert state["reconstruction_mode"] == "time_overlap_approximation"
    assert [value["query_id"] for value in state["query_records"]] == [
        "Q-1",
        "Q-2",
    ]


def test_first_query_state_sorts_unsorted_exported_query_records() -> None:
    record = _record()
    record["medication_review_trace"] = replay._trace(record)
    record["medication_review_trace"]["query_records"].reverse()

    state = replay._first_query_state(record)

    assert state is not None
    assert state["trigger_query_id"] == "Q-1"
    assert [value["query_id"] for value in state["query_records"]] == [
        "Q-1",
        "Q-2",
    ]


def test_first_query_state_accepts_v8_without_leaking_final_investigation() -> None:
    record = _record()
    trace = replay._trace(record)
    assert trace is not None
    trace.update(
        {
            "schema_version": "8.0",
            "method_family": "prim-rag-v2",
            "query_records": [
                {
                    key: value
                    for key, value in query.items()
                    if key != "relation_id"
                }
                | {
                    "investigation_id": "INV-1",
                    "retrieval_scope": "global",
                    "file_id": None,
                    "focus_plan_ids": [],
                    "focus_modifier_ids": [],
                    "atlas_companion_ids": [],
                    "returned_count": len(query.get("evidence_ids") or []),
                    "retained_count": len(query.get("evidence_ids") or []),
                    "new_evidence_ids": query.get("evidence_ids") or [],
                    "attempts": [],
                    "invalid_focus_ids": [],
                    "error_type": None,
                    "error_message": None,
                }
                for query in trace["query_records"]
            ],
            "investigations": [
                {
                    "investigation_id": "INV-1",
                    "question": "最终记录中的调查问题",
                    "origin": "agent",
                    "focus_plan_ids": [],
                    "focus_modifier_ids": [],
                    "atlas_companion_ids": [],
                    "status": "answered",
                    "query_ids": ["Q-1", "Q-2", "Q-3"],
                    "candidate_evidence_ids": [
                        "EV-Q-1",
                        "EV-Q-2",
                        "EV-Q-3",
                    ],
                    "selected_evidence_ids": ["EV-Q-3"],
                    "candidate_file_ids": ["file-1", "file-2", "file-3"],
                    "working_note": "后续 Agent 形成的结论",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:03Z",
                    "warnings": ["后续警告"],
                }
            ],
        }
    )
    record["medication_review_trace"] = trace

    state = replay._first_query_state(record)

    assert state is not None
    assert [value["query_id"] for value in state["query_records"]] == [
        "Q-1",
        "Q-2",
    ]
    investigation = state["investigations"][0]
    assert investigation["status"] == "open"
    assert investigation["query_ids"] == ["Q-1", "Q-2"]
    assert investigation["candidate_evidence_ids"] == ["EV-Q-1", "EV-Q-2"]
    assert investigation["candidate_file_ids"] == ["file-1", "file-2"]
    assert investigation["selected_evidence_ids"] == []
    assert investigation["working_note"] == ""
    assert investigation["warnings"] == []


def test_residual_metrics_reuse_evaluator_direction_and_normalization() -> None:
    record = _record()
    record["reference_documents"] = [
        "中国老年高血压管理指南（2023）",
        "第二份共识",
    ]
    record["retrieved_evidence"] = [
        {
            "source_document": (
                "【用药助手】中国老年高血压管理指南(2023).pdf_"
                "by_PaddleOCR-VL-1.6.md"
            )
        }
    ]

    metrics = replay._residual_metrics(
        record=record,
        suggested_documents=["第二份共识扩展版.md"],
    )

    assert metrics["residual_gold_documents"] == ["第二份共识"]
    assert metrics["companion_residual_hits"] == ["第二份共识"]
    assert metrics["residual_document_recall"] == 1.0


def test_residual_metrics_use_final_prim_retrieval_and_report_trigger_gap() -> None:
    record = _record()
    record["reference_documents"] = ["文档3"]

    metrics = replay._residual_metrics(
        record=record,
        suggested_documents=["文档3.md"],
    )

    assert metrics["prim_trigger_documents"] == ["文档1.md", "文档2.md"]
    assert metrics["prim_final_documents"] == [
        "文档1.md",
        "文档2.md",
        "文档3.md",
    ]
    assert metrics["trigger_point_missing_gold_documents"] == ["文档3"]
    assert metrics["companion_trigger_point_hits"] == ["文档3"]
    assert metrics["residual_gold_documents"] == []
    assert metrics["companion_residual_hits"] == []
    assert metrics["residual_document_recall"] is None


def test_external_gold_documents_override_stale_exported_labels() -> None:
    record = {
        "reference_documents": ["旧标签"],
        "input_record": {"reference_documents": ["正式金标准"]},
    }

    assert replay._gold_documents(record) == ["正式金标准"]


def test_gold_index_aligns_by_question_without_entering_reconstructed_state() -> None:
    gold = {"question": "  病例\n文本  ", "reference": "标答"}
    index = replay.GoldIndex([gold])

    matched, match_by = index.match({"question": "病例 文本"}, 99)

    assert matched is gold
    assert match_by == "question"


def test_gold_index_prefers_question_over_stale_row_index() -> None:
    wrong = {"question": "其它病例", "reference": "错误标答"}
    expected = {"question": "目标病例", "reference": "正确标答"}
    index = replay.GoldIndex([wrong, expected])

    matched, match_by = index.match(
        {"row_index": 0, "question": "目标病例"},
        0,
    )

    assert matched is expected
    assert match_by == "question"


def test_main_validates_current_atlas_before_loading_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    atlas = SimpleNamespace(
        builder_model="atlas:model",
        snapshot_hash="snapshot",
        document_cards=[],
    )

    class FakePostgresManager:
        def initialize(self) -> None:
            events.append("postgres_initialized")

        async def close(self) -> None:
            events.append("postgres_closed")

    class FakeStore:
        def __init__(self) -> None:
            events.append("store_created")

        def load_current(self, db_id: str):
            assert db_id == "db-1"
            events.append("atlas_loaded")
            return atlas

    class FakeBuilder:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model_name"] == "atlas:model"
            assert set(kwargs) == {"model_name", "store"}
            events.append("builder_created")

        async def validate_runtime(self, current) -> None:
            assert current is atlas
            events.append("atlas_validated")

    def load_model(name: str):
        assert name == "agent:model"
        events.append("model_loaded")
        return object()

    async def select_cues(**_kwargs):
        raise AssertionError("空输入不应调用选择器")

    monkeypatch.setattr(
        replay,
        "_runtime_dependencies",
        lambda: (
            FakePostgresManager(),
            load_model,
            FakeStore,
            FakeBuilder,
            select_cues,
        ),
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text("[]", encoding="utf-8")

    result = asyncio.run(
        replay.main_async(
            argparse.Namespace(
                input=input_path,
                output=output_path,
                gold=None,
                summary_output=None,
                db_id="db-1",
                model="agent:model",
                technical_retry_limit=1,
                limit=None,
            )
        )
    )

    assert result == 0
    assert events == [
        "postgres_initialized",
        "store_created",
        "atlas_loaded",
        "builder_created",
        "atlas_validated",
        "model_loaded",
        "postgres_closed",
    ]
