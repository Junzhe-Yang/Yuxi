from __future__ import annotations

import json
from argparse import Namespace

from scripts import replay_acm_top25 as script


def test_query_records_preserve_document_scope() -> None:
    calls = script._query_records(
        {
            "medication_review_trace": {
                "query_records": [
                    {
                        "query_id": "Q-1",
                        "query_text": "剂量 调整",
                        "reason": "补查阈值",
                        "retrieval_scope": "document",
                        "file_id": "file-1",
                        "investigation_id": "INV-1",
                    }
                ]
            }
        }
    )

    assert calls == [
        {
            "query_id": "Q-1",
            "query_text": "剂量 调整",
            "reason": "补查阈值",
            "retrieval_scope": "document",
            "file_id": "file-1",
            "investigation_id": "INV-1",
        }
    ]


async def test_main_replays_top25_without_llm(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class FakePostgres:
        def initialize(self) -> None:
            events.append("postgres_initialized")

        async def close(self) -> None:
            events.append("postgres_closed")

    class FakeManager:
        async def get_databases(self):
            return {
                "databases": [
                    {
                        "db_id": "db-1",
                        "name": "知识库",
                        "kb_type": "milvus",
                    }
                ]
            }

        async def aembed_texts(self, db_id, texts):
            assert db_id == "db-1"
            assert texts == ["剂量 调整"]
            return [[0.1, 0.2]]

        async def aquery(self, query, db_id, **kwargs):
            assert query == "剂量 调整"
            assert db_id == "db-1"
            assert kwargs["final_top_k"] == 25
            assert kwargs["filter_file_ids"] == ["file-1"]
            return [
                {
                    "content": "需要调整剂量。",
                    "metadata": {
                        "source": "共识.md",
                        "file_id": "file-1",
                        "chunk_id": "chunk-1",
                        "chunk_index": 7,
                    },
                    "score": 0.9,
                }
            ]

    monkeypatch.setattr(
        script,
        "_runtime_dependencies",
        lambda: (FakePostgres(), FakeManager()),
    )
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                {
                    "row_index": 0,
                    "question": "病例",
                    "search_records": [
                        {
                            "query_id": "Q-1",
                            "query_text": "剂量 调整",
                            "reason": "补查",
                            "retrieval_scope": "document",
                            "file_id": "file-1",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "top25.jsonl"
    args = Namespace(
        db_id=None,
        knowledge_name="知识库",
        records=records,
        output=output,
        limit=None,
        timeout_seconds=30,
        technical_retry_limit=0,
    )

    assert await script.main_async(args) == 0

    replay = json.loads(output.read_text(encoding="utf-8"))
    assert replay["retrieval_calls"][0]["status"] == "success"
    assert replay["retrieval_calls"][0]["retrieved_items"][0]["chunk_id"] == "chunk-1"
    summary = json.loads(
        output.with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert summary["llm_calls"] == 0
    assert events == [
        "postgres_initialized",
        "postgres_closed",
    ]
