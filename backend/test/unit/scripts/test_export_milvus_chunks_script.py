from __future__ import annotations

import asyncio
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts import export_milvus_chunks as script


class FakeIterator:
    def __init__(self, batches: list[list[dict]]) -> None:
        self.batches = iter(batches)
        self.closed = False

    def next(self) -> list[dict]:
        return next(self.batches, [])

    def close(self) -> None:
        self.closed = True


class FakeCollection:
    def __init__(self, batches: list[list[dict]], entity_counts: list[int]) -> None:
        self.iterator = FakeIterator(batches)
        self.entity_counts = iter(entity_counts)
        self.query_arguments: dict | None = None

    @property
    def num_entities(self) -> int:
        return next(self.entity_counts)

    def query_iterator(self, **kwargs):
        self.query_arguments = kwargs
        return self.iterator


def test_open_existing_collection_never_calls_create_or_rebuild(monkeypatch) -> None:
    events: list[tuple] = []

    class ExistingCollection:
        def __init__(self, *, name: str, using: str) -> None:
            events.append(("open", name, using))

        def load(self) -> None:
            events.append(("load",))

    fake_pymilvus = SimpleNamespace(
        Collection=ExistingCollection,
        utility=SimpleNamespace(
            has_collection=lambda name, using: events.append(("has", name, using)) or True,
        ),
    )
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    result = script.open_existing_collection(SimpleNamespace(connection_alias="yuxi-alias"), "kb-1")

    assert isinstance(result, ExistingCollection)
    assert events == [
        ("has", "kb-1", "yuxi-alias"),
        ("open", "kb-1", "yuxi-alias"),
        ("load",),
    ]


def test_export_collection_streams_all_batches_and_reports_metadata_gaps(tmp_path) -> None:
    collection = FakeCollection(
        batches=[
            [
                {
                    "id": "file-1_chunk_0",
                    "chunk_id": "file-1_chunk_0",
                    "file_id": "file-1",
                    "chunk_index": 0,
                    "source": "指南.md",
                    "content": "第一个实际文本块",
                }
            ],
            [
                {
                    "id": "orphan_chunk_0",
                    "chunk_id": "orphan_chunk_0",
                    "file_id": "orphan",
                    "chunk_index": 0,
                    "source": "孤立文档.md",
                    "content": "孤立文本块",
                }
            ],
        ],
        entity_counts=[2, 2],
    )
    partial_path = tmp_path / "chunks.jsonl.part"

    summary = script.export_collection_to_jsonl(
        collection=collection,
        partial_path=partial_path,
        db_id="kb-1",
        knowledge_name="用药助手-md",
        files={
            "file-1": {
                "file_id": "file-1",
                "filename": "指南.md",
                "status": "indexed",
                "content_hash": "file-hash",
                "processing_params": {"chunk_preset_id": "general"},
            },
            "file-empty": {
                "file_id": "file-empty",
                "filename": "空文档.md",
                "status": "done",
            },
        },
        batch_size=100,
    )

    records = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines()]
    assert [record["chunk_id"] for record in records] == ["file-1_chunk_0", "orphan_chunk_0"]
    assert records[0]["filename"] == "指南.md"
    assert records[0]["file_content_hash"] == "file-hash"
    assert len(records[0]["content_sha256"]) == 64
    assert records[1]["file_metadata_found"] is False
    assert summary["total_chunks"] == 2
    assert summary["orphan_file_ids"] == ["orphan"]
    assert summary["indexed_files_without_chunks"] == ["file-empty"]
    assert summary["milvus_num_entities_changed_during_export"] is False
    assert collection.query_arguments == {
        "batch_size": 100,
        "expr": 'id != ""',
        "output_fields": script.OUTPUT_FIELDS,
    }
    assert collection.iterator.closed is True


def test_export_collection_rejects_duplicate_chunk_id(tmp_path) -> None:
    collection = FakeCollection(
        batches=[
            [
                {
                    "id": "primary-1",
                    "chunk_id": "duplicate",
                    "file_id": "file-1",
                    "chunk_index": 0,
                    "source": "指南.md",
                    "content": "片段一",
                },
                {
                    "id": "primary-2",
                    "chunk_id": "duplicate",
                    "file_id": "file-1",
                    "chunk_index": 1,
                    "source": "指南.md",
                    "content": "片段二",
                },
            ]
        ],
        entity_counts=[2],
    )

    with pytest.raises(script.ChunkExportError, match="chunk_id 重复"):
        script.export_collection_to_jsonl(
            collection=collection,
            partial_path=tmp_path / "chunks.jsonl.part",
            db_id="kb-1",
            knowledge_name="用药助手-md",
            files={},
            batch_size=100,
        )

    assert collection.iterator.closed is True


def test_prepare_output_paths_keeps_previous_export_until_replacement(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    chunks_path.write_text("old chunks\n", encoding="utf-8")
    manifest_path.write_text('{"status":"complete"}\n', encoding="utf-8")

    resolved_chunks, partial_path, resolved_manifest = script._prepare_output_paths(tmp_path, overwrite=True)

    assert resolved_chunks == chunks_path
    assert resolved_manifest == manifest_path
    assert partial_path == tmp_path / "chunks.jsonl.part"
    assert chunks_path.read_text(encoding="utf-8") == "old chunks\n"
    assert manifest_path.exists()


@pytest.mark.asyncio
async def test_main_async_initializes_runtime_and_writes_complete_manifest(tmp_path) -> None:
    events: list[str] = []
    collection = FakeCollection(
        batches=[
            [
                {
                    "id": "file-1_chunk_0",
                    "chunk_id": "file-1_chunk_0",
                    "file_id": "file-1",
                    "chunk_index": 0,
                    "source": "指南.md",
                    "content": "文本块",
                }
            ]
        ],
        entity_counts=[1, 1],
    )

    class FakePostgresManager:
        def initialize(self) -> None:
            assert asyncio.get_running_loop().is_running()
            events.append("postgres_initialized")

        async def close(self) -> None:
            events.append("postgres_closed")

    class FakeManager:
        class FakeKB:
            databases_meta = {"kb-1": {}}

        async def get_databases(self) -> dict:
            events.append("databases_loaded")
            return {"databases": [{"db_id": "kb-1", "name": "用药助手-md", "kb_type": "milvus"}]}

        async def get_database_info(self, db_id: str) -> dict:
            assert db_id == "kb-1"
            return {
                "files": {
                    "file-1": {
                        "file_id": "file-1",
                        "filename": "指南.md",
                        "status": "indexed",
                    }
                }
            }

        async def aget_kb(self, db_id: str):
            assert db_id == "kb-1"
            return self.FakeKB()

    def open_collection(kb: object, db_id: str):
        assert getattr(kb, "databases_meta") == {"kb-1": {}}
        assert db_id == "kb-1"
        return collection

    result = await script.main_async(
        Namespace(
            db_id=None,
            knowledge_name="用药助手-md",
            output_dir=tmp_path,
            batch_size=100,
            overwrite=False,
        ),
        manager=FakeManager(),
        postgres_manager=FakePostgresManager(),
        collection_opener=open_collection,
    )

    assert result == 0
    assert events == ["postgres_initialized", "databases_loaded", "postgres_closed"]
    assert (tmp_path / "chunks.jsonl").exists()
    assert not (tmp_path / "chunks.jsonl.part").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["total_chunks"] == 1
