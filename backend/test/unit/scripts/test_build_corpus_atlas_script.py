from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts import build_corpus_atlas as script


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("db_id", "knowledge_name"),
    [(None, "用药助手-md"), ("db-1", None)],
)
async def test_main_async_initializes_runtime_before_loading_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
    db_id: str | None,
    knowledge_name: str | None,
) -> None:
    events: list[str] = []

    class FakePostgresManager:
        def initialize(self) -> None:
            assert asyncio.get_running_loop().is_running()
            events.append("postgres_initialized")

        async def close(self) -> None:
            events.append("postgres_closed")

    class FakeKnowledgeBase:
        async def get_databases(self) -> dict:
            assert events == ["postgres_initialized"]
            events.append("knowledge_loaded")
            return {
                "databases": [
                    {
                        "db_id": "db-1",
                        "name": "用药助手-md",
                        "kb_type": "milvus",
                    }
                ]
            }

    atlas = SimpleNamespace(
        db_id="db-1",
        knowledge_name="用药助手-md",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        embedding_model_id="embedding",
        embedding_dimension=1024,
        document_cards=[],
        section_cards=[],
        warnings=[],
    )

    class FakeBuilder:
        def __init__(self, *, store: object) -> None:
            self.store = store

        async def build(self, resolved_db_id: str) -> tuple[object, str]:
            assert resolved_db_id == "db-1"
            events.append("atlas_built")
            return atlas, "/tmp/atlas.json"

    monkeypatch.setattr(script, "pg_manager", FakePostgresManager())
    monkeypatch.setattr(script, "knowledge_base", FakeKnowledgeBase())
    monkeypatch.setattr(script, "AtlasStore", object)
    monkeypatch.setattr(script, "CorpusAtlasBuilder", FakeBuilder)

    result = await script.main_async(
        Namespace(
            db_id=db_id,
            knowledge_name=knowledge_name,
            check=False,
        )
    )

    assert result == 0
    assert events == [
        "postgres_initialized",
        "knowledge_loaded",
        "atlas_built",
        "postgres_closed",
    ]
