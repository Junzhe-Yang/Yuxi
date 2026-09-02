from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

from scripts import build_acm_corpus_atlas as script


def _args(**updates):
    values = {
        "db_id": None,
        "knowledge_name": "用药助手-md",
        "model": "provider:model",
        "check": False,
        "deep_check": False,
        "technical_retry_limit": 1,
    }
    values.update(updates)
    return Namespace(**values)


async def test_main_async_initializes_postgres_inside_running_loop(
    monkeypatch,
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
        schema_version="3.0",
        builder_version="acm-atlas-v8-whole-document",
        builder_model="provider:model",
        db_id="db-1",
        knowledge_name="用药助手-md",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        document_count=1,
        cue_count=2,
        source_count=2,
        document_audits=[],
        warnings=[],
    )

    class FakeBuilder:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {
                "model": "loaded-model",
                "model_name": "provider:model",
                "store": kwargs["store"],
                "technical_retry_limit": 1,
            }
            events.append("builder_created")

        async def build(self, db_id: str):
            assert db_id == "db-1"
            events.append("atlas_built")
            return atlas, "/tmp/atlas.json"

    monkeypatch.setattr(
        script,
        "_runtime_dependencies",
        lambda: (
            FakePostgresManager(),
            FakeKnowledgeBase(),
            lambda _name: "loaded-model",
            type("FakeStore", (), {}),
            FakeBuilder,
        ),
    )

    assert await script.main_async(_args()) == 0
    assert events == [
        "postgres_initialized",
        "builder_created",
        "atlas_built",
        "postgres_closed",
    ]


async def test_check_uses_snapshot_model_without_legacy_batch_parameters(
    monkeypatch,
) -> None:
    class FakePostgresManager:
        def initialize(self) -> None:
            pass

        async def close(self) -> None:
            pass

    class FakeKnowledgeBase:
        async def get_databases(self) -> dict:
            return {
                "databases": [
                    {"db_id": "db-1", "name": "用药助手-md", "kb_type": "milvus"}
                ]
            }

    atlas = SimpleNamespace(
        schema_version="3.0",
        builder_version="acm-atlas-v8-whole-document",
        builder_model="provider:model",
        db_id="db-1",
        knowledge_name="用药助手-md",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        document_count=1,
        cue_count=1,
        source_count=1,
        document_audits=[],
        warnings=[],
    )

    class FakeStore:
        def load_current(self, _db_id):
            return atlas

    class FakeBuilder:
        def __init__(self, **kwargs) -> None:
            assert set(kwargs) == {
                "model",
                "model_name",
                "store",
                "technical_retry_limit",
            }
            assert kwargs["model"] is None

        async def validate_current(self, current, *, deep: bool):
            assert current is atlas and deep is True

    monkeypatch.setattr(
        script,
        "_runtime_dependencies",
        lambda: (
            FakePostgresManager(),
            FakeKnowledgeBase(),
            lambda _name: None,
            FakeStore,
            FakeBuilder,
        ),
    )

    assert await script.main_async(
        _args(db_id="db-1", knowledge_name=None, model=None, check=True, deep_check=True)
    ) == 0


def test_parser_has_no_batch_or_cue_limit_options() -> None:
    help_text = script.build_parser().format_help()

    assert "batch-max-chars" not in help_text
    assert "max-batch-cues" not in help_text
    assert "max-document-cues" not in help_text
