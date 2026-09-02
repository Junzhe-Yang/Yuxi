from __future__ import annotations

import math

import pytest

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.builder import (
    AtlasBuildError,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.store import (
    AtlasStore,
)


class FakeManager:
    def __init__(self):
        self.contents = {
            "file-1": {
                "content": "# 共识甲\n\n## 肾功能调整\n\n需要调整剂量。",
                "lines": [
                    {
                        "id": "chunk-1",
                        "chunk_order_index": 0,
                        "content": "需要调整剂量。",
                    }
                ],
            },
            "file-2": {
                "lines": [
                    {
                        "id": "chunk-2",
                        "chunk_order_index": 0,
                        "content": "无标题替代方案。",
                    }
                ]
            },
        }
        self.embedding_calls = []
        self.processing_params = {"chunk_size": 1000}

    async def get_database_info(self, _db_id):
        return {
            "db_id": "db-1",
            "name": "知识库",
            "kb_type": "milvus",
            "embed_info": {"model_id": "embed-1"},
            "files": {
                "file-2": {
                    "file_id": "file-2",
                    "filename": "文档乙.md",
                    "status": "done",
                    "updated_at": "2026-01-02T00:00:00Z",
                    "processing_params": self.processing_params,
                    "is_folder": False,
                },
                "file-1": {
                    "file_id": "file-1",
                    "filename": "文档甲.md",
                    "status": "indexed",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "processing_params": self.processing_params,
                    "is_folder": False,
                },
            },
        }

    async def get_file_content(self, _db_id, file_id):
        return self.contents[file_id]

    async def aembed_texts(self, _db_id, texts):
        self.embedding_calls.append(texts)
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


async def test_builder_is_deterministic_and_batches_embeddings(tmp_path) -> None:
    manager = FakeManager()
    builder = CorpusAtlasBuilder(
        manager=manager,
        store=AtlasStore(root=tmp_path),
    )

    first, first_path = await builder.build("db-1")
    second, second_path = await builder.build("db-1")

    assert first.snapshot_hash == second.snapshot_hash
    assert first_path == second_path
    assert [card.file_id for card in first.document_cards] == ["file-1", "file-2"]
    assert first.embedding_dimension == 2
    assert len(manager.embedding_calls) == 2
    assert len(manager.embedding_calls[0]) == (len(first.document_cards) + len(first.section_cards))
    assert first.warnings == ["文档乙.md 缺少完整 Markdown，已按 chunk 拼接"]
    assert first.parameters["truncated_document_cards"] == 0
    assert first.parameters["truncated_section_cards"] == 0
    await builder.validate_current(first)

    manager.contents["file-1"]["content"] += "\n新增章节内容。"
    changed, _ = await builder.build("db-1")

    assert changed.snapshot_hash != first.snapshot_hash

    manager.processing_params = {"chunk_size": 500}
    with pytest.raises(AtlasBuildError, match="已过期"):
        await builder.validate_current(changed)


async def test_builder_rejects_non_finite_embeddings(tmp_path) -> None:
    manager = FakeManager()

    async def invalid_embeddings(_db_id, texts):
        return [[math.inf, 1.0] for _ in texts]

    manager.aembed_texts = invalid_embeddings
    builder = CorpusAtlasBuilder(
        manager=manager,
        store=AtlasStore(root=tmp_path),
    )

    with pytest.raises(AtlasBuildError, match="embedding"):
        await builder.build("db-1")
