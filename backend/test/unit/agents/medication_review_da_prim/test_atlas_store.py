from __future__ import annotations

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.models import (
    CorpusAtlas,
    DocumentCard,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.store import (
    AtlasStore,
)


def _atlas() -> CorpusAtlas:
    return CorpusAtlas(
        builder_version="test",
        snapshot_hash="abc123",
        metadata_fingerprint="meta123",
        db_id="db-1",
        knowledge_name="知识库",
        embedding_model_id="embed-1",
        embedding_dimension=2,
        built_at="2026-01-01T00:00:00Z",
        document_cards=[
            DocumentCard(
                file_id="file-1",
                file_name="共识.md",
                document_title="共识",
                routing_text="文档：共识",
                embedding=[0.1, 0.2],
            )
        ],
    )


def test_store_atomically_saves_and_loads_current_atlas(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)

    path = store.save(_atlas())
    loaded = store.load_current("db-1")

    assert path == tmp_path / "db-1" / "abc123" / "atlas.json"
    assert loaded == _atlas()
    assert not list(tmp_path.rglob("*.tmp"))
