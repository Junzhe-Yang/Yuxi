from __future__ import annotations

import json

import pytest

from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
    AtlasDocumentBuildAudit,
    AtlasDocumentCard,
    AtlasTopicCue,
    CorpusAtlas,
    calculate_snapshot_hash,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.store import (
    AtlasStore,
    AtlasStoreError,
)


def _card() -> AtlasDocumentCard:
    return AtlasDocumentCard(
        doc_id="file-1",
        file_name="文档.md",
        title="文档",
        scope_summary="范围",
        topic_cues=[
            AtlasTopicCue(
                cue_id="AT-1",
                cue_text="主题",
                source_chunk_ids=["chunk-1"],
            )
        ],
    )


def _atlas(card: AtlasDocumentCard) -> CorpusAtlas:
    snapshot_hash = calculate_snapshot_hash(
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        document_cards=[card],
    )
    return CorpusAtlas(
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash=snapshot_hash,
        metadata_fingerprint="metadata",
        source_fingerprint="source",
        db_id="db-1",
        knowledge_name="知识库",
        builder_model="provider:model",
        built_at="2026-01-01T00:00:00Z",
        document_cards=[card],
    )


def test_store_separates_snapshot_and_document_cache(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)
    card = _card()
    audit = AtlasDocumentBuildAudit(
        file_id="file-1",
        file_name="文档.md",
        document_hash="document-hash",
        cache_key="cache-key",
    )
    store.save_document_cache("db-1", "cache-key", card, audit)
    atlas = _atlas(card)
    store.save(atlas)

    assert store.load_document_cache("db-1", "cache-key") == (card, audit)
    assert store.load_current("db-1") == atlas


def test_store_rejects_snapshot_with_modified_document_cards(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)
    path = store.save(_atlas(_card()))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["document_cards"][0]["topic_cues"][0]["cue_text"] = "被修改"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AtlasStoreError, match="snapshot hash"):
        store.load_current("db-1")


def test_store_rejects_manifest_with_modified_builder_metadata(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)
    store.save(_atlas(_card()))
    path = tmp_path / "db-1" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["builder_model"] = "provider:other"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AtlasStoreError, match="元数据不一致"):
        store.load_current("db-1")


def test_store_rejects_old_schema_with_rebuild_instruction(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)
    path = store.save(_atlas(_card()))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "2.0"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AtlasStoreError, match="重新生成地图"):
        store.load_current("db-1")


def test_compact_and_document_views_are_separate() -> None:
    atlas = _atlas(_card())

    assert atlas.compact_view() == [
        {"doc_id": "file-1", "title": "文档", "scope_summary": "范围"}
    ]
    assert atlas.document_view("file-1")["topic_cues"] == [
        {"cue_id": "AT-1", "cue_text": "主题"}
    ]
    assert "source_chunk_ids" not in atlas.document_view("file-1")[
        "topic_cues"
    ][0]
