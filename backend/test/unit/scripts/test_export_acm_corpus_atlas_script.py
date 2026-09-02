from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from scripts import export_acm_corpus_atlas as script


def _atlas() -> SimpleNamespace:
    card = SimpleNamespace(
        doc_id="file-1",
        file_name="共识.md",
        title="老年用药共识",
        scope_summary="覆盖治疗选择和监测。",
        topic_cues=[
            SimpleNamespace(
                cue_id="AT-1",
                cue_text="需要监测体位性低血压。",
                source_chunk_ids=["chunk-1", "chunk-2"],
            )
        ],
    )
    audit = SimpleNamespace(
        file_id="file-1",
        input_chunk_count=12,
        input_chars=54321,
        duplicate_cue_count=1,
        unknown_source_chunk_ids=["wrong-id"],
        invocations=[SimpleNamespace(status="repaired")],
    )
    return SimpleNamespace(
        schema_version="3.0",
        builder_version="acm-atlas-v8-whole-document",
        snapshot_hash="snapshot",
        db_id="db-1",
        knowledge_name="用药助手-md",
        builder_model="provider:model",
        built_at="2026-08-13T00:00:00Z",
        document_cards=[card],
        document_audits=[audit],
        warnings=[],
        document_count=1,
        cue_count=1,
        source_count=2,
    )


def test_export_atlas_writes_two_level_review(tmp_path) -> None:
    summary = script.export_atlas(_atlas(), tmp_path, overwrite=False)

    assert summary["repaired_invocation_count"] == 1
    markdown = (tmp_path / "atlas_review.md").read_text(encoding="utf-8")
    assert "## 文档总览" in markdown
    assert "构建输入：12 chunks / 54321 字符" in markdown
    assert "需要监测体位性低血压" in markdown
    assert "`chunk-1`、`chunk-2`" in markdown
    with (tmp_path / "atlas_cues.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_chunk_ids"] == "chunk-1\nchunk-2"
    assert not (tmp_path / "atlas_candidates.csv").exists()


def test_export_atlas_refuses_to_overwrite_existing_review(tmp_path) -> None:
    (tmp_path / "atlas_review.md").write_text("old", encoding="utf-8")

    with pytest.raises(script.AtlasExportError, match="--overwrite"):
        script.export_atlas(_atlas(), tmp_path, overwrite=False)
