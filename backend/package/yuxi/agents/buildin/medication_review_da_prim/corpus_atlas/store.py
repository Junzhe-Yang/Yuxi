from __future__ import annotations

import json
import os
import re
from pathlib import Path

from yuxi import config

from .models import CorpusAtlas

SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class AtlasStoreError(RuntimeError):
    pass


class AtlasStore:
    def __init__(self, root: Path | None = None):
        self.root = root or (Path(config.save_dir) / "knowledge_base_data" / "corpus_atlas")

    def _db_dir(self, db_id: str) -> Path:
        if not SAFE_COMPONENT_RE.fullmatch(db_id):
            raise AtlasStoreError(f"非法 Atlas db_id：{db_id!r}")
        return self.root / db_id

    def save(self, atlas: CorpusAtlas) -> Path:
        db_dir = self._db_dir(atlas.db_id)
        snapshot_dir = db_dir / atlas.snapshot_hash
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        target = snapshot_dir / "atlas.json"
        temporary = snapshot_dir / "atlas.json.tmp"
        temporary.write_text(
            atlas.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

        manifest = {
            "schema_version": "1.0",
            "db_id": atlas.db_id,
            "snapshot_hash": atlas.snapshot_hash,
            "metadata_fingerprint": atlas.metadata_fingerprint,
            "embedding_model_id": atlas.embedding_model_id,
            "builder_version": atlas.builder_version,
            "built_at": atlas.built_at,
        }
        manifest_target = db_dir / "manifest.json"
        manifest_temporary = db_dir / "manifest.json.tmp"
        manifest_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(manifest_temporary, manifest_target)
        return target

    def load_current(self, db_id: str) -> CorpusAtlas:
        db_dir = self._db_dir(db_id)
        manifest_path = db_dir / "manifest.json"
        if not manifest_path.is_file():
            raise AtlasStoreError(f"知识库 {db_id} 尚未构建 Corpus Atlas")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_hash = str(manifest["snapshot_hash"])
            atlas_path = db_dir / snapshot_hash / "atlas.json"
            atlas = CorpusAtlas.model_validate_json(atlas_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - artifact may be corrupt
            raise AtlasStoreError(f"无法读取知识库 {db_id} 的 Corpus Atlas") from exc
        if atlas.db_id != db_id or atlas.snapshot_hash != snapshot_hash:
            raise AtlasStoreError("Corpus Atlas manifest 与 artifact 不一致")
        return atlas
