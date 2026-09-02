from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from yuxi import config

from .models import (
    AtlasDocumentBuildAudit,
    AtlasDocumentCard,
    CorpusAtlas,
    calculate_snapshot_hash,
)

SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class AtlasStoreError(RuntimeError):
    pass


class AtlasStore:
    def __init__(self, root: Path | None = None):
        self.root = root or (
            Path(config.save_dir)
            / "knowledge_base_data"
            / "acm_corpus_atlas"
        )

    def _db_dir(self, db_id: str) -> Path:
        if not SAFE_COMPONENT_RE.fullmatch(db_id):
            raise AtlasStoreError(f"非法 Atlas db_id：{db_id!r}")
        return self.root / db_id

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def load_document_cache(
        self,
        db_id: str,
        cache_key: str,
    ) -> tuple[AtlasDocumentCard, AtlasDocumentBuildAudit] | None:
        if not SAFE_COMPONENT_RE.fullmatch(cache_key):
            raise AtlasStoreError(f"非法 Atlas cache key：{cache_key!r}")
        path = self._db_dir(db_id) / "cache" / f"{cache_key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                AtlasDocumentCard.model_validate(payload["document_card"]),
                AtlasDocumentBuildAudit.model_validate(payload["audit"]),
            )
        except Exception as exc:  # noqa: BLE001 - cache may be corrupt
            raise AtlasStoreError(f"无法读取 Atlas 文档缓存：{path}") from exc

    def save_document_cache(
        self,
        db_id: str,
        cache_key: str,
        card: AtlasDocumentCard,
        audit: AtlasDocumentBuildAudit,
    ) -> Path:
        if not SAFE_COMPONENT_RE.fullmatch(cache_key):
            raise AtlasStoreError(f"非法 Atlas cache key：{cache_key!r}")
        path = self._db_dir(db_id) / "cache" / f"{cache_key}.json"
        self._write_json_atomic(
            path,
            {
                "document_card": card.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
            },
        )
        return path

    def save(self, atlas: CorpusAtlas) -> Path:
        db_dir = self._db_dir(atlas.db_id)
        target = db_dir / "snapshots" / atlas.snapshot_hash / "atlas.json"
        self._write_json_atomic(target, atlas.model_dump(mode="json"))
        self._write_json_atomic(
            db_dir / "manifest.json",
            {
                "schema_version": atlas.schema_version,
                "db_id": atlas.db_id,
                "snapshot_hash": atlas.snapshot_hash,
                "metadata_fingerprint": atlas.metadata_fingerprint,
                "source_fingerprint": atlas.source_fingerprint,
                "builder_version": atlas.builder_version,
                "builder_model": atlas.builder_model,
                "prompt_hashes": atlas.prompt_hashes,
                "built_at": atlas.built_at,
                "document_count": atlas.document_count,
                "cue_count": atlas.cue_count,
                "source_count": atlas.source_count,
            },
        )
        return target

    def load_current(self, db_id: str) -> CorpusAtlas:
        db_dir = self._db_dir(db_id)
        manifest_path = db_dir / "manifest.json"
        if not manifest_path.is_file():
            raise AtlasStoreError(
                f"知识库 {db_id} 尚未构建 ACM Corpus Atlas 3.0"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_hash = str(manifest["snapshot_hash"])
            atlas_path = (
                db_dir / "snapshots" / snapshot_hash / "atlas.json"
            )
            payload = json.loads(atlas_path.read_text(encoding="utf-8"))
            schema_version = str(payload.get("schema_version") or "")
            if schema_version != "3.0":
                raise AtlasStoreError(
                    "当前 ACM Corpus Atlas 不是 Schema 3.0；"
                    "请用整篇文档构建器重新生成地图"
                )
            atlas = CorpusAtlas.model_validate(payload)
        except AtlasStoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - artifact may be corrupt
            raise AtlasStoreError(
                f"无法读取知识库 {db_id} 的 ACM Corpus Atlas"
            ) from exc
        if atlas.db_id != db_id or atlas.snapshot_hash != snapshot_hash:
            raise AtlasStoreError("ACM Corpus Atlas manifest 与 artifact 不一致")
        manifest_fields = {
            "schema_version": atlas.schema_version,
            "db_id": atlas.db_id,
            "metadata_fingerprint": atlas.metadata_fingerprint,
            "source_fingerprint": atlas.source_fingerprint,
            "builder_version": atlas.builder_version,
            "builder_model": atlas.builder_model,
            "prompt_hashes": atlas.prompt_hashes,
            "built_at": atlas.built_at,
            "document_count": atlas.document_count,
            "cue_count": atlas.cue_count,
            "source_count": atlas.source_count,
        }
        if any(manifest.get(key) != value for key, value in manifest_fields.items()):
            raise AtlasStoreError("ACM Corpus Atlas manifest 与 artifact 元数据不一致")
        calculated_hash = calculate_snapshot_hash(
            metadata_fingerprint=atlas.metadata_fingerprint,
            source_fingerprint=atlas.source_fingerprint,
            document_cards=atlas.document_cards,
        )
        if calculated_hash != atlas.snapshot_hash:
            raise AtlasStoreError("ACM Corpus Atlas 内容与 snapshot hash 不一致")
        return atlas
