from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from yuxi import knowledge_base
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas import (
    AtlasStore,
    CorpusAtlasBuilder,
)
from yuxi.storage.postgres.manager import pg_manager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为一个 Milvus 知识库构建或检查 DA-PRIM Corpus Atlas。")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-id")
    target.add_argument("--knowledge-name")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查当前 Atlas 与知识库元数据是否一致。",
    )
    return parser


async def _resolve_db_id(args: argparse.Namespace) -> str:
    response = await knowledge_base.get_databases()
    databases = response.get("databases") if isinstance(response, dict) else []
    if args.db_id:
        db_id = str(args.db_id).strip()
        matches = [
            value
            for value in databases or []
            if isinstance(value, dict)
            and str(value.get("db_id") or "").strip() == db_id
            and str(value.get("kb_type") or "").lower() == "milvus"
        ]
        if len(matches) != 1:
            raise ValueError(f"Milvus 知识库 ID {db_id!r} 不存在")
        return db_id
    matches = [
        value
        for value in databases or []
        if isinstance(value, dict)
        and str(value.get("name") or "").strip() == args.knowledge_name.strip()
        and str(value.get("kb_type") or "").lower() == "milvus"
    ]
    if len(matches) != 1:
        raise ValueError(f"Milvus 知识库名称 {args.knowledge_name!r} 不存在或不唯一")
    return str(matches[0].get("db_id") or "").strip()


def _summary(atlas: Any, path: Any | None = None) -> dict[str, Any]:
    return {
        "status": "ok",
        "db_id": atlas.db_id,
        "knowledge_name": atlas.knowledge_name,
        "snapshot_hash": atlas.snapshot_hash,
        "metadata_fingerprint": atlas.metadata_fingerprint,
        "embedding_model_id": atlas.embedding_model_id,
        "embedding_dimension": atlas.embedding_dimension,
        "document_count": len(atlas.document_cards),
        "section_count": len(atlas.section_cards),
        "warning_count": len(atlas.warnings),
        "warnings": atlas.warnings,
        "path": str(path) if path is not None else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    pg_manager.initialize()
    try:
        db_id = await _resolve_db_id(args)
        store = AtlasStore()
        builder = CorpusAtlasBuilder(store=store)
        if args.check:
            atlas = store.load_current(db_id)
            await builder.validate_current(atlas)
            print(json.dumps(_summary(atlas), ensure_ascii=False, indent=2))
            return 0
        atlas, path = await builder.build(db_id)
        print(json.dumps(_summary(atlas, path), ensure_ascii=False, indent=2))
        return 0
    finally:
        await pg_manager.close()


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
