from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any


def _runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Load the application only after ``asyncio.run`` created its loop."""
    from yuxi import knowledge_base
    from yuxi.agents import load_chat_model
    from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas import (
        AtlasStore,
        CorpusAtlasBuilder,
    )
    from yuxi.storage.postgres.manager import pg_manager

    return (
        pg_manager,
        knowledge_base,
        load_chat_model,
        AtlasStore,
        CorpusAtlasBuilder,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "为一个 Milvus 知识库自动构建或检查 ACM Corpus Atlas 3.0。"
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-id")
    target.add_argument("--knowledge-name")
    parser.add_argument(
        "--model",
        help="离线 Atlas 构建模型。构建时必填；--check 时默认读取快照记录。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查当前快照。默认检查元数据；配合 --deep-check 核对全部 chunk。",
    )
    parser.add_argument(
        "--deep-check",
        action="store_true",
        help="检查时重新读取全部 Milvus chunk 并核对内容哈希。",
    )
    parser.add_argument("--technical-retry-limit", type=int, default=1)
    return parser


async def _resolve_db_id(args: argparse.Namespace, knowledge_base: Any) -> str:
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
    knowledge_name = str(args.knowledge_name or "").strip()
    matches = [
        value
        for value in databases or []
        if isinstance(value, dict)
        and str(value.get("name") or "").strip() == knowledge_name
        and str(value.get("kb_type") or "").lower() == "milvus"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Milvus 知识库名称 {knowledge_name!r} 不存在或不唯一"
        )
    return str(matches[0].get("db_id") or "").strip()


def _summary(atlas: Any, path: Any | None = None) -> dict[str, Any]:
    audits = list(atlas.document_audits or [])
    return {
        "status": "ok",
        "schema_version": atlas.schema_version,
        "builder_version": atlas.builder_version,
        "builder_model": atlas.builder_model,
        "db_id": atlas.db_id,
        "knowledge_name": atlas.knowledge_name,
        "snapshot_hash": atlas.snapshot_hash,
        "metadata_fingerprint": atlas.metadata_fingerprint,
        "source_fingerprint": atlas.source_fingerprint,
        "document_count": atlas.document_count,
        "cue_count": atlas.cue_count,
        "source_count": atlas.source_count,
        "cache_hit_count": sum(1 for value in audits if value.cache_hit),
        "input_chunk_count": sum(
            value.input_chunk_count for value in audits
        ),
        "input_chars": sum(value.input_chars for value in audits),
        "duplicate_cue_count": sum(
            value.duplicate_cue_count for value in audits
        ),
        "unknown_source_chunk_id_count": sum(
            len(value.unknown_source_chunk_ids) for value in audits
        ),
        "warning_count": len(atlas.warnings),
        "warnings": atlas.warnings,
        "path": str(path) if path is not None else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    if args.deep_check and not args.check:
        raise ValueError("--deep-check 必须与 --check 一起使用")
    (
        pg_manager,
        knowledge_base,
        load_chat_model,
        AtlasStore,
        CorpusAtlasBuilder,
    ) = _runtime_dependencies()
    pg_manager.initialize()
    try:
        db_id = await _resolve_db_id(args, knowledge_base)
        store = AtlasStore()
        current = store.load_current(db_id) if args.check else None
        model_name = str(
            args.model or (current.builder_model if current is not None else "")
        ).strip()
        if not model_name:
            raise ValueError("构建 Atlas 时必须提供 --model")
        model = None if args.check else load_chat_model(model_name)
        builder = CorpusAtlasBuilder(
            model=model,
            model_name=model_name,
            store=store,
            technical_retry_limit=args.technical_retry_limit,
        )
        if current is not None:
            await builder.validate_current(current, deep=args.deep_check)
            print(json.dumps(_summary(current), ensure_ascii=False, indent=2))
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
