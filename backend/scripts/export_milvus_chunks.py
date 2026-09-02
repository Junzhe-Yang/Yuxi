from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("export_milvus_chunks")

OUTPUT_FIELDS = ["id", "content", "source", "chunk_id", "file_id", "chunk_index"]
INDEXED_STATUSES = {"indexed", "done"}


class ChunkExportError(RuntimeError):
    """Raised when a complete and internally consistent export cannot be produced."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出一个 Yuxi Milvus 知识库中实际入库的全部文本块。")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-id", help="知识库 db_id")
    target.add_argument("--knowledge-name", help="网页中显示的知识库名称；名称必须唯一")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="输出目录；默认写入 saves/exports/milvus_chunks/<db_id>-<时间>",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Milvus 每批读取数量，默认 1000")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖目标目录中已有的导出文件")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _database_values(response: Any) -> list[dict[str, Any]]:
    values = response.get("databases", []) if isinstance(response, dict) else []
    if not isinstance(values, list):
        raise ChunkExportError("知识库列表响应中的 databases 不是列表")
    return [value for value in values if isinstance(value, dict)]


async def resolve_database(args: argparse.Namespace, manager: Any) -> dict[str, Any]:
    databases = _database_values(await manager.get_databases())
    if args.db_id:
        target = str(args.db_id).strip()
        matches = [value for value in databases if str(value.get("db_id") or "").strip() == target]
        description = f"db_id={target!r}"
    else:
        target = str(args.knowledge_name).strip()
        matches = [value for value in databases if str(value.get("name") or "").strip() == target]
        description = f"knowledge_name={target!r}"

    if not matches:
        raise ChunkExportError(f"找不到知识库：{description}")
    if len(matches) > 1:
        ids = [str(value.get("db_id") or "") for value in matches]
        raise ChunkExportError(f"知识库不唯一：{description}，匹配到 db_id={ids}")

    database = matches[0]
    if str(database.get("kb_type") or "").strip().lower() != "milvus":
        raise ChunkExportError(
            f"知识库 {database.get('name')!r} 的类型是 {database.get('kb_type')!r}，本脚本只导出 Milvus 知识库"
        )
    return database


def _normalize_file_metadata(raw_files: Any) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(raw_files, dict):
        values = raw_files.items()
    elif isinstance(raw_files, list):
        values = ((str(value.get("file_id") or ""), value) for value in raw_files if isinstance(value, dict))
    else:
        return normalized

    for key, raw in values:
        if not isinstance(raw, dict):
            continue
        file_id = str(raw.get("file_id") or key or "").strip()
        if not file_id or raw.get("is_folder"):
            continue
        value = dict(raw)
        value["file_id"] = file_id
        normalized[file_id] = value
    return normalized


def _missing_index_summary(indices: set[int]) -> tuple[int, list[int]]:
    if not indices:
        return 0, []
    ordered = sorted(indices)
    missing_count = ordered[0]
    sample: list[int] = list(range(min(ordered[0], 20)))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        gap = current - previous - 1
        if gap <= 0:
            continue
        missing_count += gap
        if len(sample) < 20:
            sample.extend(range(previous + 1, min(current, previous + 1 + 20 - len(sample))))
    return missing_count, sample


def _prepare_output_paths(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    chunks_path = output_dir / "chunks.jsonl"
    partial_path = output_dir / "chunks.jsonl.part"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (chunks_path, partial_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise ChunkExportError(f"输出目录已有导出文件（{names}）；请换目录或添加 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    return chunks_path, partial_path, manifest_path


def open_existing_collection(kb: Any, db_id: str) -> Any:
    """Open the existing collection without invoking Yuxi's create/rebuild path."""
    from pymilvus import Collection, utility

    connection_alias = str(getattr(kb, "connection_alias", "") or "").strip()
    if not connection_alias:
        raise ChunkExportError("Milvus 知识库实例缺少 connection_alias")
    if not utility.has_collection(db_id, using=connection_alias):
        raise ChunkExportError(f"Milvus collection {db_id!r} 不存在；导出脚本不会自动创建或重建它")
    collection = Collection(name=db_id, using=connection_alias)
    collection.load()
    return collection


def export_collection_to_jsonl(
    *,
    collection: Any,
    partial_path: Path,
    db_id: str,
    knowledge_name: str,
    files: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    """Stream all scalar chunk fields from a Milvus collection into JSONL."""
    observed_before = int(collection.num_entities)
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr='id != ""',
        output_fields=OUTPUT_FIELDS,
    )
    seen_primary_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    index_occurrences: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    file_counts: dict[str, int] = defaultdict(int)
    file_sources: dict[str, set[str]] = defaultdict(set)
    chunks_digest = hashlib.sha256()
    total_chunks = 0

    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as handle:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                LOGGER.info("从 Milvus 读取一批 chunk：batch=%s，累计=%s", len(batch), total_chunks + len(batch))
                for raw in batch:
                    if not isinstance(raw, dict):
                        raise ChunkExportError(f"Milvus 返回了非字典记录：{type(raw).__name__}")

                    primary_id = str(raw.get("id") or "").strip()
                    chunk_id = str(raw.get("chunk_id") or primary_id).strip()
                    file_id = str(raw.get("file_id") or "").strip()
                    source = str(raw.get("source") or "").strip()
                    content = raw.get("content")
                    try:
                        chunk_index = int(raw.get("chunk_index"))
                    except (TypeError, ValueError) as exc:
                        raise ChunkExportError(f"chunk {chunk_id or primary_id!r} 的 chunk_index 无效") from exc

                    if not primary_id or not chunk_id:
                        raise ChunkExportError("Milvus 中存在缺少 id/chunk_id 的记录")
                    if not isinstance(content, str) or not content.strip():
                        raise ChunkExportError(f"chunk {chunk_id!r} 的 content 为空或不是字符串")
                    if chunk_index < 0:
                        raise ChunkExportError(f"chunk {chunk_id!r} 的 chunk_index 为负数")
                    if primary_id in seen_primary_ids:
                        raise ChunkExportError(f"Milvus 主键重复：{primary_id}")
                    if chunk_id in seen_chunk_ids:
                        raise ChunkExportError(f"chunk_id 重复：{chunk_id}")

                    seen_primary_ids.add(primary_id)
                    seen_chunk_ids.add(chunk_id)
                    file_counts[file_id] += 1
                    file_sources[file_id].add(source)
                    index_occurrences[file_id][chunk_index] += 1

                    metadata = files.get(file_id, {})
                    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    processing_params = _json_safe(metadata.get("processing_params") or {})
                    record = {
                        "schema_version": "1.0",
                        "db_id": db_id,
                        "knowledge_name": knowledge_name,
                        "milvus_id": primary_id,
                        "chunk_id": chunk_id,
                        "file_id": file_id,
                        "filename": str(metadata.get("filename") or source),
                        "source": source,
                        "chunk_index": chunk_index,
                        "content": content,
                        "content_sha256": content_sha256,
                        "file_status": metadata.get("status"),
                        "file_content_hash": metadata.get("content_hash"),
                        "processing_params": processing_params,
                        "file_metadata_found": bool(metadata),
                    }
                    encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                    handle.write(encoded.decode("utf-8"))
                    chunks_digest.update(encoded)
                    total_chunks += 1
    finally:
        iterator.close()

    observed_after = int(collection.num_entities)
    entity_count_changed = observed_before != observed_after

    file_summaries: list[dict[str, Any]] = []
    all_file_ids = sorted(set(files) | set(file_counts))
    for file_id in all_file_ids:
        metadata = files.get(file_id, {})
        occurrences = index_occurrences.get(file_id, {})
        indices = set(occurrences)
        duplicate_indices = sorted(index for index, count in occurrences.items() if count > 1)
        missing_count, missing_sample = _missing_index_summary(indices)
        file_summaries.append(
            {
                "file_id": file_id,
                "filename": metadata.get("filename") or next(iter(file_sources.get(file_id, set())), ""),
                "status": metadata.get("status"),
                "content_hash": metadata.get("content_hash"),
                "processing_params": _json_safe(metadata.get("processing_params") or {}),
                "metadata_found": bool(metadata),
                "chunk_count": file_counts.get(file_id, 0),
                "min_chunk_index": min(indices) if indices else None,
                "max_chunk_index": max(indices) if indices else None,
                "duplicate_chunk_indices": duplicate_indices,
                "missing_chunk_index_count": missing_count,
                "missing_chunk_index_sample": missing_sample,
                "sources": sorted(file_sources.get(file_id, set())),
            }
        )

    orphan_file_ids = sorted(file_id for file_id in file_counts if file_id not in files)
    indexed_files_without_chunks = sorted(
        file_id
        for file_id, metadata in files.items()
        if str(metadata.get("status") or "").lower() in INDEXED_STATUSES and file_counts.get(file_id, 0) == 0
    )
    return {
        "total_chunks": total_chunks,
        # num_entities is useful context but is not an authoritative live-row count
        # after logical deletes and before compaction. The iterator result is the
        # exported set; a count change is reported as a warning instead of causing
        # a false failure after ordinary Yuxi re-indexing.
        "milvus_num_entities_before": observed_before,
        "milvus_num_entities_after": observed_after,
        "milvus_num_entities_changed_during_export": entity_count_changed,
        "chunks_sha256": chunks_digest.hexdigest(),
        "metadata_file_count": len(files),
        "files_with_chunks": sum(1 for count in file_counts.values() if count > 0),
        "orphan_file_ids": orphan_file_ids,
        "indexed_files_without_chunks": indexed_files_without_chunks,
        "files": file_summaries,
    }


async def main_async(
    args: argparse.Namespace,
    *,
    manager: Any | None = None,
    postgres_manager: Any | None = None,
    collection_opener: Any | None = None,
) -> int:
    if args.batch_size <= 0:
        raise ChunkExportError("--batch-size 必须大于 0")

    if (manager is None) != (postgres_manager is None):
        raise ChunkExportError("manager 与 postgres_manager 必须同时提供或同时省略")
    if manager is None:
        from yuxi import knowledge_base
        from yuxi.storage.postgres.manager import pg_manager

        manager = knowledge_base
        postgres_manager = pg_manager
    collection_opener = collection_opener or open_existing_collection

    postgres_manager.initialize()
    try:
        database = await resolve_database(args, manager)
        db_id = str(database.get("db_id") or "").strip()
        knowledge_name = str(database.get("name") or db_id)
        output_dir = args.output_dir
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = Path("saves/exports/milvus_chunks") / f"{db_id}-{timestamp}"
        chunks_path, partial_path, manifest_path = _prepare_output_paths(output_dir, args.overwrite)
        started_at = _utc_now()
        if args.overwrite:
            partial_path.unlink(missing_ok=True)
        try:
            LOGGER.info("准备导出知识库：name=%s db_id=%s", knowledge_name, db_id)
            database_info = await manager.get_database_info(db_id)
            if not isinstance(database_info, dict):
                raise ChunkExportError(f"无法读取知识库 {db_id} 的文件元数据")
            files = _normalize_file_metadata(database_info.get("files"))
            kb = await manager.aget_kb(db_id)
            if db_id not in getattr(kb, "databases_meta", {}):
                await kb._load_metadata()
            if db_id not in getattr(kb, "databases_meta", {}):
                raise ChunkExportError(f"Milvus 知识库实例未加载到 {db_id} 的元数据")
            collection = collection_opener(kb, db_id)
            summary = export_collection_to_jsonl(
                collection=collection,
                partial_path=partial_path,
                db_id=db_id,
                knowledge_name=knowledge_name,
                files=files,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            failed_manifest = {
                "schema_version": "1.0",
                "status": "failed",
                "db_id": db_id,
                "knowledge_name": knowledge_name,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "error": str(exc),
                "partial_chunks_file": partial_path.name if partial_path.exists() else None,
            }
            if args.overwrite and manifest_path.exists():
                _write_json(output_dir / "manifest.failed.json", failed_manifest)
            else:
                _write_json(manifest_path, failed_manifest)
            raise

        warnings = []
        if summary["orphan_file_ids"]:
            warnings.append("部分 chunk 的 file_id 在 PostgreSQL 文件元数据中不存在，详见 orphan_file_ids")
        if summary["indexed_files_without_chunks"]:
            warnings.append("部分 indexed/done 文件没有 Milvus chunk，详见 indexed_files_without_chunks")
        if summary["milvus_num_entities_changed_during_export"]:
            warnings.append("Milvus num_entities 在导出期间发生变化；请停止入库或重建后重新导出以冻结稳定快照")
        manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "db_id": db_id,
            "knowledge_name": knowledge_name,
            "kb_type": "milvus",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "collection_name": db_id,
            "output_fields": OUTPUT_FIELDS,
            "chunks_file": chunks_path.name,
            "chunks_file_sha256": summary.pop("chunks_sha256"),
            "warnings": warnings,
            **summary,
        }
        # Replace the data first and write the complete manifest last. The
        # previous successful export stays available until iteration succeeds.
        partial_path.replace(chunks_path)
        _write_json(manifest_path, manifest)
        (output_dir / "manifest.failed.json").unlink(missing_ok=True)
        LOGGER.info(
            "导出完成：chunks=%s files_with_chunks=%s output=%s",
            manifest["total_chunks"],
            manifest["files_with_chunks"],
            output_dir.resolve(),
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    finally:
        await postgres_manager.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(main_async(args))
    except ChunkExportError as exc:
        LOGGER.error("导出失败：%s", exc)
        return 1
    except Exception:
        LOGGER.exception("导出发生未预期错误")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
