from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AtlasExportError(RuntimeError):
    pass


OUTPUT_NAMES = ("atlas_review.md", "atlas_cues.csv", "manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把当前 ACM Corpus Atlas 3.0 导出为便于人工复核的 Markdown 和 CSV。"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-id", help="知识库 db_id")
    target.add_argument(
        "--knowledge-name",
        help="网页中显示的 Milvus 知识库名称；名称必须唯一",
    )
    target.add_argument(
        "--atlas-path",
        type=Path,
        help="直接导出指定的 atlas.json，不读取 PostgreSQL",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    from yuxi import knowledge_base
    from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.models import (
        CorpusAtlas,
        calculate_snapshot_hash,
    )
    from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.store import (
        AtlasStore,
    )
    from yuxi.storage.postgres.manager import pg_manager

    return (
        pg_manager,
        knowledge_base,
        AtlasStore,
        CorpusAtlas,
        calculate_snapshot_hash,
    )


def _write_text_atomic(path: Path, content: str, *, encoding: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding=encoding, newline="")
    temporary.replace(path)


def _write_csv_atomic(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    existing = [name for name in OUTPUT_NAMES if (path / name).exists()]
    if existing and not overwrite:
        raise AtlasExportError(
            f"输出目录已有导出文件：{existing}；如需覆盖请使用 --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)


def _markdown_review(atlas: Any) -> str:
    audits = {value.file_id: value for value in atlas.document_audits}
    lines = [
        "# ACM Corpus Atlas 人工复核",
        "",
        f"- Schema：{atlas.schema_version}",
        f"- Builder：{atlas.builder_version}",
        f"- 知识库：{atlas.knowledge_name}（{atlas.db_id}）",
        f"- 文档：{atlas.document_count}",
        f"- 主题：{atlas.cue_count}",
        "",
        "> source_chunk_ids 只用于回溯构建来源；在线回答仍必须重新进行 Milvus 检索。",
        "",
        "## 文档总览",
        "",
    ]
    for card in atlas.document_cards:
        lines.append(
            f"- **{card.title}**（doc_id=`{card.doc_id}`，主题 {len(card.topic_cues)}）："
            f"{card.scope_summary}"
        )
    for card in atlas.document_cards:
        audit = audits.get(card.doc_id)
        lines.extend(
            [
                "",
                f"## {card.title}",
                "",
                f"- 文件名：`{card.file_name}`",
                f"- doc_id / file_id：`{card.doc_id}`",
                f"- 范围摘要：{card.scope_summary}",
                f"- 主题数：{len(card.topic_cues)}",
            ]
        )
        if audit is not None:
            lines.extend(
                [
                    f"- 构建输入：{audit.input_chunk_count} chunks / {audit.input_chars} 字符",
                    f"- 重复合并：{audit.duplicate_cue_count}",
                    f"- 无效来源 ID：{len(audit.unknown_source_chunk_ids)}",
                ]
            )
        lines.extend(["", "### 主题", ""])
        if not card.topic_cues:
            lines.append("（未抽取到治疗决策主题）")
            continue
        for index, cue in enumerate(card.topic_cues, start=1):
            sources = "、".join(f"`{value}`" for value in cue.source_chunk_ids)
            lines.extend(
                [
                    f"{index}. **{cue.cue_id}** {cue.cue_text}",
                    f"   - 来源 chunk：{sources or '未标注'}",
                ]
            )
    return "\n".join(lines) + "\n"


def _cue_rows(atlas: Any) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": card.doc_id,
            "file_name": card.file_name,
            "title": card.title,
            "scope_summary": card.scope_summary,
            "cue_index": index,
            "cue_id": cue.cue_id,
            "cue_text": cue.cue_text,
            "source_chunk_ids": "\n".join(cue.source_chunk_ids),
            "source_count": len(cue.source_chunk_ids),
        }
        for card in atlas.document_cards
        for index, cue in enumerate(card.topic_cues, start=1)
    ]


def export_atlas(atlas: Any, output_dir: Path, *, overwrite: bool) -> dict[str, Any]:
    _prepare_output_dir(output_dir, overwrite=overwrite)
    rows = _cue_rows(atlas)
    repaired_count = sum(
        1
        for audit in atlas.document_audits
        for invocation in audit.invocations
        if invocation.status == "repaired"
    )
    manifest = {
        "status": "complete",
        "exported_at": datetime.now(UTC).isoformat(),
        "db_id": atlas.db_id,
        "knowledge_name": atlas.knowledge_name,
        "snapshot_hash": atlas.snapshot_hash,
        "schema_version": atlas.schema_version,
        "builder_version": atlas.builder_version,
        "builder_model": atlas.builder_model,
        "built_at": atlas.built_at,
        "document_count": atlas.document_count,
        "cue_count": atlas.cue_count,
        "source_count": atlas.source_count,
        "repaired_invocation_count": repaired_count,
        "atlas_warning_count": len(atlas.warnings),
        "files": list(OUTPUT_NAMES),
    }
    _write_text_atomic(
        output_dir / "atlas_review.md",
        _markdown_review(atlas),
        encoding="utf-8",
    )
    _write_csv_atomic(
        output_dir / "atlas_cues.csv",
        fieldnames=[
            "doc_id",
            "file_name",
            "title",
            "scope_summary",
            "cue_index",
            "cue_id",
            "cue_text",
            "source_chunk_ids",
            "source_count",
        ],
        rows=rows,
    )
    _write_text_atomic(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "output_dir": str(output_dir.resolve())}


async def _resolve_db_id(knowledge_name: str, manager: Any) -> str:
    response = await manager.get_databases()
    databases = response.get("databases", []) if isinstance(response, dict) else []
    matches = [
        value
        for value in databases
        if isinstance(value, dict)
        and str(value.get("name") or "").strip() == knowledge_name
        and str(value.get("kb_type") or "").strip().lower() == "milvus"
    ]
    if len(matches) != 1:
        raise AtlasExportError(
            f"Milvus 知识库名称 {knowledge_name!r} 不存在或不唯一"
        )
    return str(matches[0].get("db_id") or "").strip()


def _validate_atlas_file(
    path: Path,
    *,
    model_type: Any,
    calculate_snapshot_hash: Any,
) -> Any:
    if not path.is_file():
        raise AtlasExportError(f"Atlas 文件不存在：{path}")
    try:
        atlas = model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - external artifact boundary
        raise AtlasExportError(f"Atlas 文件无法解析：{path}") from exc
    actual_hash = calculate_snapshot_hash(
        metadata_fingerprint=atlas.metadata_fingerprint,
        source_fingerprint=atlas.source_fingerprint,
        document_cards=atlas.document_cards,
    )
    if actual_hash != atlas.snapshot_hash:
        raise AtlasExportError("Atlas 文件内容与 snapshot_hash 不一致")
    return atlas


async def main_async(args: argparse.Namespace) -> int:
    (
        pg_manager,
        knowledge_base,
        AtlasStore,
        CorpusAtlas,
        calculate_snapshot_hash,
    ) = _runtime_dependencies()
    if args.atlas_path is not None:
        atlas = _validate_atlas_file(
            args.atlas_path,
            model_type=CorpusAtlas,
            calculate_snapshot_hash=calculate_snapshot_hash,
        )
    else:
        if args.knowledge_name:
            pg_manager.initialize()
            try:
                db_id = await _resolve_db_id(args.knowledge_name, knowledge_base)
            finally:
                await pg_manager.close()
        else:
            db_id = str(args.db_id or "").strip()
        if not db_id:
            raise AtlasExportError("Atlas db_id 不能为空")
        atlas = AtlasStore().load_current(db_id)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (
            Path("saves/exports/acm_corpus_atlas")
            / f"{atlas.db_id}-{atlas.snapshot_hash[:12]}-{timestamp}"
        )
    summary = export_atlas(atlas, output_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async(build_parser().parse_args()))
    except AtlasExportError as exc:
        raise SystemExit(f"Atlas 导出失败：{exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
