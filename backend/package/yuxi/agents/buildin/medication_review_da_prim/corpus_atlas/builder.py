from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from yuxi import knowledge_base
from yuxi.knowledge.base import FileStatus
from yuxi.utils.datetime_utils import utc_isoformat

from .models import CorpusAtlas, DocumentCard, SectionCard
from .parser import (
    normalize_markdown,
    parse_markdown_document,
    section_lead_text,
    stable_section_id,
)
from .store import AtlasStore

ATLAS_BUILDER_VERSION = "da-prim-atlas-v1"
DOCUMENT_ROUTING_TEXT_MAX_CHARS = 5000
SECTION_ROUTING_TEXT_MAX_CHARS = 1800
LEAD_TEXT_MAX_CHARS = 600
SUCCESS_FILE_STATUSES = {FileStatus.INDEXED, FileStatus.DONE}


class AtlasBuildError(RuntimeError):
    pass


def _embedding_model_id(database: dict[str, Any]) -> str:
    embed_info = database.get("embed_info")
    if not isinstance(embed_info, dict):
        embed_info = {}
    value = embed_info.get("model_id") or embed_info.get("model")
    return str(value or "unknown")


def _embedding_dimension(database: dict[str, Any]) -> int:
    embed_info = database.get("embed_info")
    if not isinstance(embed_info, dict):
        return 0
    try:
        return max(int(embed_info.get("dimension") or 0), 0)
    except (TypeError, ValueError):
        return 0


def _files_from_database(database: dict[str, Any]) -> list[dict[str, Any]]:
    raw_files = database.get("files") or {}
    if isinstance(raw_files, dict):
        values = list(raw_files.values())
    elif isinstance(raw_files, list):
        values = raw_files
    else:
        raise AtlasBuildError("知识库 files 元数据格式无效")
    return [value for value in values if isinstance(value, dict)]


def _content_from_file_info(content_info: dict[str, Any]) -> tuple[str, bool]:
    content = content_info.get("content")
    if isinstance(content, str) and content.strip():
        return normalize_markdown(content), False
    lines = content_info.get("lines") or []
    chunks = [value for value in lines if isinstance(value, dict)]
    chunks.sort(
        key=lambda value: (
            int(value.get("chunk_order_index") or 0),
            str(value.get("id") or ""),
        )
    )
    fallback = "\n\n".join(
        str(value.get("content") or "").strip() for value in chunks if str(value.get("content") or "").strip()
    )
    return normalize_markdown(fallback), True


def _source_record(
    *,
    file_meta: dict[str, Any],
    markdown: str,
    content_info: dict[str, Any],
) -> dict[str, Any]:
    chunks = [
        {
            "id": str(value.get("id") or ""),
            "index": int(value.get("chunk_order_index") or 0),
            "content_hash": hashlib.sha256(str(value.get("content") or "").encode("utf-8")).hexdigest(),
        }
        for value in content_info.get("lines") or []
        if isinstance(value, dict)
    ]
    chunks.sort(key=lambda value: (value["index"], value["id"]))
    return {
        "file_id": str(file_meta.get("file_id") or ""),
        "file_name": str(file_meta.get("filename") or ""),
        "status": str(file_meta.get("status") or ""),
        "updated_at": str(file_meta.get("updated_at") or ""),
        "markdown_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "chunks": chunks,
    }


def metadata_fingerprint(
    *,
    db_id: str,
    embedding_model_id: str,
    embedding_dimension: int,
    files: list[dict[str, Any]],
) -> str:
    records = [
        {
            "file_id": str(value.get("file_id") or ""),
            "file_name": str(value.get("filename") or ""),
            "status": str(value.get("status") or ""),
            "updated_at": str(value.get("updated_at") or ""),
            "markdown_file": str(value.get("markdown_file") or ""),
            "content_hash": str(value.get("content_hash") or ""),
            "processing_params": value.get("processing_params") or {},
        }
        for value in files
    ]
    records.sort(key=lambda value: value["file_id"])
    payload = {
        "db_id": db_id,
        "embedding_model_id": embedding_model_id,
        "embedding_dimension": embedding_dimension,
        "builder_version": ATLAS_BUILDER_VERSION,
        "files": records,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CorpusAtlasBuilder:
    def __init__(self, *, manager: Any = knowledge_base, store: AtlasStore | None = None):
        self.manager = manager
        self.store = store or AtlasStore()

    async def validate_current(self, atlas: CorpusAtlas) -> None:
        expected_parameters = {
            "document_routing_text_max_chars": DOCUMENT_ROUTING_TEXT_MAX_CHARS,
            "section_routing_text_max_chars": SECTION_ROUTING_TEXT_MAX_CHARS,
            "lead_text_max_chars": LEAD_TEXT_MAX_CHARS,
        }
        if atlas.builder_version != ATLAS_BUILDER_VERSION or any(
            atlas.parameters.get(key) != value for key, value in expected_parameters.items()
        ):
            raise AtlasBuildError("Corpus Atlas 构建器版本或路由文本参数已变化，请重建")
        database = await self.manager.get_database_info(atlas.db_id)
        if not isinstance(database, dict):
            raise AtlasBuildError(f"知识库 {atlas.db_id} 不存在")
        embedding_model = _embedding_model_id(database)
        embedding_dimension = _embedding_dimension(database)
        files = [
            value
            for value in _files_from_database(database)
            if not value.get("is_folder") and str(value.get("status") or "") in SUCCESS_FILE_STATUSES
        ]
        current = metadata_fingerprint(
            db_id=atlas.db_id,
            embedding_model_id=embedding_model,
            embedding_dimension=embedding_dimension,
            files=files,
        )
        if current != atlas.metadata_fingerprint:
            raise AtlasBuildError("Corpus Atlas 已过期：知识库文件或 embedding 配置已变化")

    async def build(self, db_id: str) -> tuple[CorpusAtlas, Any]:
        database = await self.manager.get_database_info(db_id)
        if not isinstance(database, dict):
            raise AtlasBuildError(f"知识库 {db_id} 不存在")
        if str(database.get("kb_type") or "milvus").lower() != "milvus":
            raise AtlasBuildError("Corpus Atlas 第一版只支持 Milvus 知识库")
        embedding_model = _embedding_model_id(database)
        configured_embedding_dimension = _embedding_dimension(database)
        files = [
            value
            for value in _files_from_database(database)
            if not value.get("is_folder") and str(value.get("status") or "") in SUCCESS_FILE_STATUSES
        ]
        files.sort(key=lambda value: str(value.get("file_id") or ""))
        if not files:
            raise AtlasBuildError("知识库没有可用于 Atlas 的已索引文档")

        document_cards: list[DocumentCard] = []
        section_cards: list[SectionCard] = []
        source_records: list[dict[str, Any]] = []
        warnings: list[str] = []
        truncated_document_cards = 0
        truncated_section_cards = 0
        for file_meta in files:
            file_id = str(file_meta.get("file_id") or "").strip()
            file_name = str(file_meta.get("filename") or file_id).strip()
            if not file_id:
                raise AtlasBuildError("知识库文件缺少 file_id")
            content_info = await self.manager.get_file_content(db_id, file_id)
            if not isinstance(content_info, dict):
                raise AtlasBuildError(f"文件 {file_id} 内容格式无效")
            markdown, used_fallback = _content_from_file_info(content_info)
            if not markdown.strip():
                raise AtlasBuildError(f"文件 {file_name} 没有可用文本")
            if used_fallback:
                warnings.append(f"{file_name} 缺少完整 Markdown，已按 chunk 拼接")
            source_records.append(
                _source_record(
                    file_meta=file_meta,
                    markdown=markdown,
                    content_info=content_info,
                )
            )
            parsed = parse_markdown_document(
                file_id=file_id,
                file_name=file_name,
                markdown=markdown,
                lead_chars=LEAD_TEXT_MAX_CHARS,
            )
            document_routing_text = "\n".join(
                part
                for part in [
                    f"文档：{parsed.document_title}",
                    "章节：" + "；".join(parsed.heading_titles),
                    "表格：" + "；".join(parsed.table_titles),
                    f"开头：{parsed.lead_text}",
                ]
                if part.split("：", 1)[-1]
            )
            if len(document_routing_text) > DOCUMENT_ROUTING_TEXT_MAX_CHARS:
                truncated_document_cards += 1
                document_routing_text = document_routing_text[:DOCUMENT_ROUTING_TEXT_MAX_CHARS]
            document_cards.append(
                DocumentCard(
                    file_id=file_id,
                    file_name=file_name,
                    document_title=parsed.document_title,
                    heading_titles=parsed.heading_titles,
                    table_titles=parsed.table_titles,
                    lead_text=parsed.lead_text,
                    routing_text=document_routing_text,
                )
            )
            for ordinal, section in enumerate(parsed.sections, start=1):
                lead_text = section_lead_text(
                    section,
                    max_chars=LEAD_TEXT_MAX_CHARS,
                )
                heading_path = section.heading_path or [parsed.document_title]
                routing_text = "\n".join(
                    part
                    for part in [
                        f"文档：{parsed.document_title}",
                        "章节：" + " / ".join(heading_path),
                        "表格：" + "；".join(section.table_titles),
                        f"开头：{lead_text}",
                    ]
                    if part.split("：", 1)[-1]
                )
                if len(routing_text) > SECTION_ROUTING_TEXT_MAX_CHARS:
                    truncated_section_cards += 1
                    routing_text = routing_text[:SECTION_ROUTING_TEXT_MAX_CHARS]
                section_cards.append(
                    SectionCard(
                        section_id=stable_section_id(
                            file_id=file_id,
                            heading_path=heading_path,
                            ordinal=ordinal,
                        ),
                        file_id=file_id,
                        file_name=file_name,
                        heading_path=heading_path,
                        heading_level=section.heading_level,
                        lead_text=lead_text,
                        table_titles=section.table_titles,
                        routing_text=routing_text,
                    )
                )

        routing_texts = [card.routing_text for card in [*document_cards, *section_cards]]
        embeddings = await self.manager.aembed_texts(db_id, routing_texts)
        if len(embeddings) != len(routing_texts):
            raise AtlasBuildError("Atlas embedding 数量与卡片数量不一致")
        document_count = len(document_cards)
        document_cards = [
            card.model_copy(update={"embedding": embeddings[index]}) for index, card in enumerate(document_cards)
        ]
        section_cards = [
            card.model_copy(update={"embedding": embeddings[document_count + index]})
            for index, card in enumerate(section_cards)
        ]
        embedding_dimension = len(embeddings[0]) if embeddings else 0
        if embedding_dimension <= 0 or any(
            len(value) != embedding_dimension or not all(math.isfinite(float(item)) for item in value)
            for value in embeddings
        ):
            raise AtlasBuildError("Atlas embedding 维度无效或不一致")
        if configured_embedding_dimension > 0 and embedding_dimension != configured_embedding_dimension:
            raise AtlasBuildError(
                "Atlas embedding 维度与知识库配置不一致："
                f"configured={configured_embedding_dimension}, "
                f"actual={embedding_dimension}"
            )

        metadata_hash = metadata_fingerprint(
            db_id=db_id,
            embedding_model_id=embedding_model,
            embedding_dimension=configured_embedding_dimension,
            files=files,
        )
        parameters = {
            "document_routing_text_max_chars": DOCUMENT_ROUTING_TEXT_MAX_CHARS,
            "section_routing_text_max_chars": SECTION_ROUTING_TEXT_MAX_CHARS,
            "lead_text_max_chars": LEAD_TEXT_MAX_CHARS,
            "truncated_document_cards": truncated_document_cards,
            "truncated_section_cards": truncated_section_cards,
        }
        source_payload = {
            "db_id": db_id,
            "embedding_model_id": embedding_model,
            "embedding_dimension": embedding_dimension,
            "builder_version": ATLAS_BUILDER_VERSION,
            "parameters": parameters,
            "files": source_records,
        }
        snapshot_hash = hashlib.sha256(
            json.dumps(
                source_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        atlas = CorpusAtlas(
            builder_version=ATLAS_BUILDER_VERSION,
            snapshot_hash=snapshot_hash,
            metadata_fingerprint=metadata_hash,
            db_id=db_id,
            knowledge_name=str(database.get("name") or db_id),
            embedding_model_id=embedding_model,
            embedding_dimension=embedding_dimension,
            built_at=utc_isoformat(),
            parameters=parameters,
            document_cards=document_cards,
            section_cards=section_cards,
            warnings=warnings,
        )
        path = self.store.save(atlas)
        return atlas, path
