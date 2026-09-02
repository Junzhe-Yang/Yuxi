from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from yuxi import knowledge_base
from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    _invoke_model,
    _parse_json_object,
    merge_usage,
    message_text,
    response_usage,
)
from yuxi.knowledge.base import FileStatus
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .models import (
    AtlasDocumentBuildAudit,
    AtlasDocumentCard,
    AtlasInvocationAudit,
    AtlasSourceChunk,
    AtlasSourceRecord,
    AtlasTopicCue,
    CorpusAtlas,
    calculate_snapshot_hash,
)
from .store import AtlasStore

ATLAS_BUILDER_VERSION = "acm-atlas-v8-whole-document"
DOCUMENT_PROMPT_VERSION = "acm-atlas-whole-document-v2"
SUCCESS_FILE_STATUSES = {FileStatus.INDEXED, FileStatus.DONE}

DOCUMENT_SYSTEM_PROMPT = """你负责为一个治疗方案知识库建立两层导航地图。

你会一次看到一篇完整文档。请从全文整体理解文档，而不是逐块孤立概括。

第一层 scope_summary：用自然语言说明这篇文档主要覆盖哪些疾病、患者人群、药物或非药物治疗、方案选择与重要调整场景，使后续 Agent 能判断是否值得打开本文档。

第二层 cues：抽取原文中确实存在、可帮助审查治疗方案的具体提示。
重点包括推荐或不推荐的治疗、适用或禁用条件、剂量和疗程调整、联合治疗、相互作用、监测、风险处置及替代方案。
每条 cue 应表达一个完整且可检索的关系或决策信息，而不是只列一个实体或章节标题。

不要把流行病学数字、疾病背景、检查方法、定义、作者信息等与治疗决策无直接关系的内容抽成 cue。不要针对任何具体病例作结论，不补充文档外知识。没有治疗决策信息时，cues 可以为空。

不要限制 cue 数量：保留全文中所有有独立导航价值的治疗决策信息，但合并语义完全重复的表述。
每条 cue 尽量填写支撑它的 source_chunk_ids；只能使用输入标记中真实存在的 chunk_id。
chunk ID 仅用于离线追溯，不是在线 Evidence。

completion_marker 必须是 JSON 对象的最后一个字段，并且只能在所有 cues 都输出完成后填写固定值 ATLAS_DOCUMENT_COMPLETE。"""

DOCUMENT_USER_PROMPT_TEMPLATE = """文档文件名：{file_name}

以下是按照知识库原始顺序拼接的完整文档。chunk 边界只用于标记来源，不能把相邻 chunk 当成互不相关的文本。

{document_text}"""

JSON_REPAIR_SYSTEM_PROMPT = """你只负责修复一个已经完成的 Corpus Atlas 输出的 JSON 结构。
不得增加、删除、合并或改写其中的文档摘要、主题内容和 source_chunk_ids。只修复 JSON 语法、字段名和字段类型。"""
COMPLETION_MARKER = "ATLAS_DOCUMENT_COMPLETE"
COMPLETION_MARKER_TAIL = re.compile(
    rf'"completion_marker"\s*:\s*"{COMPLETION_MARKER}"'
    r"\s*}\s*(?:```)?\s*$"
)


class AtlasBuildError(RuntimeError):
    pass


class DraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentCueDraft(DraftModel):
    cue_text: str = Field(min_length=1)
    source_chunk_ids: list[str] = Field(default_factory=list)


class DocumentAtlasEnvelope(DraftModel):
    scope_summary: str = Field(min_length=1)
    cues: list[DocumentCueDraft] = Field(default_factory=list)
    completion_marker: Literal["ATLAS_DOCUMENT_COMPLETE"]


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _files_from_database(database: dict[str, Any]) -> list[dict[str, Any]]:
    raw_files = database.get("files") or {}
    if isinstance(raw_files, dict):
        values = list(raw_files.values())
    elif isinstance(raw_files, list):
        values = raw_files
    else:
        raise AtlasBuildError("知识库 files 元数据格式无效")
    files = [
        value
        for value in values
        if isinstance(value, dict)
        and not value.get("is_folder")
        and str(value.get("status") or "") in SUCCESS_FILE_STATUSES
    ]
    return sorted(files, key=lambda value: str(value.get("file_id") or ""))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunks_from_content(
    content_info: dict[str, Any],
    *,
    file_name: str,
) -> list[dict[str, Any]]:
    raw_lines = content_info.get("lines")
    if not isinstance(raw_lines, list):
        raise AtlasBuildError(f"文档 {file_name} 的 chunk 列表格式无效")

    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_lines):
        if not isinstance(raw, dict):
            raise AtlasBuildError(
                f"文档 {file_name} 的第 {position + 1} 个 chunk 格式无效"
            )
        chunk_id = str(raw.get("id") or raw.get("chunk_id") or "").strip()
        if not chunk_id:
            raise AtlasBuildError(
                f"文档 {file_name} 的第 {position + 1} 个 chunk 缺少 chunk_id"
            )
        raw_content = raw.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise AtlasBuildError(f"文档 {file_name} 的 chunk {chunk_id} 内容为空")
        if chunk_id in seen_ids:
            raise AtlasBuildError(
                f"文档 {file_name} 存在重复 chunk_id：{chunk_id}"
            )
        seen_ids.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": _int(
                    raw.get("chunk_order_index", raw.get("chunk_index")),
                    position,
                ),
                "content": raw_content,
                "content_hash": _sha256_text(raw_content),
            }
        )
    chunks.sort(key=lambda value: (value["chunk_index"], value["chunk_id"]))
    if not chunks:
        raise AtlasBuildError(f"文档 {file_name} 没有可用的已索引 chunk")
    return chunks


def _document_title(file_name: str) -> str:
    """Use the actual indexed filename instead of guessing from OCR text."""
    return Path(file_name).stem.strip() or file_name or "未命名文档"


def _render_document(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                (
                    "===== chunk_id="
                    f"{chunk['chunk_id']} chunk_index={chunk['chunk_index']} ====="
                ),
                chunk["content"],
            ]
        )
        for chunk in chunks
    )


def _source_record(
    file_meta: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> AtlasSourceRecord:
    file_id = str(file_meta.get("file_id") or "").strip()
    file_name = str(file_meta.get("filename") or file_id).strip()
    source_chunks = [
        AtlasSourceChunk(
            chunk_id=value["chunk_id"],
            chunk_index=value["chunk_index"],
            content_hash=value["content_hash"],
            content_chars=len(value["content"]),
        )
        for value in chunks
    ]
    document_hash = _sha256_text(
        _stable_json(
            {
                "file_id": file_id,
                "file_name": file_name,
                "chunks": [
                    {
                        "chunk_id": value.chunk_id,
                        "chunk_index": value.chunk_index,
                        "content_hash": value.content_hash,
                    }
                    for value in source_chunks
                ],
            }
        )
    )
    return AtlasSourceRecord(
        file_id=file_id,
        file_name=file_name,
        status=str(file_meta.get("status") or ""),
        updated_at=str(file_meta.get("updated_at") or ""),
        metadata_content_hash=str(file_meta.get("content_hash") or ""),
        document_hash=document_hash,
        chunks=source_chunks,
    )


def metadata_fingerprint(
    *,
    db_id: str,
    files: list[dict[str, Any]],
    parameters: dict[str, Any],
    builder_model: str,
    prompt_hashes: dict[str, str],
) -> str:
    return _sha256_text(
        _stable_json(
            {
                "db_id": db_id,
                "builder_version": ATLAS_BUILDER_VERSION,
                "builder_model": builder_model,
                "prompt_hashes": prompt_hashes,
                "parameters": parameters,
                "files": [
                    {
                        "file_id": str(value.get("file_id") or ""),
                        "file_name": str(value.get("filename") or ""),
                        "status": str(value.get("status") or ""),
                        "updated_at": str(value.get("updated_at") or ""),
                        "content_hash": str(value.get("content_hash") or ""),
                        "markdown_file": str(value.get("markdown_file") or ""),
                        "processing_params": value.get("processing_params") or {},
                    }
                    for value in files
                ],
            }
        )
    )


def source_fingerprint(source_records: list[AtlasSourceRecord]) -> str:
    return _sha256_text(
        _stable_json(
            [value.model_dump(mode="json") for value in source_records]
        )
    )


def _schema_instructions(model_type: type[BaseModel]) -> str:
    schema = json.dumps(
        model_type.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\n只返回一个符合下方 JSON Schema 的 JSON 对象。"
        "不要使用 Markdown 代码围栏，不要附加解释或其它文本。"
        "字段名和必填字段必须严格遵守 Schema。\n\n"
        f"目标 JSON Schema：\n{schema}"
    )


def _finish_reason(response: Any) -> str:
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get("finish_reason") or metadata.get("stop_reason")
    if not value and isinstance(metadata.get("choices"), list):
        choices = metadata["choices"]
        if choices and isinstance(choices[0], dict):
            value = choices[0].get("finish_reason")
    return str(value or "").strip().lower()


def _has_terminal_completion_marker(raw_output: str) -> bool:
    """Only accept the marker when it closes the returned JSON object."""
    return bool(COMPLETION_MARKER_TAIL.search(raw_output))


async def _invoke_document_model(
    *,
    model: Any,
    file_name: str,
    document_text: str,
    technical_retry_limit: int,
) -> tuple[DocumentAtlasEnvelope, AtlasInvocationAudit]:
    started_at = utc_isoformat()
    started = time.monotonic()
    raw_output: str | None = None
    repair_raw_output: str | None = None
    errors: list[str] = []
    usage: dict[str, Any] = {}
    user_prompt = DOCUMENT_USER_PROMPT_TEMPLATE.format(
        file_name=file_name,
        document_text=document_text,
    )
    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=user_prompt
                    + _schema_instructions(DocumentAtlasEnvelope)
                ),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        raw_output = message_text(response)
        usage = merge_usage(usage, response_usage(response))
        finish_reason = _finish_reason(response)
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            raise AtlasBuildError(
                f"文档 {file_name} 的 Atlas 输出被模型截断；"
                "请提高该模型/网关的最大输出 token 后重建，程序不会静默丢失主题"
            )
        if not _has_terminal_completion_marker(raw_output):
            raise AtlasBuildError(
                f"文档 {file_name} 的 Atlas 输出缺少末尾完成标记，"
                "可能已被网关截断；程序不会把不完整主题列表修成合法 JSON"
            )
        payload = _parse_json_object(raw_output)
        if list(payload)[-1:] != ["completion_marker"]:
            raise AtlasBuildError(
                f"文档 {file_name} 的 completion_marker 不是 JSON 最后字段"
            )
        parsed = DocumentAtlasEnvelope.model_validate(payload)
        return parsed, AtlasInvocationAudit(
            status="success",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output,
            usage=usage,
        )
    except AtlasBuildError:
        raise
    except Exception as exc:  # noqa: BLE001 - model/schema boundary
        errors.append(f"{type(exc).__name__}: {exc}")

    repair_prompt = (
        "下方输出的内容已经生成完毕，但 JSON 格式或字段不合法。"
        "只修复结构，不得改写或补充内容。\n\n"
        f"校验错误：\n{errors[-1]}\n\n"
        f"待修复输出：\n{raw_output or '<无输出>'}"
        + _schema_instructions(DocumentAtlasEnvelope)
    )
    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=JSON_REPAIR_SYSTEM_PROMPT),
                HumanMessage(content=repair_prompt),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        repair_raw_output = message_text(response)
        usage = merge_usage(usage, response_usage(response))
        finish_reason = _finish_reason(response)
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            raise AtlasBuildError("JSON 修复输出再次被模型截断")
        if not _has_terminal_completion_marker(repair_raw_output):
            raise AtlasBuildError("JSON 修复输出缺少末尾完成标记")
        payload = _parse_json_object(repair_raw_output)
        if list(payload)[-1:] != ["completion_marker"]:
            raise AtlasBuildError("JSON 修复的 completion_marker 不是最后字段")
        parsed = DocumentAtlasEnvelope.model_validate(payload)
        return parsed, AtlasInvocationAudit(
            status="repaired",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            raw_output=raw_output,
            repair_raw_output=repair_raw_output,
            validation_errors=errors,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 - model/schema boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        logger.error(
            f"ACM Atlas document output failed: file={file_name} "
            f"errors={errors} raw_output={raw_output!r} "
            f"repair_raw_output={repair_raw_output!r}"
        )
        raise AtlasBuildError(
            f"文档 {file_name} 的 Atlas 输出无法解析：{errors[-1]}"
        ) from exc


def _cue_id(
    *,
    file_id: str,
    cue_text: str,
    source_chunk_ids: list[str],
) -> str:
    digest = _sha256_text(
        _stable_json(
            {
                "file_id": file_id,
                "cue_text": cue_text,
                "source_chunk_ids": source_chunk_ids,
            }
        )
    )[:16]
    return f"AT-{digest.upper()}"


def _finalize_document_card(
    *,
    file_id: str,
    file_name: str,
    envelope: DocumentAtlasEnvelope,
    chunks: list[dict[str, Any]],
) -> tuple[AtlasDocumentCard, int, list[str]]:
    known_chunk_ids = {value["chunk_id"] for value in chunks}
    unknown_source_ids: list[str] = []
    cues: list[AtlasTopicCue] = []
    position_by_text: dict[str, int] = {}
    duplicate_count = 0

    for draft in envelope.cues:
        cue_text = draft.cue_text.strip()
        if not cue_text:
            continue
        source_ids: list[str] = []
        for raw_id in draft.source_chunk_ids:
            chunk_id = str(raw_id).strip()
            if not chunk_id or chunk_id in source_ids:
                continue
            if chunk_id not in known_chunk_ids:
                unknown_source_ids.append(chunk_id)
                continue
            source_ids.append(chunk_id)
        position = position_by_text.get(cue_text)
        if position is not None:
            duplicate_count += 1
            existing = cues[position]
            merged_sources = list(
                dict.fromkeys([*existing.source_chunk_ids, *source_ids])
            )
            cues[position] = existing.model_copy(
                update={"source_chunk_ids": merged_sources}
            )
            continue
        position_by_text[cue_text] = len(cues)
        cues.append(
            AtlasTopicCue(
                cue_id=_cue_id(
                    file_id=file_id,
                    cue_text=cue_text,
                    source_chunk_ids=source_ids,
                ),
                cue_text=cue_text,
                source_chunk_ids=source_ids,
            )
        )

    return (
        AtlasDocumentCard(
            doc_id=file_id,
            file_name=file_name,
            title=_document_title(file_name),
            scope_summary=envelope.scope_summary.strip(),
            topic_cues=cues,
        ),
        duplicate_count,
        list(dict.fromkeys(unknown_source_ids)),
    )


class CorpusAtlasBuilder:
    def __init__(
        self,
        *,
        model: Any | None = None,
        model_name: str = "",
        manager: Any = knowledge_base,
        store: AtlasStore | None = None,
        technical_retry_limit: int = 1,
    ):
        self.model = model
        self.model_name = model_name.strip()
        self.manager = manager
        self.store = store or AtlasStore()
        self.technical_retry_limit = technical_retry_limit
        self.prompt_versions = {"document_extract": DOCUMENT_PROMPT_VERSION}
        self.prompt_hashes = {
            "document_extract": _sha256_text(
                _stable_json(
                    {
                        "system_prompt": DOCUMENT_SYSTEM_PROMPT,
                        "user_prompt_template": DOCUMENT_USER_PROMPT_TEMPLATE,
                        "schema_instructions": _schema_instructions(
                            DocumentAtlasEnvelope
                        ),
                        "json_repair_system_prompt": JSON_REPAIR_SYSTEM_PROMPT,
                    }
                )
            )
        }
        self.parameters = {"document_input_mode": "whole_document"}

    async def _database(
        self,
        db_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        database = await self.manager.get_database_info(db_id)
        if not isinstance(database, dict):
            raise AtlasBuildError(f"知识库 {db_id} 不存在")
        if str(database.get("kb_type") or "milvus").lower() != "milvus":
            raise AtlasBuildError("ACM Corpus Atlas 只支持 Milvus 知识库")
        files = _files_from_database(database)
        if not files:
            raise AtlasBuildError("知识库没有可用于 Atlas 的已索引文档")
        return database, files

    async def _source_state(
        self,
        db_id: str,
    ) -> tuple[
        dict[str, Any],
        list[
            tuple[
                dict[str, Any],
                dict[str, Any],
                list[dict[str, Any]],
                AtlasSourceRecord,
            ]
        ],
    ]:
        database, files = await self._database(db_id)
        values = []
        for file_meta in files:
            file_id = str(file_meta.get("file_id") or "").strip()
            file_name = str(file_meta.get("filename") or file_id).strip()
            if not file_id:
                raise AtlasBuildError("知识库文档缺少 file_id")
            try:
                content_info = await self.manager.get_file_content(db_id, file_id)
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                raise AtlasBuildError(
                    f"读取文档 {file_name} 的已索引 chunk 失败：{exc}"
                ) from exc
            if not isinstance(content_info, dict):
                raise AtlasBuildError(f"文档 {file_name} 内容格式无效")
            chunks = _chunks_from_content(content_info, file_name=file_name)
            values.append(
                (
                    file_meta,
                    content_info,
                    chunks,
                    _source_record(file_meta, chunks),
                )
            )
        return database, values

    async def validate_current(
        self,
        atlas: CorpusAtlas,
        *,
        deep: bool = False,
    ) -> None:
        if atlas.schema_version != "3.0":
            raise AtlasBuildError("ACM Corpus Atlas Schema 不是 3.0，请重建")
        if atlas.builder_version != ATLAS_BUILDER_VERSION:
            raise AtlasBuildError("ACM Corpus Atlas 构建器版本已变化，请重建")
        if atlas.builder_model != self.model_name:
            raise AtlasBuildError("ACM Corpus Atlas 构建模型已变化，请重建")
        if atlas.prompt_hashes != self.prompt_hashes:
            raise AtlasBuildError("ACM Corpus Atlas 提示版本已变化，请重建")
        if atlas.parameters != self.parameters:
            raise AtlasBuildError("ACM Corpus Atlas 构建参数已变化，请重建")
        self._validate_snapshot_structure(atlas)
        _, files = await self._database(atlas.db_id)
        current = metadata_fingerprint(
            db_id=atlas.db_id,
            files=files,
            parameters=self.parameters,
            builder_model=self.model_name,
            prompt_hashes=self.prompt_hashes,
        )
        if current != atlas.metadata_fingerprint:
            raise AtlasBuildError(
                "ACM Corpus Atlas 已过期：知识库文档或 chunk 已变化"
            )
        if deep:
            _, source_values = await self._source_state(atlas.db_id)
            current_source = source_fingerprint(
                [value[3] for value in source_values]
            )
            if current_source != atlas.source_fingerprint:
                raise AtlasBuildError(
                    "ACM Corpus Atlas 已过期：已索引 chunk 内容已变化"
                )
            self._validate_source_references(atlas, source_values)

    @staticmethod
    def _validate_snapshot_structure(atlas: CorpusAtlas) -> None:
        if not atlas.document_cards or not atlas.source_records:
            raise AtlasBuildError("ACM Corpus Atlas 快照不完整，请重建")
        card_ids = [value.doc_id for value in atlas.document_cards]
        source_ids = [value.file_id for value in atlas.source_records]
        if (
            len(card_ids) != len(set(card_ids))
            or len(source_ids) != len(set(source_ids))
            or set(card_ids) != set(source_ids)
        ):
            raise AtlasBuildError(
                "ACM Corpus Atlas 文档卡与来源清单不一致，请重建"
            )
        cue_ids = [
            cue.cue_id
            for card in atlas.document_cards
            for cue in card.topic_cues
        ]
        if len(cue_ids) != len(set(cue_ids)):
            raise AtlasBuildError("ACM Corpus Atlas cue ID 重复，请重建")

    async def validate_runtime(self, atlas: CorpusAtlas) -> None:
        """Cheap online validation that does not re-read every Milvus chunk."""
        await self.validate_current(atlas, deep=False)

    @staticmethod
    def _validate_source_references(
        atlas: CorpusAtlas,
        source_values: list[
            tuple[
                dict[str, Any],
                dict[str, Any],
                list[dict[str, Any]],
                AtlasSourceRecord,
            ]
        ],
    ) -> None:
        chunk_ids_by_file = {
            value[3].file_id: {chunk["chunk_id"] for chunk in value[2]}
            for value in source_values
        }
        if {value.file_id for value in atlas.source_records} != set(
            chunk_ids_by_file
        ):
            raise AtlasBuildError(
                "ACM Corpus Atlas source_records 与当前知识库文档不一致"
            )
        for card in atlas.document_cards:
            known = chunk_ids_by_file.get(card.doc_id)
            if known is None:
                raise AtlasBuildError(
                    f"Atlas 文档卡引用未知文档：{card.doc_id}"
                )
            for cue in card.topic_cues:
                unknown = [
                    value for value in cue.source_chunk_ids if value not in known
                ]
                if unknown:
                    raise AtlasBuildError(
                        f"Atlas cue {cue.cue_id} 引用未知 chunk：{unknown}"
                    )

    async def _build_document(
        self,
        *,
        db_id: str,
        chunks: list[dict[str, Any]],
        source_record: AtlasSourceRecord,
    ) -> tuple[AtlasDocumentCard, AtlasDocumentBuildAudit]:
        if self.model is None:
            raise AtlasBuildError("构建 Atlas 时必须提供离线 LLM")
        file_id = source_record.file_id
        file_name = source_record.file_name
        cache_key = _sha256_text(
            _stable_json(
                {
                    "document_hash": source_record.document_hash,
                    "builder_version": ATLAS_BUILDER_VERSION,
                    "builder_model": self.model_name,
                    "prompt_hashes": self.prompt_hashes,
                    "parameters": self.parameters,
                }
            )
        )
        cached = self.store.load_document_cache(db_id, cache_key)
        if cached is not None:
            card, audit = cached
            return card, audit.model_copy(update={"cache_hit": True})

        started = time.monotonic()
        document_text = _render_document(chunks)
        logger.info(
            f"ACM Atlas whole-document extract start: file={file_name} "
            f"chunks={len(chunks)} chars={len(document_text)}"
        )
        envelope, invocation = await _invoke_document_model(
            model=self.model,
            file_name=file_name,
            document_text=document_text,
            technical_retry_limit=self.technical_retry_limit,
        )
        card, duplicate_count, unknown_source_ids = _finalize_document_card(
            file_id=file_id,
            file_name=file_name,
            envelope=envelope,
            chunks=chunks,
        )
        warnings: list[str] = []
        if duplicate_count:
            warnings.append(
                f"合并 {duplicate_count} 条文本完全重复的主题；未做语义裁剪"
            )
        if unknown_source_ids:
            warnings.append(
                f"忽略 {len(unknown_source_ids)} 个模型写错的 source_chunk_id；"
                "主题仍保留且不会作为在线 Evidence"
            )
        audit = AtlasDocumentBuildAudit(
            file_id=file_id,
            file_name=file_name,
            document_hash=source_record.document_hash,
            cache_key=cache_key,
            input_chunk_count=len(chunks),
            input_chars=len(document_text),
            extracted_cue_count=len(envelope.cues),
            retained_cue_count=len(card.topic_cues),
            duplicate_cue_count=duplicate_count,
            unknown_source_chunk_ids=unknown_source_ids,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            invocations=[invocation],
            warnings=warnings,
        )
        self.store.save_document_cache(db_id, cache_key, card, audit)
        logger.info(
            f"ACM Atlas whole-document extract complete: file={file_name} "
            f"cues={len(card.topic_cues)} duplicates={duplicate_count} "
            f"unknown_sources={len(unknown_source_ids)}"
        )
        return card, audit

    async def build(self, db_id: str) -> tuple[CorpusAtlas, Path]:
        if not self.model_name:
            raise AtlasBuildError("构建 Atlas 时必须记录 builder model 名称")
        database, source_values = await self._source_state(db_id)
        source_records = [value[3] for value in source_values]
        cards: list[AtlasDocumentCard] = []
        audits: list[AtlasDocumentBuildAudit] = []
        for index, value in enumerate(source_values, start=1):
            _, _, chunks, source_record = value
            logger.info(
                f"ACM Atlas document start: {index}/{len(source_values)} "
                f"file={source_record.file_name} chunks={len(chunks)}"
            )
            card, audit = await self._build_document(
                db_id=db_id,
                chunks=chunks,
                source_record=source_record,
            )
            cards.append(card)
            audits.append(audit)
            logger.info(
                f"ACM Atlas document complete: {index}/{len(source_values)} "
                f"file={source_record.file_name} cues={len(card.topic_cues)} "
                f"cache_hit={audit.cache_hit}"
            )

        fingerprint = metadata_fingerprint(
            db_id=db_id,
            files=[value[0] for value in source_values],
            parameters=self.parameters,
            builder_model=self.model_name,
            prompt_hashes=self.prompt_hashes,
        )
        source_hash = source_fingerprint(source_records)
        snapshot_hash = calculate_snapshot_hash(
            metadata_fingerprint=fingerprint,
            source_fingerprint=source_hash,
            document_cards=cards,
        )
        atlas = CorpusAtlas(
            builder_version=ATLAS_BUILDER_VERSION,
            snapshot_hash=snapshot_hash,
            metadata_fingerprint=fingerprint,
            source_fingerprint=source_hash,
            db_id=db_id,
            knowledge_name=str(database.get("name") or db_id),
            builder_model=self.model_name,
            built_at=utc_isoformat(),
            prompt_versions=self.prompt_versions,
            prompt_hashes=self.prompt_hashes,
            parameters=self.parameters,
            source_records=source_records,
            document_cards=cards,
            document_audits=audits,
            warnings=[
                f"{audit.file_name}: {warning}"
                for audit in audits
                for warning in audit.warnings
            ],
        )
        self._validate_snapshot_structure(atlas)
        self._validate_source_references(atlas, source_values)
        path = self.store.save(atlas)
        logger.info(
            f"ACM Atlas build complete: db_id={db_id} documents={len(cards)} "
            f"cues={atlas.cue_count} "
            f"cache_hits={sum(1 for value in audits if value.cache_hit)} "
            f"snapshot={snapshot_hash}"
        )
        return atlas, path
