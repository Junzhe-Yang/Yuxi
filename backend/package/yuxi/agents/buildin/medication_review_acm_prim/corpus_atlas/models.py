from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AtlasModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtlasTopicCue(AtlasModel):
    cue_id: str
    cue_text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class AtlasDocumentCard(AtlasModel):
    doc_id: str
    file_name: str
    title: str
    scope_summary: str
    topic_cues: list[AtlasTopicCue] = Field(default_factory=list)


class AtlasSourceChunk(AtlasModel):
    chunk_id: str
    chunk_index: int
    content_hash: str
    content_chars: int


class AtlasSourceRecord(AtlasModel):
    file_id: str
    file_name: str
    status: str
    updated_at: str = ""
    metadata_content_hash: str = ""
    document_hash: str
    chunks: list[AtlasSourceChunk] = Field(min_length=1)


class AtlasInvocationAudit(AtlasModel):
    stage: Literal["document_extract"] = "document_extract"
    status: Literal["success", "repaired", "failed"]
    started_at: str
    elapsed_ms: int
    raw_output: str | None = None
    repair_raw_output: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class AtlasDocumentBuildAudit(AtlasModel):
    file_id: str
    file_name: str
    document_hash: str
    cache_key: str
    cache_hit: bool = False
    input_chunk_count: int = 0
    input_chars: int = 0
    extracted_cue_count: int = 0
    retained_cue_count: int = 0
    duplicate_cue_count: int = 0
    unknown_source_chunk_ids: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    invocations: list[AtlasInvocationAudit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def calculate_snapshot_hash(
    *,
    metadata_fingerprint: str,
    source_fingerprint: str,
    document_cards: list[AtlasDocumentCard],
) -> str:
    payload = {
        "metadata_fingerprint": metadata_fingerprint,
        "source_fingerprint": source_fingerprint,
        "document_cards": [
            value.model_dump(mode="json") for value in document_cards
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CorpusAtlas(AtlasModel):
    schema_version: Literal["3.0"] = "3.0"
    builder_version: str
    snapshot_hash: str
    metadata_fingerprint: str
    source_fingerprint: str
    db_id: str
    knowledge_name: str
    builder_model: str
    built_at: str
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    document_count: int = 0
    cue_count: int = 0
    source_count: int = 0
    source_records: list[AtlasSourceRecord] = Field(default_factory=list)
    document_cards: list[AtlasDocumentCard] = Field(default_factory=list)
    document_audits: list[AtlasDocumentBuildAudit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        del __context
        document_count = len(self.document_cards)
        cue_count = sum(len(card.topic_cues) for card in self.document_cards)
        source_count = sum(
            len(cue.source_chunk_ids)
            for card in self.document_cards
            for cue in card.topic_cues
        )
        recorded = (self.document_count, self.cue_count, self.source_count)
        actual = (document_count, cue_count, source_count)
        if recorded != (0, 0, 0) and recorded != actual:
            raise ValueError(
                "Corpus Atlas 汇总计数与文档卡不一致："
                f"recorded={recorded}, actual={actual}"
            )
        self.document_count = document_count
        self.cue_count = cue_count
        self.source_count = source_count

    def compact_view(self) -> list[dict[str, Any]]:
        """Return the document-level map shown before the Agent opens a card."""
        return [
            {
                "doc_id": card.doc_id,
                "title": card.title,
                "scope_summary": card.scope_summary,
            }
            for card in self.document_cards
        ]

    def document_view(self, doc_id: str) -> dict[str, Any]:
        for card in self.document_cards:
            if card.doc_id == doc_id:
                return {
                    "doc_id": card.doc_id,
                    "title": card.title,
                    "scope_summary": card.scope_summary,
                    "topic_cues": [
                        {
                            "cue_id": cue.cue_id,
                            "cue_text": cue.cue_text,
                        }
                        for cue in card.topic_cues
                    ],
                }
        raise KeyError(doc_id)

    def selector_view(self) -> list[dict[str, Any]]:
        """Detailed view retained only for offline historical selector replay."""
        return [self.document_view(card.doc_id) for card in self.document_cards]

    def cue_index(self) -> dict[str, tuple[AtlasDocumentCard, AtlasTopicCue]]:
        return {
            cue.cue_id: (card, cue)
            for card in self.document_cards
            for cue in card.topic_cues
        }
