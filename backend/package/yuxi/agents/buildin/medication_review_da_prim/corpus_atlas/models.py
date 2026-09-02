from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AtlasModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentCard(AtlasModel):
    file_id: str
    file_name: str
    document_title: str
    heading_titles: list[str] = Field(default_factory=list)
    table_titles: list[str] = Field(default_factory=list)
    lead_text: str = ""
    routing_text: str
    embedding: list[float] = Field(default_factory=list)


class SectionCard(AtlasModel):
    section_id: str
    file_id: str
    file_name: str
    heading_path: list[str] = Field(default_factory=list)
    heading_level: int
    lead_text: str = ""
    table_titles: list[str] = Field(default_factory=list)
    routing_text: str
    embedding: list[float] = Field(default_factory=list)


class CorpusAtlas(AtlasModel):
    schema_version: Literal["1.0"] = "1.0"
    builder_version: str
    snapshot_hash: str
    metadata_fingerprint: str
    db_id: str
    knowledge_name: str
    embedding_model_id: str
    embedding_dimension: int
    built_at: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    document_cards: list[DocumentCard] = Field(default_factory=list)
    section_cards: list[SectionCard] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
