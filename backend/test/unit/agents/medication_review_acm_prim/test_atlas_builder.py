from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.builder import (
    ATLAS_BUILDER_VERSION,
    AtlasBuildError,
    CorpusAtlasBuilder,
)
from yuxi.agents.buildin.medication_review_acm_prim.corpus_atlas.store import (
    AtlasStore,
)


class FakeModel:
    def __init__(self, responses: list[AIMessage | str]):
        self.responses = list(responses)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        value = self.responses.pop(0)
        return value if isinstance(value, AIMessage) else AIMessage(content=value)


class FakeManager:
    def __init__(self):
        self.updated_at = "2026-01-01T00:00:00Z"
        self.chunks = [
            {
                "id": "chunk-2",
                "chunk_order_index": 1,
                "content": "治疗期间需要关注体位性低血压。",
            },
            {
                "id": "chunk-1",
                "chunk_order_index": 0,
                "content": "α1受体阻滞剂与其他降压药合用时降压作用增强。",
            },
        ]

    async def get_database_info(self, _db_id):
        return {
            "db_id": "db-1",
            "name": "知识库",
            "kb_type": "milvus",
            "files": {
                "file-1": {
                    "file_id": "file-1",
                    "filename": "【用药助手】老年用药共识.md",
                    "status": "indexed",
                    "updated_at": self.updated_at,
                    "content_hash": "meta-hash",
                    "is_folder": False,
                }
            },
        }

    async def get_file_content(self, _db_id, _file_id):
        return {"lines": self.chunks}


def _document_response() -> str:
    return json.dumps(
        {
            "scope_summary": "覆盖老年患者降压治疗的联合用药和低血压监测。",
            "cues": [
                {
                    "cue_text": "α1受体阻滞剂与其他降压药合用时降压作用增强。",
                    "source_chunk_ids": ["chunk-1"],
                },
                {
                    "cue_text": "治疗期间需要关注体位性低血压。",
                    "source_chunk_ids": ["chunk-2"],
                },
            ],
            "completion_marker": "ATLAS_DOCUMENT_COMPLETE",
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_builder_sends_the_whole_ordered_document_once(tmp_path) -> None:
    manager = FakeManager()
    model = FakeModel([_document_response()])
    builder = CorpusAtlasBuilder(
        model=model,
        model_name="provider:model",
        manager=manager,
        store=AtlasStore(root=tmp_path),
    )

    atlas, _ = await builder.build("db-1")

    assert atlas.schema_version == "3.0"
    assert atlas.builder_version == ATLAS_BUILDER_VERSION
    assert atlas.parameters == {"document_input_mode": "whole_document"}
    assert len(model.calls) == 1
    prompt = model.calls[0][1].content
    assert "【用药助手】老年用药共识.md" in prompt
    assert prompt.index("chunk_id=chunk-1") < prompt.index("chunk_id=chunk-2")
    assert "降压作用增强" in prompt
    assert "体位性低血压" in prompt
    card = atlas.document_cards[0]
    assert card.title == "【用药助手】老年用药共识"
    assert [value.source_chunk_ids for value in card.topic_cues] == [
        ["chunk-1"],
        ["chunk-2"],
    ]
    assert atlas.document_audits[0].input_chunk_count == 2
    assert len(atlas.document_audits[0].invocations) == 1


@pytest.mark.asyncio
async def test_builder_keeps_cue_when_source_chunk_id_is_wrong(tmp_path) -> None:
    response = json.dumps(
        {
            "scope_summary": "覆盖降压治疗。",
            "cues": [
                {
                    "cue_text": "联合用药可能增强降压作用。",
                    "source_chunk_ids": ["not-a-real-chunk"],
                }
            ],
            "completion_marker": "ATLAS_DOCUMENT_COMPLETE",
        },
        ensure_ascii=False,
    )
    builder = CorpusAtlasBuilder(
        model=FakeModel([response]),
        model_name="provider:model",
        manager=FakeManager(),
        store=AtlasStore(root=tmp_path),
    )

    atlas, _ = await builder.build("db-1")

    assert atlas.document_cards[0].topic_cues[0].source_chunk_ids == []
    audit = atlas.document_audits[0]
    assert audit.unknown_source_chunk_ids == ["not-a-real-chunk"]
    assert "主题仍保留" in audit.warnings[0]


@pytest.mark.asyncio
async def test_builder_merges_only_exact_duplicate_cues(tmp_path) -> None:
    response = json.dumps(
        {
            "scope_summary": "覆盖降压治疗。",
            "cues": [
                {"cue_text": "同一主题", "source_chunk_ids": ["chunk-1"]},
                {"cue_text": "同一主题", "source_chunk_ids": ["chunk-2"]},
                {"cue_text": "相近但不同的主题", "source_chunk_ids": []},
            ],
            "completion_marker": "ATLAS_DOCUMENT_COMPLETE",
        },
        ensure_ascii=False,
    )
    builder = CorpusAtlasBuilder(
        model=FakeModel([response]),
        model_name="provider:model",
        manager=FakeManager(),
        store=AtlasStore(root=tmp_path),
    )

    atlas, _ = await builder.build("db-1")

    assert [value.cue_text for value in atlas.document_cards[0].topic_cues] == [
        "同一主题",
        "相近但不同的主题",
    ]
    assert atlas.document_cards[0].topic_cues[0].source_chunk_ids == [
        "chunk-1",
        "chunk-2",
    ]
    assert atlas.document_audits[0].duplicate_cue_count == 1


@pytest.mark.asyncio
async def test_builder_repairs_json_structure_once(tmp_path) -> None:
    model = FakeModel(
        [
            json.dumps(
                {
                    "scope_summary": "治疗范围",
                    "cues": "字段类型错误",
                    "completion_marker": "ATLAS_DOCUMENT_COMPLETE",
                },
                ensure_ascii=False,
            ),
            _document_response(),
        ]
    )
    builder = CorpusAtlasBuilder(
        model=model,
        model_name="provider:model",
        manager=FakeManager(),
        store=AtlasStore(root=tmp_path),
    )

    atlas, _ = await builder.build("db-1")

    assert len(model.calls) == 2
    assert atlas.document_audits[0].invocations[0].status == "repaired"
    assert "只修复 JSON" in model.calls[1][0].content


@pytest.mark.asyncio
async def test_builder_fails_explicitly_when_output_is_truncated(tmp_path) -> None:
    model = FakeModel(
        [
            AIMessage(
                content='{"scope_summary":"未结束',
                response_metadata={"finish_reason": "length"},
            )
        ]
    )
    builder = CorpusAtlasBuilder(
        model=model,
        model_name="provider:model",
        manager=FakeManager(),
        store=AtlasStore(root=tmp_path),
    )

    with pytest.raises(AtlasBuildError, match="输出被模型截断"):
        await builder.build("db-1")

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_builder_requires_completion_marker_as_last_json_field(tmp_path) -> None:
    model = FakeModel(
        [
            json.dumps(
                {
                    "completion_marker": "ATLAS_DOCUMENT_COMPLETE",
                    "scope_summary": "摘要",
                    "cues": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    builder = CorpusAtlasBuilder(
        model=model,
        model_name="provider:model",
        manager=FakeManager(),
        store=AtlasStore(root=tmp_path),
    )

    with pytest.raises(AtlasBuildError, match="末尾完成标记"):
        await builder.build("db-1")

    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_builder_reuses_unchanged_document_cache(tmp_path) -> None:
    store = AtlasStore(root=tmp_path)
    first_model = FakeModel([_document_response()])
    await CorpusAtlasBuilder(
        model=first_model,
        model_name="provider:model",
        manager=FakeManager(),
        store=store,
    ).build("db-1")

    second_model = FakeModel([])
    atlas, _ = await CorpusAtlasBuilder(
        model=second_model,
        model_name="provider:model",
        manager=FakeManager(),
        store=store,
    ).build("db-1")

    assert second_model.calls == []
    assert atlas.document_audits[0].cache_hit is True
