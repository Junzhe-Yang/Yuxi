from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.models import (
    CorpusAtlas,
    DocumentCard,
    SectionCard,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.router import (
    route_case,
    route_query_documents,
)


class FakeEmbeddingManager:
    async def aembed_texts(self, _db_id, texts):
        values = []
        for text in texts:
            if "患者事实" in text:
                values.append([0.0, 1.0])
            else:
                values.append([1.0, 0.0])
        return values


def _atlas() -> CorpusAtlas:
    documents = [
        DocumentCard(
            file_id="doc-a",
            file_name="甲.md",
            document_title="甲",
            routing_text="甲",
            embedding=[1.0, 0.0],
        ),
        DocumentCard(
            file_id="doc-b",
            file_name="乙.md",
            document_title="乙",
            routing_text="乙",
            embedding=[0.0, 1.0],
        ),
        DocumentCard(
            file_id="doc-c",
            file_name="丙.md",
            document_title="丙",
            routing_text="丙",
            embedding=[0.7, 0.7],
        ),
    ]
    sections = [
        SectionCard(
            section_id=f"sec-{index}",
            file_id=document.file_id,
            file_name=document.file_name,
            heading_path=[document.document_title, "章节"],
            heading_level=2,
            routing_text="章节",
            embedding=document.embedding,
        )
        for index, document in enumerate(documents, start=1)
    ]
    return CorpusAtlas(
        builder_version="test",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        db_id="db",
        knowledge_name="知识库",
        embedding_model_id="embed",
        embedding_dimension=2,
        built_at="2026-01-01T00:00:00Z",
        document_cards=documents,
        section_cards=sections,
    )


@pytest.mark.asyncio
async def test_case_route_normalizes_modifier_type_weight() -> None:
    common = {
        "manager": FakeEmbeddingManager(),
        "db_id": "db",
        "atlas": _atlas(),
        "raw_case_text": "病例和方案",
        "plan_anchors": [{"element_id": "PE001", "source_span": "方案要素"}],
    }
    one = await route_case(
        **common,
        patient_modifiers=[{"modifier_id": "PM001", "source_span": "患者事实一"}],
    )
    two = await route_case(
        **common,
        patient_modifiers=[
            {"modifier_id": "PM001", "source_span": "患者事实一"},
            {"modifier_id": "PM002", "source_span": "患者事实二"},
        ],
    )

    one_scores = {value.file_id: value.score for value in one.record.ranked_documents}
    two_scores = {value.file_id: value.score for value in two.record.ranked_documents}
    assert two_scores == pytest.approx(one_scores)
    assert len(one.record.map_sections) == 3


@pytest.mark.asyncio
async def test_query_route_can_inject_explicit_opportunity_document() -> None:
    computation = await route_case(
        manager=FakeEmbeddingManager(),
        db_id="db",
        atlas=_atlas(),
        raw_case_text="病例和方案",
        plan_anchors=[{"element_id": "PE001", "source_span": "方案要素"}],
        patient_modifiers=[],
    )

    documents, injected = route_query_documents(
        atlas=_atlas(),
        case_route=computation.record,
        query_embedding=[1.0, 0.0],
        opportunity_file_id="doc-b",
        top_k=2,
    )

    assert injected == "doc-b"
    assert documents[-1].file_id == "doc-b"
    assert documents[-1].injected is True
