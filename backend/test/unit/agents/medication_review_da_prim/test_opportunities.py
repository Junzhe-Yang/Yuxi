from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.models import (
    CorpusAtlas,
    DocumentCard,
    SectionCard,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.opportunities import (
    build_retrieval_opportunities,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.router import (
    route_case,
)
from yuxi.agents.buildin.medication_review_da_prim.atlas_memory import (
    build_atlas_memory,
)


class SameEmbeddingManager:
    async def aembed_texts(self, _db_id, texts):
        return [[1.0, 0.0] for _ in texts]


def _atlas() -> CorpusAtlas:
    document = DocumentCard(
        file_id="doc-a",
        file_name="共识.md",
        document_title="共识",
        routing_text="共识",
        embedding=[1.0, 0.0],
    )
    sections = [
        SectionCard(
            section_id="sec-parent",
            file_id="doc-a",
            file_name="共识.md",
            heading_path=["共识", "剂量调整"],
            heading_level=2,
            routing_text="剂量调整",
            embedding=[1.0, 0.0],
        ),
        SectionCard(
            section_id="sec-child",
            file_id="doc-a",
            file_name="共识.md",
            heading_path=["共识", "剂量调整", "肾功能"],
            heading_level=3,
            routing_text="肾功能",
            embedding=[0.9, 0.1],
        ),
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
        document_cards=[document],
        section_cards=sections,
    )


@pytest.mark.asyncio
async def test_plan_modifier_opportunity_is_non_conclusive_and_deduplicated() -> None:
    plans = [{"element_id": "PE001", "source_span": "方案甲"}]
    modifiers = [{"modifier_id": "PM001", "source_span": "肾功能减退"}]
    computation = await route_case(
        manager=SameEmbeddingManager(),
        db_id="db",
        atlas=_atlas(),
        raw_case_text="方案甲，患者肾功能减退",
        plan_anchors=plans,
        patient_modifiers=modifiers,
    )

    opportunities = build_retrieval_opportunities(
        atlas=_atlas(),
        computation=computation,
        plan_anchors=plans,
        patient_modifiers=modifiers,
    )
    memory = build_atlas_memory(
        case_route=computation.record,
        opportunities=opportunities,
        include_opportunities=True,
    )

    assert len(opportunities) == 1
    assert opportunities[0].opportunity_type == "plan_modifier"
    assert opportunities[0].focus_plan_ids == ["PE001"]
    assert opportunities[0].focus_modifier_ids == ["PM001"]
    assert "不表示临床关系或合理性结论" in memory
    assert opportunities[0].opportunity_id in memory


@pytest.mark.asyncio
async def test_single_plan_does_not_force_an_opportunity() -> None:
    plans = [{"element_id": "PE001", "source_span": "方案甲"}]
    computation = await route_case(
        manager=SameEmbeddingManager(),
        db_id="db",
        atlas=_atlas(),
        raw_case_text="方案甲",
        plan_anchors=plans,
        patient_modifiers=[],
    )

    opportunities = build_retrieval_opportunities(
        atlas=_atlas(),
        computation=computation,
        plan_anchors=plans,
        patient_modifiers=[],
    )

    assert opportunities == []
