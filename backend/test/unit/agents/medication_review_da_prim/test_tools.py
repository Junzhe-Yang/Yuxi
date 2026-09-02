from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_da_prim.context import (
    MedicationReviewDaPrimContext,
)
from yuxi.agents.buildin.medication_review_da_prim.models import (
    RetrievalOpportunity,
)
from yuxi.agents.buildin.medication_review_da_prim.tools import (
    search_review_kb_da,
    search_review_kb_da_route,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrievalOutcome,
    RetrieverSelection,
    ensure_runtime_resources,
)


def _runtime(context, *, opportunities=None):
    ensure_runtime_resources(context)
    return SimpleNamespace(
        context=context,
        state={
            "plan_anchors": [
                {"element_id": "PE001"},
                {"element_id": "PE002"},
            ],
            "patient_modifiers": [{"modifier_id": "PM001"}],
            "retrieval_opportunities": opportunities or [],
            "relation_investigations": [],
            "evidence_store": {},
        },
        tool_call_id="call-da-search",
    )


def test_only_full_tool_schema_exposes_opportunity_id() -> None:
    route_properties = set(search_review_kb_da_route.args_schema.model_json_schema()["properties"])
    full_properties = set(search_review_kb_da.args_schema.model_json_schema()["properties"])

    assert "opportunity_id" not in route_properties
    assert "opportunity_id" in full_properties


@pytest.mark.asyncio
async def test_map_profile_preserves_flat_retrieval() -> None:
    calls = []

    async def retriever(query, **kwargs):
        calls.append((query, kwargs))
        return []

    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="map",
        evidence_excerpt_chars=600,
    )
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db",
        retriever=retriever,
        snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
    )

    result = await search_review_kb_da.coroutine(
        query_text="方案甲是否适用",
        reason="核验方案",
        focus_plan_ids=["PE001"],
        runtime=_runtime(context),
    )

    assert len(calls) == 1
    assert result.update["query_records"][0].status == "success_empty"
    assert "routed_retrieval_records" not in result.update


@pytest.mark.asyncio
async def test_full_profile_adopts_opportunity_as_soft_focus(
    monkeypatch,
) -> None:
    opportunity = RetrievalOpportunity(
        opportunity_id="OP-001",
        opportunity_type="plan_modifier",
        section_id="section-1",
        file_id="file-1",
        file_name="共识.md",
        focus_plan_ids=["PE002"],
        focus_modifier_ids=["PM001"],
        score=0.8,
    )
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="full",
        evidence_excerpt_chars=600,
    )
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db",
        retriever=lambda *_args, **_kwargs: None,
        snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
    )
    captured = []

    def make_strategy(*, opportunity_id):
        assert opportunity_id == "OP-001"

        async def strategy(request):
            captured.append(request)
            return RetrievalOutcome(diagnostic_record={"retrieval_record_id": "RR-1"})

        return strategy

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.tools.make_routed_retrieval_strategy",
        make_strategy,
    )

    result = await search_review_kb_da.coroutine(
        query_text="方案甲与肾功能关系",
        reason="核验关系",
        focus_plan_ids=["PE001"],
        opportunity_id="op-001",
        runtime=_runtime(context, opportunities=[opportunity]),
    )

    assert captured[0].focus_plan_ids == ["PE001", "PE002"]
    assert captured[0].focus_modifier_ids == ["PM001"]
    assert result.update["adopted_opportunity_ids"] == ["OP-001"]
    assert result.update["routed_retrieval_records"] == [{"retrieval_record_id": "RR-1"}]


@pytest.mark.asyncio
async def test_unknown_opportunity_is_ignored_and_warned(monkeypatch) -> None:
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="route",
        evidence_excerpt_chars=600,
    )
    context._prim_retriever_selection = RetrieverSelection(
        db_id="db",
        retriever=lambda *_args, **_kwargs: None,
        snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
    )

    def make_strategy(*, opportunity_id):
        assert opportunity_id is None

        async def strategy(_request):
            return RetrievalOutcome()

        return strategy

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.tools.make_routed_retrieval_strategy",
        make_strategy,
    )

    result = await search_review_kb_da.coroutine(
        query_text="通用查询",
        reason="补充证据",
        opportunity_id="op-missing",
        runtime=_runtime(context),
    )

    assert "adopted_opportunity_ids" not in result.update
    assert any("OP-MISSING" in value for value in result.update["warnings"])
