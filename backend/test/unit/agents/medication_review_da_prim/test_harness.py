from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_da_prim.context import (
    MedicationReviewDaPrimContext,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.models import (
    CorpusAtlas,
    DocumentCard,
)
from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.router import (
    CaseRoutingComputation,
)
from yuxi.agents.buildin.medication_review_da_prim.harness import (
    DaReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_da_prim.models import (
    CaseRouteRecord,
    RankedDocument,
)
from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    disabled_anchor_audit,
)
from yuxi.agents.buildin.medication_review_prim.extraction import (
    disabled_modifier_audit,
)
from yuxi.agents.buildin.medication_review_prim.harness import (
    ReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimTrace,
    PrimCoverageReport,
    ReflectionReport,
)


def _atlas(snapshot_hash: str = "snapshot") -> CorpusAtlas:
    return CorpusAtlas(
        builder_version="test",
        snapshot_hash=snapshot_hash,
        metadata_fingerprint="metadata",
        db_id="db",
        knowledge_name="知识库",
        embedding_model_id="embed",
        embedding_dimension=2,
        built_at="2026-01-01T00:00:00Z",
        document_cards=[
            DocumentCard(
                file_id="file-1",
                file_name="共识.md",
                document_title="共识",
                routing_text="共识",
                embedding=[1.0, 0.0],
            )
        ],
    )


def _route(snapshot_hash: str = "snapshot") -> CaseRouteRecord:
    return CaseRouteRecord(
        atlas_snapshot_hash=snapshot_hash,
        ranked_documents=[
            RankedDocument(
                file_id="file-1",
                file_name="共识.md",
                document_title="共识",
                rank=1,
                score=0.9,
                max_similarity=0.9,
                source_view_ids=["VIEW-CASE"],
            )
        ],
    )


@pytest.mark.asyncio
async def test_da_preflight_validates_atlas_before_parent_extraction(monkeypatch) -> None:
    atlas = _atlas()
    events = []

    async def resolve(_context):
        events.append("resolve")
        return SimpleNamespace(db_id="db")

    def load_current(_self, db_id):
        assert db_id == "db"
        events.append("load")
        return atlas

    async def validate_current(_self, value):
        assert value is atlas
        events.append("validate")

    async def parent_before(_self, _state, runtime):
        assert runtime.context._da_prim_atlas is atlas
        events.append("parent")
        return {"parent": True}

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.resolve_milvus_retriever",
        resolve,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.AtlasStore.load_current",
        load_current,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.CorpusAtlasBuilder.validate_current",
        validate_current,
    )
    monkeypatch.setattr(ReviewHarnessMiddleware, "abefore_agent", parent_before)
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="route",
    )

    result = await DaReviewHarnessMiddleware(model=object()).abefore_agent(
        {},
        SimpleNamespace(context=context),
    )

    assert result == {"parent": True}
    assert events == ["resolve", "load", "validate", "parent"]


@pytest.mark.asyncio
async def test_initial_state_loads_validated_atlas_and_routes_case(
    monkeypatch,
) -> None:
    atlas = _atlas()
    route = _route()
    route_calls = []

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.AtlasStore.load_current",
        lambda _self, db_id: atlas if db_id == "db" else None,
    )

    async def validate_current(_self, value):
        assert value is atlas

    async def fake_route_case(**kwargs):
        route_calls.append(kwargs)
        return CaseRoutingComputation(record=route, view_embeddings={})

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.CorpusAtlasBuilder.validate_current",
        validate_current,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.route_case",
        fake_route_case,
    )
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="route",
    )
    update = {
        "knowledge_base_snapshot": {"db_id": "db"},
        "raw_case_text": "病例",
        "plan_anchors": [],
        "patient_modifiers": [],
    }

    result = await DaReviewHarnessMiddleware(model=object()).augment_initial_state(
        state={},
        update=update,
        runtime=SimpleNamespace(context=context),
    )

    assert route_calls[0]["raw_case_text"] == "病例"
    assert result["atlas_profile"] == "route"
    assert result["case_route_record"] is route
    assert result["retrieval_opportunities"] == []
    assert context._da_prim_atlas is atlas


@pytest.mark.asyncio
async def test_resumed_thread_rejects_changed_atlas(monkeypatch) -> None:
    atlas = _atlas("new-snapshot")
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.AtlasStore.load_current",
        lambda _self, _db_id: atlas,
    )

    async def validate_current(_self, _value):
        return None

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.CorpusAtlasBuilder.validate_current",
        validate_current,
    )
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="route",
    )

    with pytest.raises(ValueError, match="快照已变化"):
        await DaReviewHarnessMiddleware(model=object()).augment_initial_state(
            state={"case_route_record": _route("old-snapshot")},
            update={"knowledge_base_snapshot": {"db_id": "db"}},
            runtime=SimpleNamespace(context=context),
        )


@pytest.mark.asyncio
async def test_resumed_thread_rejects_changed_atlas_profile(monkeypatch) -> None:
    atlas = _atlas()
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.AtlasStore.load_current",
        lambda _self, _db_id: atlas,
    )

    async def validate_current(_self, _value):
        return None

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.harness.CorpusAtlasBuilder.validate_current",
        validate_current,
    )
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="full",
    )

    with pytest.raises(ValueError, match="不能切换 atlas_profile"):
        await DaReviewHarnessMiddleware(model=object()).augment_initial_state(
            state={
                "atlas_profile": "route",
                "case_route_record": _route(),
            },
            update={"knowledge_base_snapshot": {"db_id": "db"}},
            runtime=SimpleNamespace(context=context),
        )


def test_finalize_trace_upgrades_schema_without_changing_prim_payload() -> None:
    base = MedicationReviewPrimTrace(
        method_version="prim-rag-v1-full-vector-top5",
        requested_profile="full",
        effective_profile="full",
        run_status="completed",
        completion_reason="answer_generated",
        review_run_id="run-1",
        raw_question_hash="hash",
        plan_extraction=disabled_anchor_audit(),
        modifier_extraction=disabled_modifier_audit(),
        coverage_report=PrimCoverageReport(),
        reflection_report=ReflectionReport(enabled=True),
        final_answer_hash="answer-hash",
    )
    context = MedicationReviewDaPrimContext(
        knowledges=["知识库"],
        atlas_profile="full",
    )

    trace = DaReviewHarnessMiddleware(model=object()).finalize_trace(
        base_trace=base,
        state={
            "atlas_snapshot": {"snapshot_hash": "snapshot"},
            "case_route_record": _route(),
            "retrieval_opportunities": [],
            "adopted_opportunity_ids": [],
            "routed_retrieval_records": [],
        },
        context=context,
    )

    assert trace.schema_version == "6.0"
    assert trace.method_family == "da-prim-rag-v1"
    assert trace.method_version == "da-prim-rag-v1-full-vector-top5"
    assert trace.requested_profile == base.requested_profile
    assert trace.case_route_record.atlas_snapshot_hash == "snapshot"
