from __future__ import annotations

from types import SimpleNamespace

import pytest

from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.models import (
    CorpusAtlas,
    DocumentCard,
)
from yuxi.agents.buildin.medication_review_da_prim.models import (
    CaseRouteRecord,
    RankedDocument,
)
from yuxi.agents.buildin.medication_review_da_prim.routed_retrieval import (
    make_routed_retrieval_strategy,
)
from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrievalRequest,
    RetrieverSelection,
    ensure_runtime_resources,
)


def _atlas_and_route() -> tuple[CorpusAtlas, CaseRouteRecord]:
    cards = [
        DocumentCard(
            file_id=f"file-{index}",
            file_name=f"文档{index}.md",
            document_title=f"文档{index}",
            routing_text=f"文档{index}",
            embedding=[1.0, float(index) / 10],
        )
        for index in range(1, 7)
    ]
    ranked = [
        RankedDocument(
            file_id=card.file_id,
            file_name=card.file_name,
            document_title=card.document_title,
            rank=index,
            score=1.0 / index,
            max_similarity=1.0 / index,
        )
        for index, card in enumerate(cards, start=1)
    ]
    atlas = CorpusAtlas(
        builder_version="test",
        snapshot_hash="snapshot",
        metadata_fingerprint="metadata",
        db_id="db",
        knowledge_name="知识库",
        embedding_model_id="embed",
        embedding_dimension=2,
        built_at="2026-01-01T00:00:00Z",
        document_cards=cards,
    )
    return atlas, CaseRouteRecord(
        atlas_snapshot_hash="snapshot",
        ranked_documents=ranked,
    )


def _chunk(file_id: str, chunk_id: str, score: float) -> dict:
    return {
        "content": f"{file_id}-{chunk_id} evidence",
        "metadata": {
            "source": f"{file_id}.md",
            "file_id": file_id,
            "chunk_id": chunk_id,
            "chunk_index": 0,
        },
        "score": score,
        "distance": score,
    }


@pytest.mark.asyncio
async def test_routed_retrieval_batches_embedding_and_reuses_query_vector(
    monkeypatch,
) -> None:
    atlas, case_route = _atlas_and_route()
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
        technical_retry_limit=0,
    )
    ensure_runtime_resources(context)
    context._da_prim_atlas = atlas
    embed_calls = []
    search_calls = []

    async def aembed_texts(db_id, texts):
        embed_calls.append((db_id, texts))
        return [[1.0, 0.0] for _ in texts]

    async def aquery(_query, db_id, **kwargs):
        search_calls.append((db_id, kwargs))
        file_ids = kwargs.get("filter_file_ids")
        if file_ids is None:
            return [
                _chunk("file-1", "shared", 0.95),
                _chunk("outside", "global-only", 0.9),
            ]
        file_id = file_ids[0]
        if file_id == "file-1":
            return [_chunk("file-1", "shared", 0.95)]
        return [_chunk(file_id, "local", 0.8)]

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aembed_texts",
        aembed_texts,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aquery",
        aquery,
    )
    request = RetrievalRequest(
        selection=RetrieverSelection(
            db_id="db",
            retriever=lambda *_args, **_kwargs: None,
            snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
        ),
        context=context,
        query_id="Q-TEST",
        query_text="方案甲 肾功能",
        state={
            "case_route_record": case_route,
            "plan_anchors": [],
            "patient_modifiers": [],
            "retrieval_opportunities": [],
        },
    )

    outcome = await make_routed_retrieval_strategy(opportunity_id=None)(request)

    assert len(embed_calls) == 1
    assert len(search_calls) == 7
    assert all(value[1]["query_embedding"] == [1.0, 0.0] for value in search_calls)
    assert search_calls[0][1]["filter_file_ids"] is None
    assert [value[1]["filter_file_ids"] for value in search_calls[1:]] == [[f"file-{index}"] for index in range(1, 7)]
    assert len(outcome.chunks) == 5
    assert outcome.diagnostic_record.embedding_batch_count == 1
    assert outcome.diagnostic_record.backend_search_count == 7
    shared = next(value for value in outcome.diagnostic_record.fused_candidates if value.chunk_id == "shared")
    assert shared.retrieval_paths == ["global", "local"]


@pytest.mark.asyncio
async def test_one_local_failure_keeps_other_branches(monkeypatch) -> None:
    atlas, case_route = _atlas_and_route()
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
        technical_retry_limit=0,
    )
    ensure_runtime_resources(context)
    context._da_prim_atlas = atlas

    async def aembed_texts(_db_id, _texts):
        return [[1.0, 0.0] for _ in _texts]

    async def aquery(_query, _db_id, **kwargs):
        file_ids = kwargs.get("filter_file_ids")
        if file_ids == ["file-3"]:
            raise RuntimeError("branch unavailable")
        file_id = file_ids[0] if file_ids else "file-1"
        return [_chunk(file_id, "chunk", 0.8)]

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aembed_texts",
        aembed_texts,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aquery",
        aquery,
    )
    request = RetrievalRequest(
        selection=RetrieverSelection(
            db_id="db",
            retriever=SimpleNamespace(),
            snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
        ),
        context=context,
        query_id="Q-PARTIAL",
        query_text="查询",
        state={"case_route_record": case_route},
    )

    outcome = await make_routed_retrieval_strategy(opportunity_id=None)(request)

    assert outcome.chunks
    assert outcome.attempts[0].status == "success"
    assert any("local:file-3" in value for value in outcome.diagnostic_record.degraded_reasons)


@pytest.mark.asyncio
async def test_embedding_failure_retries_and_is_traced(monkeypatch) -> None:
    atlas, case_route = _atlas_and_route()
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
        technical_retry_limit=1,
    )
    ensure_runtime_resources(context)
    context._da_prim_atlas = atlas
    embed_calls = 0

    async def aembed_texts(_db_id, texts):
        nonlocal embed_calls
        embed_calls += 1
        if embed_calls == 1:
            raise RuntimeError("temporary embedding outage")
        return [[1.0, 0.0] for _ in texts]

    async def aquery(_query, _db_id, **kwargs):
        file_ids = kwargs.get("filter_file_ids")
        file_id = file_ids[0] if file_ids else "file-1"
        return [_chunk(file_id, "chunk", 0.8)]

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aembed_texts",
        aembed_texts,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.routed_retrieval.knowledge_base.aquery",
        aquery,
    )
    request = RetrievalRequest(
        selection=RetrieverSelection(
            db_id="db",
            retriever=SimpleNamespace(),
            snapshot={"db_id": "db", "name": "知识库", "kb_type": "milvus"},
        ),
        context=context,
        query_id="Q-RETRY",
        query_text="查询",
        state={"case_route_record": case_route},
    )

    outcome = await make_routed_retrieval_strategy(opportunity_id=None)(request)

    assert outcome.chunks
    assert embed_calls == 2
    assert outcome.diagnostic_record.embedding_batch_count == 2
