from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review_d0.context import MedicationReviewD0Context
from yuxi.agents.buildin.medication_review.models import QueryBundle
from yuxi.agents.buildin.medication_review import retrieval


def _context() -> MedicationReviewD0Context:
    return MedicationReviewD0Context(
        user_id="1",
        thread_id="thread-1",
        knowledges=["处方知识库"],
        per_query_top_k=3,
        retrieval_concurrency=1,
        retrieval_timeout_seconds=30,
        max_query_bundles=128,
    )


def _bundle(bundle_id: str = "QB:MP:M001") -> QueryBundle:
    return QueryBundle(
        bundle_id=bundle_id,
        slot_ids=["MP:M001"],
        query_text="老年患者使用链霉素时，主要禁忌、慎用、不良反应和监测要求是什么？",
        template_id="medication_profile",
        core_entity_ids=["M001"],
    )


@pytest.mark.asyncio
async def test_retrieval_forces_vector_top3_and_disables_reranker(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    async def fake_resolve(_context):
        return [{"db_id": "db-1", "name": "处方知识库", "kb_type": "milvus"}]

    async def fake_retriever(query_text, **kwargs):
        calls.append({"query_text": query_text, **kwargs})
        return [
            {
                "content": "证据片段",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                },
                "score": 0.91,
                "distance": 0.91,
            }
        ]

    monkeypatch.setattr(retrieval, "resolve_visible_knowledge_bases_for_context", fake_resolve)
    monkeypatch.setattr(
        retrieval.knowledge_base,
        "get_retrievers",
        lambda: {
            "db-1": {
                "name": "处方知识库",
                "retriever": fake_retriever,
                "metadata": {
                    "embed_info": {"model_id": "bge-m3", "api_key": "must-not-leak"},
                    "query_params": {
                        "options": {
                            "search_mode": "hybrid",
                            "use_reranker": True,
                            "similarity_threshold": 0.3,
                            "include_distances": False,
                            "bm25_weight": 0.5,
                        },
                    },
                },
            }
        },
    )

    bundles, records, evidence, snapshot, usage = await retrieval.retrieve_query_bundles(
        [_bundle()],
        _context(),
        "review-1",
        "case-1",
    )

    assert calls == [
        {
            "query_text": _bundle().query_text,
            "search_mode": "vector",
            "final_top_k": 3,
            "use_reranker": False,
            "include_distances": True,
            "raise_on_error": True,
            "use_async_embedding": True,
        }
    ]
    assert records[0].status == "success"
    assert bundles[0].evidence_ids == [evidence[0].evidence_id]
    assert evidence[0].occurrences[0].rank == 1
    assert evidence[0].occurrences[0].score == 0.91
    assert evidence[0].occurrences[0].distance == 0.91
    assert snapshot["query_params"]["search_mode"] == "vector"
    assert snapshot["query_params"]["use_reranker"] is False
    assert snapshot["query_params"]["include_distances"] is True
    assert snapshot["query_params"]["similarity_threshold"] == 0.3
    assert "api_key" not in str(snapshot)
    assert usage["unique_evidence_count"] == 1


@pytest.mark.asyncio
async def test_duplicate_chunk_is_stored_once_with_all_query_occurrences(monkeypatch: pytest.MonkeyPatch):
    async def fake_resolve(_context):
        return [{"db_id": "db-1", "name": "处方知识库", "kb_type": "milvus"}]

    async def fake_retriever(_query_text, **_kwargs):
        return [
            {
                "content": "同一证据",
                "metadata": {"source": "a.md", "file_id": "f", "chunk_id": "c", "chunk_index": 1},
                "score": 0.8,
            }
        ]

    monkeypatch.setattr(retrieval, "resolve_visible_knowledge_bases_for_context", fake_resolve)
    monkeypatch.setattr(
        retrieval.knowledge_base,
        "get_retrievers",
        lambda: {
            "db-1": {
                "retriever": fake_retriever,
                "metadata": {"query_params": {}},
            }
        },
    )
    second = _bundle("QB:MP:M002").model_copy(update={"slot_ids": ["MP:M002"]})

    _bundles, _records, evidence, _snapshot, _usage = await retrieval.retrieve_query_bundles(
        [_bundle(), second],
        _context(),
        "review-1",
        "case-1",
    )

    assert len(evidence) == 1
    assert [item.bundle_id for item in evidence[0].occurrences] == ["QB:MP:M001", "QB:MP:M002"]


@pytest.mark.asyncio
async def test_embedding_error_is_not_reported_as_empty_recall(monkeypatch: pytest.MonkeyPatch):
    class MilvusEmbeddingError(RuntimeError):
        pass

    async def fake_resolve(_context):
        return [{"db_id": "db-1", "name": "处方知识库", "kb_type": "milvus"}]

    async def fake_retriever(_query_text, **_kwargs):
        raise MilvusEmbeddingError("cpu encoder unavailable")

    monkeypatch.setattr(retrieval, "resolve_visible_knowledge_bases_for_context", fake_resolve)
    monkeypatch.setattr(
        retrieval.knowledge_base,
        "get_retrievers",
        lambda: {"db-1": {"retriever": fake_retriever, "metadata": {"query_params": {}}}},
    )

    _bundles, records, evidence, _snapshot, _usage = await retrieval.retrieve_query_bundles(
        [_bundle()],
        _context(),
        "review-1",
        "case-1",
    )

    assert records[0].status == "embedding_error"
    assert records[0].error_message == "cpu encoder unavailable"
    assert evidence == []


@pytest.mark.asyncio
async def test_invalid_query_never_calls_retriever(monkeypatch: pytest.MonkeyPatch):
    called = False

    async def fake_resolve(_context):
        return [{"db_id": "db-1", "name": "处方知识库", "kb_type": "milvus"}]

    async def fake_retriever(_query_text, **_kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(retrieval, "resolve_visible_knowledge_bases_for_context", fake_resolve)
    monkeypatch.setattr(
        retrieval.knowledge_base,
        "get_retrievers",
        lambda: {"db-1": {"retriever": fake_retriever, "metadata": {"query_params": {}}}},
    )
    invalid = _bundle().model_copy(
        update={"validation_status": "invalid", "validation_errors": ["missing entity"]}
    )

    _bundles, records, _evidence, _snapshot, _usage = await retrieval.retrieve_query_bundles(
        [invalid],
        _context(),
        "review-1",
        "case-1",
    )

    assert called is False
    assert records[0].status == "invalid_query"


def test_context_rejects_non_top3_method_change():
    context = _context()
    context.per_query_top_k = 5

    with pytest.raises(retrieval.MedicationReviewConfigError, match="Top-3"):
        retrieval.validate_context(context)
