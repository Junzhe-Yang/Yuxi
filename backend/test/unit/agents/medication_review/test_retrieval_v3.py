from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review.context import MedicationReviewContext
from yuxi.agents.buildin.medication_review.models import EvidenceItemV3, SearchSubquery
from yuxi.agents.buildin.medication_review.retrieval import (
    RetrieverSelection,
    open_evidence_window_v3,
    retrieve_subquery,
)


def _context() -> MedicationReviewContext:
    return MedicationReviewContext(
        user_id="user-1",
        thread_id="thread-1",
        knowledges=["处方知识库"],
        retrieval_top_k=5,
        retrieval_timeout_seconds=30,
        technical_retry_limit=1,
    )


@pytest.mark.asyncio
async def test_subquery_forces_sequential_vector_top5_and_retries(monkeypatch):
    calls: list[dict] = []

    async def retriever(query_text, **kwargs):
        calls.append({"query_text": query_text, **kwargs})
        if len(calls) == 1:
            raise RuntimeError("temporary backend error")
        return [
            {
                "content": "来源直接证据",
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 7,
                },
                "score": 0.9,
            }
        ]

    async def selection(_context):
        return RetrieverSelection(
            db_id="db-1",
            retriever=retriever,
            snapshot={"query_params": {"search_mode": "vector", "final_top_k": 5}},
        )

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.retrieval.resolve_milvus_retriever_v2",
        selection,
    )
    subquery = SearchSubquery(
        query_id="Q001",
        query_text="对于该患者，当前治疗方案的给药剂量是否符合来源建议？",
        linked_element_ids=["PE001"],
        search_reason="核验给药剂量",
    )

    result = await retrieve_subquery(
        subquery=subquery,
        context=_context(),
        review_run_id="run-1",
        case_id="CASE-1",
    )

    assert result.status == "success"
    assert [item.status for item in result.attempts] == ["backend_error", "success"]
    assert calls[-1]["search_mode"] == "vector"
    assert calls[-1]["final_top_k"] == 5
    assert calls[-1]["use_reranker"] is False
    assert calls[-1]["use_async_embedding"] is True
    assert result.candidates[0].occurrences[0].query_id == "Q001"


@pytest.mark.asyncio
async def test_open_retries_and_returns_only_adjacent_chunks(monkeypatch):
    async def selection(_context):
        async def unused_retriever(*_args, **_kwargs):
            return []

        return RetrieverSelection(db_id="db-1", retriever=unused_retriever, snapshot={})

    calls = 0

    async def file_content(_db_id, _file_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return {
            "lines": [
                {
                    "id": f"chunk-{index}",
                    "chunk_order_index": index,
                    "content": f"片段{index}",
                }
                for index in range(10)
            ]
        }

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.retrieval.resolve_milvus_retriever_v2",
        selection,
    )
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.retrieval.knowledge_base.get_file_content",
        file_content,
    )
    parent = EvidenceItemV3(
        evidence_id="EV001",
        content_hash="parent-hash",
        raw_text="片段5",
        source_document="共识.md",
        file_id="file-1",
        chunk_id="chunk-5",
        chunk_index=5,
    )

    result = await open_evidence_window_v3(
        parent=parent,
        context=_context(),
        window_before=1,
        window_after=1,
    )

    assert result.status == "success"
    assert result.attempt_count == 2
    assert [item.chunk_index for item in result.candidates] == [4, 5, 6]
    assert all(item.source_method == "open" for item in result.candidates)
    assert all(item.parent_content_hash == "parent-hash" for item in result.candidates)
