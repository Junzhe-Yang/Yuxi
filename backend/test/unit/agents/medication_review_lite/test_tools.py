from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from yuxi.agents.buildin.medication_review_lite.context import (
    MedicationReviewLiteContext,
)
from yuxi.agents.buildin.medication_review_lite.tools import (
    RetrieverSelection,
    _full_chunk_excerpt,
    ensure_runtime_resources,
    search_review_kb,
)


def test_full_chunk_profile_view_does_not_truncate() -> None:
    raw = "完整原文" * 1000

    excerpt = _full_chunk_excerpt(raw)

    assert excerpt.text == raw
    assert excerpt.start == 0
    assert excerpt.end == len(raw)


@pytest.mark.asyncio
async def test_search_tool_returns_readable_cards_and_full_trace_state() -> None:
    calls: list[dict] = []
    raw_text = (
        "背景内容。" * 100
        + "关键方案证据：需要缩短评估间隔。"
        + "后续内容。" * 100
    )

    async def retriever(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [
            {
                "content": raw_text,
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 8,
                },
                "score": 0.91,
            }
        ]

    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile="m2",
        evidence_excerpt_chars=600,
    )
    ensure_runtime_resources(context)
    context._pat_rag_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"name": "知识库", "kb_type": "milvus"},
    )
    runtime = SimpleNamespace(
        context=context,
        state={
            "plan_anchors": [
                {
                    "element_id": "PE001",
                    "label": "评估时点",
                    "source_span": "计划较晚评估",
                }
            ],
            "evidence_store": {},
        },
        tool_call_id="call-1",
    )

    result = await search_review_kb.coroutine(
        query_text="评估间隔是否需要缩短",
        reason="核验评估时点",
        focus_element_ids=["PE001"],
        runtime=runtime,
    )

    assert isinstance(result, Command)
    assert calls[0]["search_mode"] == "vector"
    assert calls[0]["final_top_k"] == 5
    assert calls[0]["use_reranker"] is False
    tool_message = result.update["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert "关键方案证据：需要缩短评估间隔" in tool_message.content
    evidence = next(iter(result.update["evidence_store"].values()))
    assert evidence.raw_text == raw_text
    assert evidence.occurrences[0].shown_excerpt in tool_message.content
    assert result.update["search_count"] == 1


@pytest.mark.asyncio
async def test_search_lock_serializes_same_run_embedding_requests() -> None:
    active = 0
    peak = 0

    async def retriever(query: str, **kwargs):
        nonlocal active, peak
        del query, kwargs
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return []

    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile="m2",
    )
    ensure_runtime_resources(context)
    context._pat_rag_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"name": "知识库"},
    )

    async def run(call_id: str):
        runtime = SimpleNamespace(
            context=context,
            state={"plan_anchors": [], "evidence_store": {}},
            tool_call_id=call_id,
        )
        return await search_review_kb.coroutine(
            query_text=f"查询 {call_id}",
            reason="核验",
            focus_element_ids=[],
            runtime=runtime,
        )

    await asyncio.gather(run("call-1"), run("call-2"))

    assert peak == 1
