from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_lite.context import (
    MedicationReviewLiteContext,
)
from yuxi.agents.buildin.medication_review_lite import graph as graph_module
from yuxi.agents.buildin.medication_review_lite.evidence import (
    content_identity_hash,
    evidence_id_from_hash,
)
from yuxi.agents.buildin.medication_review_lite.harness import (
    ReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    MedicationReviewLiteState,
)
from yuxi.agents.buildin.medication_review_lite.tools import (
    RetrieverSelection,
    ensure_runtime_resources,
    open_review_evidence,
    search_review_kb,
)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> ToolCallingFakeModel:
        del tools, tool_choice, kwargs
        return self


@pytest.mark.asyncio
async def test_graph_can_be_rebuilt_without_runtime_knowledge_for_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    agent = object.__new__(graph_module.MedicationReviewLiteAgent)

    async def get_checkpointer():
        return None

    agent._get_checkpointer = get_checkpointer
    monkeypatch.setattr(
        graph_module,
        "load_chat_model",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        graph_module,
        "create_agent",
        lambda **_kwargs: sentinel,
    )

    graph = await agent.get_graph()

    assert graph is sentinel


def _graph(model: ToolCallingFakeModel):
    return create_agent(
        model=model,
        tools=[search_review_kb, open_review_evidence],
        middleware=[
            ReviewHarnessMiddleware(model=model),
            ToolCallLimitMiddleware(
                tool_name="search_review_kb",
                run_limit=8,
                exit_behavior="continue",
            ),
            ToolCallLimitMiddleware(
                tool_name="open_review_evidence",
                run_limit=2,
                exit_behavior="continue",
            ),
        ],
        state_schema=MedicationReviewLiteState,
        context_schema=MedicationReviewLiteContext,
    )


@pytest.mark.asyncio
async def test_native_agent_searches_then_returns_one_traced_answer() -> None:
    assert "runtime" in search_review_kb._injected_args_keys
    raw_evidence = "方案甲适用于当前情况，应继续随访。"
    evidence_id = evidence_id_from_hash(
        content_identity_hash(
            raw_text=raw_evidence,
            db_id="db-1",
            file_id="file-1",
            chunk_id="chunk-1",
            chunk_index=1,
        )
    )
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content=(
                    '{"anchors":[{"source_span":"方案甲","label":"方案甲",'
                    '"kind":"explicit_regimen_or_other"}]}'
                )
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_review_kb",
                        "args": {
                            "query_text": "方案甲适用条件与调整原则",
                            "reason": "核验方案甲",
                            "focus_element_ids": ["PE001"],
                        },
                        "id": "call-search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "②【逐项判断】\n"
                    "■ 【PE001】方案甲\n"
                    f"判断：合理。依据：[{evidence_id}]\n\n"
                    "③【正面判断汇总】\n方案甲合理。\n\n"
                    "④【负面判断汇总】\n无。\n\n"
                    "⑤【综合建议】\n继续随访。"
                )
            ),
        ]
    )

    retriever_calls: list[str] = []

    async def retriever(query: str, **kwargs: Any) -> list[dict[str, Any]]:
        retriever_calls.append(query)
        del query, kwargs
        return [
            {
                "content": raw_evidence,
                "metadata": {
                    "source": "共识.md",
                    "file_id": "file-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                },
                "score": 0.9,
            }
        ]

    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile="m2",
    )
    ensure_runtime_resources(context)
    context._pat_rag_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    result = await _graph(model).ainvoke(
        {"messages": [HumanMessage(content="患者采用方案甲。")]},
        context=context,
    )

    terminal_messages = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]
    assert any(
        isinstance(message, ToolMessage) for message in result["messages"]
    ), result["messages"]
    assert retriever_calls == ["方案甲适用条件与调整原则"], [
        message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert len(terminal_messages) == 1
    final = terminal_messages[0]
    assert "①【原方案要素清单】" in final.content
    assert "⑥【依据清单】" in final.content
    trace = final.additional_kwargs["medication_review_trace"]
    assert trace["schema_version"] == "4.0"
    assert trace["experiment_profile"] == "m2"
    assert len(result["search_records"]) == 1, trace
    assert len(trace["search_records"]) == 1, result
    assert trace["budgets"]["executed_search_calls"] == 1
    assert trace["evidence_store"][0]["raw_text"] == raw_evidence


@pytest.mark.parametrize("profile", ["b1", "m1", "m2", "m3"])
@pytest.mark.asyncio
async def test_every_experiment_profile_generates_a_final_answer(
    profile: str,
) -> None:
    final_body = (
        "②【逐项判断】\n"
        + (
            "■ 【PE001】方案甲\n判断：证据不足。\n\n"
            if profile != "b1"
            else "■ 方案甲\n判断：证据不足。\n\n"
        )
        + "③【正面判断汇总】\n无。\n\n"
        "④【负面判断汇总】\n证据不足。\n\n"
        "⑤【综合建议】\n补充证据后复核。"
    )
    responses = [AIMessage(content=final_body)]
    if profile != "b1":
        responses.insert(
            0,
            AIMessage(
                content=(
                    '{"anchors":[{"source_span":"方案甲","label":"方案甲",'
                    '"kind":"explicit_regimen_or_other"}]}'
                )
            ),
        )
    model = ToolCallingFakeModel(responses=responses)
    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile=profile,
    )
    ensure_runtime_resources(context)
    context._pat_rag_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=lambda *_args, **_kwargs: [],
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )

    result = await _graph(model).ainvoke(
        {"messages": [HumanMessage(content="患者采用方案甲。")]},
        context=context,
    )

    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content
    trace = final.additional_kwargs["medication_review_trace"]
    assert trace["experiment_profile"] == profile
    assert trace["run_status"] in {"completed", "partial"}
