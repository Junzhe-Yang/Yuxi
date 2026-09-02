from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    content_identity_hash,
    evidence_id_from_hash,
)
from yuxi.agents.buildin.medication_review_prim.harness import (
    ReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimState,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    RetrieverSelection,
    coverage_reflection,
    ensure_runtime_resources,
    open_review_evidence,
    search_review_kb_relation,
    update_investigation,
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


def _graph(model: ToolCallingFakeModel):
    return create_agent(
        model=model,
        tools=[
            search_review_kb_relation,
            open_review_evidence,
            update_investigation,
            coverage_reflection,
        ],
        middleware=[
            ReviewHarnessMiddleware(model=model),
        ],
        state_schema=MedicationReviewPrimState,
        context_schema=MedicationReviewPrimContext,
    )


@pytest.mark.asyncio
async def test_full_profile_reflects_once_then_returns_trace_v8() -> None:
    raw_case = "患者采用方案甲和方案乙，且肾功能减退。"
    raw_evidence = "肾功能减退时方案乙需要结合患者情况复核。"
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
                    '{"anchors":['
                    '{"source_span":"方案甲","label":"方案甲",'
                    '"kind":"explicit_regimen_or_other"},'
                    '{"source_span":"方案乙","label":"方案乙",'
                    '"kind":"explicit_regimen_or_other"}]}'
                )
            ),
            AIMessage(
                content='{"modifiers":[{"source_span":"肾功能减退"}]}'
            ),
            AIMessage(
                content=(
                    "②【逐项判断】\n"
                    "■ 【PE001】方案甲\n判断：证据不足。\n\n"
                    "③【正面判断汇总】\n无。\n\n"
                    "④【负面判断汇总】\n无。\n\n"
                    "⑤【综合建议】\n继续核定。"
                )
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_review_kb",
                        "args": {
                            "query_text": "肾功能减退时方案乙是否需要调整",
                            "reason": "补查遗漏方案要素",
                            "question": (
                                "肾功能减退是否改变方案乙的适用性？"
                            ),
                            "focus_plan_ids": ["PE002"],
                            "focus_modifier_ids": ["PM001"],
                        },
                        "id": "call-reflection-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "②【逐项判断】\n"
                    "■ 【PE001】方案甲\n判断：证据不足。\n"
                    f"■ 【PE002】方案乙\n判断：需复核。[{evidence_id}]\n\n"
                    "③【正面判断汇总】\n无。\n\n"
                    "④【负面判断汇总】\n无。\n\n"
                    "⑤【综合建议】\n继续核定。"
                )
            ),
        ]
    )
    context = MedicationReviewPrimContext(
        knowledges=["知识库"],
        experiment_profile="full",
    )
    ensure_runtime_resources(context)
    async def retriever(query: str, **kwargs: Any):
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

    context._prim_retriever_selection = RetrieverSelection(
        db_id="db-1",
        retriever=retriever,
        snapshot={"db_id": "db-1", "name": "知识库", "kb_type": "milvus"},
    )
    context._prim_allowed_file_ids = {"file-1"}

    result = await _graph(model).ainvoke(
        {"messages": [HumanMessage(content=raw_case)]},
        context=context,
    )

    reflection_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "coverage_reflection"
    ]
    terminal = [
        message
        for message in result["messages"]
        if isinstance(message, AIMessage) and not message.tool_calls
    ]
    assert len(reflection_messages) == 1
    assert len(terminal) == 1
    trace = terminal[0].additional_kwargs["medication_review_trace"]
    assert trace["schema_version"] == "8.0"
    assert trace["requested_profile"] == "full"
    assert trace["effective_profile"] == "full"
    assert trace["reflection_report"]["triggered"] is True
    assert trace["reflection_report"]["completed"] is True
    assert trace["reflection_report"]["search_count_after"] == 1
    assert trace["reflection_report"]["second_draft"]
    assert trace["coverage_report"]["missing_after_reflection"] == []
    assert len(trace["query_records"]) == 1
    assert len(trace["investigations"]) == 1
    assert trace["investigations"][0]["status"] == "open"
