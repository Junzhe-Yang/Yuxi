from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)

from yuxi.agents import BaseAgent, load_chat_model

from .context import MedicationReviewLiteContext
from .harness import ReviewHarnessMiddleware
from .models import MedicationReviewLiteState
from .tools import open_review_evidence, search_review_kb


class MedicationReviewLiteAgent(BaseAgent):
    name = "老年治疗方案合理性审查（PAT-RAG 实验）"
    description = (
        "使用方案锚点防漏，由原生 Agent 自主执行 Milvus 向量检索和原文打开，"
        "直接生成带 Evidence ID 的完整治疗方案审查答案。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewLiteContext
    metadata = {
        "examples": [
            "审查该患者完整治疗方案的合理性，并给出逐项判断、修正建议和依据。"
        ],
        "method_family": "pat-rag-v1",
        "trace_schema_version": "4.0",
    }

    async def get_graph(
        self,
        context: MedicationReviewLiteContext | None = None,
        **kwargs,
    ):
        del kwargs
        context = context or self.context_schema()
        model = load_chat_model(fully_specified_name=context.model)
        return create_agent(
            model=model,
            tools=[search_review_kb, open_review_evidence],
            system_prompt="",
            middleware=[
                ReviewHarnessMiddleware(model=model),
                ToolCallLimitMiddleware(
                    tool_name="search_review_kb",
                    run_limit=context.max_search_calls,
                    exit_behavior="continue",
                ),
                ToolCallLimitMiddleware(
                    tool_name="open_review_evidence",
                    run_limit=context.max_open_calls,
                    exit_behavior="continue",
                ),
                ModelRetryMiddleware(),
            ],
            state_schema=MedicationReviewLiteState,
            context_schema=MedicationReviewLiteContext,
            checkpointer=await self._get_checkpointer(),
            name="medication_review_lite",
        )
