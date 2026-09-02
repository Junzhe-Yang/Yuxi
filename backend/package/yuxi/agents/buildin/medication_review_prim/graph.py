from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from yuxi.agents import BaseAgent, load_chat_model

from .context import MedicationReviewPrimContext
from .harness import ReviewHarnessMiddleware
from .models import MedicationReviewPrimState
from .tools import (
    coverage_reflection,
    open_review_evidence,
    search_tool_for_profile,
    update_investigation,
)


class MedicationReviewPrimAgent(BaseAgent):
    name = "老年治疗方案关系调查审查（PRIM-RAG 实验）"
    description = (
        "保留原生 Agent 自主检索，在不同实验组中逐步加入方案节点、患者修饰节点、"
        "证据问题记忆、文档内补查和一次可检索软反思。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewPrimContext
    metadata = {
        "examples": [
            "审查该患者完整治疗方案的合理性，并给出逐项判断、修正建议和依据。"
        ],
        "method_family": "prim-rag-v2",
        "trace_schema_version": "8.0",
        "profiles": ["b1", "m1", "m2", "m3", "full"],
    }

    async def get_graph(
        self,
        context: MedicationReviewPrimContext | None = None,
        **kwargs,
    ):
        del kwargs
        context = context or self.context_schema()
        model = load_chat_model(fully_specified_name=context.model)
        search_tool = search_tool_for_profile(context.experiment_profile)
        return create_agent(
            model=model,
            tools=[
                search_tool,
                open_review_evidence,
                update_investigation,
                coverage_reflection,
            ],
            system_prompt="",
            middleware=[
                ReviewHarnessMiddleware(model=model),
                ModelRetryMiddleware(),
            ],
            state_schema=MedicationReviewPrimState,
            context_schema=MedicationReviewPrimContext,
            checkpointer=await self._get_checkpointer(),
            name="medication_review_prim",
        )
