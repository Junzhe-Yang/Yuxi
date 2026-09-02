from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from yuxi.agents import BaseAgent, load_chat_model
from yuxi.agents.buildin.medication_review_prim.tools import (
    coverage_reflection,
    open_review_evidence,
    update_investigation,
)

from .context import MedicationReviewDaPrimContext
from .harness import DaReviewHarnessMiddleware
from .models import MedicationReviewDaPrimState
from .tools import search_review_kb_da


class MedicationReviewDaPrimAgent(BaseAgent):
    name = "文档地图引导的老年治疗方案调查审查（DA-PRIM 实验）"
    description = (
        "保留 PRIM-RAG Agent 自主检索，在语料地图引导下增加全局与"
        "候选文档内双路径向量检索，并可选提供非结论性调查机会。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewDaPrimContext
    metadata = {
        "examples": ["审查患者完整治疗方案，给出逐项判断、替代建议和依据。"],
        "method_family": "da-prim-rag-v1",
        "trace_schema_version": "6.0",
        "atlas_profiles": ["map", "route", "full"],
    }

    async def get_graph(
        self,
        context: MedicationReviewDaPrimContext | None = None,
        **kwargs,
    ):
        del kwargs
        context = context or self.context_schema()
        model = load_chat_model(fully_specified_name=context.model)
        return create_agent(
            model=model,
            tools=[
                search_review_kb_da,
                open_review_evidence,
                update_investigation,
                coverage_reflection,
            ],
            system_prompt="",
            middleware=[
                DaReviewHarnessMiddleware(model=model),
                ModelRetryMiddleware(),
            ],
            state_schema=MedicationReviewDaPrimState,
            context_schema=MedicationReviewDaPrimContext,
            checkpointer=await self._get_checkpointer(),
            name="medication_review_da_prim",
        )
