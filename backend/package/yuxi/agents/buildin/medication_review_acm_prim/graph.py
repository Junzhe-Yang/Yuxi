from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

from yuxi.agents import BaseAgent, load_chat_model
from yuxi.agents.buildin.medication_review_prim.tools import (
    coverage_reflection,
    open_review_evidence,
)

from .context import MedicationReviewAcmPrimContext
from .harness import AcmReviewHarnessMiddleware
from .models import MedicationReviewAcmPrimState
from .tools import (
    adaptive_coverage_checkpoint,
    extend_investigation_agenda,
    open_atlas_document,
    search_review_kb_acm_dispatch,
    set_investigation_agenda,
    submit_coverage_gap_assessment,
    update_acm_investigation,
    v7_effort_checkpoint,
)


class MedicationReviewAcmPrimAgent(BaseAgent):
    name = "关系导航式用药调查智能体（RNMI）"
    # name = "Atlas 导航治疗方案审查（ACM-PRIM 实验）"
    description = (
        "在 PRIM-RAG full 自主调查中提供两层 Corpus Atlas：先查看文档范围，"
        "再按需打开治疗主题；可切换 V7 调查努力合同和 Top-25 深度消融，"
        "也可使用无固定调查数和搜索数的自适应覆盖协议；"
        "正式证据仍由 Milvus 纯向量检索产生。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewAcmPrimContext
    metadata = {
        "examples": ["审查该患者完整治疗方案的合理性，并给出逐项判断、修正建议和依据。"],
        "method_family": "acm-prim-rag-v3/acm-prim-rag-v7/acm-prim-rag-v8",
        "trace_schema_version": "10.0/11.0/12.0",
        "atlas_schema_version": "3.0",
    }

    async def get_graph(
        self,
        context: MedicationReviewAcmPrimContext | None = None,
        **kwargs,
    ):
        del kwargs
        context = context or self.context_schema()
        model = load_chat_model(fully_specified_name=context.model)
        return create_agent(
            model=model,
            tools=[
                search_review_kb_acm_dispatch,
                set_investigation_agenda,
                extend_investigation_agenda,
                submit_coverage_gap_assessment,
                open_atlas_document,
                open_review_evidence,
                update_acm_investigation,
                coverage_reflection,
                v7_effort_checkpoint,
                adaptive_coverage_checkpoint,
            ],
            system_prompt="",
            middleware=[
                AcmReviewHarnessMiddleware(model=model),
                ModelRetryMiddleware(),
            ],
            state_schema=MedicationReviewAcmPrimState,
            context_schema=MedicationReviewAcmPrimContext,
            checkpointer=await self._get_checkpointer(),
            name="medication_review_acm_prim",
        )
