from dataclasses import dataclass, field
from typing import Annotated

from yuxi.agents import BaseContext

from .models import AgendaMode, RunMode, SynthesisMode
from .prompt import PEA_RAG_DEFAULT_SYSTEM_PROMPT


@dataclass(kw_only=True)
class MedicationReviewContext(BaseContext):
    system_prompt: Annotated[str, {"__template_metadata__": {"kind": "prompt"}}] = field(
        default=PEA_RAG_DEFAULT_SYSTEM_PROMPT,
        metadata={
            "name": "治疗方案审查补充提示词",
            "description": "补充术语、病例格式和审查侧重点；不能覆盖结构与引用约束。",
        },
    )
    knowledges: Annotated[list[str] | None, {"__template_metadata__": {"kind": "knowledges"}}] = field(
        default=None,
        metadata={
            "name": "Milvus 知识库",
            "description": "必须且只能选择一个当前用户可访问的 Milvus 知识库。",
            "type": "list",
        },
    )
    run_mode: RunMode = field(
        default="full",
        metadata={
            "name": "运行模式",
            "description": "full 生成正式答案；其它选项用于远程阶段诊断。",
            "type": "select",
            "options": [
                "full",
                "stop_after_plan",
                "stop_after_agenda",
                "stop_after_retrieval",
                "stop_after_claims",
            ],
        },
    )
    agenda_mode: AgendaMode = field(
        default="dynamic",
        metadata={
            "name": "审查议程模式",
            "description": "none 直接自主检索；dynamic 先生成自由审查问题。",
            "type": "select",
            "options": ["none", "dynamic"],
        },
    )
    synthesis_mode: SynthesisMode = field(
        default="claims",
        metadata={
            "name": "综合输入模式",
            "description": "direct_chunks 直接综合片段；claims 先抽取来源 Claim。",
            "type": "select",
            "options": ["direct_chunks", "claims"],
        },
    )
    diagnostic_trace: bool = field(
        default=False,
        metadata={
            "name": "保存诊断原始输出",
            "description": "仅在小规模远程冒烟时启用；正式批量默认关闭。",
        },
    )
    retrieval_timeout_seconds: int = field(
        default=300,
        metadata={"name": "单查询超时（秒）", "description": "允许范围 30–900。", "type": "number"},
    )
    max_search_calls: int = field(
        default=8,
        metadata={"name": "最大向量子查询数", "description": "按实际执行的子查询计数。", "type": "number"},
    )
    max_open_calls: int = field(
        default=2,
        metadata={"name": "最大打开原文次数", "description": "按证据打开相邻片段的预算。", "type": "number"},
    )
    max_agent_steps: int = field(
        default=12,
        metadata={"name": "最大逻辑 Agent 步数", "description": "技术失败不计入逻辑步数。", "type": "number"},
    )
    retrieval_top_k: int = field(
        default=5,
        metadata={"name": "每次向量检索 Top-K", "description": "主实验固定为 5。", "type": "number"},
    )
    max_final_evidence_tokens: int = field(
        default=12000,
        metadata={"name": "最终证据 Token 上限", "description": "Claim和综合共用的证据预算。", "type": "number"},
    )
    max_review_questions: int = field(
        default=8,
        metadata={"name": "最大动态审查问题数", "description": "允许范围 1–12。", "type": "number"},
    )
    max_claim_evidence: int = field(
        default=15,
        metadata={"name": "最大最终证据片段数", "description": "允许范围 1–30。", "type": "number"},
    )
    max_subqueries_per_action: int = field(
        default=3,
        metadata={"name": "单次工具最大子查询数", "description": "允许范围 1–3，按顺序执行。", "type": "number"},
    )
    technical_retry_limit: int = field(
        default=1,
        metadata={"name": "技术重试次数", "description": "不消耗逻辑查询预算。", "type": "number"},
    )
    plan_repair_limit: int = field(
        default=1,
        metadata={"name": "初始方案修复次数", "description": "允许范围 0–1。", "type": "number"},
    )

    tools: list[str] = field(default_factory=list, metadata={"hide": True})
    mcps: list[str] = field(default_factory=list, metadata={"hide": True})
    skills: list[str] = field(default_factory=list, metadata={"hide": True})
    subagents_model: str = field(default="", metadata={"hide": True})
    subagents: list[str] = field(default_factory=list, metadata={"hide": True})
    summary_threshold: int = field(default=100, metadata={"hide": True})
