from dataclasses import dataclass, field
from typing import Annotated

from yuxi.agents import BaseContext

from .models import ExperimentProfile
from .prompt import DEFAULT_REVIEW_SYSTEM_PROMPT


@dataclass(kw_only=True)
class MedicationReviewPrimContext(BaseContext):
    system_prompt: Annotated[
        str,
        {"__template_metadata__": {"kind": "prompt"}},
    ] = field(
        default=DEFAULT_REVIEW_SYSTEM_PROMPT,
        metadata={
            "name": "治疗方案审查补充提示词",
            "description": (
                "用于补充角色、病例格式和审查重点。PRIM-RAG 的 PE/PM/INV/EV "
                "协议由代码保留。"
            ),
        },
    )
    knowledges: Annotated[
        list[str] | None,
        {"__template_metadata__": {"kind": "knowledges"}},
    ] = field(
        default=None,
        metadata={
            "name": "Milvus 知识库",
            "description": "必须且只能选择一个当前用户可访问的 Milvus 知识库。",
            "type": "list",
        },
    )
    experiment_profile: ExperimentProfile = field(
        default="full",
        metadata={
            "name": "PRIM-RAG 实验组",
            "description": (
                "b1=Evidence Card baseline；m1=方案节点；m2=方案+患者节点；"
                "m3=m2+证据调查记忆；full=m3+一次可检索软反思。"
            ),
            "type": "select",
            "options": ["b1", "m1", "m2", "m3", "full"],
        },
    )
    max_search_calls: int = field(
        default=8,
        metadata={
            "name": "最大向量检索次数",
            "description": (
                "远程实验可按统一配置提高；达到上限后仍允许 open 和生成最终答案。"
            ),
            "type": "number",
        },
    )
    max_open_calls: int = field(
        default=2,
        metadata={
            "name": "最大打开原文次数",
            "description": "达到上限后仍允许基于已有证据回答。",
            "type": "number",
        },
    )
    evidence_excerpt_chars: int = field(
        default=800,
        metadata={
            "name": "证据卡片目标字符数",
            "description": "所有实验组统一使用，允许范围 600–1000。",
            "type": "number",
        },
    )
    retrieval_timeout_seconds: int = field(
        default=600,
        metadata={
            "name": "单次检索超时（秒）",
            "description": "CPU 向量模型建议保留较长超时，允许范围 30–900。",
            "type": "number",
        },
    )
    technical_retry_limit: int = field(
        default=1,
        metadata={
            "name": "技术重试次数",
            "description": "Provider、embedding 或检索后端技术错误的重试次数。",
            "type": "number",
        },
    )

    tools: list[str] = field(default_factory=list, metadata={"hide": True})
    mcps: list[str] = field(default_factory=list, metadata={"hide": True})
    skills: list[str] = field(default_factory=list, metadata={"hide": True})
    subagents_model: str = field(default="", metadata={"hide": True})
    subagents: list[str] = field(default_factory=list, metadata={"hide": True})
    summary_threshold: int = field(default=100, metadata={"hide": True})


def validate_context_values(context: MedicationReviewPrimContext) -> None:
    selected = [
        str(value).strip()
        for value in (context.knowledges or [])
        if str(value).strip()
    ]
    if len(selected) != 1:
        raise ValueError("PRIM-RAG 必须且只能选择一个 Milvus 知识库")
    if context.experiment_profile not in {"b1", "m1", "m2", "m3", "full"}:
        raise ValueError(f"未知 experiment_profile：{context.experiment_profile}")
    if context.max_search_calls < 1:
        raise ValueError("max_search_calls 必须大于等于 1")
    if context.max_open_calls < 0:
        raise ValueError("max_open_calls 必须大于等于 0")
    if not 600 <= context.evidence_excerpt_chars <= 1000:
        raise ValueError("evidence_excerpt_chars 必须在 600–1000 之间")
    if not 30 <= context.retrieval_timeout_seconds <= 900:
        raise ValueError("retrieval_timeout_seconds 必须在 30–900 之间")
    if not 0 <= context.technical_retry_limit <= 3:
        raise ValueError("technical_retry_limit 必须在 0–3 之间")


def profile_uses_plans(profile: ExperimentProfile) -> bool:
    return profile != "b1"


def profile_uses_modifiers(profile: ExperimentProfile) -> bool:
    return profile in {"m2", "m3", "full"}


def profile_uses_investigations(profile: ExperimentProfile) -> bool:
    return profile in {"m3", "full"}


# Historical internal name retained for code importing the old helper.
profile_uses_relations = profile_uses_investigations


def profile_uses_reflection(profile: ExperimentProfile) -> bool:
    return profile == "full"
