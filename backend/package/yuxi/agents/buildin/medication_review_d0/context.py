from dataclasses import dataclass, field
from typing import Annotated

from yuxi.agents import BaseContext

from .prompt import DEFAULT_SYSTEM_PROMPT


@dataclass(kw_only=True)
class MedicationReviewD0Context(BaseContext):
    """Frozen configuration surface for the deterministic D0 baseline."""

    system_prompt: Annotated[
        str,
        {"__template_metadata__": {"kind": "prompt"}},
    ] = field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "name": "病例抽取补充提示词",
            "description": "只用于补充病例格式、别名和抽取要求，不改变关系枚举和检索方法。",
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
    per_query_top_k: int = field(
        default=3,
        metadata={
            "name": "每关系返回片段数",
            "description": "D0 固定为 3。",
            "type": "number",
        },
    )
    retrieval_concurrency: int = field(
        default=1,
        metadata={
            "name": "检索并发",
            "description": "CPU Embedding 建议保持为 1，最大为 2。",
            "type": "number",
        },
    )
    retrieval_timeout_seconds: int = field(
        default=300,
        metadata={
            "name": "单查询超时（秒）",
            "description": "允许范围 30–900。",
            "type": "number",
        },
    )
    max_query_bundles: int = field(
        default=128,
        metadata={
            "name": "单病例查询上限",
            "description": "超限时整例停止，不静默截断。",
            "type": "number",
        },
    )

    tools: list[str] = field(default_factory=list, metadata={"hide": True})
    mcps: list[str] = field(default_factory=list, metadata={"hide": True})
    skills: list[str] = field(default_factory=list, metadata={"hide": True})
    subagents_model: str = field(default="", metadata={"hide": True})
    subagents: list[str] = field(default_factory=list, metadata={"hide": True})
    summary_threshold: int = field(default=100, metadata={"hide": True})
