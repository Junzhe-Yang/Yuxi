from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yuxi.agents.buildin.medication_review_acm_prim.context import (
    MedicationReviewAcmPrimContext,
    validate_acm_context,
)

MODEL_CONTEXT_WINDOW_TOKENS = 262_144


@dataclass(kw_only=True)
class MedicationReviewAcmBoundedContext(MedicationReviewAcmPrimContext):
    """Fixed adaptive configuration for the independent bounded agent."""

    max_search_calls: int = field(
        default=50,
        metadata={
            "name": "检索防循环保护",
            "description": (
                "只作为异常运行的安全保护，不是目标次数、最低次数或完成条件；"
                "正常搜索数由未解决 evidence obligation 决定。"
            ),
            "type": "number",
        },
    )
    max_open_calls: int = field(
        default=10,
        metadata={
            "name": "相邻原文打开防循环保护",
            "description": "只作为异常运行保护，不要求用满。",
            "type": "number",
        },
    )
    experiment_profile: Literal["full"] = field(default="full", metadata={"hide": True})
    acm_protocol: Literal["adaptive_coverage"] = field(
        default="adaptive_coverage",
        metadata={"hide": True},
    )
    v7_experiment_arm: Literal["a0"] = field(default="a0", metadata={"hide": True})
    v7_retrieval_depth: Literal["shadow_top25"] = field(
        default="shadow_top25",
        metadata={"hide": True},
    )
    model_context_window_tokens: int = field(
        default=MODEL_CONTEXT_WINDOW_TOKENS,
        metadata={"hide": True},
    )
    action_output_tokens: int = field(default=1_536, metadata={"hide": True})
    action_output_absolute_limit: int = field(default=2_048, metadata={"hide": True})
    final_output_tokens: int = field(default=8_192, metadata={"hide": True})
    bounded_force_tool_choice: bool = field(
        default=False,
        metadata={
            "name": "强制当前工具",
            "description": (
                "仅在当前 OpenAI-compatible endpoint 已验证支持 tool_choice 时启用；"
                "关闭时仍由控制器限制可见工具并阻止提前结束。"
            ),
            "type": "boolean",
        },
    )
    phase_observation_targets: dict[str, int] = field(
        default_factory=lambda: {
            "PROPOSE_INITIAL_AGENDA": 24_000,
            "EXTEND_AGENDA": 24_000,
            "SEARCH_ACTIVE_OBLIGATION": 32_000,
            "REVIEW_ACTIVE_OBLIGATION": 56_000,
            "CLOSE_ACTIVE_INVESTIGATION": 64_000,
            "AUDIT_COVERAGE": 64_000,
            "DRAFT_FINAL": 96_000,
        },
        metadata={"hide": True},
    )


def validate_bounded_context(context: MedicationReviewAcmBoundedContext) -> None:
    validate_acm_context(context)
    if context.acm_protocol != "adaptive_coverage":
        raise ValueError("ACM Bounded Agent 只支持 adaptive_coverage")
    if context.v7_retrieval_depth != "shadow_top25":
        raise ValueError("ACM Bounded Agent 固定后台 Top-25、Agent 可见 Top-10")
    if context.model_context_window_tokens != MODEL_CONTEXT_WINDOW_TOKENS:
        raise ValueError("ACM Bounded Agent 的模型上下文固定为 262144 token")
    if not 1 <= context.action_output_tokens <= context.action_output_absolute_limit:
        raise ValueError("动作输出预算必须为正且不超过动作绝对上限")
    if context.action_output_absolute_limit >= context.model_context_window_tokens:
        raise ValueError("动作输出上限必须小于模型上下文")
    if not context.action_output_absolute_limit <= context.final_output_tokens < context.model_context_window_tokens:
        raise ValueError("最终输出预算必须不小于动作上限且小于模型上下文")
