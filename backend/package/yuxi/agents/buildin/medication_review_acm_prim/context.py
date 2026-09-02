from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
    validate_context_values,
)

V7ExperimentArm = Literal["a0", "a1", "a2_k2", "a2_k3"]
V7RetrievalDepth = Literal["top10", "shadow_top25", "visible_top25"]
AcmProtocol = Literal["legacy_v7", "adaptive_coverage"]
V7_MINIMUM_SEARCH_CALLS = 6


@dataclass(kw_only=True)
class MedicationReviewAcmPrimContext(MedicationReviewPrimContext):
    experiment_profile: Literal["full"] = field(
        default="full",
        metadata={"hide": True},
    )
    acm_protocol: AcmProtocol = field(
        default="legacy_v7",
        metadata={
            "name": "ACM 调查协议",
            "description": ("legacy_v7=复现既有 A0/A1/A2 实验；adaptive_coverage=按动态调查覆盖与证据缺口调度。"),
            "type": "select",
            "options": ["legacy_v7", "adaptive_coverage"],
        },
    )
    v7_experiment_arm: V7ExperimentArm = field(
        default="a0",
        metadata={
            "name": "V7 调查努力实验臂",
            "description": (
                "a0=当前 ACM；a1=至少 6 次有效搜索；"
                "a2_k2/a2_k3=固定 2/3 个调查并执行初始与互补探查。"
                "仅 legacy_v7 生效，adaptive_coverage 会忽略此字段。"
            ),
            "type": "select",
            "options": ["a0", "a1", "a2_k2", "a2_k3"],
        },
    )
    v7_retrieval_depth: V7RetrievalDepth = field(
        default="top10",
        metadata={
            "name": "V7 向量检索深度",
            "description": (
                "top10=当前行为；shadow_top25=后台取 25、Agent 只看前 10；"
                "visible_top25=Agent 查看全部 25，仅用于独立消融。"
                "adaptive_coverage 固定使用 shadow_top25。"
            ),
            "type": "select",
            "options": ["top10", "shadow_top25", "visible_top25"],
        },
    )
    max_search_calls: int = field(
        default=50,
        metadata={
            "name": "最大向量检索次数",
            "description": (
                "默认 50，仅防止失控循环，且允许配置得更高。"
                "legacy V7 的 6 次是最低努力；adaptive_coverage 不设最低搜索次数。"
            ),
            "type": "number",
        },
    )


def validate_acm_context(context: MedicationReviewAcmPrimContext) -> None:
    validate_context_values(context)
    if context.experiment_profile != "full":
        raise ValueError("ACM-PRIM 的基础 PRIM profile 必须固定为 full")
    if context.acm_protocol not in {"legacy_v7", "adaptive_coverage"}:
        raise ValueError(f"未知 acm_protocol：{context.acm_protocol}")
    if context.v7_experiment_arm not in {"a0", "a1", "a2_k2", "a2_k3"}:
        raise ValueError(f"未知 v7_experiment_arm：{context.v7_experiment_arm}")
    if context.v7_retrieval_depth not in {
        "top10",
        "shadow_top25",
        "visible_top25",
    }:
        raise ValueError(f"未知 v7_retrieval_depth：{context.v7_retrieval_depth}")
    if context.acm_protocol == "adaptive_coverage" and context.v7_retrieval_depth != "shadow_top25":
        raise ValueError("adaptive_coverage 固定后台 Top-25、Agent 可见 Top-10，v7_retrieval_depth 必须为 shadow_top25")
    if (
        context.acm_protocol == "legacy_v7"
        and context.v7_experiment_arm != "a0"
        and context.max_search_calls < V7_MINIMUM_SEARCH_CALLS
    ):
        raise ValueError("V7 A1/A2 的 max_search_calls 必须大于等于 6")


def adaptive_coverage_enabled(
    context: MedicationReviewAcmPrimContext,
) -> bool:
    return context.acm_protocol == "adaptive_coverage"


def v7_required_investigation_count(
    arm: V7ExperimentArm,
) -> int:
    if arm == "a2_k2":
        return 2
    if arm == "a2_k3":
        return 3
    return 0


def v7_retrieval_depths(depth: V7RetrievalDepth) -> tuple[int, int]:
    if depth == "shadow_top25":
        return 25, 10
    if depth == "visible_top25":
        return 25, 25
    return 10, 10
