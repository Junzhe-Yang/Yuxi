from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from yuxi.agents.buildin.medication_review_prim.context import (
    MedicationReviewPrimContext,
    validate_context_values,
)

from .models import AtlasProfile


@dataclass(kw_only=True)
class MedicationReviewDaPrimContext(MedicationReviewPrimContext):
    experiment_profile: Literal["full"] = field(
        default="full",
        metadata={"hide": True},
    )
    atlas_profile: AtlasProfile = field(
        default="full",
        metadata={
            "name": "DA-PRIM Atlas 实验组",
            "description": ("map=仅显示语料地图；route=地图+双路径检索；" "full=地图+双路径检索+调查机会。"),
            "type": "select",
            "options": ["map", "route", "full"],
        },
    )


def validate_da_context(context: MedicationReviewDaPrimContext) -> None:
    validate_context_values(context)
    if context.experiment_profile != "full":
        raise ValueError("DA-PRIM 的基础 PRIM profile 必须固定为 full")
    if context.atlas_profile not in {"map", "route", "full"}:
        raise ValueError(f"未知 atlas_profile：{context.atlas_profile}")


def profile_uses_routed_retrieval(profile: AtlasProfile) -> bool:
    return profile in {"route", "full"}


def profile_uses_opportunities(profile: AtlasProfile) -> bool:
    return profile == "full"
