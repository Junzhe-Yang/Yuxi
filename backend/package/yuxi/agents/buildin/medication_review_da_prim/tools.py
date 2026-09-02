from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import Field

from yuxi.agents.buildin.medication_review_prim.tools import (
    _search_review_kb_impl,
)
from yuxi.agents.buildin.medication_review_prim.models import RetrievalScope

from .context import (
    MedicationReviewDaPrimContext,
    profile_uses_routed_retrieval,
)
from .models import RetrievalOpportunity
from .routed_retrieval import make_routed_retrieval_strategy


def _opportunity(
    runtime: ToolRuntime,
    opportunity_id: str | None,
) -> RetrievalOpportunity | None:
    requested = (opportunity_id or "").strip().upper()
    if not requested:
        return None
    state = runtime.state if isinstance(runtime.state, dict) else {}
    for raw in state.get("retrieval_opportunities") or []:
        try:
            value = raw if isinstance(raw, RetrievalOpportunity) else RetrievalOpportunity.model_validate(raw)
        except Exception:  # noqa: BLE001 - corrupted checkpoint entry
            continue
        if value.opportunity_id == requested:
            return value
    return None


async def _search_review_kb_da_impl(
    *,
    query_text: str,
    reason: str,
    focus_plan_ids: list[str] | None,
    focus_modifier_ids: list[str] | None,
    question: str | None,
    investigation_id: str | None,
    retrieval_scope: RetrievalScope,
    file_id: str | None,
    opportunity_id: str | None,
    runtime: ToolRuntime,
) -> Command:
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    context: MedicationReviewDaPrimContext = runtime.context
    opportunity = _opportunity(runtime, opportunity_id)
    requested_opportunity = (opportunity_id or "").strip().upper() or None
    plans = list(focus_plan_ids or [])
    modifiers = list(focus_modifier_ids or [])
    if opportunity is not None:
        plans = list(dict.fromkeys([*plans, *opportunity.focus_plan_ids]))
        modifiers = list(dict.fromkeys([*modifiers, *opportunity.focus_modifier_ids]))
    strategy = None
    if (
        profile_uses_routed_retrieval(context.atlas_profile)
        and retrieval_scope == "global"
    ):
        strategy = make_routed_retrieval_strategy(opportunity_id=(opportunity.opportunity_id if opportunity else None))
    result = await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        focus_plan_ids=plans,
        focus_modifier_ids=modifiers,
        question=question,
        investigation_id=investigation_id,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        runtime=runtime,
        retrieval_strategy=strategy,
    )
    update = dict(result.update or {})
    warnings = list(update.get("warnings") or [])
    if requested_opportunity and opportunity is None:
        warnings.append(f"忽略未知 opportunity_id：{requested_opportunity}")
    if opportunity is not None:
        update["adopted_opportunity_ids"] = [opportunity.opportunity_id]
    update["warnings"] = warnings
    return Command(update=update)


@tool("search_review_kb")
async def search_review_kb_da_route(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    question: Annotated[str, Field(min_length=1, max_length=500)] | None = None,
    investigation_id: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """自主检索 Milvus，并保留关系调查、补查和打开原文能力。"""
    return await _search_review_kb_da_impl(
        query_text=query_text,
        reason=reason,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        question=question,
        investigation_id=investigation_id,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        opportunity_id=None,
        runtime=runtime,
    )


@tool("search_review_kb")
async def search_review_kb_da(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    question: Annotated[str, Field(min_length=1, max_length=500)] | None = None,
    investigation_id: str | None = None,
    opportunity_id: (
        Annotated[
            str,
            Field(description="可选：若采用语料地图中的调查机会，传入对应 OP ID。"),
        ]
        | None
    ) = None,
    runtime: ToolRuntime = None,
) -> Command:
    """自主检索 Milvus；可采用 Atlas 调查机会，但机会不代表关系成立。"""
    return await _search_review_kb_da_impl(
        query_text=query_text,
        reason=reason,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        question=question,
        investigation_id=investigation_id,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        opportunity_id=opportunity_id,
        runtime=runtime,
    )
