from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import Field

from .context import MedicationReviewContext
from .evidence_board import merge_evidence
from .models import (
    EvidenceItemV3,
    EvidenceOpenRecordV3,
    EvidenceSearchRecordV3,
    ReviewQuestion,
    SearchSubquery,
    SearchSubqueryDraft,
    StrictModel,
)
from .retrieval import open_evidence_window_v3, retrieve_subquery


class SearchEvidenceInput(StrictModel):
    subqueries: list[SearchSubqueryDraft] = Field(min_length=1, max_length=3)


class OpenEvidenceSourceInput(StrictModel):
    evidence_id: str
    reason: str
    window_before: int = Field(default=1, ge=0, le=2)
    window_after: int = Field(default=1, ge=0, le=2)


class FinishRetrievalInput(StrictModel):
    reason: str
    priority_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_question_ids: list[str] = Field(default_factory=list)


@tool("search_evidence", args_schema=SearchEvidenceInput)
async def search_evidence_tool(subqueries: list[SearchSubqueryDraft]) -> str:
    """按顺序执行 1–3 条单一临床命题式自然语言向量查询。"""
    return json.dumps(
        {
            "subqueries": [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in subqueries
            ]
        },
        ensure_ascii=False,
    )


@tool("open_evidence_source", args_schema=OpenEvidenceSourceInput)
async def open_evidence_source_tool(
    evidence_id: str,
    reason: str,
    window_before: int = 1,
    window_after: int = 1,
) -> str:
    """打开一个已召回 Evidence 的相邻 chunk，以补全条件、表头或上下文。"""
    return json.dumps(locals(), ensure_ascii=False)


@tool("finish_retrieval", args_schema=FinishRetrievalInput)
async def finish_retrieval_tool(
    reason: str,
    priority_evidence_ids: list[str] | None = None,
    unresolved_question_ids: list[str] | None = None,
) -> str:
    """结束证据探索，可标记优先 Evidence 和仍未解决的动态问题。"""
    return json.dumps(
        {
            "reason": reason,
            "priority_evidence_ids": priority_evidence_ids or [],
            "unresolved_question_ids": unresolved_question_ids or [],
        },
        ensure_ascii=False,
    )


AGENT_TOOLS = [
    search_evidence_tool,
    open_evidence_source_tool,
    finish_retrieval_tool,
]


@dataclass(frozen=True)
class ToolExecutionResult:
    updates: dict[str, Any]
    content: str
    route: Literal["agent", "prepare_evidence"]


def _next_query_number(records: list[dict[str, Any]]) -> int:
    return len(records) + 1


def _valid_ids(state: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    question_ids = {
        str(item.get("question_id"))
        for item in state.get("review_agenda") or []
        if item.get("question_id")
    }
    element_ids = {
        str(item.get("element_id"))
        for item in state.get("plan_elements") or []
        if item.get("element_id")
    }
    fact_ids = {
        str(item.get("fact_id"))
        for item in state.get("patient_facts") or []
        if item.get("fact_id")
    }
    return question_ids, element_ids, fact_ids


def _canonicalize_subquery(
    *,
    draft: SearchSubqueryDraft,
    query_id: str,
    state: dict[str, Any],
) -> SearchSubquery:
    query_text = " ".join(draft.query_text.split()).strip()
    if not 4 <= len(query_text) <= 500:
        raise ValueError("query_text 长度必须在 4–500 字符之间")
    question_ids, element_ids, fact_ids = _valid_ids(state)
    return SearchSubquery(
        **draft.model_dump(
            exclude={
                "query_text",
                "linked_question_ids",
                "linked_element_ids",
                "linked_patient_fact_ids",
            }
        ),
        query_id=query_id,
        query_text=query_text,
        linked_question_ids=[
            value for value in dict.fromkeys(draft.linked_question_ids)
            if value in question_ids
        ],
        linked_element_ids=[
            value for value in dict.fromkeys(draft.linked_element_ids)
            if value in element_ids
        ],
        linked_patient_fact_ids=[
            value for value in dict.fromkeys(draft.linked_patient_fact_ids)
            if value in fact_ids
        ],
    )


def _update_questions(
    *,
    questions: list[dict[str, Any]],
    subquery: SearchSubquery,
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    linked = set(subquery.linked_question_ids)
    for value in questions:
        question = ReviewQuestion.model_validate(value)
        if question.question_id not in linked:
            updated.append(question.model_dump(mode="json"))
            continue
        updated.append(
            question.model_copy(
                update={
                    "status": "searched",
                    "query_ids": list(
                        dict.fromkeys([*question.query_ids, subquery.query_id])
                    ),
                    "evidence_ids": list(
                        dict.fromkeys([*question.evidence_ids, *evidence_ids])
                    ),
                }
            ).model_dump(mode="json")
        )
    return updated


async def _execute_search(
    *,
    args: dict[str, Any],
    state: dict[str, Any],
    context: MedicationReviewContext,
) -> ToolExecutionResult:
    payload = SearchEvidenceInput.model_validate(args)
    records = list(state.get("search_records") or [])
    evidence = [
        EvidenceItemV3.model_validate(item) for item in state.get("evidence") or []
    ]
    questions = list(state.get("review_agenda") or [])
    remaining = max(context.max_search_calls - int(state.get("executed_query_count") or 0), 0)
    allowed = min(remaining, context.max_subqueries_per_action)
    result_records: list[EvidenceSearchRecordV3] = []
    executed_count = int(state.get("executed_query_count") or 0)
    technical_attempts = int(state.get("technical_attempt_count") or 0)
    snapshot = dict(state.get("knowledge_base_snapshot") or {})

    for index, draft in enumerate(payload.subqueries):
        query_id = f"Q{_next_query_number([*records, *[item.model_dump() for item in result_records]]):03d}"
        try:
            subquery = _canonicalize_subquery(
                draft=draft,
                query_id=query_id,
                state=state,
            )
        except ValueError as exc:
            subquery = SearchSubquery(
                **draft.model_dump(),
                query_id=query_id,
            )
            result_records.append(
                EvidenceSearchRecordV3(
                    subquery=subquery,
                    status="invalid_query",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        if index >= allowed:
            result_records.append(
                EvidenceSearchRecordV3(
                    subquery=subquery,
                    status="skipped_budget",
                )
            )
            continue
        retrieval = await retrieve_subquery(
            subquery=subquery,
            context=context,
            review_run_id=str(state["review_run_id"]),
            case_id=str((state.get("patient_case") or {}).get("case_id") or ""),
        )
        technical_attempts += len(retrieval.attempts)
        snapshot = retrieval.knowledge_base_snapshot
        merged = merge_evidence(existing=evidence, candidates=retrieval.candidates)
        evidence = merged.evidence
        record = EvidenceSearchRecordV3(
            subquery=subquery,
            status=retrieval.status,
            attempts=retrieval.attempts,
            evidence_ids=merged.candidate_evidence_ids,
            new_evidence_ids=merged.new_evidence_ids,
            returned_count=retrieval.returned_count,
            duplicate_ratio=merged.duplicate_ratio,
            error_type=retrieval.error_type,
            error_message=retrieval.error_message,
        )
        result_records.append(record)
        if retrieval.status in {"success", "success_empty"}:
            executed_count += 1
        if retrieval.status in {"success", "success_empty"}:
            questions = _update_questions(
                questions=questions,
                subquery=subquery,
                evidence_ids=merged.candidate_evidence_ids,
            )

    route: Literal["agent", "prepare_evidence"] = "agent"
    if executed_count >= context.max_search_calls:
        route = "prepare_evidence"
    all_technical_failed = bool(result_records) and all(
        item.status == "technical_failed" for item in result_records
    )
    consecutive_errors = (
        int(state.get("consecutive_tool_error_count") or 0) + 1
        if all_technical_failed
        else 0
    )
    if consecutive_errors >= 3:
        route = "prepare_evidence"
    summary = {
        "ok": any(
            item.status in {"success", "success_empty"} for item in result_records
        ),
        "executed": sum(
            item.status in {"success", "success_empty"} for item in result_records
        ),
        "technical_failed": sum(
            item.status == "technical_failed" for item in result_records
        ),
        "new_evidence_ids": [
            value for item in result_records for value in item.new_evidence_ids
        ],
        "remaining_query_budget": max(context.max_search_calls - executed_count, 0),
    }
    return ToolExecutionResult(
        updates={
            "search_records": [
                *records,
                *[item.model_dump(mode="json") for item in result_records],
            ],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "review_agenda": questions,
            "executed_query_count": executed_count,
            "technical_attempt_count": technical_attempts,
            "logical_step_count": (
                max(int(state.get("logical_step_count") or 0) - 1, 0)
                if all_technical_failed
                else int(state.get("logical_step_count") or 0)
            ),
            "consecutive_tool_error_count": consecutive_errors,
            "knowledge_base_snapshot": snapshot,
            "last_tool_summary": summary,
            "degraded": bool(state.get("degraded")) or consecutive_errors >= 3,
            "warnings": [
                *list(state.get("warnings") or []),
                *(
                    ["连续三次检索后端技术失败，已停止证据探索并继续局部降级"]
                    if consecutive_errors >= 3
                    else []
                ),
            ],
        },
        content=json.dumps(summary, ensure_ascii=False),
        route=route,
    )


async def _execute_open(
    *,
    args: dict[str, Any],
    state: dict[str, Any],
    context: MedicationReviewContext,
) -> ToolExecutionResult:
    payload = OpenEvidenceSourceInput.model_validate(args)
    evidence = [
        EvidenceItemV3.model_validate(item) for item in state.get("evidence") or []
    ]
    by_id = {item.evidence_id: item for item in evidence}
    parent = by_id.get(payload.evidence_id)
    if parent is None:
        raise ValueError(f"未知 Evidence ID：{payload.evidence_id}")
    open_records = list(state.get("open_records") or [])
    if len(open_records) >= context.max_open_calls:
        summary = {"ok": False, "error": "open_budget_exhausted"}
        return ToolExecutionResult(
            updates={
                "last_tool_summary": summary,
            },
            content=json.dumps(summary, ensure_ascii=False),
            route="agent",
        )
    result = await open_evidence_window_v3(
        parent=parent,
        context=context,
        window_before=payload.window_before,
        window_after=payload.window_after,
    )
    merged = merge_evidence(existing=evidence, candidates=result.candidates)
    open_id = f"OP{len(open_records) + 1:03d}"
    record = EvidenceOpenRecordV3(
        open_id=open_id,
        parent_evidence_id=payload.evidence_id,
        reason=payload.reason,
        window_before=payload.window_before,
        window_after=payload.window_after,
        status=result.status,
        started_at=result.started_at,
        elapsed_ms=result.elapsed_ms,
        attempt_count=result.attempt_count,
        evidence_ids=merged.candidate_evidence_ids,
        new_evidence_ids=merged.new_evidence_ids,
        error_type=result.error_type,
        error_message=result.error_message,
    )
    technical_failure = result.status in {"timeout", "backend_error"}
    consecutive_errors = (
        int(state.get("consecutive_tool_error_count") or 0) + 1
        if technical_failure
        else 0
    )
    summary = {
        "ok": result.status in {"success", "success_empty"},
        "status": result.status,
        "attempt_count": result.attempt_count,
        "new_evidence_ids": merged.new_evidence_ids,
    }
    return ToolExecutionResult(
        updates={
            "open_records": [*open_records, record.model_dump(mode="json")],
            "evidence": [item.model_dump(mode="json") for item in merged.evidence],
            "technical_attempt_count": int(state.get("technical_attempt_count") or 0)
            + result.attempt_count,
            "logical_step_count": (
                max(int(state.get("logical_step_count") or 0) - 1, 0)
                if technical_failure
                else int(state.get("logical_step_count") or 0)
            ),
            "consecutive_tool_error_count": consecutive_errors,
            "last_tool_summary": summary,
            "degraded": bool(state.get("degraded")) or consecutive_errors >= 3,
            "warnings": [
                *list(state.get("warnings") or []),
                *(
                    ["连续三次原文打开技术失败，已停止证据探索并继续局部降级"]
                    if consecutive_errors >= 3
                    else []
                ),
            ],
        },
        content=json.dumps(summary, ensure_ascii=False),
        route="prepare_evidence" if consecutive_errors >= 3 else "agent",
    )


def _execute_finish(
    *,
    args: dict[str, Any],
    state: dict[str, Any],
) -> ToolExecutionResult:
    payload = FinishRetrievalInput.model_validate(args)
    evidence_ids = {
        str(item.get("evidence_id"))
        for item in state.get("evidence") or []
        if item.get("evidence_id")
    }
    question_ids = {
        str(item.get("question_id"))
        for item in state.get("review_agenda") or []
        if item.get("question_id")
    }
    priority = [
        value for value in dict.fromkeys(payload.priority_evidence_ids)
        if value in evidence_ids
    ]
    unresolved = [
        value for value in dict.fromkeys(payload.unresolved_question_ids)
        if value in question_ids
    ]
    finish = {
        "reason": payload.reason,
        "priority_evidence_ids": priority,
        "unresolved_question_ids": unresolved,
    }
    return ToolExecutionResult(
        updates={
            "finish_retrieval": finish,
            "last_tool_summary": {"ok": True, **finish},
        },
        content=json.dumps({"ok": True, **finish}, ensure_ascii=False),
        route="prepare_evidence",
    )


async def execute_agent_tool(
    *,
    name: str,
    args: dict[str, Any],
    state: dict[str, Any],
    context: MedicationReviewContext,
) -> ToolExecutionResult:
    if name == "search_evidence":
        return await _execute_search(args=args, state=state, context=context)
    if name == "open_evidence_source":
        return await _execute_open(args=args, state=state, context=context)
    if name == "finish_retrieval":
        return _execute_finish(args=args, state=state)
    raise ValueError(f"不支持的 Agent 工具：{name}")
