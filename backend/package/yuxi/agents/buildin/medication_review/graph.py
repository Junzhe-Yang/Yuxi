from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from yuxi.agents import BaseAgent, load_chat_model
from yuxi.utils import logger
from yuxi.utils.datetime_utils import utc_isoformat

from .claim_extraction import extract_claims
from .context import MedicationReviewContext
from .evidence_board import select_evidence
from .extraction import CasePlanExtractionError, extract_case_and_plan
from .models import (
    AgentDecisionRecordV3,
    EvidenceClaim,
    EvidenceItemV3,
    MedicationReviewStateV3,
    PatientCase,
    ReviewQuestion,
    TreatmentPlanElement,
    V3_DEFAULT_METHOD_VERSION,
)
from .prompt import AGENT_DECISION_PROMPT, AGENT_DECISION_SYSTEM_PROMPT
from .rendering import render_review_v3
from .retrieval import validate_v3_context
from .review_agenda import build_review_agenda
from .review_synthesis import synthesize_review
from .tools import AGENT_TOOLS, execute_agent_tool
from .trace import build_trace, safe_context_snapshot


def _latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
        else:
            dumped = message.model_dump() if hasattr(message, "model_dump") else message
            if not isinstance(dumped, dict) or dumped.get("type") not in {"human", "user"}:
                continue
            content = dumped.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str) else str(item.get("text") or "")
                for item in content
                if isinstance(item, (str, dict))
            )
    raise CasePlanExtractionError("没有找到本次运行的用户病例")


def _error(stage: str, exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "stage": stage}


def _merge_usage(
    current: dict[str, Any],
    stage: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    usage = dict(current)
    stages = dict(usage.get("llm_usage") or {})
    previous = dict(stages.get(stage) or {})
    for key, value in values.items():
        if isinstance(value, int | float) and isinstance(previous.get(key), int | float):
            previous[key] += value
        else:
            previous[key] = value
    stages[stage] = previous
    usage["llm_usage"] = stages
    return usage


def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return dict(usage)
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(value, dict):
            return dict(value)
    return {}


async def initialize_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    raw_question = _latest_human_text(list(state.get("messages") or []))
    try:
        validate_v3_context(runtime.context)
        errors: list[dict[str, Any]] = []
        completion_reason = "not_completed"
    except Exception as exc:  # noqa: BLE001
        errors = [_error("validate_context", exc)]
        completion_reason = "invalid_config"
    return {
        "review_run_id": str(uuid.uuid4()),
        "stage": "initialized",
        "run_mode": runtime.context.run_mode,
        "agenda_mode": runtime.context.agenda_mode,
        "synthesis_mode": runtime.context.synthesis_mode,
        "raw_question": raw_question,
        "raw_question_hash": hashlib.sha256(
            " ".join(raw_question.split()).encode("utf-8")
        ).hexdigest(),
        "patient_case": None,
        "patient_facts": [],
        "plan_extraction": {},
        "plan_elements": [],
        "review_agenda": [],
        "agenda_audit": {},
        "search_records": [],
        "open_records": [],
        "finish_retrieval": {},
        "evidence": [],
        "evidence_selection": {},
        "selected_evidence_ids": [],
        "evidence_claims": [],
        "claim_extraction": {},
        "review_synthesis": {},
        "synthesis_fatal": False,
        "local_validation_events": [],
        "agent_steps": [],
        "final_review": None,
        "rendered_answer": "",
        "run_status": "failed",
        "completion_reason": completion_reason,
        "knowledge_base_snapshot": {},
        "agent_config_snapshot": safe_context_snapshot(runtime.context),
        "usage": {},
        "warnings": [],
        "errors": errors,
        "logical_step_count": 0,
        "technical_attempt_count": 0,
        "agent_model_error_count": 0,
        "consecutive_tool_error_count": 0,
        "executed_query_count": 0,
        "degraded": False,
        "pending_tool_call": None,
        "tool_route": "agent",
        "last_tool_summary": None,
    }


def route_after_initialize(state: MedicationReviewStateV3) -> str:
    return "finalize" if state.get("completion_reason") == "invalid_config" else "extract_plan"


async def extract_plan_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    context = runtime.context
    try:
        result = await extract_case_and_plan(
            raw_question=str(state["raw_question"]),
            model=load_chat_model(context.model),
            system_prompt=context.system_prompt,
            plan_repair_limit=context.plan_repair_limit,
            technical_retry_limit=context.technical_retry_limit,
            retain_raw_output=context.diagnostic_trace,
        )
        usage = dict(state.get("usage") or {})
        for stage in ("extraction", "verification"):
            audit = getattr(result.audit, stage)
            if audit is not None:
                usage = _merge_usage(usage, stage, audit.usage)
        return {
            "stage": "plan_extracted",
            "patient_case": result.patient_case.model_dump(mode="json"),
            "patient_facts": result.patient_facts,
            "plan_elements": [
                item.model_dump(mode="json") for item in result.plan_elements
            ],
            "plan_extraction": result.audit.model_dump(mode="json"),
            "warnings": [*list(state.get("warnings") or []), *result.warnings],
            "usage": usage,
            "degraded": bool(state.get("degraded")) or result.degraded,
        }
    except Exception as exc:  # noqa: BLE001 - trace must preserve extraction failure
        audit: dict[str, Any] = {}
        if isinstance(exc, CasePlanExtractionError):
            audit = exc.audit.model_dump(mode="json")
        return {
            "stage": "plan_failed",
            "run_status": "failed",
            "completion_reason": "plan_extraction_failed",
            "plan_extraction": audit,
            "errors": [*list(state.get("errors") or []), _error("extract_plan", exc)],
        }


def route_after_extraction(
    state: MedicationReviewStateV3,
) -> str:
    if not state.get("patient_case") or not state.get("plan_elements"):
        return "finalize"
    if state.get("run_mode") == "stop_after_plan":
        return "finalize_debug"
    if state.get("agenda_mode") == "dynamic":
        return "build_agenda"
    return "agent"


async def build_agenda_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    elements = [
        TreatmentPlanElement.model_validate(item)
        for item in state.get("plan_elements") or []
    ]
    result = await build_review_agenda(
        model=load_chat_model(runtime.context.model),
        raw_case_text=str(state["raw_question"]),
        patient_facts=list(state.get("patient_facts") or []),
        elements=elements,
        system_prompt=runtime.context.system_prompt,
        max_questions=runtime.context.max_review_questions,
        technical_retry_limit=runtime.context.technical_retry_limit,
        retain_raw_output=runtime.context.diagnostic_trace,
    )
    usage = dict(state.get("usage") or {})
    if isinstance(result.audit.get("usage"), dict):
        usage = _merge_usage(usage, "review_agenda", result.audit["usage"])
    return {
        "stage": "agenda_built",
        "review_agenda": [
            item.model_dump(mode="json") for item in result.questions
        ],
        "agenda_audit": result.audit,
        "warnings": [*list(state.get("warnings") or []), *result.warnings],
        "usage": usage,
    }


def route_after_agenda(
    state: MedicationReviewStateV3,
) -> str:
    return (
        "finalize_debug"
        if state.get("run_mode") == "stop_after_agenda"
        else "agent"
    )


def _remaining_budgets(
    state: MedicationReviewStateV3,
    context: MedicationReviewContext,
) -> dict[str, int]:
    return {
        "subqueries": max(
            context.max_search_calls - int(state.get("executed_query_count") or 0),
            0,
        ),
        "open": max(
            context.max_open_calls - len(state.get("open_records") or []),
            0,
        ),
        "logical_steps": max(
            context.max_agent_steps - int(state.get("logical_step_count") or 0),
            0,
        ),
    }


def _decision_context(
    state: MedicationReviewStateV3,
    context: MedicationReviewContext,
) -> dict[str, Any]:
    evidence = [
        EvidenceItemV3.model_validate(item) for item in state.get("evidence") or []
    ]
    evidence_index = [
        {
            "evidence_id": item.evidence_id,
            "source_document": item.source_document,
            "chunk_index": item.chunk_index,
            "source_method": item.source_method,
            "query_ids": list(
                dict.fromkeys(value.query_id for value in item.occurrences)
            ),
            "excerpt": " ".join(item.raw_text.split())[:260],
        }
        for item in evidence[-40:]
    ]
    remaining = _remaining_budgets(state, context)
    actions = ["finish_retrieval"]
    if remaining["subqueries"] > 0:
        actions.insert(0, "search_evidence")
    if remaining["open"] > 0 and evidence:
        actions.insert(0, "open_evidence_source")
    return {
        "patient_case": state.get("patient_case"),
        "patient_facts": state.get("patient_facts") or [],
        "plan_elements": state.get("plan_elements") or [],
        "review_questions": state.get("review_agenda") or [],
        "evidence_index": evidence_index,
        "latest_tool_result": state.get("last_tool_summary"),
        "remaining_budgets": remaining,
        "available_actions": actions,
    }


def _is_auth_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        value in text
        for value in (
            "authentication",
            "unauthorized",
            "permission",
            "invalid api key",
            "invalid_api_key",
            "401",
            "403",
        )
    )


async def agent_decision_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    context = runtime.context
    logical_step = int(state.get("logical_step_count") or 0) + 1
    if logical_step > context.max_agent_steps:
        return {"tool_route": "prepare_evidence", "stage": "retrieval_budget_exhausted"}
    started_at = utc_isoformat()
    started = time.monotonic()
    model = load_chat_model(context.model).bind_tools(AGENT_TOOLS)
    response: Any | None = None
    last_error: BaseException | None = None
    technical_calls = 0
    for _attempt in range(context.technical_retry_limit + 1):
        try:
            response = await model.ainvoke(
                [
                    {"role": "system", "content": AGENT_DECISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": AGENT_DECISION_PROMPT.format(
                            system_prompt=context.system_prompt,
                            decision_context=json.dumps(
                                _decision_context(state, context),
                                ensure_ascii=False,
                            ),
                        ),
                    },
                ]
            )
            break
        except Exception as exc:  # noqa: BLE001 - heterogeneous model providers
            technical_calls += 1
            last_error = exc
            if _is_auth_error(exc):
                break
    if response is None:
        auth_error = bool(last_error and _is_auth_error(last_error))
        errors = (
            3
            if auth_error
            else int(state.get("agent_model_error_count") or 0) + 1
        )
        record = AgentDecisionRecordV3(
            step=logical_step,
            action="model_error",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            status="technical_error",
            error_type=type(last_error).__name__ if last_error else "UnknownError",
            error_message=str(last_error or "未知模型错误"),
        )
        return {
            "agent_steps": [
                *list(state.get("agent_steps") or []),
                record.model_dump(mode="json"),
            ],
            "technical_attempt_count": int(state.get("technical_attempt_count") or 0)
            + technical_calls,
            "agent_model_error_count": errors,
            "errors": [
                *list(state.get("errors") or []),
                _error("agent_decision", last_error or RuntimeError("未知模型错误")),
            ],
            "tool_route": "prepare_evidence" if errors >= 3 else "agent",
            "degraded": True,
            "last_tool_summary": {"technical_error": str(last_error or "")},
        }

    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    usage = _merge_usage(
        dict(state.get("usage") or {}),
        "agent_decision",
        _response_usage(response),
    )
    tool_calls = list(response.tool_calls or [])
    available = set(_decision_context(state, context)["available_actions"])
    if len(tool_calls) != 1 or tool_calls[0].get("name") not in available:
        summary = (
            f"期望一个可用工具调用，实际调用数={len(tool_calls)}，"
            f"可用工具={sorted(available)}"
        )
        record = AgentDecisionRecordV3(
            step=logical_step,
            action="protocol_error",
            started_at=started_at,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            status="protocol_error",
            summary=summary,
        )
        return {
            "agent_steps": [
                *list(state.get("agent_steps") or []),
                record.model_dump(mode="json"),
            ],
            "logical_step_count": logical_step,
            "tool_route": (
                "prepare_evidence"
                if logical_step >= context.max_agent_steps
                else "agent"
            ),
            "last_tool_summary": {"protocol_error": summary},
            "usage": usage,
        }

    call = tool_calls[0]
    record = AgentDecisionRecordV3(
        step=logical_step,
        action=str(call["name"]),
        tool_call_id=str(call["id"]),
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        status="success",
        args=dict(call.get("args") or {}),
    )
    return {
        "messages": [response],
        "agent_steps": [
            *list(state.get("agent_steps") or []),
            record.model_dump(mode="json"),
        ],
        "logical_step_count": logical_step,
        "pending_tool_call": {
            "id": str(call["id"]),
            "name": str(call["name"]),
            "args": dict(call.get("args") or {}),
        },
        "tool_route": "execute_tool",
        "stage": "execute_tool",
        "usage": usage,
    }


def route_after_agent(state: MedicationReviewStateV3) -> str:
    return str(state.get("tool_route") or "agent")


async def execute_tool_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    call = state.get("pending_tool_call")
    if not isinstance(call, dict):
        return {
            "tool_route": "agent",
            "last_tool_summary": {"protocol_error": "缺少待执行工具调用"},
        }
    try:
        result = await execute_agent_tool(
            name=str(call["name"]),
            args=dict(call.get("args") or {}),
            state=dict(state),
            context=runtime.context,
        )
        updates = dict(result.updates)
        updates.update(
            {
                "messages": [
                    ToolMessage(
                        content=result.content,
                        tool_call_id=str(call["id"]),
                        name=str(call["name"]),
                    )
                ],
                "pending_tool_call": None,
                "tool_route": result.route,
                "stage": (
                    "retrieval_prepared"
                    if result.route == "prepare_evidence"
                    else "agent"
                ),
            }
        )
        return updates
    except Exception as exc:  # noqa: BLE001 - malformed tool call is local and recoverable
        content = json.dumps(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            ensure_ascii=False,
        )
        return {
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=str(call["id"]),
                    name=str(call["name"]),
                )
            ],
            "errors": [
                *list(state.get("errors") or []),
                _error(f"tool:{call['name']}", exc),
            ],
            "pending_tool_call": None,
            "tool_route": "agent",
            "stage": "agent",
            "last_tool_summary": json.loads(content),
        }


def route_after_tool(state: MedicationReviewStateV3) -> str:
    return str(state.get("tool_route") or "agent")


async def prepare_evidence_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    evidence = [
        EvidenceItemV3.model_validate(item) for item in state.get("evidence") or []
    ]
    finish = state.get("finish_retrieval") or {}
    selection = select_evidence(
        evidence=evidence,
        priority_evidence_ids=list(finish.get("priority_evidence_ids") or []),
        max_evidence=runtime.context.max_claim_evidence,
        max_tokens=runtime.context.max_final_evidence_tokens,
    )
    usage = dict(state.get("usage") or {})
    usage.update(
        {
            "logical_agent_steps": int(state.get("logical_step_count") or 0),
            "technical_attempts": int(state.get("technical_attempt_count") or 0),
            "executed_subqueries": int(state.get("executed_query_count") or 0),
            "open_calls": len(state.get("open_records") or []),
            "unique_evidence_count": len(evidence),
            "selected_evidence_count": len(selection.selected_evidence_ids),
            "selected_evidence_estimated_tokens": selection.estimated_tokens,
        }
    )
    return {
        "stage": "evidence_prepared",
        "evidence_selection": selection.model_dump(mode="json"),
        "selected_evidence_ids": selection.selected_evidence_ids,
        "warnings": [
            *list(state.get("warnings") or []),
            *selection.warnings,
        ],
        "usage": usage,
    }


def route_after_prepare(
    state: MedicationReviewStateV3,
) -> str:
    if state.get("run_mode") == "stop_after_retrieval":
        return "finalize_debug"
    if state.get("synthesis_mode") == "claims":
        return "extract_claims"
    return "synthesize"


async def extract_claims_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    selected = set(state.get("selected_evidence_ids") or [])
    evidence = [
        EvidenceItemV3.model_validate(item)
        for item in state.get("evidence") or []
        if item.get("evidence_id") in selected
    ]
    elements = [
        TreatmentPlanElement.model_validate(item)
        for item in state.get("plan_elements") or []
    ]
    questions = [
        ReviewQuestion.model_validate(item)
        for item in state.get("review_agenda") or []
    ]
    result = await extract_claims(
        model=load_chat_model(runtime.context.model),
        evidence=evidence,
        elements=elements,
        questions=questions,
        technical_retry_limit=runtime.context.technical_retry_limit,
        retain_raw_output=runtime.context.diagnostic_trace,
    )
    usage = dict(state.get("usage") or {})
    if isinstance(result.audit.get("usage"), dict):
        usage = _merge_usage(usage, "claim_extraction", result.audit["usage"])
    return {
        "stage": "claims_extracted",
        "evidence_claims": [
            item.model_dump(mode="json") for item in result.claims
        ],
        "claim_extraction": result.audit,
        "warnings": [*list(state.get("warnings") or []), *result.warnings],
        "degraded": bool(state.get("degraded")) or result.degraded,
        "usage": usage,
    }


def route_after_claims(
    state: MedicationReviewStateV3,
) -> str:
    return (
        "finalize_debug"
        if state.get("run_mode") == "stop_after_claims"
        else "synthesize"
    )


async def synthesize_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    patient_case = PatientCase.model_validate(state["patient_case"])
    elements = [
        TreatmentPlanElement.model_validate(item)
        for item in state.get("plan_elements") or []
    ]
    questions = [
        ReviewQuestion.model_validate(item)
        for item in state.get("review_agenda") or []
    ]
    selected = set(state.get("selected_evidence_ids") or [])
    evidence = [
        EvidenceItemV3.model_validate(item)
        for item in state.get("evidence") or []
        if item.get("evidence_id") in selected
    ]
    claims = [
        EvidenceClaim.model_validate(item)
        for item in state.get("evidence_claims") or []
    ]
    finish = state.get("finish_retrieval") or {}
    unresolved_ids = list(finish.get("unresolved_question_ids") or [])
    question_by_id = {item.question_id: item for item in questions}
    unresolved = [
        question_by_id[value].question_text
        for value in unresolved_ids
        if value in question_by_id
    ]
    result = await synthesize_review(
        model=load_chat_model(runtime.context.model),
        raw_case_text=str(state["raw_question"]),
        patient_case=patient_case,
        patient_facts=list(state.get("patient_facts") or []),
        elements=elements,
        questions=questions,
        evidence=evidence,
        claims=claims,
        unresolved_questions=unresolved,
        synthesis_mode=runtime.context.synthesis_mode,
        system_prompt=runtime.context.system_prompt,
        technical_retry_limit=runtime.context.technical_retry_limit,
        retain_raw_output=runtime.context.diagnostic_trace,
    )
    usage = dict(state.get("usage") or {})
    if isinstance(result.audit.get("usage"), dict):
        usage = _merge_usage(usage, "review_synthesis", result.audit["usage"])
    return {
        "stage": "synthesis_failed" if result.fatal else "review_synthesized",
        "review_synthesis": result.audit,
        "local_validation_events": [
            item.model_dump(mode="json") for item in result.validation_events
        ],
        "final_review": (
            None if result.fatal else result.review.model_dump(mode="json")
        ),
        "synthesis_fatal": result.fatal,
        "run_status": "failed" if result.fatal else state.get("run_status", "failed"),
        "completion_reason": (
            "review_synthesis_provider_failed"
            if result.fatal
            else state.get("completion_reason", "not_completed")
        ),
        "errors": [
            *list(state.get("errors") or []),
            *(
                [
                    {
                        "type": "ReviewSynthesisProviderError",
                        "message": "最终综合模型调用失败",
                        "stage": "review_synthesis",
                    }
                ]
                if result.fatal
                else []
            ),
        ],
        "warnings": [*list(state.get("warnings") or []), *result.warnings],
        "degraded": bool(state.get("degraded")) or result.degraded,
        "usage": usage,
    }


def route_after_synthesis(state: MedicationReviewStateV3) -> str:
    return "finalize" if state.get("synthesis_fatal") else "render_answer"


async def render_answer_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    del runtime
    from .models import ReviewSynthesis

    review = ReviewSynthesis.model_validate(state["final_review"])
    elements = [
        TreatmentPlanElement.model_validate(item)
        for item in state.get("plan_elements") or []
    ]
    evidence = [
        EvidenceItemV3.model_validate(item)
        for item in state.get("evidence") or []
    ]
    claims = [
        EvidenceClaim.model_validate(item)
        for item in state.get("evidence_claims") or []
    ]
    rendered = render_review_v3(
        review=review,
        plan_elements=elements,
        evidence_items=evidence,
        claims=claims,
    )
    return {
        "stage": "answer_rendered",
        "rendered_answer": rendered,
        "run_status": "partial" if state.get("degraded") else "completed",
        "completion_reason": (
            "answer_rendered_with_local_degradation"
            if state.get("degraded")
            else "answer_rendered"
        ),
    }


async def finalize_debug_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    stage = str(state.get("stage") or "unknown")
    content = "\n".join(
        [
            "PEA-RAG v2 阶段诊断已按配置停止。",
            "",
            f"- run_mode：{runtime.context.run_mode}",
            f"- last_completed_stage：{stage}",
            f"- 方案要素：{len(state.get('plan_elements') or [])}",
            f"- 动态问题：{len(state.get('review_agenda') or [])}",
            f"- 已执行子查询：{state.get('executed_query_count', 0)}",
            f"- 唯一 Evidence：{len(state.get('evidence') or [])}",
            f"- 选定 Evidence：{len(state.get('selected_evidence_ids') or [])}",
            f"- Claims：{len(state.get('evidence_claims') or [])}",
            "",
            "本次为 debug_stopped 诊断记录，不构成治疗方案审查答案。",
        ]
    )
    return {
        "stage": stage,
        "rendered_answer": content,
        "run_status": "debug_stopped",
        "completion_reason": runtime.context.run_mode,
    }


async def finalize_node(
    state: MedicationReviewStateV3,
    runtime: Runtime[MedicationReviewContext],
) -> dict[str, Any]:
    answer = state.get("rendered_answer")
    if not answer:
        answer = "治疗方案审查未完成。请查看 medication_review_trace 中的错误详情。"
    trace = build_trace(state=dict(state), context=runtime.context)
    message = AIMessage(
        content=answer,
        additional_kwargs={"medication_review_trace": trace.model_dump(mode="json")},
    )
    return {
        "messages": [message],
        "usage": trace.usage,
        "agent_config_snapshot": trace.agent_config_snapshot,
    }


class MedicationReviewAgent(BaseAgent):
    name = "老年治疗方案合理性审查（PEA-RAG v2 实验）"
    description = (
        "解析完整治疗方案，由检索 Agent 自主执行 Milvus 向量检索，"
        "再通过来源 Claim、患者级综合和局部降级生成六段式答案。"
    )
    capabilities: list[str] = []
    context_schema = MedicationReviewContext
    metadata = {
        "examples": ["审查该患者完整治疗方案的合理性，并给出逐项判断和依据。"],
        "method_version": V3_DEFAULT_METHOD_VERSION,
    }

    async def get_graph(
        self,
        context: MedicationReviewContext | None = None,
        **kwargs,
    ):
        del context, kwargs
        if self.graph is not None:
            return self.graph
        workflow = StateGraph(
            MedicationReviewStateV3,
            context_schema=MedicationReviewContext,
        )
        workflow.add_node("initialize", initialize_node)
        workflow.add_node("extract_plan", extract_plan_node)
        workflow.add_node("build_agenda", build_agenda_node)
        workflow.add_node("agent", agent_decision_node)
        workflow.add_node("execute_tool", execute_tool_node)
        workflow.add_node("prepare_evidence", prepare_evidence_node)
        workflow.add_node("extract_claims", extract_claims_node)
        workflow.add_node("synthesize", synthesize_node)
        workflow.add_node("render_answer", render_answer_node)
        workflow.add_node("finalize_debug", finalize_debug_node)
        workflow.add_node("finalize", finalize_node)

        workflow.add_edge(START, "initialize")
        workflow.add_conditional_edges(
            "initialize",
            route_after_initialize,
            {"extract_plan": "extract_plan", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "extract_plan",
            route_after_extraction,
            {
                "build_agenda": "build_agenda",
                "agent": "agent",
                "finalize_debug": "finalize_debug",
                "finalize": "finalize",
            },
        )
        workflow.add_conditional_edges(
            "build_agenda",
            route_after_agenda,
            {"agent": "agent", "finalize_debug": "finalize_debug"},
        )
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "agent": "agent",
                "execute_tool": "execute_tool",
                "prepare_evidence": "prepare_evidence",
            },
        )
        workflow.add_conditional_edges(
            "execute_tool",
            route_after_tool,
            {"agent": "agent", "prepare_evidence": "prepare_evidence"},
        )
        workflow.add_conditional_edges(
            "prepare_evidence",
            route_after_prepare,
            {
                "extract_claims": "extract_claims",
                "synthesize": "synthesize",
                "finalize_debug": "finalize_debug",
            },
        )
        workflow.add_conditional_edges(
            "extract_claims",
            route_after_claims,
            {"synthesize": "synthesize", "finalize_debug": "finalize_debug"},
        )
        workflow.add_conditional_edges(
            "synthesize",
            route_after_synthesis,
            {"render_answer": "render_answer", "finalize": "finalize"},
        )
        workflow.add_edge("render_answer", "finalize")
        workflow.add_edge("finalize_debug", "finalize")
        workflow.add_edge("finalize", END)
        self.graph = workflow.compile(checkpointer=await self._get_checkpointer())
        return self.graph
