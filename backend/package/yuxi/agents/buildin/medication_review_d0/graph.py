from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from yuxi.agents import BaseAgent, load_chat_model
from yuxi.utils.datetime_utils import utc_isoformat

from .models import (
    METHOD_VERSION,
    MedicationReviewState,
    MedicationReviewTrace,
    PatientCase,
    QueryBundle,
    RetrievalRecord,
    TraceError,
)
from .planning import (
    CaseExtractionError,
    build_review_plan,
    extract_patient_case,
)
from .retrieval import (
    MedicationReviewConfigError,
    retrieve_query_bundles,
    validate_context,
)

from .context import MedicationReviewD0Context


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
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
    raise CaseExtractionError("没有找到本次运行的用户病例")


def _trace_error(stage: str, exc: BaseException) -> dict[str, Any]:
    return TraceError(type=type(exc).__name__, message=str(exc), stage=stage).model_dump()


def _agent_config_snapshot(context: MedicationReviewD0Context) -> dict[str, Any]:
    return {
        "model": context.model,
        "knowledges": list(context.knowledges or []),
        "per_query_top_k": context.per_query_top_k,
        "retrieval_concurrency": context.retrieval_concurrency,
        "retrieval_timeout_seconds": context.retrieval_timeout_seconds,
        "max_query_bundles": context.max_query_bundles,
        "system_prompt_hash": hashlib.sha256(context.system_prompt.encode("utf-8")).hexdigest(),
    }


async def parse_case_node(
    state: MedicationReviewState,
    runtime: Runtime[MedicationReviewD0Context],
) -> dict[str, Any]:
    context = runtime.context
    review_run_id = str(uuid.uuid4())
    reset: dict[str, Any] = {
        "review_run_id": review_run_id,
        "run_status": "parse_failed",
        "patient_case": None,
        "review_slots": [],
        "query_bundles": [],
        "retrieval_records": [],
        "evidence": [],
        "knowledge_base_snapshot": {},
        "agent_config_snapshot": _agent_config_snapshot(context),
        "usage": {},
        "warnings": [],
        "errors": [],
        "extraction_mode": "not_started",
    }
    try:
        raw_question = _latest_human_text(list(state.get("messages") or []))
        raw_hash = hashlib.sha256(" ".join(raw_question.split()).encode("utf-8")).hexdigest()
        reset["raw_question"] = raw_question
        reset["raw_question_hash"] = raw_hash
        try:
            direct_payload = json.loads(raw_question)
        except json.JSONDecodeError:
            direct_payload = None
        is_structured_case = isinstance(direct_payload, dict) and any(
            key in direct_payload for key in ("medications", "diagnoses", "renal_function")
        )
        model = None if is_structured_case else load_chat_model(context.model)
        patient_case, extraction_mode, warnings = await extract_patient_case(
            raw_question=raw_question,
            model=model,
            system_prompt=context.system_prompt,
        )
        reset.update(
            {
                "run_status": "parsed",
                "patient_case": patient_case.model_dump(mode="json"),
                "warnings": warnings,
                "extraction_mode": extraction_mode,
            }
        )
    except Exception as exc:  # noqa: BLE001 - finalize must persist parse failures
        reset["errors"] = [_trace_error("parse_case", exc)]
    return reset


def route_after_parse(state: MedicationReviewState) -> str:
    return "plan" if state.get("run_status") == "parsed" and state.get("patient_case") else "finalize"


async def build_review_plan_node(
    state: MedicationReviewState,
    runtime: Runtime[MedicationReviewD0Context],
) -> dict[str, Any]:
    context = runtime.context
    try:
        validate_context(context)
        patient_case = PatientCase.model_validate(state["patient_case"])
        slots, bundles = build_review_plan(patient_case)
        usage = dict(state.get("usage") or {})
        usage.update(
            {
                "review_slot_count": len(slots),
                "query_count": len(bundles),
                "query_budget": context.max_query_bundles,
            }
        )
        if len(bundles) > context.max_query_bundles:
            started_at = utc_isoformat()
            records = [
                RetrievalRecord(
                    bundle_id=bundle.bundle_id,
                    query_text=bundle.query_text,
                    slot_ids=bundle.slot_ids,
                    status="skipped_budget",
                    started_at=started_at,
                    elapsed_ms=0,
                    error_type="plan_budget_exceeded",
                    error_message=f"查询数 {len(bundles)} 超过预算 {context.max_query_bundles}",
                )
                for bundle in bundles
            ]
            return {
                "run_status": "plan_budget_exceeded",
                "review_slots": [item.model_dump(mode="json") for item in slots],
                "query_bundles": [item.model_dump(mode="json") for item in bundles],
                "retrieval_records": [item.model_dump(mode="json") for item in records],
                "usage": usage,
                "errors": [
                    TraceError(
                        type="plan_budget_exceeded",
                        message=f"查询数 {len(bundles)} 超过预算 {context.max_query_bundles}",
                        stage="build_review_plan",
                    ).model_dump()
                ],
            }
        return {
            "run_status": "planned",
            "review_slots": [item.model_dump(mode="json") for item in slots],
            "query_bundles": [item.model_dump(mode="json") for item in bundles],
            "usage": usage,
        }
    except MedicationReviewConfigError as exc:
        return {"run_status": "invalid_config", "errors": [_trace_error("build_review_plan", exc)]}
    except Exception as exc:  # noqa: BLE001 - finalize records plan failures
        return {"run_status": "plan_failed", "errors": [_trace_error("build_review_plan", exc)]}


def route_after_plan(state: MedicationReviewState) -> str:
    return "retrieve" if state.get("run_status") == "planned" else "finalize"


async def retrieve_bundles_node(
    state: MedicationReviewState,
    runtime: Runtime[MedicationReviewD0Context],
) -> dict[str, Any]:
    try:
        patient_case = PatientCase.model_validate(state["patient_case"])
        bundles = [QueryBundle.model_validate(item) for item in state.get("query_bundles") or []]
        updated_bundles, records, evidence, kb_snapshot, retrieval_usage = await retrieve_query_bundles(
            bundles=bundles,
            context=runtime.context,
            review_run_id=str(state["review_run_id"]),
            case_id=patient_case.case_id,
        )
        success_count = sum(record.status in {"success", "success_empty"} for record in records)
        failure_count = len(records) - success_count
        if success_count == 0:
            run_status = "retrieval_failed"
        elif failure_count:
            run_status = "partial"
        else:
            run_status = "completed"
        usage = {**dict(state.get("usage") or {}), **retrieval_usage}
        return {
            "run_status": run_status,
            "query_bundles": [item.model_dump(mode="json") for item in updated_bundles],
            "retrieval_records": [item.model_dump(mode="json") for item in records],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "knowledge_base_snapshot": kb_snapshot,
            "usage": usage,
        }
    except MedicationReviewConfigError as exc:
        return {"run_status": "invalid_config", "errors": [_trace_error("retrieve_bundles", exc)]}
    except Exception as exc:  # noqa: BLE001 - finalize must persist retrieval setup failures
        return {"run_status": "retrieval_failed", "errors": [_trace_error("retrieve_bundles", exc)]}


def _build_report(trace: MedicationReviewTrace) -> str:
    usage = trace.usage
    status_counts = usage.get("status_counts") if isinstance(usage.get("status_counts"), dict) else {}
    success = int(status_counts.get("success", 0))
    empty = int(status_counts.get("success_empty", 0))
    failed = max(int(usage.get("query_count", 0)) - success - empty, 0)
    case = trace.patient_case or {}
    kb_name = trace.knowledge_base_snapshot.get("name") or "未完成知识库选择"
    if trace.run_status == "completed":
        heading = "处方关系覆盖检索已完成（实验性，不构成临床用药结论）"
    elif trace.run_status == "partial":
        heading = "处方关系覆盖检索部分完成（实验性，不构成临床用药结论）"
    else:
        heading = "处方关系覆盖处理未完成（实验性，不构成临床用药结论）"
    lines = [
        heading,
        "",
        f"- 运行状态：{trace.run_status}",
        f"- 解析药物：{len(case.get('medications') or [])}",
        f"- 解析疾病：{len(case.get('diagnoses') or [])}",
        f"- 审查槽位：{len(trace.review_slots)}",
        f"- 查询束：{len(trace.query_bundles)}",
        f"- 成功/空结果/失败：{success}/{empty}/{failed}",
        f"- 唯一召回片段：{len(trace.evidence)}",
        f"- 知识库：{kb_name}（milvus/vector）",
        f"- 方法版本：{trace.method_version}",
        "",
        "详细槽位、查询和 Top-3 片段已保存在 medication_review_trace。",
    ]
    if trace.errors:
        lines.extend(["", "错误："])
        lines.extend(f"- [{item.get('stage')}] {item.get('message')}" for item in trace.errors)
    return "\n".join(lines)


async def finalize_node(
    state: MedicationReviewState,
    runtime: Runtime[MedicationReviewD0Context],
) -> dict[str, Any]:
    del runtime
    usage = dict(state.get("usage") or {})
    trace = MedicationReviewTrace(
        review_run_id=str(state.get("review_run_id") or uuid.uuid4()),
        run_status=state.get("run_status", "retrieval_failed"),
        case_id=(state.get("patient_case") or {}).get("case_id") if state.get("patient_case") else None,
        patient_case=state.get("patient_case"),
        review_slots=list(state.get("review_slots") or []),
        query_bundles=list(state.get("query_bundles") or []),
        retrieval_records=list(state.get("retrieval_records") or []),
        evidence=list(state.get("evidence") or []),
        knowledge_base_snapshot=dict(state.get("knowledge_base_snapshot") or {}),
        agent_config_snapshot=dict(state.get("agent_config_snapshot") or {}),
        usage=usage,
        warnings=list(state.get("warnings") or []),
        errors=list(state.get("errors") or []),
    )
    serialized = trace.model_dump_json()
    usage["trace_bytes"] = len(serialized.encode("utf-8"))
    trace = trace.model_copy(update={"usage": usage})
    message = AIMessage(
        content=_build_report(trace),
        additional_kwargs={"medication_review_trace": trace.model_dump(mode="json")},
    )
    return {"messages": [message], "usage": usage}


class MedicationReviewD0Agent(BaseAgent):
    name = "处方关系固定检索 D0（仅诊断）"
    description = "将老年患者病例拆分为处方审查关系并执行可审计的 Milvus 纯向量检索；不形成临床裁决。"
    capabilities: list[str] = []
    context_schema = MedicationReviewD0Context
    metadata = {
        "examples": [
            "分析该老年患者处方中的药物关系，并检索每个关系的候选证据。",
        ],
        "method_version": METHOD_VERSION,
    }

    async def get_graph(self, context: MedicationReviewD0Context | None = None, **kwargs):
        del context, kwargs
        if self.graph is not None:
            return self.graph
        workflow = StateGraph(MedicationReviewState, context_schema=MedicationReviewD0Context)
        workflow.add_node("parse_case", parse_case_node)
        workflow.add_node("build_review_plan", build_review_plan_node)
        workflow.add_node("retrieve_bundles", retrieve_bundles_node)
        workflow.add_node("finalize", finalize_node)
        workflow.add_edge(START, "parse_case")
        workflow.add_conditional_edges(
            "parse_case",
            route_after_parse,
            {"plan": "build_review_plan", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "build_review_plan",
            route_after_plan,
            {"retrieve": "retrieve_bundles", "finalize": "finalize"},
        )
        workflow.add_edge("retrieve_bundles", "finalize")
        workflow.add_edge("finalize", END)
        self.graph = workflow.compile(checkpointer=await self._get_checkpointer())
        return self.graph
