from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from .anchor_extraction import (
    disabled_anchor_audit,
    extract_plan_anchors,
    merge_usage,
    message_text,
    response_usage,
)
from .context import (
    MedicationReviewLiteContext,
    profile_uses_anchors,
    profile_uses_coverage_patch,
    validate_context_values,
)
from .evidence import (
    build_evidence_section,
    build_plan_section,
    extract_all_element_ids,
    extract_evidence_ids,
    extract_item_element_ids,
    insert_coverage_patch,
    strip_program_owned_sections,
)
from .models import (
    AnchorExtractionAudit,
    CoverageReport,
    EvidenceItem,
    MedicationReviewLiteState,
    MedicationReviewLiteTrace,
    OpenRecord,
    PlanAnchor,
    RunStatus,
    SearchRecord,
    TraceError,
)
from .prompt import (
    PROMPT_VERSION,
    build_agent_system_prompt,
    build_coverage_patch_prompt,
)
from .tools import (
    ensure_runtime_resources,
    resolve_milvus_retriever,
)


def _as_anchor(value: PlanAnchor | dict[str, Any]) -> PlanAnchor:
    return value if isinstance(value, PlanAnchor) else PlanAnchor.model_validate(value)


def _as_search_record(value: SearchRecord | dict[str, Any]) -> SearchRecord:
    return (
        value
        if isinstance(value, SearchRecord)
        else SearchRecord.model_validate(value)
    )


def _as_open_record(value: OpenRecord | dict[str, Any]) -> OpenRecord:
    return (
        value
        if isinstance(value, OpenRecord)
        else OpenRecord.model_validate(value)
    )


def _as_evidence(value: EvidenceItem | dict[str, Any]) -> EvidenceItem:
    return (
        value
        if isinstance(value, EvidenceItem)
        else EvidenceItem.model_validate(value)
    )


def _anchors_from_state(state: dict[str, Any]) -> list[PlanAnchor]:
    result: list[PlanAnchor] = []
    for value in state.get("plan_anchors") or []:
        try:
            result.append(_as_anchor(value))
        except Exception:  # noqa: BLE001 - trace corrupted state without hiding run
            continue
    return result


def _search_records_from_state(state: dict[str, Any]) -> list[SearchRecord]:
    result: list[SearchRecord] = []
    for value in state.get("search_records") or []:
        try:
            result.append(_as_search_record(value))
        except Exception:  # noqa: BLE001
            continue
    return sorted(result, key=lambda value: (value.started_at, value.record_id))


def _open_records_from_state(state: dict[str, Any]) -> list[OpenRecord]:
    result: list[OpenRecord] = []
    for value in state.get("open_records") or []:
        try:
            result.append(_as_open_record(value))
        except Exception:  # noqa: BLE001
            continue
    return sorted(result, key=lambda value: (value.started_at, value.record_id))


def _evidence_from_state(state: dict[str, Any]) -> dict[str, EvidenceItem]:
    result: dict[str, EvidenceItem] = {}
    raw = state.get("evidence_store")
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        try:
            result[str(key).upper()] = _as_evidence(value)
        except Exception:  # noqa: BLE001
            continue
    return result


def _latest_human_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_text(message).strip()
        if isinstance(message, dict) and message.get("type") in {
            "human",
            "user",
        }:
            return message_text(message.get("content", "")).strip()
    return ""


def _last_ai_message(
    response: ModelResponse,
) -> tuple[int, AIMessage] | None:
    for index in range(len(response.result) - 1, -1, -1):
        message = response.result[index]
        if isinstance(message, AIMessage):
            return index, message
    return None


def _coverage(
    *,
    answer_body: str,
    anchors: list[PlanAnchor],
    evidence_store: dict[str, EvidenceItem],
    before_item_ids: list[str] | None = None,
    missing_before: list[str] | None = None,
    patch_attempted: bool = False,
    patch_succeeded: bool = False,
    patch_usage: dict[str, Any] | None = None,
) -> CoverageReport:
    expected = [anchor.element_id for anchor in anchors]
    item_ids, degraded = extract_item_element_ids(answer_body)
    item_counts = Counter(item_ids)
    all_element_ids = extract_all_element_ids(answer_body)
    unknown_element_ids = list(
        dict.fromkeys(
            value for value in all_element_ids if value not in expected
        )
    )
    duplicate_ids = [
        value for value, count in item_counts.items() if count > 1
    ]
    missing = [value for value in expected if value not in item_counts]
    cited = extract_evidence_ids(answer_body)
    unknown_evidence = [
        value for value in cited if value not in evidence_store
    ]
    warnings: list[str] = []
    if degraded:
        warnings.append("无法稳定识别第②至第③部分边界，覆盖检查已退化为全文检查")
    if missing:
        warnings.append("逐项判断仍遗漏：" + "、".join(missing))
    if unknown_element_ids:
        warnings.append("回答引用未知方案锚点：" + "、".join(unknown_element_ids))
    if duplicate_ids:
        warnings.append("逐项判断重复锚点：" + "、".join(duplicate_ids))
    if unknown_evidence:
        warnings.append("回答引用未知 Evidence：" + "、".join(unknown_evidence))
    return CoverageReport(
        expected_element_ids=expected,
        item_element_ids_before_patch=before_item_ids or item_ids,
        item_element_ids_after_patch=item_ids,
        missing_before_patch=missing_before if missing_before is not None else missing,
        missing_after_patch=missing,
        unknown_element_ids=unknown_element_ids,
        duplicate_item_element_ids=duplicate_ids,
        cited_evidence_ids=cited,
        unknown_evidence_ids=unknown_evidence,
        section_parse_degraded=degraded,
        patch_attempted=patch_attempted,
        patch_succeeded=patch_succeeded,
        patch_usage=patch_usage or {},
        warnings=warnings,
    )


def _patch_is_acceptable(
    patch_text: str,
    requested_ids: list[str],
) -> bool:
    item_ids, _degraded = extract_item_element_ids(patch_text)
    if set(item_ids) != set(requested_ids):
        return False
    if any(
        marker in patch_text
        for marker in (
            "③【正面判断汇总】",
            "④【负面判断汇总】",
            "⑤【综合建议】",
            "⑥【依据清单】",
        )
    ):
        return False
    return True


def _message_usage(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return dict(usage)
    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(value, dict):
            return dict(value)
    return {}


def _aggregate_usage(
    *,
    messages: list[Any],
    final_message: AIMessage,
    anchor_audit: AnchorExtractionAudit,
    patch_usage: dict[str, Any],
) -> dict[str, Any]:
    agent_usage: dict[str, Any] = {}
    for message in [*messages, final_message]:
        agent_usage = merge_usage(agent_usage, _message_usage(message))
    total = merge_usage(anchor_audit.usage, agent_usage)
    total = merge_usage(total, patch_usage)
    return {
        "available": bool(total),
        "anchor_extraction": anchor_audit.usage,
        "agent": agent_usage,
        "coverage_patch": patch_usage,
        "total": total,
    }


def _trace_errors(
    search_records: list[SearchRecord],
    open_records: list[OpenRecord],
) -> list[TraceError]:
    errors: list[TraceError] = []
    for record in [*search_records, *open_records]:
        if record.status not in {"technical_failed", "invalid_source"}:
            continue
        errors.append(
            TraceError(
                stage=(
                    "search_review_kb"
                    if isinstance(record, SearchRecord)
                    else "open_review_evidence"
                ),
                error_type=record.error_type or record.status,
                message=record.error_message or record.status,
                record_id=record.record_id,
            )
        )
    return errors


def _run_status(
    *,
    profile: str,
    answer_body: str,
    anchor_audit: AnchorExtractionAudit,
    coverage: CoverageReport,
    errors: list[TraceError],
) -> RunStatus:
    if not answer_body.strip():
        return "failed"
    if profile != "b1" and anchor_audit.status in {
        "failed",
        "no_valid_anchor",
    }:
        return "partial"
    if anchor_audit.dropped_drafts:
        return "partial"
    if (
        coverage.missing_after_patch
        or coverage.unknown_element_ids
        or coverage.duplicate_item_element_ids
        or coverage.unknown_evidence_ids
        or coverage.section_parse_degraded
        or errors
    ):
        return "partial"
    return "completed"


def _anchor_audit_from_state(state: dict[str, Any]) -> AnchorExtractionAudit:
    raw = state.get("anchor_extraction")
    if isinstance(raw, AnchorExtractionAudit):
        return raw
    if isinstance(raw, dict):
        return AnchorExtractionAudit.model_validate(raw)
    return disabled_anchor_audit()


class ReviewHarnessMiddleware(
    AgentMiddleware[MedicationReviewLiteState, MedicationReviewLiteContext]
):
    state_schema = MedicationReviewLiteState

    def __init__(self, *, model: Any):
        super().__init__()
        self.model = model

    async def abefore_agent(
        self,
        state: MedicationReviewLiteState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        context: MedicationReviewLiteContext = runtime.context
        validate_context_values(context)
        ensure_runtime_resources(context)
        selection = await resolve_milvus_retriever(context)

        raw_case_text = _latest_human_text(list(state.get("messages") or []))
        if not raw_case_text:
            raise ValueError("PAT-RAG 没有收到可审查的用户病例文本")
        raw_hash = hashlib.sha256(raw_case_text.encode("utf-8")).hexdigest()
        previous_hash = str(state.get("raw_question_hash") or "")
        previous_profile = str(state.get("experiment_profile") or "")
        if previous_hash and previous_hash != raw_hash:
            raise ValueError(
                "PAT-RAG 实验要求每个病例使用独立 thread；当前 thread 已包含另一病例"
            )
        if previous_profile and previous_profile != context.experiment_profile:
            raise ValueError(
                "同一 PAT-RAG thread 不能切换 experiment_profile，请新建会话"
            )
        if previous_hash == raw_hash and state.get("anchor_extraction"):
            return {"knowledge_base_snapshot": selection.snapshot}

        if profile_uses_anchors(context.experiment_profile):
            anchors, audit = await extract_plan_anchors(
                model=self.model,
                raw_text=raw_case_text,
                technical_retry_limit=context.technical_retry_limit,
            )
        else:
            anchors, audit = [], disabled_anchor_audit()

        warnings: list[str] = []
        if audit.status in {"failed", "no_valid_anchor"}:
            warnings.append(
                "方案锚点不可用；本轮已退化为原始病例驱动的自主 Agent"
            )
        if audit.dropped_drafts:
            warnings.append(
                f"有 {len(audit.dropped_drafts)} 个锚点 draft 无法回指原文"
            )
        return {
            "review_run_id": str(uuid.uuid4()),
            "experiment_profile": context.experiment_profile,
            "raw_case_text": raw_case_text,
            "raw_question_hash": raw_hash,
            "plan_anchors": anchors,
            "anchor_extraction": audit,
            "evidence_store": {},
            "search_records": [],
            "open_records": [],
            "knowledge_base_snapshot": selection.snapshot,
            "search_count": 0,
            "open_count": 0,
            "technical_attempts": 0,
            "warnings": warnings,
        }

    async def awrap_model_call(
        self,
        request: ModelRequest[MedicationReviewLiteContext],
        handler: Callable[
            [ModelRequest[MedicationReviewLiteContext]],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse | ExtendedModelResponse:
        context: MedicationReviewLiteContext = request.runtime.context
        state = request.state if isinstance(request.state, dict) else {}
        anchors = _anchors_from_state(state)
        dynamic_prompt = build_agent_system_prompt(
            user_prompt=context.system_prompt,
            profile=context.experiment_profile,
            anchors=anchors,
            search_remaining=(
                context.max_search_calls - int(state.get("search_count") or 0)
            ),
            open_remaining=(
                context.max_open_calls - int(state.get("open_count") or 0)
            ),
        )
        model_request = request.override(
            system_message=SystemMessage(content=dynamic_prompt)
        )
        response = await handler(model_request)
        final_pair = _last_ai_message(response)
        if final_pair is None:
            return response
        message_index, candidate_message = final_pair
        if candidate_message.tool_calls:
            return response

        original_body = strip_program_owned_sections(
            message_text(candidate_message)
        )
        evidence_store = _evidence_from_state(state)
        before_coverage = _coverage(
            answer_body=original_body,
            anchors=anchors,
            evidence_store=evidence_store,
        )
        final_body = original_body
        patch_attempted = False
        patch_succeeded = False
        patch_usage: dict[str, Any] = {}

        if (
            profile_uses_coverage_patch(context.experiment_profile)
            and anchors
            and before_coverage.missing_after_patch
        ):
            patch_attempted = True
            missing_set = set(before_coverage.missing_after_patch)
            missing_anchors = [
                anchor
                for anchor in anchors
                if anchor.element_id in missing_set
            ]
            patch_prompt = build_coverage_patch_prompt(
                missing_anchors=missing_anchors,
                existing_answer=original_body,
            )
            patch_request = model_request.override(
                messages=[
                    *model_request.messages,
                    candidate_message,
                    HumanMessage(content=patch_prompt),
                ],
                system_message=SystemMessage(
                    content=(
                        "你只执行一次遗漏方案要素的局部补写。"
                        "不得调用工具，不得重写已有答案，不得输出其它章节。"
                    )
                ),
                tools=[],
                tool_choice=None,
            )
            try:
                patch_response = await handler(patch_request)
                patch_pair = _last_ai_message(patch_response)
                if patch_pair is not None:
                    patch_message = patch_pair[1]
                    patch_usage = response_usage(patch_message)
                    patch_text = strip_program_owned_sections(
                        message_text(patch_message)
                    )
                    if (
                        not patch_message.tool_calls
                        and _patch_is_acceptable(
                            patch_text,
                            before_coverage.missing_after_patch,
                        )
                    ):
                        final_body = insert_coverage_patch(
                            original_body,
                            patch_text,
                        )
                        patch_succeeded = True
            except Exception:  # noqa: BLE001 - preserve original final answer
                patch_succeeded = False

        coverage = _coverage(
            answer_body=final_body,
            anchors=anchors,
            evidence_store=evidence_store,
            before_item_ids=before_coverage.item_element_ids_after_patch,
            missing_before=before_coverage.missing_after_patch,
            patch_attempted=patch_attempted,
            patch_succeeded=patch_succeeded,
            patch_usage=patch_usage,
        )
        plan_section = build_plan_section(anchors)
        evidence_section = build_evidence_section(
            cited_evidence_ids=coverage.cited_evidence_ids,
            unknown_evidence_ids=coverage.unknown_evidence_ids,
            evidence_store=evidence_store,
        )
        if final_body:
            final_answer = "\n\n".join(
                [plan_section, final_body, evidence_section]
            )
            completion_reason = "model_final"
        else:
            final_answer = (
                "治疗方案审查未完成：模型没有生成最终回答。"
                "请查看 medication_review_trace 中的错误和运行状态。"
            )
            completion_reason = "empty_model_response"

        search_records = _search_records_from_state(state)
        open_records = _open_records_from_state(state)
        errors = _trace_errors(search_records, open_records)
        anchor_audit = _anchor_audit_from_state(state)
        status = _run_status(
            profile=context.experiment_profile,
            answer_body=final_body,
            anchor_audit=anchor_audit,
            coverage=coverage,
            errors=errors,
        )
        warnings = list(
            dict.fromkeys(
                [
                    *(state.get("warnings") or []),
                    *coverage.warnings,
                    *(
                        ["覆盖补写失败，已保留原始候选答案"]
                        if patch_attempted and not patch_succeeded
                        else []
                    ),
                ]
            )
        )
        prompt_hash = hashlib.sha256(
            dynamic_prompt.encode("utf-8")
        ).hexdigest()
        usage = _aggregate_usage(
            messages=model_request.messages,
            final_message=candidate_message,
            anchor_audit=anchor_audit,
            patch_usage=patch_usage,
        )
        trace = MedicationReviewLiteTrace(
            method_version=(
                f"pat-rag-v1-{context.experiment_profile}-vector-top5"
            ),
            experiment_profile=context.experiment_profile,
            run_status=status,
            completion_reason=completion_reason,
            review_run_id=str(state.get("review_run_id") or ""),
            raw_question_hash=str(state.get("raw_question_hash") or ""),
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash,
            plan_anchors=anchors,
            anchor_extraction=anchor_audit,
            knowledge_base_snapshot=dict(
                state.get("knowledge_base_snapshot") or {}
            ),
            search_records=search_records,
            open_records=open_records,
            evidence_store=sorted(
                evidence_store.values(),
                key=lambda value: value.evidence_id,
            ),
            coverage_report=coverage,
            cited_evidence_ids=coverage.cited_evidence_ids,
            unknown_evidence_ids=coverage.unknown_evidence_ids,
            budgets={
                "max_search_calls": context.max_search_calls,
                "max_open_calls": context.max_open_calls,
                "executed_search_calls": int(
                    state.get("search_count") or 0
                ),
                "executed_open_calls": int(state.get("open_count") or 0),
                "technical_attempts": int(
                    state.get("technical_attempts") or 0
                ),
            },
            usage=usage,
            warnings=warnings,
            errors=errors,
            final_answer_hash=hashlib.sha256(
                final_answer.encode("utf-8")
            ).hexdigest(),
        )

        additional_kwargs = dict(candidate_message.additional_kwargs or {})
        additional_kwargs["medication_review_trace"] = trace.model_dump(
            mode="json"
        )
        final_message = candidate_message.model_copy(
            update={
                "content": final_answer,
                "additional_kwargs": additional_kwargs,
            }
        )
        result_messages = list(response.result)
        result_messages[message_index] = final_message
        final_response = ModelResponse(
            result=result_messages,
            structured_response=response.structured_response,
        )
        return ExtendedModelResponse(
            model_response=final_response,
            command=Command(
                update={
                    "coverage_report": coverage,
                    "final_answer": final_answer,
                    "run_status": status,
                    "warnings": warnings,
                    "errors": errors,
                }
            ),
        )
