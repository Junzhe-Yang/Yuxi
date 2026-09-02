from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command

from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    disabled_anchor_audit,
    extract_plan_anchors,
    merge_usage,
    message_text,
)
from yuxi.agents.buildin.medication_review_lite.evidence import (
    build_evidence_section,
    build_plan_section,
    extract_all_element_ids,
    extract_evidence_ids,
    extract_item_element_ids,
    strip_program_owned_sections,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    AnchorExtractionAudit,
    EvidenceItem,
    OpenRecord,
    PlanAnchor,
    TraceError,
)

from .context import (
    MedicationReviewPrimContext,
    profile_uses_modifiers,
    profile_uses_plans,
    profile_uses_reflection,
    validate_context_values,
)
from .extraction import (
    disabled_modifier_audit,
    extract_patient_modifiers,
)
from .memory import (
    anchors_from_state,
    build_investigation_memory,
    investigations_from_state,
    modifiers_from_state,
    queries_from_state,
    uninvestigated_plan_ids,
)
from .models import (
    ExperimentProfile,
    MedicationReviewPrimState,
    MedicationReviewPrimTrace,
    ModifierExtractionAudit,
    PrimCoverageReport,
    QueryRecord,
    ReflectionReport,
    RunStatus,
)
from .prompt import (
    MODIFIER_PROMPT_VERSION,
    PROMPT_VERSION,
    build_agent_system_prompt,
)
from .tools import (
    ensure_runtime_resources,
    resolve_milvus_retriever,
    search_tool_for_profile,
)


@dataclass(frozen=True)
class PreFinalInterruption:
    """A recoverable middleware request to continue the Agent loop."""

    tool_name: str
    tool_call_id: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    state_update: dict[str, Any] = field(default_factory=dict)


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


def _as_evidence(value: EvidenceItem | dict[str, Any]) -> EvidenceItem:
    return value if isinstance(value, EvidenceItem) else EvidenceItem.model_validate(value)


def _evidence_from_state(state: dict[str, Any]) -> dict[str, EvidenceItem]:
    result: dict[str, EvidenceItem] = {}
    raw = state.get("evidence_store")
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        try:
            result[str(key).upper()] = _as_evidence(value)
        except Exception:  # noqa: BLE001 - corrupted state remains trace-local
            continue
    return result


def _open_records_from_state(state: dict[str, Any]) -> list[OpenRecord]:
    result: list[OpenRecord] = []
    for raw in state.get("open_records") or []:
        try:
            result.append(raw if isinstance(raw, OpenRecord) else OpenRecord.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    return sorted(result, key=lambda value: (value.started_at, value.record_id))


def _plan_audit_from_state(
    state: dict[str, Any],
) -> AnchorExtractionAudit:
    raw = state.get("plan_extraction")
    if isinstance(raw, AnchorExtractionAudit):
        return raw
    if isinstance(raw, dict):
        return AnchorExtractionAudit.model_validate(raw)
    return disabled_anchor_audit()


def _modifier_audit_from_state(
    state: dict[str, Any],
) -> ModifierExtractionAudit:
    raw = state.get("modifier_extraction")
    if isinstance(raw, ModifierExtractionAudit):
        return raw
    if isinstance(raw, dict):
        return ModifierExtractionAudit.model_validate(raw)
    return disabled_modifier_audit()


def _reflection_from_state(
    state: dict[str, Any],
    *,
    enabled: bool,
) -> ReflectionReport:
    raw = state.get("reflection_report")
    if isinstance(raw, ReflectionReport):
        return raw
    if isinstance(raw, dict):
        return ReflectionReport.model_validate(raw)
    return ReflectionReport(enabled=enabled)


def _strip_section_six(text: str) -> str:
    value = text.strip()
    section_six = re.search(r"⑥\s*【依据清单】", value)
    return value[: section_six.start()].strip() if section_six else value


def _coverage(
    *,
    answer_body: str,
    anchors: list[PlanAnchor],
    evidence_store: dict[str, EvidenceItem],
    before_item_ids: list[str] | None = None,
    missing_before: list[str] | None = None,
    reflection_attempted: bool = False,
) -> PrimCoverageReport:
    expected = [anchor.element_id for anchor in anchors]
    cited = extract_evidence_ids(answer_body)
    unknown_evidence = [value for value in cited if value not in evidence_store]
    if not expected:
        warnings = ["回答引用未知 Evidence：" + "、".join(unknown_evidence)] if unknown_evidence else []
        return PrimCoverageReport(
            cited_evidence_ids=cited,
            unknown_evidence_ids=unknown_evidence,
            reflection_attempted=reflection_attempted,
            warnings=warnings,
        )

    item_ids, degraded = extract_item_element_ids(answer_body)
    item_counts = Counter(item_ids)
    all_element_ids = extract_all_element_ids(answer_body)
    unknown_element_ids = list(dict.fromkeys(value for value in all_element_ids if value not in expected))
    duplicate_ids = [value for value, count in item_counts.items() if count > 1]
    missing = [value for value in expected if value not in item_counts]
    warnings: list[str] = []
    if degraded:
        warnings.append("无法稳定识别第②至第③部分边界，覆盖检查已退化为全文检查")
    if missing:
        warnings.append("逐项判断仍遗漏：" + "、".join(missing))
    if unknown_element_ids:
        warnings.append("回答引用未知方案节点：" + "、".join(unknown_element_ids))
    if duplicate_ids:
        warnings.append("逐项判断重复方案节点：" + "、".join(duplicate_ids))
    if unknown_evidence:
        warnings.append("回答引用未知 Evidence：" + "、".join(unknown_evidence))
    missing_before_values = missing_before if missing_before is not None else missing
    return PrimCoverageReport(
        expected_element_ids=expected,
        item_element_ids_before_reflection=before_item_ids or item_ids,
        item_element_ids_after_reflection=item_ids,
        missing_before_reflection=missing_before_values,
        missing_after_reflection=missing,
        unknown_element_ids=unknown_element_ids,
        duplicate_item_element_ids=duplicate_ids,
        cited_evidence_ids=cited,
        unknown_evidence_ids=unknown_evidence,
        section_parse_degraded=degraded,
        reflection_attempted=reflection_attempted,
        reflection_succeeded=(reflection_attempted and bool(missing_before_values) and not missing),
        warnings=warnings,
    )


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
    plan_audit: AnchorExtractionAudit,
    modifier_audit: ModifierExtractionAudit,
    first_draft_usage: dict[str, Any],
) -> dict[str, Any]:
    agent_usage: dict[str, Any] = {}
    for message in [*messages, final_message]:
        agent_usage = merge_usage(agent_usage, _message_usage(message))
    if first_draft_usage:
        agent_usage = merge_usage(agent_usage, first_draft_usage)
    total = merge_usage(plan_audit.usage, modifier_audit.usage)
    total = merge_usage(total, agent_usage)
    return {
        "available": bool(total),
        "plan_extraction": plan_audit.usage,
        "modifier_extraction": modifier_audit.usage,
        "agent": agent_usage,
        "total": total,
    }


def _trace_errors(
    queries: list[QueryRecord],
    open_records: list[OpenRecord],
) -> list[TraceError]:
    errors: list[TraceError] = []
    for record in queries:
        if record.status != "technical_failed":
            continue
        errors.append(
            TraceError(
                stage="search_review_kb",
                error_type=record.error_type or record.status,
                message=record.error_message or record.status,
                record_id=record.query_id,
            )
        )
    for record in open_records:
        if record.status not in {"technical_failed", "invalid_source"}:
            continue
        errors.append(
            TraceError(
                stage="open_review_evidence",
                error_type=record.error_type or record.status,
                message=record.error_message or record.status,
                record_id=record.record_id,
            )
        )
    return errors


def _run_status(
    *,
    answer_body: str,
    plan_audit: AnchorExtractionAudit,
    modifier_audit: ModifierExtractionAudit,
    coverage: PrimCoverageReport,
    errors: list[TraceError],
) -> RunStatus:
    if not answer_body.strip():
        return "failed"
    if (
        plan_audit.status in {"failed", "no_valid_anchor"}
        or modifier_audit.status == "failed"
        or plan_audit.dropped_drafts
        or modifier_audit.dropped_drafts
        or coverage.unknown_evidence_ids
        or errors
    ):
        return "partial"
    return "completed"


def _effective_profile(
    *,
    requested: ExperimentProfile,
    plan_audit: AnchorExtractionAudit,
    modifier_audit: ModifierExtractionAudit,
) -> ExperimentProfile:
    if requested == "b1":
        return "b1"
    # Investigation memory is the core of m3/full and does not require a
    # successful auxiliary extraction. Keep the raw case and the core Agent
    # loop alive; the extraction audit still makes the run explicitly partial.
    if requested in {"m3", "full"}:
        return requested
    if plan_audit.status not in {"success", "repaired"}:
        return "b1"
    if requested == "m1":
        return "m1"
    # An empty modifier list is a valid result: some cases simply have no
    # additional patient fact that changes the review. Keep InvestigationItem
    # memory and reflection enabled for those cases. Only a technical/schema
    # failure changes the effective ablation profile.
    if modifier_audit.status not in {
        "success",
        "repaired",
        "no_valid_modifier",
    }:
        return "m1"
    return requested


class ReviewHarnessMiddleware(AgentMiddleware[MedicationReviewPrimState, MedicationReviewPrimContext]):
    state_schema = MedicationReviewPrimState

    def __init__(self, *, model: Any):
        super().__init__()
        self.model = model

    async def augment_initial_state(
        self,
        *,
        state: dict[str, Any],
        update: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any]:
        del state, update, runtime
        return {}

    def augment_investigation_memory(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
        memory_text: str,
    ) -> str:
        del state, context
        return memory_text

    def project_model_messages(
        self,
        *,
        messages: list[Any],
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> list[Any]:
        del state, context
        return messages

    def finalize_trace(
        self,
        *,
        base_trace: MedicationReviewPrimTrace,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> Any:
        del state, context
        return base_trace

    async def prepare_candidate_state(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
        candidate_body: str,
    ) -> dict[str, Any]:
        del state, context, candidate_body
        return {}

    def build_reflection_supplement(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> str:
        del state, context
        return ""

    def reflection_state_update(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> dict[str, Any]:
        del state, context
        return {}

    def project_search_tool(
        self,
        effective_profile: ExperimentProfile,
        context: MedicationReviewPrimContext | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        del context, state
        return search_tool_for_profile(effective_profile)

    def tool_is_visible(
        self,
        *,
        tool_name: str,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> bool:
        del tool_name, state, context
        return True

    async def prepare_pre_final_interruption(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
        candidate_body: str,
    ) -> PreFinalInterruption | None:
        del state, context, candidate_body
        return None

    def investigation_tools_enabled(
        self,
        *,
        effective_profile: ExperimentProfile,
        state: dict[str, Any],
        context: MedicationReviewPrimContext,
    ) -> bool:
        del state, context
        return effective_profile in {"m3", "full"}

    async def abefore_agent(
        self,
        state: MedicationReviewPrimState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        context: MedicationReviewPrimContext = runtime.context
        validate_context_values(context)
        ensure_runtime_resources(context)
        selection = await resolve_milvus_retriever(context)

        raw_case_text = _latest_human_text(list(state.get("messages") or []))
        if not raw_case_text:
            raise ValueError("PRIM-RAG 没有收到可审查的用户病例文本")
        raw_hash = hashlib.sha256(raw_case_text.encode("utf-8")).hexdigest()
        previous_hash = str(state.get("raw_question_hash") or "")
        previous_profile = str(state.get("requested_profile") or "")
        if previous_hash and previous_hash != raw_hash:
            raise ValueError("PRIM-RAG 实验要求每个病例使用独立 thread；当前 thread 已包含另一病例")
        if previous_profile and previous_profile != context.experiment_profile:
            raise ValueError("同一 PRIM-RAG thread 不能切换 experiment_profile，请新建会话")
        if previous_hash == raw_hash and state.get("plan_extraction"):
            update = {"knowledge_base_snapshot": selection.snapshot}
            update.update(
                await self.augment_initial_state(
                    state=dict(state),
                    update=update,
                    runtime=runtime,
                )
            )
            return update

        if profile_uses_plans(context.experiment_profile):
            anchors, plan_audit = await extract_plan_anchors(
                model=self.model,
                raw_text=raw_case_text,
                technical_retry_limit=context.technical_retry_limit,
            )
        else:
            anchors, plan_audit = [], disabled_anchor_audit()

        if profile_uses_modifiers(context.experiment_profile) and plan_audit.status in {"success", "repaired"}:
            modifiers, modifier_audit = await extract_patient_modifiers(
                model=self.model,
                raw_text=raw_case_text,
                technical_retry_limit=context.technical_retry_limit,
            )
        else:
            modifiers, modifier_audit = [], disabled_modifier_audit()

        effective = _effective_profile(
            requested=context.experiment_profile,
            plan_audit=plan_audit,
            modifier_audit=modifier_audit,
        )
        warnings: list[str] = []
        if effective != context.experiment_profile:
            warnings.append(f"预处理不可用；实验组由 {context.experiment_profile} " f"技术降级为 {effective}")
        if plan_audit.status in {"failed", "no_valid_anchor"}:
            warnings.append(
                "方案要素抽取不可用；保留原病例并继续 Agent 调查，"
                "本次运行标记为 partial"
            )
        if modifier_audit.status == "failed":
            warnings.append(
                "患者事实抽取失败；保留原病例并继续 Agent 调查，"
                "本次运行标记为 partial"
            )
        if plan_audit.dropped_drafts:
            warnings.append(f"有 {len(plan_audit.dropped_drafts)} 个方案 draft 无法回指原文")
        if modifier_audit.dropped_drafts:
            warnings.append(f"有 {len(modifier_audit.dropped_drafts)} 个患者事实 draft 无法回指原文")
        update = {
            "review_run_id": str(uuid.uuid4()),
            "requested_profile": context.experiment_profile,
            "effective_profile": effective,
            "raw_case_text": raw_case_text,
            "raw_question_hash": raw_hash,
            "plan_anchors": anchors,
            "plan_extraction": plan_audit,
            "patient_modifiers": modifiers,
            "modifier_extraction": modifier_audit,
            "evidence_store": {},
            "query_records": [],
            "investigations": [],
            "deferred_knowledge_calls": [],
            "open_records": [],
            "knowledge_base_snapshot": selection.snapshot,
            "search_count": 0,
            "open_count": 0,
            "technical_attempts": 0,
            "reflection_attempted": False,
            "reflection_report": ReflectionReport(enabled=profile_uses_reflection(effective)),
            "warnings": warnings,
        }
        update.update(
            await self.augment_initial_state(
                state=dict(state),
                update=update,
                runtime=runtime,
            )
        )
        return update

    async def awrap_model_call(
        self,
        request: ModelRequest[MedicationReviewPrimContext],
        handler: Callable[
            [ModelRequest[MedicationReviewPrimContext]],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse | ExtendedModelResponse:
        context: MedicationReviewPrimContext = request.runtime.context
        state = request.state if isinstance(request.state, dict) else {}
        requested = context.experiment_profile
        effective = str(state.get("effective_profile") or requested)
        if effective not in {"b1", "m1", "m2", "m3", "full"}:
            effective = requested
        effective_profile: ExperimentProfile = effective
        anchors = anchors_from_state(state)
        modifiers = modifiers_from_state(state)
        queries = queries_from_state(state)
        investigations = investigations_from_state(state)
        reflection = _reflection_from_state(
            state,
            enabled=profile_uses_reflection(effective_profile),
        )
        current_uninvestigated_ids = uninvestigated_plan_ids(
            anchors,
            investigations,
        )
        missing_set = set(current_uninvestigated_ids)
        reflection_missing = [
            anchor for anchor in anchors if anchor.element_id in missing_set
        ]
        memory_text = build_investigation_memory(
            profile=effective_profile,
            anchors=anchors,
            modifiers=modifiers,
            queries=queries,
            investigations=investigations,
        )
        memory_text = self.augment_investigation_memory(
            state=state,
            context=context,
            memory_text=memory_text,
        )
        dynamic_prompt = build_agent_system_prompt(
            user_prompt=context.system_prompt,
            requested_profile=requested,
            effective_profile=effective_profile,
            anchors=anchors,
            modifiers=modifiers,
            memory_text=memory_text,
            search_remaining=(context.max_search_calls - int(state.get("search_count") or 0)),
            open_remaining=(context.max_open_calls - int(state.get("open_count") or 0)),
            reflection_draft=reflection.first_draft or "",
            reflection_missing=reflection_missing,
            reflection_open_investigation_ids=(
                [
                    value.investigation_id
                    for value in investigations
                    if value.status == "open"
                ]
            ),
            reflection_supplement=self.build_reflection_supplement(
                state=state,
                context=context,
            ),
        )
        visible_tools = []
        search_exhausted = int(state.get("search_count") or 0) >= (
            context.max_search_calls
        )
        open_exhausted = int(state.get("open_count") or 0) >= (
            context.max_open_calls
        )
        for value in request.tools:
            name = getattr(value, "name", "")
            if not self.tool_is_visible(
                tool_name=name,
                state=state,
                context=context,
            ):
                continue
            if name == "coverage_reflection":
                continue
            if name == "open_review_evidence" and open_exhausted:
                continue
            if name == "update_investigation":
                if self.investigation_tools_enabled(
                    effective_profile=effective_profile,
                    state=state,
                    context=context,
                ):
                    visible_tools.append(value)
                continue
            if name == "search_review_kb":
                if search_exhausted:
                    continue
                visible_tools.append(
                    self.project_search_tool(
                        effective_profile,
                        context,
                        state,
                    )
                )
            else:
                visible_tools.append(value)
        projected_messages = self.project_model_messages(
            messages=request.messages,
            state=state,
            context=context,
        )
        model_request = request.override(
            messages=projected_messages,
            system_message=SystemMessage(content=dynamic_prompt),
            tools=visible_tools,
        )
        response = await handler(model_request)
        final_pair = _last_ai_message(response)
        if final_pair is None:
            return response
        message_index, candidate_message = final_pair
        if candidate_message.tool_calls:
            return response

        raw_candidate = message_text(candidate_message)
        candidate_body = (
            strip_program_owned_sections(raw_candidate)
            if effective_profile != "b1" and anchors
            else _strip_section_six(raw_candidate)
        )
        interruption = await self.prepare_pre_final_interruption(
            state=state,
            context=context,
            candidate_body=candidate_body,
        )
        if interruption is not None:
            synthetic = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": interruption.tool_name,
                        "args": interruption.tool_args,
                        "id": interruption.tool_call_id,
                        "type": "tool_call",
                    }
                ],
            )
            result_messages = list(response.result)
            result_messages[message_index] = synthetic
            return ExtendedModelResponse(
                model_response=ModelResponse(
                    result=result_messages,
                    structured_response=response.structured_response,
                ),
                command=Command(update=interruption.state_update),
            )
        candidate_state_update = await self.prepare_candidate_state(
            state=state,
            context=context,
            candidate_body=candidate_body,
        )
        effective_state = {**state, **candidate_state_update}
        evidence_store = _evidence_from_state(effective_state)
        candidate_coverage = _coverage(
            answer_body=candidate_body,
            anchors=anchors,
            evidence_store=evidence_store,
            reflection_attempted=bool(effective_state.get("reflection_attempted")),
        )

        reflection_supplement = self.build_reflection_supplement(
            state=effective_state,
            context=context,
        )
        effective_investigations = investigations_from_state(effective_state)
        open_investigation_ids = [
            value.investigation_id
            for value in effective_investigations
            if value.status == "open"
        ]
        uninvestigated_ids = uninvestigated_plan_ids(
            anchors,
            effective_investigations,
        )
        investigation_triggered = bool(
            open_investigation_ids or uninvestigated_ids
        )
        supplement_triggered = bool(reflection_supplement.strip())

        if (
            profile_uses_reflection(effective_profile)
            and not effective_state.get("reflection_attempted")
            and (investigation_triggered or supplement_triggered)
        ):
            tool_call_id = _stable_reflection_call_id(
                str(state.get("review_run_id") or ""),
                candidate_body,
            )
            trigger_reason = (
                "investigation_gaps_and_companion_cues"
                if investigation_triggered and supplement_triggered
                else "investigation_gaps"
                if investigation_triggered
                else "remaining_companion_cues"
            )
            first_report = ReflectionReport(
                enabled=True,
                triggered=True,
                trigger_reason=trigger_reason,
                first_draft=candidate_body,
                first_draft_hash=hashlib.sha256(candidate_body.encode("utf-8")).hexdigest(),
                first_draft_usage=_message_usage(candidate_message),
                missing_before=candidate_coverage.missing_after_reflection,
                open_investigation_ids_before=open_investigation_ids,
                uninvestigated_plan_ids_before=uninvestigated_ids,
                search_count_before=int(effective_state.get("search_count") or 0),
                open_count_before=int(effective_state.get("open_count") or 0),
                evidence_ids_before=sorted(evidence_store),
            )
            synthetic = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "coverage_reflection",
                        "args": {},
                        "id": tool_call_id,
                        "type": "tool_call",
                    }
                ],
            )
            result_messages = list(response.result)
            result_messages[message_index] = synthetic
            return ExtendedModelResponse(
                model_response=ModelResponse(
                    result=result_messages,
                    structured_response=response.structured_response,
                ),
                command=Command(
                    update={
                        **candidate_state_update,
                        **self.reflection_state_update(
                            state=effective_state,
                            context=context,
                        ),
                        "reflection_attempted": True,
                        "reflection_report": first_report,
                        "draft_answer": candidate_body,
                    }
                ),
            )

        fallback_to_first = (
            bool(effective_state.get("reflection_attempted"))
            and not candidate_body.strip()
            and bool(reflection.first_draft)
        )
        final_body = reflection.first_draft or "" if fallback_to_first else candidate_body
        before_item_ids: list[str] | None = None
        missing_before: list[str] | None = None
        if reflection.triggered and reflection.first_draft:
            first_coverage = _coverage(
                answer_body=reflection.first_draft,
                anchors=anchors,
                evidence_store=evidence_store,
            )
            before_item_ids = first_coverage.item_element_ids_after_reflection
            missing_before = reflection.missing_before
        coverage = _coverage(
            answer_body=final_body,
            anchors=anchors,
            evidence_store=evidence_store,
            before_item_ids=before_item_ids,
            missing_before=missing_before,
            reflection_attempted=bool(effective_state.get("reflection_attempted")),
        )
        evidence_section = build_evidence_section(
            cited_evidence_ids=coverage.cited_evidence_ids,
            unknown_evidence_ids=coverage.unknown_evidence_ids,
            evidence_store=evidence_store,
        )
        if final_body:
            sections = [final_body, evidence_section]
            if effective_profile != "b1" and anchors:
                sections.insert(0, build_plan_section(anchors))
            final_answer = "\n\n".join(sections)
            completion_reason = (
                "reflection_final"
                if effective_state.get("reflection_attempted")
                else "model_final"
            )
        else:
            final_answer = (
                "治疗方案审查未完成：模型没有生成最终回答。" "请查看 medication_review_trace 中的错误和运行状态。"
            )
            completion_reason = "empty_model_response"

        current_evidence_ids = sorted(evidence_store)
        evidence_added = [value for value in current_evidence_ids if value not in reflection.evidence_ids_before]
        final_investigations = investigations_from_state(effective_state)
        final_open_investigation_ids = [
            value.investigation_id
            for value in final_investigations
            if value.status == "open"
        ]
        final_uninvestigated_plan_ids = uninvestigated_plan_ids(
            anchors,
            final_investigations,
        )
        final_reflection = reflection.model_copy(
            update={
                "second_draft": (
                    candidate_body
                    if effective_state.get("reflection_attempted")
                    else None
                ),
                "second_draft_hash": (
                    hashlib.sha256(candidate_body.encode("utf-8")).hexdigest()
                    if effective_state.get("reflection_attempted") and candidate_body
                    else None
                ),
                "search_count_after": int(effective_state.get("search_count") or 0),
                "open_count_after": int(effective_state.get("open_count") or 0),
                "evidence_ids_added": evidence_added,
                "missing_after": coverage.missing_after_reflection,
                "open_investigation_ids_after": (
                    final_open_investigation_ids
                ),
                "uninvestigated_plan_ids_after": (
                    final_uninvestigated_plan_ids
                ),
                "completed": bool(effective_state.get("reflection_attempted")),
                "fallback_to_first_draft": fallback_to_first,
                "warnings": (["反思轮未生成正文，已保留第一版答案"] if fallback_to_first else []),
            }
        )
        open_records = _open_records_from_state(effective_state)
        errors = _trace_errors(queries, open_records)
        plan_audit = _plan_audit_from_state(state)
        modifier_audit = _modifier_audit_from_state(state)
        status = _run_status(
            answer_body=final_body,
            plan_audit=plan_audit,
            modifier_audit=modifier_audit,
            coverage=coverage,
            errors=errors,
        )
        warnings = list(
            dict.fromkeys(
                [
                    *(effective_state.get("warnings") or []),
                    *coverage.warnings,
                    *final_reflection.warnings,
                ]
            )
        )
        prompt_hash = hashlib.sha256(dynamic_prompt.encode("utf-8")).hexdigest()
        usage = _aggregate_usage(
            messages=model_request.messages,
            final_message=candidate_message,
            plan_audit=plan_audit,
            modifier_audit=modifier_audit,
            first_draft_usage=reflection.first_draft_usage,
        )
        trace = MedicationReviewPrimTrace(
            method_version=(f"prim-rag-v2-{requested}-vector-top10"),
            requested_profile=requested,
            effective_profile=effective_profile,
            run_status=status,
            completion_reason=completion_reason,
            review_run_id=str(effective_state.get("review_run_id") or ""),
            raw_question_hash=str(
                effective_state.get("raw_question_hash") or ""
            ),
            prompt_versions={
                "agent": PROMPT_VERSION,
                "modifier_extraction": MODIFIER_PROMPT_VERSION,
                "plan_extraction": "pat-rag-anchor-v1",
            },
            prompt_hashes={"agent_final": prompt_hash},
            plan_anchors=anchors,
            plan_extraction=plan_audit,
            patient_modifiers=modifiers,
            modifier_extraction=modifier_audit,
            knowledge_base_snapshot=dict(
                effective_state.get("knowledge_base_snapshot") or {}
            ),
            query_records=queries,
            investigations=final_investigations,
            deferred_knowledge_calls=list(
                effective_state.get("deferred_knowledge_calls") or []
            ),
            open_records=open_records,
            evidence_store=sorted(
                evidence_store.values(),
                key=lambda value: value.evidence_id,
            ),
            coverage_report=coverage,
            reflection_report=final_reflection,
            cited_evidence_ids=coverage.cited_evidence_ids,
            unknown_evidence_ids=coverage.unknown_evidence_ids,
            budgets={
                "max_search_calls": context.max_search_calls,
                "max_open_calls": context.max_open_calls,
                "executed_search_calls": int(
                    effective_state.get("search_count") or 0
                ),
                "executed_open_calls": int(
                    effective_state.get("open_count") or 0
                ),
                "technical_attempts": int(
                    effective_state.get("technical_attempts") or 0
                ),
            },
            usage=usage,
            warnings=warnings,
            errors=errors,
            final_answer_hash=hashlib.sha256(final_answer.encode("utf-8")).hexdigest(),
        )
        try:
            trace = self.finalize_trace(
                base_trace=trace,
                state=effective_state,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001 - answer must survive trace extension failure
            trace_error = TraceError(
                stage="trace_finalization",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            errors = [*errors, trace_error]
            warnings = list(
                dict.fromkeys(
                    [
                        *warnings,
                        "扩展 Trace 构建失败；已保留基础 Trace 和最终答案",
                    ]
                )
            )
            status = "partial"
            trace = trace.model_copy(
                update={
                    "run_status": status,
                    "warnings": warnings,
                    "errors": errors,
                }
            )
        status = getattr(trace, "run_status", status)

        additional_kwargs = dict(candidate_message.additional_kwargs or {})
        try:
            additional_kwargs["medication_review_trace"] = trace.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 - answer must survive trace failure
            trace_version = str(getattr(trace, "schema_version", "unknown"))
            trace_error = TraceError(
                stage="trace_serialization",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            errors = [*errors, trace_error]
            warnings = list(
                dict.fromkeys(
                    [
                        *warnings,
                        f"Trace {trace_version} 序列化失败；最终答案已保留",
                    ]
                )
            )
            status = "partial"
            additional_kwargs["medication_review_trace_error"] = trace_error.model_dump(mode="json")
        final_message = candidate_message.model_copy(
            update={
                "content": final_answer,
                "additional_kwargs": additional_kwargs,
            }
        )
        result_messages = list(response.result)
        result_messages[message_index] = final_message
        return ExtendedModelResponse(
            model_response=ModelResponse(
                result=result_messages,
                structured_response=response.structured_response,
            ),
            command=Command(
                update={
                    **candidate_state_update,
                    "coverage_report": coverage,
                    "reflection_report": final_reflection,
                    "final_answer": final_answer,
                    "run_status": status,
                    "warnings": warnings,
                    "errors": errors,
                }
            ),
        )


def _stable_reflection_call_id(review_run_id: str, draft: str) -> str:
    source = f"{review_run_id}\0{draft}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"REFLECT-{digest}"
