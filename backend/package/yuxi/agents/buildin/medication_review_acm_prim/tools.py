from __future__ import annotations

import hashlib
import uuid
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import Field

from yuxi.agents.buildin.medication_review_prim.memory import (
    investigations_from_state,
    queries_from_state,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    InvestigationItem,
    InvestigationStatus,
    QueryRecord,
    RetrievalScope,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    _search_review_kb_impl,
    update_investigation,
)
from yuxi.utils.datetime_utils import utc_isoformat

from .adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
    SUCCESSFUL_SEARCH_STATUSES as ADAPTIVE_SUCCESSFUL_SEARCH_STATUSES,
)
from .adaptive import (
    adaptive_agenda_from_state,
    adaptive_attempted_obligation_probes,
    adaptive_meta_from_state,
    adaptive_obligation_support_map,
    adaptive_probe_records_from_state,
    adaptive_recoveries_from_state,
    adaptive_route_key,
    adaptive_successful_obligation_probes,
    build_adaptive_coverage_report,
    build_adaptive_state_fingerprint,
    adaptive_query_contract_violations,
    normalize_text,
)
from .context import (
    MedicationReviewAcmPrimContext,
    adaptive_coverage_enabled,
    v7_required_investigation_count,
)
from .experiment import (
    SUCCESSFUL_SEARCH_STATUSES,
    agenda_from_state,
    build_v7_contract_report,
    probe_records_from_state,
)
from .memory import (
    ADAPTIVE_ATLAS_DOCUMENT_OPEN_INSTRUCTIONS,
    ATLAS_DOCUMENT_OPEN_INSTRUCTIONS,
)
from .models import (
    AdaptiveAgendaItem,
    AdaptiveAgendaRevision,
    AdaptiveCoverageAuditEntry,
    AdaptiveGapAssessment,
    AdaptiveInvestigationAgenda,
    AdaptiveInvestigationMeta,
    AdaptiveObligationSupport,
    AdaptiveProbeRecord,
    AdaptiveRecoveryRequirement,
    AdaptiveRetrievalIntent,
    AdaptiveReviewOutcome,
    AcmAgendaItemDraft,
    AtlasDocumentOpenRecord,
    V7AgendaItem,
    V7InvestigationAgenda,
    V7ProbePass,
    V7ProbeRecord,
)


def _tool_message(
    *,
    runtime: ToolRuntime,
    tool_name: str,
    content: str,
    warning: bool = False,
) -> Command:
    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    update = {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        ]
    }
    if warning:
        update["warnings"] = [content]
    return Command(update=update)


def _known_case_ids(state: dict) -> tuple[set[str], set[str]]:
    plan_ids = {
        str(raw.get("element_id") if isinstance(raw, dict) else getattr(raw, "element_id", ""))
        for raw in state.get("plan_anchors") or []
    } - {""}
    modifier_ids = {
        str(raw.get("modifier_id") if isinstance(raw, dict) else getattr(raw, "modifier_id", ""))
        for raw in state.get("patient_modifiers") or []
    } - {""}
    return plan_ids, modifier_ids


def _state_evidence(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("evidence_store")
    return raw if isinstance(raw, dict) else {}


def _evidence_file_id(value: Any) -> str | None:
    raw = value.get("file_id") if isinstance(value, dict) else getattr(value, "file_id", None)
    return str(raw).strip() if raw else None


def _adaptive_agenda_command(
    *,
    items: list[AcmAgendaItemDraft],
    reason: str,
    runtime: ToolRuntime,
    initial: bool,
    tool_name: str,
) -> Command:
    state = runtime.state if isinstance(runtime.state, dict) else {}
    existing = adaptive_agenda_from_state(state)
    if initial and existing is not None:
        return _tool_message(
            runtime=runtime,
            tool_name=tool_name,
            content=(f"自适应议程 [{existing.agenda_id}] 已建立；如发现新缺口，请调用 extend_investigation_agenda。"),
            warning=True,
        )
    if not initial and existing is None:
        return _tool_message(
            runtime=runtime,
            tool_name=tool_name,
            content="尚未建立自适应议程，请先调用 set_investigation_agenda。",
            warning=True,
        )

    normalized_items = [
        value if isinstance(value, AcmAgendaItemDraft) else AcmAgendaItemDraft.model_validate(value) for value in items
    ]
    if not normalized_items:
        return _tool_message(
            runtime=runtime,
            tool_name=tool_name,
            content="至少需要提交一个有实质意义的调查。",
            warning=True,
        )

    plan_ids, modifier_ids = _known_case_ids(state)
    evidence = _state_evidence(state)
    existing_items = list(existing.items) if existing else []
    existing_questions = {normalize_text(value.question) for value in existing_items}
    existing_scopes = {normalize_text(value.distinct_scope) for value in existing_items}
    errors: list[str] = []
    questions = [normalize_text(value.question) for value in normalized_items]
    scopes = [normalize_text(value.distinct_scope) for value in normalized_items]
    if len(set(questions)) != len(questions) or any(value in existing_questions for value in questions):
        errors.append("调查问题与现有或本次其他调查完全重复")
    if len(set(scopes)) != len(scopes) or any(value in existing_scopes for value in scopes):
        errors.append("distinct_scope 与现有或本次其他调查完全重复")

    unknown_plan_ids = sorted(
        {value for item in normalized_items for value in item.focus_plan_ids if value not in plan_ids}
    )
    unknown_modifier_ids = sorted(
        {value for item in normalized_items for value in item.focus_modifier_ids if value not in modifier_ids}
    )
    unknown_evidence_ids = sorted(
        {
            value.strip().upper()
            for item in normalized_items
            for value in item.parent_evidence_ids
            if value.strip().upper() not in evidence
        }
    )
    if unknown_plan_ids:
        errors.append("未知 PlanAnchor：" + "、".join(unknown_plan_ids))
    if unknown_modifier_ids:
        errors.append("未知 PatientModifier：" + "、".join(unknown_modifier_ids))
    if unknown_evidence_ids:
        errors.append("未知 parent Evidence：" + "、".join(unknown_evidence_ids))
    if any(not value.why_it_matters.strip() for value in normalized_items):
        errors.append("每个调查都必须说明 why_it_matters")
    for index, item in enumerate(normalized_items, start=1):
        if item.investigation_kind is None:
            errors.append(f"第 {index} 个调查必须指定 investigation_kind")
        elif item.investigation_kind == "current_regimen_review" and len(set(item.focus_plan_ids)) != 1:
            errors.append(f"第 {index} 个 current_regimen_review 必须且只能关联一个 PlanAnchor")
        elif item.investigation_kind == "improvement_plan" and not item.focus_plan_ids:
            errors.append(f"第 {index} 个 improvement_plan 必须关联至少一个 PlanAnchor")
        obligations = [normalize_text(value) for value in item.evidence_obligations if normalize_text(value)]
        if not obligations:
            errors.append(f"第 {index} 个调查必须列出 evidence_obligations")
        elif len(set(obligations)) != len(obligations):
            errors.append(f"第 {index} 个调查的 evidence_obligations 存在重复")

    if initial:
        requested_plan_ids = {
            value
            for item in normalized_items
            if item.investigation_kind == "current_regimen_review"
            for value in item.focus_plan_ids
        }
        missing_plan_ids = sorted(plan_ids - requested_plan_ids)
        if missing_plan_ids:
            errors.append("尚未建立 current_regimen_review 的 PlanAnchor：" + "、".join(missing_plan_ids))

    if errors:
        return _tool_message(
            runtime=runtime,
            tool_name=tool_name,
            content="自适应议程未更新：" + "；".join(errors),
            warning=True,
        )

    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    now = utc_isoformat()
    revision = (existing.revision + 1) if existing else 1
    agenda_items: list[AdaptiveAgendaItem] = []
    investigations: list[InvestigationItem] = []
    for index, draft in enumerate(normalized_items, start=1):
        digest = hashlib.sha256(f"{tool_call_id}\0{index}\0{draft.question}".encode()).hexdigest()[:12].upper()
        investigation_id = f"INV-ACM-{digest}"
        parent_evidence_ids = list(
            dict.fromkeys(value.strip().upper() for value in draft.parent_evidence_ids if value.strip())
        )
        focus_plan_ids = list(dict.fromkeys(draft.focus_plan_ids))
        focus_modifier_ids = list(dict.fromkeys(draft.focus_modifier_ids))
        agenda_items.append(
            AdaptiveAgendaItem(
                investigation_id=investigation_id,
                question=draft.question.strip(),
                why_it_matters=draft.why_it_matters.strip(),
                distinct_scope=draft.distinct_scope.strip(),
                decision_tags=list(dict.fromkeys(draft.decision_tags)),
                investigation_kind=draft.investigation_kind,
                evidence_obligations=list(
                    dict.fromkeys(value.strip() for value in draft.evidence_obligations if value.strip())
                ),
                focus_plan_ids=focus_plan_ids,
                focus_modifier_ids=focus_modifier_ids,
                parent_evidence_ids=parent_evidence_ids,
                created_at=now,
                revision=revision,
            )
        )
        parent_file_ids = [
            file_id for evidence_id in parent_evidence_ids if (file_id := _evidence_file_id(evidence[evidence_id]))
        ]
        investigations.append(
            InvestigationItem(
                investigation_id=investigation_id,
                question=draft.question.strip(),
                focus_plan_ids=focus_plan_ids,
                focus_modifier_ids=focus_modifier_ids,
                candidate_evidence_ids=parent_evidence_ids,
                candidate_file_ids=list(dict.fromkeys(parent_file_ids)),
                status="open",
                created_at=now,
                updated_at=now,
            )
        )

    agenda_id = existing.agenda_id if existing else f"AGENDA-ACM-{tool_call_id}"
    revision_record = AdaptiveAgendaRevision(
        revision=revision,
        tool_call_id=tool_call_id,
        created_at=now,
        reason=reason.strip(),
        added_investigation_ids=[value.investigation_id for value in agenda_items],
    )
    agenda = AdaptiveInvestigationAgenda(
        agenda_id=agenda_id,
        created_at=existing.created_at if existing else now,
        updated_at=now,
        revision=revision,
        items=[*existing_items, *agenda_items],
        revisions=[*(existing.revisions if existing else []), revision_record],
    )
    content = "\n".join(
        [
            f"自适应调查议程 [{agenda.agenda_id}] 已更新到 revision {revision}。",
            *[
                f"- [{value.investigation_id}] {value.question}；意义：{value.why_it_matters}；"
                f"类型：{value.investigation_kind}；"
                f"证据义务：{' | '.join(value.evidence_obligations)}"
                for value in agenda_items
            ],
            "下一步：按调度顺序让每个证据义务各执行一次定向真实检索。",
        ]
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            ],
            "adaptive_agenda": agenda,
            "investigations": investigations,
        }
    )


@tool
async def set_investigation_agenda(
    items: Annotated[list[AcmAgendaItemDraft], Field(min_length=1)],
    runtime: ToolRuntime = None,
) -> Command:
    """建立调查议程；自适应协议接受任意个有实质意义的不同调查。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("set_investigation_agenda 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    if adaptive_coverage_enabled(context):
        return _adaptive_agenda_command(
            items=items,
            reason="initial coverage agenda",
            runtime=runtime,
            initial=True,
            tool_name="set_investigation_agenda",
        )
    required_count = v7_required_investigation_count(context.v7_experiment_arm)
    if not required_count:
        return _tool_message(
            runtime=runtime,
            tool_name="set_investigation_agenda",
            content="当前实验臂不使用固定调查议程，本次调用未执行。",
            warning=True,
        )
    state = runtime.state if isinstance(runtime.state, dict) else {}
    existing = agenda_from_state(state)
    if existing is not None:
        return _tool_message(
            runtime=runtime,
            tool_name="set_investigation_agenda",
            content=(f"议程 [{existing.agenda_id}] 已建立，不能在同一病例中重置。请按现有调查继续。"),
        )
    if len(items) != required_count:
        return _tool_message(
            runtime=runtime,
            tool_name="set_investigation_agenda",
            content=(f"议程需要恰好 {required_count} 个调查，实际收到 {len(items)} 个。请重新提交。"),
            warning=True,
        )

    plan_ids, modifier_ids = _known_case_ids(state)
    requested_plan_ids = {value for item in items for value in item.focus_plan_ids if value in plan_ids}
    missing_plan_ids = sorted(plan_ids - requested_plan_ids)
    normalized_questions = [" ".join(item.question.casefold().split()) for item in items]
    errors: list[str] = []
    if missing_plan_ids:
        errors.append("尚未覆盖 PlanAnchor：" + "、".join(missing_plan_ids))
    if len(set(normalized_questions)) != len(normalized_questions):
        errors.append("调查问题存在完全重复")
    if errors:
        return _tool_message(
            runtime=runtime,
            tool_name="set_investigation_agenda",
            content="议程未建立：" + "；".join(errors),
            warning=True,
        )

    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    now = utc_isoformat()
    agenda_items: list[V7AgendaItem] = []
    investigations: list[InvestigationItem] = []
    for index, draft in enumerate(items, start=1):
        digest = hashlib.sha256(f"{tool_call_id}\0{index}\0{draft.question}".encode()).hexdigest()[:12].upper()
        investigation_id = f"INV-V7-{digest}"
        agenda_items.append(
            V7AgendaItem(
                investigation_id=investigation_id,
                question=draft.question.strip(),
                focus_plan_ids=list(dict.fromkeys(value for value in draft.focus_plan_ids if value in plan_ids)),
                focus_modifier_ids=list(
                    dict.fromkeys(value for value in draft.focus_modifier_ids if value in modifier_ids)
                ),
                distinct_scope=draft.distinct_scope.strip(),
            )
        )
        investigations.append(
            InvestigationItem(
                investigation_id=investigation_id,
                question=draft.question.strip(),
                focus_plan_ids=list(dict.fromkeys(value for value in draft.focus_plan_ids if value in plan_ids)),
                focus_modifier_ids=list(
                    dict.fromkeys(value for value in draft.focus_modifier_ids if value in modifier_ids)
                ),
                status="open",
                created_at=now,
                updated_at=now,
            )
        )
    agenda = V7InvestigationAgenda(
        agenda_id=f"AGENDA-{tool_call_id}",
        experiment_arm=context.v7_experiment_arm,
        required_count=required_count,
        created_at=now,
        items=agenda_items,
    )
    content = "\n".join(
        [
            f"V7 调查议程 [{agenda.agenda_id}] 已建立。",
            *[f"- [{item.investigation_id}] {item.question}；区别：{item.distinct_scope}" for item in agenda.items],
            "下一步：先让所有调查各执行一次 initial_probe。",
        ]
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name="set_investigation_agenda",
                )
            ],
            "v7_agenda": agenda,
            "investigations": investigations,
        }
    )


@tool
async def extend_investigation_agenda(
    items: Annotated[list[AcmAgendaItemDraft], Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1, max_length=800)],
    runtime: ToolRuntime = None,
) -> Command:
    """把覆盖审计发现的新问题追加到自适应议程，不重写已有调查。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("extend_investigation_agenda 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    if not adaptive_coverage_enabled(context):
        return _tool_message(
            runtime=runtime,
            tool_name="extend_investigation_agenda",
            content="当前协议不支持动态扩展议程。",
            warning=True,
        )
    return _adaptive_agenda_command(
        items=items,
        reason=reason,
        runtime=runtime,
        initial=False,
        tool_name="extend_investigation_agenda",
    )


def _append_tool_notice(update: dict[str, Any], notice: str) -> None:
    messages = update.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    first = messages[0]
    if isinstance(first, ToolMessage):
        messages[0] = first.model_copy(update={"content": f"{first.content}\n\n{notice}"})


async def _run_adaptive_search(
    *,
    query_text: str,
    reason: str,
    investigation_id: str,
    uncovered_aspect: str,
    retrieval_intent: AdaptiveRetrievalIntent,
    retrieval_scope: RetrievalScope,
    file_id: str | None,
    runtime: ToolRuntime,
) -> Command:
    context: MedicationReviewAcmPrimContext = runtime.context
    state = runtime.state if isinstance(runtime.state, dict) else {}
    agenda = adaptive_agenda_from_state(state)
    if agenda is None:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=("尚未建立自适应调查议程。请先调用 set_investigation_agenda，本次搜索未执行。"),
            warning=True,
        )
    normalized_id = investigation_id.strip().upper()
    agenda_item = next(
        (value for value in agenda.items if value.investigation_id == normalized_id),
        None,
    )
    if agenda_item is None:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(f"搜索必须绑定现有议程调查，未知 investigation_id={normalized_id or '<空>'}。本次搜索未执行。"),
            warning=True,
        )

    normalized_obligation = normalize_text(uncovered_aspect)
    obligation_by_normalized = {
        normalize_text(value): value for value in agenda_item.evidence_obligations if normalize_text(value)
    }
    if normalized_obligation not in obligation_by_normalized:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                "uncovered_aspect 必须精确对应当前调查的一个 evidence obligation。"
                "可选项：" + " | ".join(agenda_item.evidence_obligations)
            ),
            warning=True,
        )
    current_meta = next(
        (value for value in adaptive_meta_from_state(state) if value.investigation_id == normalized_id),
        None,
    )
    if normalized_obligation in adaptive_obligation_support_map(current_meta):
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                f"证据义务 [{obligation_by_normalized[normalized_obligation]}] 已绑定直接 Evidence；"
                "请处理仍未支持的义务。"
            ),
            warning=True,
        )

    report = build_adaptive_coverage_report(state, context)
    if normalized_id not in report.eligible_investigation_ids:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                "当前必须优先处理："
                f"{'、'.join(report.eligible_investigation_ids) or '无'}。"
                f"调查 [{normalized_id}] 暂不可搜索。"
            ),
            warning=True,
        )
    unprobed_for_investigation = {
        normalize_text(value)
        for value in report.unprobed_evidence_obligations.get(normalized_id, [])
        if normalize_text(value)
    }
    if unprobed_for_investigation and normalized_obligation not in unprobed_for_investigation:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                f"调查 [{normalized_id}] 仍有尚未定向检索的证据义务："
                + " | ".join(report.unprobed_evidence_obligations[normalized_id])
                + "。请先处理这些义务。"
            ),
            warning=True,
        )

    query_violations = adaptive_query_contract_violations(query_text)
    if query_violations:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                "本次搜索未执行：query_text 不符合单轴短查询合同："
                + "；".join(query_violations)
                + "。当前 evidence obligation："
                + obligation_by_normalized[normalized_obligation]
                + "。请只保留一个主体、单一待查属性和至多一个患者限定，"
                "不要枚举预计答案。"
            ),
            warning=True,
        )

    normalized_file_id = (file_id or "").strip() or None
    if retrieval_intent == "adjacent_context":
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=("adjacent_context 不是向量搜索意图；请对已知 Evidence ID 调用 open_review_evidence。"),
            warning=True,
        )
    if retrieval_intent == "source_discovery":
        if retrieval_scope != "global" or normalized_file_id is not None:
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=("source_discovery 必须使用 retrieval_scope=global，且不能传 file_id。"),
                warning=True,
            )
    elif retrieval_intent == "within_document_localization" and (
        retrieval_scope != "document" or normalized_file_id is None
    ):
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=("within_document_localization 必须使用 retrieval_scope=document 并提供 file_id。"),
            warning=True,
        )

    pending_recoveries = [
        value
        for value in adaptive_recoveries_from_state(state)
        if value.investigation_id == normalized_id and value.status == "pending"
    ]
    if pending_recoveries and retrieval_intent != "source_discovery":
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(f"调查 [{normalized_id}] 存在文档内零新增证据，下一次成功检索必须是 source_discovery/global。"),
            warning=True,
        )

    successful_queries = [
        value for value in queries_from_state(state) if value.status in ADAPTIVE_SUCCESSFUL_SEARCH_STATUSES
    ]
    successful_for_investigation = [value for value in successful_queries if value.investigation_id == normalized_id]
    evidence = _state_evidence(state)
    parent_file_ids = {
        value
        for evidence_id in agenda_item.parent_evidence_ids
        if (value := _evidence_file_id(evidence.get(evidence_id)))
    }
    if (
        not successful_for_investigation
        and retrieval_intent != "source_discovery"
        and normalized_file_id not in parent_file_ids
    ):
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                "该调查尚无成功检索，也没有与此 file_id 对应的 parent "
                "Evidence；首次检索必须使用 source_discovery/global。"
            ),
            warning=True,
        )

    investigation = next(
        (value for value in investigations_from_state(state) if value.investigation_id == normalized_id),
        None,
    )
    opened_doc_ids = {
        str(value.get("doc_id") if isinstance(value, dict) else getattr(value, "doc_id", ""))
        for value in state.get("atlas_document_open_records") or []
    } - {""}
    allowed_file_ids = {
        *(investigation.candidate_file_ids if investigation else []),
        *parent_file_ids,
        *opened_doc_ids,
    }
    if retrieval_intent == "within_document_localization" and normalized_file_id not in allowed_file_ids:
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                f"file_id={normalized_file_id!r} 尚未由该调查的候选证据、parent Evidence 或已打开 Atlas 文档建立来源。"
            ),
            warning=True,
        )

    normalized_query = normalize_text(query_text)
    if any(
        value.investigation_id == normalized_id and normalize_text(value.query_text) == normalized_query
        for value in successful_queries
    ):
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content="该查询已成功执行过；请围绕尚未覆盖的方面改写。",
            warning=True,
        )
    route_key = adaptive_route_key(
        investigation_id=normalized_id,
        uncovered_aspect=uncovered_aspect,
        retrieval_scope=retrieval_scope,
        file_id=normalized_file_id,
    )
    duplicate_route = any(
        value.route_key == route_key and value.status == "success" for value in adaptive_probe_records_from_state(state)
    )
    recovery_global = bool(pending_recoveries and retrieval_intent == "source_discovery")
    if duplicate_route and not recovery_global:
        candidate_file_ids = investigation.candidate_file_ids if investigation else []
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content=(
                "同一调查、证据义务与检索范围的路线已成功执行。"
                + (
                    "已有候选 file_id=" + "、".join(candidate_file_ids) + "；若仍缺直接证据，请对其中最可能的来源执行 "
                    "within_document_localization/document。"
                    if retrieval_scope == "global" and candidate_file_ids
                    else "请改变检索路线或明确保留语料不足边界。"
                )
            ),
            warning=True,
        )

    result = await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=normalized_file_id,
        question=None,
        focus_plan_ids=agenda_item.focus_plan_ids,
        focus_modifier_ids=agenda_item.focus_modifier_ids,
        investigation_id=normalized_id,
        runtime=runtime,
    )
    update = result.update if isinstance(result.update, dict) else {}
    raw_records = update.get("query_records")
    if not isinstance(raw_records, list) or not raw_records:
        return result
    query_record = (
        raw_records[0] if isinstance(raw_records[0], QueryRecord) else QueryRecord.model_validate(raw_records[0])
    )
    previous_candidate_evidence_ids = set(
        investigation.candidate_evidence_ids if investigation else agenda_item.parent_evidence_ids
    )
    novel_investigation_evidence_ids = [
        value for value in query_record.evidence_ids if value not in previous_candidate_evidence_ids
    ]
    document_redundant = (
        retrieval_scope == "document"
        and query_record.status in ADAPTIVE_SUCCESSFUL_SEARCH_STATUSES
        and not novel_investigation_evidence_ids
    )
    probe_record = AdaptiveProbeRecord(
        probe_record_id=f"APROBE-{query_record.query_id}",
        query_id=query_record.query_id,
        investigation_id=normalized_id,
        uncovered_aspect=uncovered_aspect.strip(),
        retrieval_intent=retrieval_intent,
        route_key=route_key,
        status=query_record.status,
        redundant=document_redundant,
    )
    recovery_updates: list[AdaptiveRecoveryRequirement] = []
    if query_record.status in ADAPTIVE_SUCCESSFUL_SEARCH_STATUSES:
        if retrieval_intent == "source_discovery" and pending_recoveries:
            recovery_updates.extend(
                value.model_copy(
                    update={
                        "status": "resolved",
                        "resolved_by_query_id": query_record.query_id,
                    }
                )
                for value in pending_recoveries
            )
            _append_tool_notice(
                update,
                "已完成强制全库恢复；此前文档内零新增证据不再阻塞该调查。",
            )
        elif document_redundant:
            recovery = AdaptiveRecoveryRequirement(
                recovery_id=f"RECOVERY-{query_record.query_id}",
                investigation_id=normalized_id,
                source_query_id=query_record.query_id,
                file_id=normalized_file_id,
                uncovered_aspect=uncovered_aspect.strip(),
                status="pending",
                created_at=utc_isoformat(),
            )
            recovery_updates.append(recovery)
            _append_tool_notice(
                update,
                "本次文档内检索没有为该调查新增 Evidence；下一次必须执行 source_discovery/global 恢复检索。",
            )
    return Command(
        update={
            **update,
            "adaptive_probe_records": [probe_record],
            "adaptive_recovery_requirements": recovery_updates,
        }
    )


@tool("search_review_kb")
async def search_review_kb_adaptive(
    query_text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=160,
            description=(
                "仅表达当前一个 evidence obligation 的单轴中文关键词；使用空格分隔，"
                "最多 6 个概念块：一个主体、单一待查属性、至多一个患者限定；"
                "不要枚举候选答案或混入其它审计维度。"
            ),
        ),
    ],
    reason: Annotated[str, Field(min_length=1)],
    investigation_id: Annotated[str, Field(min_length=1)],
    uncovered_aspect: Annotated[str, Field(min_length=1, max_length=500)],
    retrieval_intent: AdaptiveRetrievalIntent,
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """按自适应覆盖合同执行来源发现或文档内定位检索。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    if not adaptive_coverage_enabled(context):
        return _tool_message(
            runtime=runtime,
            tool_name="search_review_kb",
            content="当前运行未启用 adaptive_coverage 协议。",
            warning=True,
        )
    return await _run_adaptive_search(
        query_text=query_text,
        reason=reason,
        investigation_id=investigation_id,
        uncovered_aspect=uncovered_aspect,
        retrieval_intent=retrieval_intent,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        runtime=runtime,
    )


async def _run_acm_search(
    *,
    query_text: str,
    reason: str,
    retrieval_scope: RetrievalScope,
    file_id: str | None,
    question: str | None,
    focus_plan_ids: list[str] | None,
    focus_modifier_ids: list[str] | None,
    investigation_id: str | None,
    probe_pass: V7ProbePass | None,
    uncovered_aspect: str,
    runtime: ToolRuntime,
) -> Command:
    context: MedicationReviewAcmPrimContext = runtime.context
    state = runtime.state if isinstance(runtime.state, dict) else {}
    required_count = v7_required_investigation_count(context.v7_experiment_arm)
    agenda_item = None
    if required_count:
        agenda = agenda_from_state(state)
        if agenda is None:
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=("尚未建立 V7 调查议程。请先调用 set_investigation_agenda，本次搜索未执行。"),
            )
        normalized_id = (investigation_id or "").strip().upper()
        agenda_item = next(
            (value for value in agenda.items if value.investigation_id == normalized_id),
            None,
        )
        if agenda_item is None:
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=(
                    f"搜索必须绑定现有议程调查，未知 investigation_id={normalized_id or '<空>'}。本次搜索未执行。"
                ),
            )
        report = build_v7_contract_report(state, context)
        if report.missing_initial_probe_ids:
            expected_pass: V7ProbePass = "initial_probe"
            eligible_ids = report.missing_initial_probe_ids
        elif report.missing_complementary_probe_ids:
            expected_pass = "complementary_probe"
            eligible_ids = report.missing_complementary_probe_ids
        else:
            expected_pass = "adaptive_probe"
            eligible_ids = report.agenda_investigation_ids
        if probe_pass != expected_pass or normalized_id not in eligible_ids:
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=(f"当前应执行 {expected_pass}，可选调查：{'、'.join(eligible_ids)}。本次搜索未执行。"),
            )
        if probe_pass == "complementary_probe" and not uncovered_aspect.strip():
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=("complementary_probe 必须说明首轮仍未覆盖的证据方面，本次搜索未执行。"),
            )
        if probe_pass == "complementary_probe":
            successful_initial_query_ids = {
                value.query_id
                for value in probe_records_from_state(state)
                if value.investigation_id == normalized_id
                and value.probe_pass == "initial_probe"
                and value.status in SUCCESSFUL_SEARCH_STATUSES
            }
            normalized_query = " ".join(query_text.casefold().split())
            initial_queries = {
                " ".join(value.query_text.casefold().split())
                for value in queries_from_state(state)
                if value.query_id in successful_initial_query_ids
            }
            if normalized_query in initial_queries:
                return _tool_message(
                    runtime=runtime,
                    tool_name="search_review_kb",
                    content=(
                        "complementary_probe 不能逐字重复该调查的 initial_probe "
                        "查询。请围绕 uncovered_aspect 改变证据问题；本次搜索未执行。"
                    ),
                )
        investigation_id = normalized_id
        question = None
        focus_plan_ids = agenda_item.focus_plan_ids
        focus_modifier_ids = agenda_item.focus_modifier_ids

    result = await _search_review_kb_impl(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        question=question,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        investigation_id=investigation_id,
        runtime=runtime,
    )
    if not required_count or agenda_item is None or probe_pass is None:
        return result
    update = result.update if isinstance(result.update, dict) else {}
    records = update.get("query_records")
    if not isinstance(records, list) or not records:
        return result
    query_record = records[0]
    probe_record = V7ProbeRecord(
        probe_record_id=f"PROBE-{query_record.query_id}",
        query_id=query_record.query_id,
        investigation_id=agenda_item.investigation_id,
        probe_pass=probe_pass,
        uncovered_aspect=uncovered_aspect.strip(),
        status=query_record.status,
    )
    return Command(update={**update, "v7_probe_records": [probe_record]})


@tool("search_review_kb")
async def search_review_kb_acm_dispatch(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    question: Annotated[str, Field(min_length=1, max_length=500)] | None = None,
    focus_plan_ids: list[str] | None = None,
    focus_modifier_ids: list[str] | None = None,
    investigation_id: str | None = None,
    probe_pass: V7ProbePass | None = None,
    uncovered_aspect: Annotated[str, Field(max_length=500)] = "",
    retrieval_intent: AdaptiveRetrievalIntent | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """ACM 检索执行器；运行时按实验臂采用当前或 V7 调查协议。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    if adaptive_coverage_enabled(context):
        if not investigation_id or not uncovered_aspect.strip() or not retrieval_intent:
            return _tool_message(
                runtime=runtime,
                tool_name="search_review_kb",
                content=("adaptive_coverage 搜索必须提供 investigation_id、uncovered_aspect 和 retrieval_intent。"),
                warning=True,
            )
        return await _run_adaptive_search(
            query_text=query_text,
            reason=reason,
            investigation_id=investigation_id,
            uncovered_aspect=uncovered_aspect,
            retrieval_intent=retrieval_intent,
            retrieval_scope=retrieval_scope,
            file_id=file_id,
            runtime=runtime,
        )
    return await _run_acm_search(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        question=question,
        focus_plan_ids=focus_plan_ids,
        focus_modifier_ids=focus_modifier_ids,
        investigation_id=investigation_id,
        probe_pass=probe_pass,
        uncovered_aspect=uncovered_aspect,
        runtime=runtime,
    )


@tool("search_review_kb")
async def search_review_kb_v7(
    query_text: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    investigation_id: Annotated[str, Field(min_length=1)],
    probe_pass: V7ProbePass,
    uncovered_aspect: Annotated[str, Field(max_length=500)] = "",
    retrieval_scope: RetrievalScope = "global",
    file_id: str | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """按 V7 议程执行一项初始、互补或自适应 Milvus 向量搜索。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("search_review_kb 缺少 ToolRuntime")
    return await _run_acm_search(
        query_text=query_text,
        reason=reason,
        retrieval_scope=retrieval_scope,
        file_id=file_id,
        question=None,
        focus_plan_ids=None,
        focus_modifier_ids=None,
        investigation_id=investigation_id,
        probe_pass=probe_pass,
        uncovered_aspect=uncovered_aspect,
        runtime=runtime,
    )


async def _update_adaptive_investigation(
    *,
    investigation_id: str,
    status: InvestigationStatus,
    selected_evidence_ids: list[str] | None,
    working_note: str,
    resolved_aspects: list[str] | None,
    remaining_aspects: list[str] | None,
    obligation_supports: list[AdaptiveObligationSupport] | None,
    review_outcome: AdaptiveReviewOutcome | None,
    residual_uncertainty: str | None,
    closure_reason: str | None,
    runtime: ToolRuntime,
) -> Command:
    context: MedicationReviewAcmPrimContext = runtime.context
    state = runtime.state if isinstance(runtime.state, dict) else {}
    normalized_id = investigation_id.strip().upper()
    agenda = adaptive_agenda_from_state(state)
    agenda_item = next(
        (value for value in (agenda.items if agenda else []) if value.investigation_id == normalized_id),
        None,
    )
    if agenda_item is None:
        return _tool_message(
            runtime=runtime,
            tool_name="update_investigation",
            content=f"未知自适应议程调查：{normalized_id}。",
            warning=True,
        )
    investigation = next(
        (value for value in investigations_from_state(state) if value.investigation_id == normalized_id),
        None,
    )
    current_meta = next(
        (value for value in adaptive_meta_from_state(state) if value.investigation_id == normalized_id),
        None,
    )
    obligation_by_normalized = {
        normalize_text(value): value for value in agenda_item.evidence_obligations if normalize_text(value)
    }
    raw_supports = (
        obligation_supports
        if obligation_supports is not None
        else current_meta.obligation_supports
        if current_meta
        else []
    )
    normalized_supports: list[AdaptiveObligationSupport] = []
    support_by_normalized: dict[str, AdaptiveObligationSupport] = {}
    errors: list[str] = []
    for raw_support in raw_supports:
        support = (
            raw_support
            if isinstance(raw_support, AdaptiveObligationSupport)
            else AdaptiveObligationSupport.model_validate(raw_support)
        )
        normalized_obligation = normalize_text(support.obligation)
        if normalized_obligation not in obligation_by_normalized:
            errors.append(f"obligation_supports 包含未知义务：{support.obligation}")
            continue
        if normalized_obligation in support_by_normalized:
            errors.append(f"obligation_supports 重复：{support.obligation}")
            continue
        normalized_support = support.model_copy(
            update={
                "obligation": obligation_by_normalized[normalized_obligation],
                "evidence_ids": list(
                    dict.fromkeys(value.strip().upper() for value in support.evidence_ids if value.strip())
                ),
            }
        )
        support_by_normalized[normalized_obligation] = normalized_support
        normalized_supports.append(normalized_support)
    supported_obligations = [value for key, value in obligation_by_normalized.items() if key in support_by_normalized]
    missing_support_obligations = [
        value for key, value in obligation_by_normalized.items() if key not in support_by_normalized
    ]
    resolved = supported_obligations
    remaining = missing_support_obligations
    if resolved_aspects is not None and {
        normalize_text(value) for value in resolved_aspects if normalize_text(value)
    } != {normalize_text(value) for value in resolved}:
        errors.append("resolved_aspects 必须与 obligation_supports 已支持的义务一致")
    if remaining_aspects is not None and {
        normalize_text(value) for value in remaining_aspects if normalize_text(value)
    } != {normalize_text(value) for value in remaining}:
        errors.append("remaining_aspects 必须与尚未支持的 evidence obligations 一致")
    uncertainty = (
        residual_uncertainty
        if residual_uncertainty is not None
        else current_meta.residual_uncertainty
        if current_meta
        else ""
    ).strip()
    normalized_closure_reason = (
        closure_reason if closure_reason is not None else current_meta.closure_reason if current_meta else ""
    ).strip()
    effective_review_outcome = (
        review_outcome if review_outcome is not None else current_meta.review_outcome if current_meta else None
    )
    pending_recoveries = [
        value
        for value in adaptive_recoveries_from_state(state)
        if value.investigation_id == normalized_id and value.status == "pending"
    ]
    effective_selected = (
        selected_evidence_ids
        if selected_evidence_ids is not None
        else investigation.selected_evidence_ids
        if investigation
        else []
    )
    effective_working_note = working_note.strip() or (investigation.working_note if investigation else "")
    selected_set = {value.strip().upper() for value in effective_selected if value.strip()}
    query_by_id = {value.query_id: value for value in queries_from_state(state)}
    direct_evidence_by_obligation: dict[str, set[str]] = {}
    for probe in adaptive_probe_records_from_state(state):
        if probe.investigation_id != normalized_id or probe.status not in ADAPTIVE_SUCCESSFUL_SEARCH_STATUSES:
            continue
        query = query_by_id.get(probe.query_id)
        if query is None:
            continue
        direct_evidence_by_obligation.setdefault(
            normalize_text(probe.uncovered_aspect),
            set(),
        ).update(value.strip().upper() for value in query.evidence_ids if value.strip())
    changed = True
    while changed:
        changed = False
        for raw_open in state.get("open_records") or []:
            open_investigation_id = (
                raw_open.get("investigation_id")
                if isinstance(raw_open, dict)
                else getattr(raw_open, "investigation_id", None)
            )
            if (open_investigation_id or "").strip().upper() != normalized_id:
                continue
            parent_evidence_id = (
                str(
                    raw_open.get("parent_evidence_id")
                    if isinstance(raw_open, dict)
                    else getattr(raw_open, "parent_evidence_id", "")
                )
                .strip()
                .upper()
            )
            opened_evidence_ids = {
                str(value).strip().upper()
                for value in (
                    raw_open.get("evidence_ids", [])
                    if isinstance(raw_open, dict)
                    else getattr(raw_open, "evidence_ids", [])
                )
                if str(value).strip()
            }
            for evidence_ids in direct_evidence_by_obligation.values():
                before = len(evidence_ids)
                if parent_evidence_id in evidence_ids:
                    evidence_ids.update(opened_evidence_ids)
                changed = changed or len(evidence_ids) > before
    for support in normalized_supports:
        unknown_selected = [value for value in support.evidence_ids if value not in selected_set]
        if unknown_selected:
            errors.append(
                f"义务 [{support.obligation}] 的 Evidence 未包含在 selected_evidence_ids："
                + "、".join(unknown_selected)
            )
        obligation_evidence_ids = direct_evidence_by_obligation.get(
            normalize_text(support.obligation),
            set(),
        )
        if not obligation_evidence_ids.intersection(support.evidence_ids):
            errors.append(
                f"义务 [{support.obligation}] 至少需要绑定一条由该义务定向 probe 返回或从其打开的相邻 Evidence"
            )
    successfully_probed_obligations = adaptive_successful_obligation_probes(state).get(normalized_id, set())
    attempted_obligations = adaptive_attempted_obligation_probes(state).get(normalized_id, set())
    unsuccessfully_probed_obligations = [
        value for key, value in obligation_by_normalized.items() if key not in successfully_probed_obligations
    ]
    unattempted_obligations = [
        value for key, value in obligation_by_normalized.items() if key not in attempted_obligations
    ]
    if status == "answered":
        if not effective_selected:
            errors.append("answered 必须选择至少一个候选 Evidence ID")
        if not effective_working_note:
            errors.append("answered 必须写明 working_note")
        if unsuccessfully_probed_obligations:
            errors.append("answered 前仍有未成功定向检索的证据义务：" + " | ".join(unsuccessfully_probed_obligations))
        if missing_support_obligations:
            errors.append("answered 前仍有未绑定直接 Evidence 的义务：" + " | ".join(missing_support_obligations))
        if pending_recoveries:
            errors.append("仍有强制全库恢复任务，不能标记 answered")
        if agenda_item.investigation_kind == "current_regimen_review" and effective_review_outcome is None:
            errors.append("current_regimen_review 标记 answered 时必须给出 review_outcome=appropriate/adjust/avoid")
    elif status == "insufficient":
        if not uncertainty:
            errors.append("insufficient 必须说明 residual_uncertainty")
        if not normalized_closure_reason:
            errors.append("insufficient 必须说明 closure_reason")
        has_probe = any(value.investigation_id == normalized_id for value in adaptive_probe_records_from_state(state))
        budget_exhausted = int(state.get("search_count") or 0) >= context.max_search_calls
        if not has_probe and not budget_exhausted:
            errors.append("insufficient 前必须至少执行一次真实检索尝试")
        if unattempted_obligations and not budget_exhausted:
            errors.append("insufficient 前仍有未实际尝试的证据义务：" + " | ".join(unattempted_obligations))
        if not missing_support_obligations:
            errors.append("全部证据义务均已支持时不应标记 insufficient")
        if pending_recoveries and not budget_exhausted:
            errors.append("仍有强制全库恢复任务，不能标记 insufficient")
    elif status == "dismissed" and not normalized_closure_reason:
        errors.append("dismissed 必须说明 closure_reason")
    if agenda_item.investigation_kind != "current_regimen_review" and review_outcome is not None:
        errors.append("review_outcome 只能用于 current_regimen_review")
    if errors:
        return _tool_message(
            runtime=runtime,
            tool_name="update_investigation",
            content="调查状态未更新：" + "；".join(errors),
            warning=True,
        )

    result = await update_investigation.coroutine(
        investigation_id=normalized_id,
        status=status,
        selected_evidence_ids=selected_evidence_ids,
        working_note=effective_working_note,
        runtime=runtime,
    )
    update = result.update if isinstance(result.update, dict) else {}
    if not update.get("investigations"):
        return result
    meta = AdaptiveInvestigationMeta(
        investigation_id=normalized_id,
        resolved_aspects=resolved,
        remaining_aspects=remaining,
        obligation_supports=normalized_supports,
        review_outcome=(
            effective_review_outcome
            if status == "answered" and agenda_item.investigation_kind == "current_regimen_review"
            else None
        ),
        residual_uncertainty=uncertainty,
        closure_reason=normalized_closure_reason,
        updated_at=utc_isoformat(),
    )
    recovery_updates: list[AdaptiveRecoveryRequirement] = []
    if status == "insufficient" and pending_recoveries:
        recovery_updates = [value.model_copy(update={"status": "closed_insufficient"}) for value in pending_recoveries]
        _append_tool_notice(
            update,
            "搜索预算已耗尽；待恢复任务以 closed_insufficient 留痕关闭。",
        )
    if (
        status == "answered"
        and agenda_item.investigation_kind == "current_regimen_review"
        and effective_review_outcome in {"adjust", "avoid"}
    ):
        _append_tool_notice(
            update,
            "该现用药已判定需调整/避免；若尚无关联同一 PlanAnchor 的 "
            "improvement_plan，请调用 extend_investigation_agenda 追加具体纠正、"
            "替代/非药物干预与随访义务。",
        )
    return Command(
        update={
            **update,
            "adaptive_investigation_meta": [meta],
            "adaptive_recovery_requirements": recovery_updates,
        }
    )


@tool("update_investigation")
async def update_acm_investigation(
    investigation_id: Annotated[str, Field(min_length=1)],
    status: InvestigationStatus,
    selected_evidence_ids: list[str] | None = None,
    working_note: Annotated[str, Field(max_length=800)] = "",
    resolved_aspects: list[str] | None = None,
    remaining_aspects: list[str] | None = None,
    obligation_supports: list[AdaptiveObligationSupport] | None = None,
    review_outcome: AdaptiveReviewOutcome | None = None,
    residual_uncertainty: Annotated[str | None, Field(max_length=800)] = None,
    closure_reason: Annotated[str | None, Field(max_length=800)] = None,
    runtime: ToolRuntime = None,
) -> Command:
    """更新调查；按当前协议校验关闭所需证据、缺口与恢复状态。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("update_investigation 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    state = runtime.state if isinstance(runtime.state, dict) else {}
    if adaptive_coverage_enabled(context):
        return await _update_adaptive_investigation(
            investigation_id=investigation_id,
            status=status,
            selected_evidence_ids=selected_evidence_ids,
            working_note=working_note,
            resolved_aspects=resolved_aspects,
            remaining_aspects=remaining_aspects,
            obligation_supports=obligation_supports,
            review_outcome=review_outcome,
            residual_uncertainty=residual_uncertainty,
            closure_reason=closure_reason,
            runtime=runtime,
        )
    if v7_required_investigation_count(context.v7_experiment_arm) and status != "open":
        normalized_id = investigation_id.strip().upper()
        report = build_v7_contract_report(state, context)
        if normalized_id in report.missing_initial_probe_ids:
            return _tool_message(
                runtime=runtime,
                tool_name="update_investigation",
                content=(f"调查 [{normalized_id}] 尚未完成 initial_probe，不能关闭。"),
            )
        if normalized_id in report.missing_complementary_probe_ids:
            return _tool_message(
                runtime=runtime,
                tool_name="update_investigation",
                content=(f"调查 [{normalized_id}] 尚未完成 complementary_probe，不能关闭。"),
            )
    return await update_investigation.coroutine(
        investigation_id=investigation_id,
        status=status,
        selected_evidence_ids=selected_evidence_ids,
        working_note=working_note,
        runtime=runtime,
    )


@tool
async def submit_coverage_gap_assessment(
    material_gap_found: bool,
    rationale: Annotated[str, Field(min_length=1, max_length=1200)],
    coverage_audit: Annotated[list[AdaptiveCoverageAuditEntry], Field(min_length=1)],
    unsupported_investigation_ids: list[str] | None = None,
    unsupported_obligations: dict[str, list[str]] | None = None,
    proposed_items: list[AcmAgendaItemDraft] | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """逐维度审计临床与证据覆盖，并重开旧调查或追加实质性新调查。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("submit_coverage_gap_assessment 缺少 ToolRuntime")
    context: MedicationReviewAcmPrimContext = runtime.context
    if not adaptive_coverage_enabled(context):
        return _tool_message(
            runtime=runtime,
            tool_name="submit_coverage_gap_assessment",
            content="当前运行未启用 adaptive_coverage 协议。",
            warning=True,
        )
    state = runtime.state if isinstance(runtime.state, dict) else {}
    agenda = adaptive_agenda_from_state(state)
    if agenda is None:
        return _tool_message(
            runtime=runtime,
            tool_name="submit_coverage_gap_assessment",
            content="尚未建立调查议程，不能执行全局缺口审计。",
            warning=True,
        )
    report = build_adaptive_coverage_report(state, context)
    preconditions: list[str] = []
    if report.missing_current_regimen_review_plan_ids:
        preconditions.append("尚未逐项审查 PlanAnchor：" + "、".join(report.missing_current_regimen_review_plan_ids))
    if report.missing_improvement_plan_ids:
        preconditions.append(
            "已判定需调整/避免但缺 improvement_plan：" + "、".join(report.missing_improvement_plan_ids)
        )
    if report.unprobed_investigation_ids:
        preconditions.append("尚未真实检索：" + "、".join(report.unprobed_investigation_ids))
    if report.unprobed_evidence_obligations:
        preconditions.append(
            "尚未定向检索证据义务："
            + "；".join(
                f"{investigation_id}={' | '.join(values)}"
                for investigation_id, values in report.unprobed_evidence_obligations.items()
            )
        )
    if report.open_investigation_ids:
        preconditions.append("尚未关闭：" + "、".join(report.open_investigation_ids))
    if report.unsupported_evidence_obligations:
        preconditions.append(
            "尚未处理证据义务："
            + "；".join(
                f"{investigation_id}={' | '.join(values)}"
                for investigation_id, values in report.unsupported_evidence_obligations.items()
            )
        )
    if report.pending_recovery_ids:
        preconditions.append("尚待全库恢复：" + "、".join(report.pending_recovery_ids))
    if preconditions:
        return _tool_message(
            runtime=runtime,
            tool_name="submit_coverage_gap_assessment",
            content="全局缺口审计尚不可提交：" + "；".join(preconditions),
            warning=True,
        )

    unsupported = list(
        dict.fromkeys(value.strip().upper() for value in (unsupported_investigation_ids or []) if value.strip())
    )
    normalized_unsupported_obligations = {
        investigation_id.strip().upper(): list(dict.fromkeys(value.strip() for value in values if value.strip()))
        for investigation_id, values in (unsupported_obligations or {}).items()
        if investigation_id.strip()
    }
    proposed = proposed_items or []
    agenda_ids = {value.investigation_id for value in agenda.items}
    normalized_audit = [
        value if isinstance(value, AdaptiveCoverageAuditEntry) else AdaptiveCoverageAuditEntry.model_validate(value)
        for value in coverage_audit
    ]
    audit_dimensions = [value.dimension for value in normalized_audit]
    audit_investigation_ids = {
        investigation_id.strip().upper()
        for value in normalized_audit
        for investigation_id in value.investigation_ids
        if investigation_id.strip()
    }
    unknown_ids = sorted(
        {
            value
            for value in [*unsupported, *normalized_unsupported_obligations, *audit_investigation_ids]
            if value not in agenda_ids
        }
    )
    errors: list[str] = []
    if unknown_ids:
        errors.append("未知 Investigation ID：" + "、".join(unknown_ids))
    missing_dimensions = [value for value in ADAPTIVE_AUDIT_DIMENSIONS if value not in audit_dimensions]
    duplicate_dimensions = sorted({value for value in audit_dimensions if audit_dimensions.count(value) > 1})
    if missing_dimensions:
        errors.append("coverage_audit 缺少维度：" + "、".join(missing_dimensions))
    if duplicate_dimensions:
        errors.append("coverage_audit 重复维度：" + "、".join(duplicate_dimensions))
    if any(value.status == "covered" and not value.investigation_ids for value in normalized_audit):
        errors.append("coverage_audit 的 covered 维度必须引用至少一个 Investigation ID")
    gap_dimensions = [value.dimension for value in normalized_audit if value.status == "gap"]
    if gap_dimensions and not material_gap_found:
        errors.append("coverage_audit 存在 gap 时 material_gap_found 必须为 true")
    unknown_unsupported_keys = sorted(set(normalized_unsupported_obligations) - set(unsupported))
    if unknown_unsupported_keys:
        errors.append(
            "unsupported_obligations 的 Investigation 必须同时列入 unsupported_investigation_ids："
            + "、".join(unknown_unsupported_keys)
        )
    if (unsupported or proposed) and not material_gap_found:
        errors.append("存在 unsupported investigation 或 proposed item 时，material_gap_found 必须为 true")
    if material_gap_found and not unsupported and not proposed:
        errors.append("material_gap_found=true 时必须重开旧 Investigation 或追加 proposed item")
    agenda_item_by_id = {value.investigation_id: value for value in agenda.items}
    for investigation_id, obligations in normalized_unsupported_obligations.items():
        item = agenda_item_by_id.get(investigation_id)
        if item is None:
            continue
        known_obligations = {normalize_text(value) for value in item.evidence_obligations}
        unknown_obligations = [value for value in obligations if normalize_text(value) not in known_obligations]
        if unknown_obligations:
            errors.append(f"调查 [{investigation_id}] 包含未知 evidence obligation：" + " | ".join(unknown_obligations))
    if errors:
        return _tool_message(
            runtime=runtime,
            tool_name="submit_coverage_gap_assessment",
            content="全局缺口审计未保存：" + "；".join(errors),
            warning=True,
        )

    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    fingerprint = build_adaptive_state_fingerprint(state)
    agenda_update: dict[str, Any] = {}
    proposed_ids: list[str] = []
    if proposed:
        append_result = _adaptive_agenda_command(
            items=proposed,
            reason=f"coverage gap assessment: {rationale.strip()}",
            runtime=runtime,
            initial=False,
            tool_name="submit_coverage_gap_assessment",
        )
        agenda_update = append_result.update if isinstance(append_result.update, dict) else {}
        appended_agenda = agenda_update.get("adaptive_agenda")
        if not isinstance(appended_agenda, AdaptiveInvestigationAgenda):
            return append_result
        proposed_ids = [
            value.investigation_id for value in appended_agenda.items if value.investigation_id not in agenda_ids
        ]

    normalized_audit = [
        value.model_copy(
            update={
                "investigation_ids": list(
                    dict.fromkeys(
                        investigation_id.strip().upper()
                        for investigation_id in value.investigation_ids
                        if investigation_id.strip()
                    )
                )
            }
        )
        for value in normalized_audit
    ]
    investigation_by_id = {value.investigation_id: value for value in investigations_from_state(state)}
    meta_by_id = {value.investigation_id: value for value in adaptive_meta_from_state(state)}
    reopened_investigations: list[InvestigationItem] = []
    reopened_meta: list[AdaptiveInvestigationMeta] = []
    now = utc_isoformat()
    for investigation_id in unsupported:
        item = agenda_item_by_id[investigation_id]
        investigation = investigation_by_id.get(investigation_id)
        if investigation is None:
            continue
        target_values = normalized_unsupported_obligations.get(
            investigation_id,
            item.evidence_obligations,
        )
        target_obligations = {normalize_text(value) for value in target_values}
        current_meta = meta_by_id.get(investigation_id)
        retained_supports = [
            value
            for value in (current_meta.obligation_supports if current_meta else [])
            if normalize_text(value.obligation) not in target_obligations
        ]
        retained_support_map = {normalize_text(value.obligation): value for value in retained_supports}
        resolved = [value for value in item.evidence_obligations if normalize_text(value) in retained_support_map]
        remaining = [value for value in item.evidence_obligations if normalize_text(value) not in retained_support_map]
        reopened_investigations.append(
            investigation.model_copy(
                update={
                    "status": "open",
                    "working_note": (f"全局覆盖审计判定证据不足，需重新定位：{' | '.join(target_values)}"),
                    "updated_at": now,
                }
            )
        )
        reopened_meta.append(
            AdaptiveInvestigationMeta(
                investigation_id=investigation_id,
                resolved_aspects=resolved,
                remaining_aspects=remaining,
                obligation_supports=retained_supports,
                review_outcome=None,
                residual_uncertainty="",
                closure_reason="",
                updated_at=now,
            )
        )

    assessment = AdaptiveGapAssessment(
        assessment_id=f"GAP-{tool_call_id}",
        created_at=now,
        state_fingerprint=fingerprint,
        material_gap_found=material_gap_found,
        rationale=rationale.strip(),
        unsupported_investigation_ids=unsupported,
        unsupported_obligations=normalized_unsupported_obligations,
        proposed_investigation_ids=proposed_ids,
        coverage_audit=normalized_audit,
    )
    if proposed or unsupported:
        notices: list[str] = []
        if unsupported:
            notices.append("本次审计已重开证据不足的调查：" + "、".join(unsupported) + "。")
        if proposed:
            notices.append("本次审计已原子追加新调查。")
        notices.append("完成这些调查后必须重新提交缺口审计。")
        _append_tool_notice(
            agenda_update,
            "".join(notices),
        )
        if not agenda_update.get("messages"):
            agenda_update["messages"] = [
                ToolMessage(
                    content="".join(notices),
                    tool_call_id=tool_call_id,
                    name="submit_coverage_gap_assessment",
                )
            ]
        return Command(
            update={
                **agenda_update,
                "investigations": [
                    *(agenda_update.get("investigations") or []),
                    *reopened_investigations,
                ],
                "adaptive_investigation_meta": [
                    *(agenda_update.get("adaptive_investigation_meta") or []),
                    *reopened_meta,
                ],
                "adaptive_gap_assessments": [assessment],
            }
        )
    content = (
        f"全局缺口审计 [{assessment.assessment_id}] 已保存：material_gap_found={str(material_gap_found).lower()}。"
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name="submit_coverage_gap_assessment",
                )
            ],
            "adaptive_gap_assessments": [assessment],
        }
    )


@tool
async def adaptive_coverage_checkpoint(
    notice: Annotated[str, Field(min_length=1, max_length=3000)],
    runtime: ToolRuntime = None,
) -> Command:
    """向 Agent 返回自适应覆盖合同未完成项；仅由 Harness 调用。"""
    if runtime is None:
        raise RuntimeError("adaptive_coverage_checkpoint 缺少 ToolRuntime")
    return _tool_message(
        runtime=runtime,
        tool_name="adaptive_coverage_checkpoint",
        content=notice,
    )


@tool
async def v7_effort_checkpoint(
    notice: Annotated[str, Field(min_length=1, max_length=2000)],
    runtime: ToolRuntime = None,
) -> Command:
    """向 Agent 返回 V7 未完成项；仅由 Harness 调用。"""
    if runtime is None:
        raise RuntimeError("v7_effort_checkpoint 缺少 ToolRuntime")
    return _tool_message(
        runtime=runtime,
        tool_name="v7_effort_checkpoint",
        content=notice,
    )


@tool
async def open_atlas_document(
    doc_id: Annotated[str, Field(min_length=1)],
    reason: Annotated[str, Field(min_length=1)],
    runtime: ToolRuntime = None,
) -> Command:
    """打开 Atlas 中某篇文档的详细治疗主题；主题仅用于导航，不是答案证据。"""
    if runtime is None or runtime.context is None:
        raise RuntimeError("open_atlas_document 缺少 ToolRuntime")
    atlas = getattr(runtime.context, "_acm_prim_atlas", None)
    if atlas is None or not hasattr(atlas, "document_view"):
        raise RuntimeError("当前运行没有可用的 Corpus Atlas")
    normalized_id = doc_id.strip()
    try:
        document = atlas.document_view(normalized_id)
    except KeyError:
        available = "、".join(value.doc_id for value in atlas.document_cards)
        content = f"Atlas 中不存在 doc_id={normalized_id!r}。可用 doc_id：{available}"
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=str(runtime.tool_call_id or uuid.uuid4()),
                        name="open_atlas_document",
                    )
                ],
                "warnings": [content],
            }
        )

    tool_call_id = str(runtime.tool_call_id or uuid.uuid4())
    topic_lines = [f"- [{value['cue_id']}] {value['cue_text']}" for value in document["topic_cues"]]
    content = "\n".join(
        [
            "【Atlas 文档主题】",
            f"文档：{document['title']}（file_id={document['doc_id']}）",
            f"范围摘要：{document['scope_summary']}",
            (
                ADAPTIVE_ATLAS_DOCUMENT_OPEN_INSTRUCTIONS
                if adaptive_coverage_enabled(runtime.context)
                else ATLAS_DOCUMENT_OPEN_INSTRUCTIONS
            ).format(doc_id=document["doc_id"]),
            "主题列表：",
            *(topic_lines or ["- 本文档没有抽取到详细治疗决策主题。"]),
        ]
    )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name="open_atlas_document",
                )
            ],
            "atlas_document_open_records": [
                AtlasDocumentOpenRecord(
                    record_id=f"AOPEN-{tool_call_id}",
                    tool_call_id=tool_call_id,
                    doc_id=document["doc_id"],
                    title=document["title"],
                    reason=reason.strip(),
                    topic_count=len(document["topic_cues"]),
                    cue_ids=[value["cue_id"] for value in document["topic_cues"]],
                )
            ],
        }
    )
