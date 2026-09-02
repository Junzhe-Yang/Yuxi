from __future__ import annotations

import hashlib
import json
import uuid
from typing import Annotated, Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from pydantic import Field

from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
    adaptive_meta_from_state,
    adaptive_query_contract_violations,
)
from yuxi.agents.buildin.medication_review_acm_prim.models import (
    AcmAgendaItemDraft,
    AdaptiveCoverageAuditEntry,
    AdaptiveObligationSupport,
    AdaptiveReviewOutcome,
)
from yuxi.agents.buildin.medication_review_acm_prim.tools import (
    _adaptive_agenda_command,
    _run_adaptive_search,
    _update_adaptive_investigation,
    open_atlas_document,
    submit_coverage_gap_assessment,
)
from yuxi.agents.buildin.medication_review_lite.models import EvidenceItem
from yuxi.agents.buildin.medication_review_prim.memory import investigations_from_state
from yuxi.agents.buildin.medication_review_prim.tools import open_review_evidence
from yuxi.utils.datetime_utils import utc_isoformat

from .models import (
    ActionAttempt,
    ActionDirective,
    BoundedAgendaItemDraft,
    BoundedAuditEntry,
    BoundedObligationJudgment,
    SemanticOutcome,
    ToolOutcome,
)

_ERROR_OUTCOMES = {
    "REJECTED",
    "INVALID_ARGUMENT",
    "NEEDS_INPUT",
    "RETRYABLE_ERROR",
    "FATAL_ERROR",
}


def _state(runtime: ToolRuntime) -> dict[str, Any]:
    return runtime.state if isinstance(runtime.state, dict) else {}


def _directive(runtime: ToolRuntime) -> ActionDirective:
    state = _state(runtime)
    raw = state.get("action_directive") or getattr(
        runtime.context,
        "_acm_bounded_directive",
        None,
    )
    if raw is None:
        raise RuntimeError("ACM Bounded tool 缺少 ActionDirective")
    return raw if isinstance(raw, ActionDirective) else ActionDirective.model_validate(raw)


def _canonical_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str))


def _action_fingerprint(
    directive: ActionDirective,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    payload = {
        "directive_id": directive.directive_id,
        "state_version": directive.state_version,
        "tool_name": tool_name,
        "arguments": _canonical_arguments(arguments),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _message_text(command: Command | None) -> str:
    if command is None or not isinstance(command.update, dict):
        return ""
    messages = command.update.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = getattr(messages[0], "content", "")
    return content if isinstance(content, str) else str(content)


def _repeated_nonprogress_attempt(
    runtime: ToolRuntime,
    fingerprint: str,
) -> ActionAttempt | None:
    for raw in _state(runtime).get("bounded_action_attempts") or []:
        try:
            attempt = raw if isinstance(raw, ActionAttempt) else ActionAttempt.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
        if attempt.fingerprint == fingerprint and not attempt.state_changed:
            return attempt
    return None


def _outcome_command(
    *,
    runtime: ToolRuntime,
    tool_name: str,
    arguments: dict[str, Any],
    semantic_outcome: SemanticOutcome,
    reason_code: str | None,
    message_for_model: str,
    base_command: Command | None = None,
    state_changed: bool = False,
    executed_backend: bool = False,
    transport_status: Literal["COMPLETED", "FAILED"] = "COMPLETED",
    retryable_by_model: bool = False,
    technical_retryable: bool = False,
    recovery_action: str | None = None,
    evidence_ids: list[str] | None = None,
    extra_update: dict[str, Any] | None = None,
) -> Command:
    directive = _directive(runtime)
    call_id = str(runtime.tool_call_id or uuid.uuid4())
    before = int(_state(runtime).get("bounded_state_version") or 0)
    after = before + (1 if state_changed else 0)
    fingerprint = _action_fingerprint(directive, tool_name, arguments)
    outcome = ToolOutcome(
        call_id=call_id,
        directive_id=directive.directive_id,
        tool_name=tool_name,
        transport_status=transport_status,
        semantic_outcome=semantic_outcome,
        reason_code=reason_code,
        retryable_by_model=retryable_by_model,
        technical_retryable=technical_retryable,
        state_changed=state_changed,
        executed_backend=executed_backend,
        state_version_before=before,
        state_version_after=after,
        recovery_action=recovery_action,
        message_for_model=message_for_model,
        next_legal_actions=list(directive.allowed_actions),
        investigation_id=directive.active_investigation_id,
        obligation=directive.active_obligation,
        evidence_ids=list(evidence_ids or []),
    )
    attempt = ActionAttempt(
        fingerprint=fingerprint,
        directive_id=directive.directive_id,
        state_version=directive.state_version,
        tool_name=tool_name,
        canonical_arguments=_canonical_arguments(arguments),
        semantic_outcome=semantic_outcome,
        reason_code=reason_code,
        state_changed=state_changed,
        created_at=utc_isoformat(),
    )
    update = dict(base_command.update) if base_command is not None and isinstance(base_command.update, dict) else {}
    original = _message_text(base_command)
    envelope = json.dumps(
        {"tool_outcome": outcome.model_dump(mode="json")},
        ensure_ascii=False,
        sort_keys=True,
    )
    content = envelope + (f"\n\n{original}" if original else f"\n\n{message_for_model}")
    previous_messages = update.get("messages")
    previous = previous_messages[0] if isinstance(previous_messages, list) and previous_messages else None
    if isinstance(previous, ToolMessage):
        message = previous.model_copy(
            update={
                "content": content,
                "name": tool_name,
                "status": "error" if semantic_outcome in _ERROR_OUTCOMES else "success",
            }
        )
    else:
        message = ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=tool_name,
            status="error" if semantic_outcome in _ERROR_OUTCOMES else "success",
        )
    update.update(extra_update or {})
    update.update(
        {
            "messages": [message],
            "bounded_tool_outcomes": [outcome],
            "bounded_action_attempts": [attempt],
        }
    )
    if state_changed:
        update["bounded_state_version"] = 1
    if semantic_outcome in _ERROR_OUTCOMES:
        update["warnings"] = [message_for_model]
    return Command(update=update)


def _rejection_if_unreachable(
    *,
    runtime: ToolRuntime,
    tool_name: str,
    arguments: dict[str, Any],
) -> Command | None:
    directive = _directive(runtime)
    if tool_name not in directive.allowed_actions:
        return _outcome_command(
            runtime=runtime,
            tool_name=tool_name,
            arguments=arguments,
            semantic_outcome="REJECTED",
            reason_code="ACTION_NOT_LEGAL_FOR_DIRECTIVE",
            message_for_model=f"当前 directive 不允许 {tool_name}。",
        )
    fingerprint = _action_fingerprint(directive, tool_name, arguments)
    if previous := _repeated_nonprogress_attempt(runtime, fingerprint):
        repeated_rejection = previous.semantic_outcome in {
            "REJECTED",
            "INVALID_ARGUMENT",
        }
        return _outcome_command(
            runtime=runtime,
            tool_name=tool_name,
            arguments=arguments,
            semantic_outcome="REJECTED",
            reason_code=("REPEATED_REJECTED_ACTION" if repeated_rejection else "REPEATED_NON_PROGRESS_ACTION"),
            message_for_model=(
                "相同状态下的相同非法动作已被拒绝；必须修改语义参数或等待状态变化。"
                if repeated_rejection
                else "相同状态下的无状态变化动作已经执行过；必须选择能推进 ledger 的动作。"
            ),
        )
    return None


def _agenda_drafts(
    values: list[BoundedAgendaItemDraft],
    directive: ActionDirective,
) -> tuple[list[AcmAgendaItemDraft], list[str]]:
    drafts: list[AcmAgendaItemDraft] = []
    unknown_aliases: list[str] = []
    for value in values:
        parent_ids: list[str] = []
        for alias in value.parent_evidence_aliases:
            evidence_id = directive.evidence_aliases.get(alias.upper())
            if evidence_id is None:
                unknown_aliases.append(alias)
            else:
                parent_ids.append(evidence_id)
        drafts.append(
            AcmAgendaItemDraft(
                question=value.question,
                why_it_matters=value.why_it_matters,
                distinct_scope=value.distinct_scope,
                investigation_kind=value.investigation_kind,
                evidence_obligations=value.evidence_obligations,
                focus_plan_ids=value.focus_plan_ids,
                focus_modifier_ids=value.focus_modifier_ids,
                decision_tags=value.decision_tags,
                parent_evidence_ids=parent_ids,
            )
        )
    return drafts, unknown_aliases


@tool
async def propose_initial_agenda(
    items: Annotated[list[BoundedAgendaItemDraft], Field(min_length=1)],
    runtime: ToolRuntime = None,
) -> Command:
    """一次提出所有临床上有实质区别的调查；数量由病例决定，不设固定 K。"""
    if runtime is None:
        raise RuntimeError("propose_initial_agenda 缺少 ToolRuntime")
    arguments = {"items": [value.model_dump(mode="json") for value in items]}
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="propose_initial_agenda",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    drafts, unknown = _agenda_drafts(items, directive)
    if unknown:
        return _outcome_command(
            runtime=runtime,
            tool_name="propose_initial_agenda",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_PARENT_EVIDENCE_ALIAS",
            message_for_model="未知 parent Evidence alias：" + "、".join(unknown),
            retryable_by_model=True,
            recovery_action="REPAIR_ALIAS_ONCE",
        )
    command = _adaptive_agenda_command(
        items=drafts,
        reason="initial bounded coverage agenda",
        runtime=runtime,
        initial=True,
        tool_name="propose_initial_agenda",
    )
    changed = isinstance(command.update, dict) and "adaptive_agenda" in command.update
    return _outcome_command(
        runtime=runtime,
        tool_name="propose_initial_agenda",
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed else "REJECTED",
        reason_code=None if changed else "AGENDA_CONTRACT_REJECTED",
        message_for_model="初始调查议程已建立。" if changed else _message_text(command),
        base_command=command,
        state_changed=changed,
        retryable_by_model=not changed,
        recovery_action=None if changed else "REPAIR_AGENDA_ONCE",
    )


@tool("extend_investigation_agenda_bounded")
async def extend_investigation_agenda_bounded(
    items: Annotated[list[BoundedAgendaItemDraft], Field(min_length=1)],
    rationale: Annotated[str, Field(min_length=1, max_length=800)],
    runtime: ToolRuntime = None,
) -> Command:
    """只追加当前覆盖缺口要求的新调查，不重写已有议程。"""
    if runtime is None:
        raise RuntimeError("extend_investigation_agenda_bounded 缺少 ToolRuntime")
    arguments = {
        "items": [value.model_dump(mode="json") for value in items],
        "rationale": rationale,
    }
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="extend_investigation_agenda_bounded",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    drafts, unknown = _agenda_drafts(items, directive)
    if unknown:
        return _outcome_command(
            runtime=runtime,
            tool_name="extend_investigation_agenda_bounded",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_PARENT_EVIDENCE_ALIAS",
            message_for_model="未知 parent Evidence alias：" + "、".join(unknown),
            retryable_by_model=True,
            recovery_action="REPAIR_ALIAS_ONCE",
        )
    command = _adaptive_agenda_command(
        items=drafts,
        reason=rationale,
        runtime=runtime,
        initial=False,
        tool_name="extend_investigation_agenda_bounded",
    )
    changed = isinstance(command.update, dict) and "adaptive_agenda" in command.update
    return _outcome_command(
        runtime=runtime,
        tool_name="extend_investigation_agenda_bounded",
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed else "REJECTED",
        reason_code=None if changed else "AGENDA_EXTENSION_REJECTED",
        message_for_model="调查议程已追加。" if changed else _message_text(command),
        base_command=command,
        state_changed=changed,
        retryable_by_model=not changed,
        recovery_action=None if changed else "REPAIR_AGENDA_ONCE",
    )


@tool
async def search_active_obligation(
    query_text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=160,
            description=(
                "当前义务的 2—6 个中文语料关键词：一个主体、一个属性、至多一个患者限定；"
                "不要复制义务整句、解释原因、枚举答案或混合属性轴。"
            ),
        ),
    ],
    runtime: ToolRuntime = None,
) -> Command:
    """检索当前 controller 绑定的唯一 evidence obligation。"""
    if runtime is None:
        raise RuntimeError("search_active_obligation 缺少 ToolRuntime")
    arguments = {"query_text": query_text}
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="search_active_obligation",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    if not directive.active_investigation_id or not directive.active_obligation:
        return _outcome_command(
            runtime=runtime,
            tool_name="search_active_obligation",
            arguments=arguments,
            semantic_outcome="NEEDS_INPUT",
            reason_code="DIRECTIVE_TARGET_MISSING",
            message_for_model="当前 directive 缺少 active investigation 或 obligation。",
        )
    violations = adaptive_query_contract_violations(query_text)
    if violations:
        return _outcome_command(
            runtime=runtime,
            tool_name="search_active_obligation",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="QUERY_SHAPE_INVALID",
            message_for_model="；".join(violations),
            retryable_by_model=True,
            recovery_action="REPAIR_QUERY_ONCE",
        )
    try:
        command = await _run_adaptive_search(
            query_text=query_text,
            reason=directive.active_obligation,
            investigation_id=directive.active_investigation_id,
            uncovered_aspect=directive.active_obligation,
            retrieval_intent=directive.bound_retrieval_intent or "source_discovery",
            retrieval_scope=directive.bound_retrieval_scope or "global",
            file_id=directive.bound_file_id,
            runtime=runtime,
        )
    except Exception as exc:  # noqa: BLE001 - exported outcome owns technical failure
        return _outcome_command(
            runtime=runtime,
            tool_name="search_active_obligation",
            arguments=arguments,
            semantic_outcome="RETRYABLE_ERROR",
            reason_code=type(exc).__name__,
            message_for_model=f"检索技术失败：{exc}",
            transport_status="FAILED",
            technical_retryable=True,
            recovery_action="TECHNICAL_RETRY",
        )
    update = command.update if isinstance(command.update, dict) else {}
    records = update.get("query_records")
    if not isinstance(records, list) or not records:
        return _outcome_command(
            runtime=runtime,
            tool_name="search_active_obligation",
            arguments=arguments,
            semantic_outcome="REJECTED",
            reason_code="SEARCH_CONTRACT_REJECTED",
            message_for_model=_message_text(command),
            base_command=command,
            retryable_by_model=True,
            recovery_action="REPAIR_SEARCH_ONCE",
        )
    record = records[0]
    status = record.status if hasattr(record, "status") else str(record.get("status") or "")
    evidence_ids = list(record.evidence_ids if hasattr(record, "evidence_ids") else record.get("evidence_ids") or [])
    semantic: SemanticOutcome = (
        "SUCCESS" if status == "success" else "NO_RESULT" if status == "success_empty" else "RETRYABLE_ERROR"
    )
    return _outcome_command(
        runtime=runtime,
        tool_name="search_active_obligation",
        arguments=arguments,
        semantic_outcome=semantic,
        reason_code=None if semantic in {"SUCCESS", "NO_RESULT"} else "RETRIEVAL_TECHNICAL_FAILED",
        message_for_model=("检索完成。" if semantic == "SUCCESS" else "检索已执行但没有结果。"),
        base_command=command,
        state_changed=True,
        executed_backend=True,
        technical_retryable=semantic == "RETRYABLE_ERROR",
        evidence_ids=evidence_ids,
    )


def _resolve_evidence_aliases(
    aliases: list[str],
    directive: ActionDirective,
) -> tuple[list[str], list[str]]:
    evidence_ids: list[str] = []
    unknown: list[str] = []
    for raw in aliases:
        alias = raw.strip().upper()
        evidence_id = directive.evidence_aliases.get(alias)
        if evidence_id is None:
            unknown.append(raw)
        else:
            evidence_ids.append(evidence_id)
    return list(dict.fromkeys(evidence_ids)), unknown


@tool
async def read_active_evidence(
    evidence_aliases: Annotated[list[str], Field(min_length=1)],
    runtime: ToolRuntime = None,
) -> Command:
    """从 Evidence Store 重读当前 directive 中 E alias 的不可变精确原文；不重新检索。"""
    if runtime is None:
        raise RuntimeError("read_active_evidence 缺少 ToolRuntime")
    arguments = {"evidence_aliases": evidence_aliases}
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="read_active_evidence",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    evidence_ids, unknown = _resolve_evidence_aliases(evidence_aliases, directive)
    if unknown:
        return _outcome_command(
            runtime=runtime,
            tool_name="read_active_evidence",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_EVIDENCE_ALIAS",
            message_for_model="未知 Evidence alias：" + "、".join(unknown),
            retryable_by_model=True,
            recovery_action="REPAIR_ALIAS_ONCE",
        )
    store: dict[str, EvidenceItem] = {}
    for evidence_id, raw in (_state(runtime).get("evidence_store") or {}).items():
        try:
            store[str(evidence_id).upper()] = raw if isinstance(raw, EvidenceItem) else EvidenceItem.model_validate(raw)
        except Exception:  # noqa: BLE001
            continue
    blocks = []
    for alias in evidence_aliases:
        evidence_id = directive.evidence_aliases[alias.strip().upper()]
        item = store.get(evidence_id)
        if item is not None:
            blocks.append(f"[{alias.upper()} -> {evidence_id}] hash={item.content_hash}\n原文：\n{item.raw_text}")
    base = Command(
        update={
            "messages": [
                ToolMessage(
                    content="\n\n".join(blocks),
                    tool_call_id=str(runtime.tool_call_id or uuid.uuid4()),
                    name="read_active_evidence",
                )
            ]
        }
    )
    return _outcome_command(
        runtime=runtime,
        tool_name="read_active_evidence",
        arguments=arguments,
        semantic_outcome="SUCCESS",
        reason_code=None,
        message_for_model="已从 Evidence Store 读取精确原文。",
        base_command=base,
        evidence_ids=evidence_ids,
    )


@tool
async def open_active_evidence(
    evidence_alias: str,
    runtime: ToolRuntime = None,
) -> Command:
    """围绕当前 E alias 打开同一文档的相邻原文；Investigation 由 controller 绑定。"""
    if runtime is None:
        raise RuntimeError("open_active_evidence 缺少 ToolRuntime")
    arguments = {"evidence_alias": evidence_alias}
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="open_active_evidence",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    evidence_ids, unknown = _resolve_evidence_aliases([evidence_alias], directive)
    if unknown or not directive.active_investigation_id:
        return _outcome_command(
            runtime=runtime,
            tool_name="open_active_evidence",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_EVIDENCE_ALIAS",
            message_for_model="Evidence alias 不属于当前候选集合。",
            retryable_by_model=True,
            recovery_action="REPAIR_ALIAS_ONCE",
        )
    try:
        command = await open_review_evidence.coroutine(
            evidence_id=evidence_ids[0],
            reason=directive.active_obligation or "查看当前义务的相邻原文",
            investigation_id=directive.active_investigation_id,
            window_before=1,
            window_after=1,
            runtime=runtime,
        )
    except Exception as exc:  # noqa: BLE001
        return _outcome_command(
            runtime=runtime,
            tool_name="open_active_evidence",
            arguments=arguments,
            semantic_outcome="RETRYABLE_ERROR",
            reason_code=type(exc).__name__,
            message_for_model=f"打开相邻原文失败：{exc}",
            transport_status="FAILED",
            technical_retryable=True,
        )
    update = command.update if isinstance(command.update, dict) else {}
    records = update.get("open_records")
    changed = isinstance(records, list) and bool(records)
    opened_ids = []
    if changed:
        record = records[0]
        opened_ids = list(record.evidence_ids if hasattr(record, "evidence_ids") else record.get("evidence_ids") or [])
    return _outcome_command(
        runtime=runtime,
        tool_name="open_active_evidence",
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed and opened_ids else "NO_RESULT" if changed else "REJECTED",
        reason_code=None if changed else "OPEN_CONTRACT_REJECTED",
        message_for_model="相邻原文已打开。" if opened_ids else _message_text(command),
        base_command=command,
        state_changed=changed,
        executed_backend=changed,
        retryable_by_model=not changed,
        evidence_ids=opened_ids,
    )


@tool
async def open_active_atlas_document(
    document_alias: str,
    runtime: ToolRuntime = None,
) -> Command:
    """打开当前 directive 提供的未读 Atlas 文档 alias；Atlas 只用于导航。"""
    if runtime is None:
        raise RuntimeError("open_active_atlas_document 缺少 ToolRuntime")
    arguments = {"document_alias": document_alias}
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="open_active_atlas_document",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    doc_id = directive.atlas_document_aliases.get(document_alias.strip().upper())
    if doc_id is None:
        return _outcome_command(
            runtime=runtime,
            tool_name="open_active_atlas_document",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_ATLAS_ALIAS",
            message_for_model="Atlas alias 不在当前未读文档集合中。",
            retryable_by_model=True,
        )
    command = await open_atlas_document.coroutine(
        doc_id=doc_id,
        reason=directive.active_obligation or "改善当前检索式",
        runtime=runtime,
    )
    changed = isinstance(command.update, dict) and bool(command.update.get("atlas_document_open_records"))
    return _outcome_command(
        runtime=runtime,
        tool_name="open_active_atlas_document",
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed else "REJECTED",
        reason_code=None if changed else "ATLAS_OPEN_REJECTED",
        message_for_model="Atlas 文档主题已打开。" if changed else _message_text(command),
        base_command=command,
        state_changed=changed,
        retryable_by_model=not changed,
    )


@tool
async def record_active_obligation_support(
    evidence_aliases: list[str],
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"],
    rationale: Annotated[str, Field(min_length=1, max_length=800)],
    runtime: ToolRuntime = None,
) -> Command:
    """记录当前义务的证据判断；target、候选归属和 provenance 由 controller 绑定。"""
    if runtime is None:
        raise RuntimeError("record_active_obligation_support 缺少 ToolRuntime")
    arguments = {
        "evidence_aliases": evidence_aliases,
        "verdict": verdict,
        "rationale": rationale,
    }
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="record_active_obligation_support",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    if not directive.active_investigation_id or not directive.active_obligation:
        return _outcome_command(
            runtime=runtime,
            tool_name="record_active_obligation_support",
            arguments=arguments,
            semantic_outcome="NEEDS_INPUT",
            reason_code="DIRECTIVE_TARGET_MISSING",
            message_for_model="当前 directive 没有可判断的 obligation。",
        )
    evidence_ids, unknown = _resolve_evidence_aliases(evidence_aliases, directive)
    if unknown:
        return _outcome_command(
            runtime=runtime,
            tool_name="record_active_obligation_support",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="EVIDENCE_OUTSIDE_CANDIDATE_SET",
            message_for_model="请选择当前 directive 中的 Evidence alias：" + "、".join(directive.evidence_aliases),
            retryable_by_model=True,
            recovery_action="REPAIR_EVIDENCE_SELECTION_ONCE",
        )
    if verdict in {"SUPPORTED", "CONTRADICTED"} and not evidence_ids:
        return _outcome_command(
            runtime=runtime,
            tool_name="record_active_obligation_support",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="EVIDENCE_REQUIRED_FOR_VERDICT",
            message_for_model=f"{verdict} 必须选择至少一个当前 Evidence alias。",
            retryable_by_model=True,
        )
    if verdict == "INSUFFICIENT" and evidence_ids:
        return _outcome_command(
            runtime=runtime,
            tool_name="record_active_obligation_support",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="INSUFFICIENT_MUST_NOT_BIND_SUPPORT",
            message_for_model="INSUFFICIENT 不绑定支持 Evidence；请清空 evidence_aliases。",
            retryable_by_model=True,
        )
    state = _state(runtime)
    investigation = next(
        (
            value
            for value in investigations_from_state(state)
            if value.investigation_id == directive.active_investigation_id
        ),
        None,
    )
    current_meta = next(
        (
            value
            for value in adaptive_meta_from_state(state)
            if value.investigation_id == directive.active_investigation_id
        ),
        None,
    )
    supports = list(current_meta.obligation_supports if current_meta else [])
    supports = [
        value
        for value in supports
        if value.obligation.strip().casefold() != directive.active_obligation.strip().casefold()
    ]
    if verdict != "INSUFFICIENT":
        supports.append(
            AdaptiveObligationSupport(
                obligation=directive.active_obligation,
                evidence_ids=evidence_ids,
            )
        )
    selected = list(
        dict.fromkeys(
            [
                *(investigation.selected_evidence_ids if investigation else []),
                *(value for support in supports for value in support.evidence_ids),
            ]
        )
    )
    command = await _update_adaptive_investigation(
        investigation_id=directive.active_investigation_id,
        status="open",
        selected_evidence_ids=selected,
        working_note=rationale,
        resolved_aspects=None,
        remaining_aspects=None,
        obligation_supports=supports,
        review_outcome=None,
        residual_uncertainty=None,
        closure_reason=None,
        runtime=runtime,
    )
    changed = isinstance(command.update, dict) and bool(command.update.get("investigations"))
    if not changed:
        return _outcome_command(
            runtime=runtime,
            tool_name="record_active_obligation_support",
            arguments=arguments,
            semantic_outcome="REJECTED",
            reason_code="EVIDENCE_PROVENANCE_REJECTED",
            message_for_model=_message_text(command),
            base_command=command,
            retryable_by_model=True,
            recovery_action="REPAIR_EVIDENCE_SELECTION_ONCE",
        )
    call_id = str(runtime.tool_call_id or uuid.uuid4())
    judgment = BoundedObligationJudgment(
        judgment_id=f"JUDG-{call_id}",
        tool_call_id=call_id,
        investigation_id=directive.active_investigation_id,
        obligation=directive.active_obligation,
        verdict=verdict,
        evidence_ids=evidence_ids,
        rationale=rationale,
        state_version_after=int(state.get("bounded_state_version") or 0) + 1,
        created_at=utc_isoformat(),
    )
    return _outcome_command(
        runtime=runtime,
        tool_name="record_active_obligation_support",
        arguments=arguments,
        semantic_outcome="SUCCESS",
        reason_code=None,
        message_for_model="当前义务判断已写入 Working Ledger。",
        base_command=command,
        state_changed=True,
        evidence_ids=evidence_ids,
        extra_update={"bounded_obligation_judgments": [judgment]},
    )


async def _close_investigation(
    *,
    runtime: ToolRuntime,
    tool_name: str,
    status: Literal["answered", "insufficient"],
    conclusion: str,
    residual_uncertainty: str | None,
    closure_reason: str | None,
    review_outcome: AdaptiveReviewOutcome | None,
) -> Command:
    arguments = {
        "status": status,
        "conclusion": conclusion,
        "residual_uncertainty": residual_uncertainty,
        "closure_reason": closure_reason,
        "review_outcome": review_outcome,
    }
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name=tool_name,
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    if not directive.active_investigation_id:
        return _outcome_command(
            runtime=runtime,
            tool_name=tool_name,
            arguments=arguments,
            semantic_outcome="NEEDS_INPUT",
            reason_code="DIRECTIVE_TARGET_MISSING",
            message_for_model="当前 directive 缺少 active investigation。",
        )
    state = _state(runtime)
    current_meta = next(
        (
            value
            for value in adaptive_meta_from_state(state)
            if value.investigation_id == directive.active_investigation_id
        ),
        None,
    )
    supports = list(current_meta.obligation_supports if current_meta else [])
    selected = list(dict.fromkeys(value for support in supports for value in support.evidence_ids))
    command = await _update_adaptive_investigation(
        investigation_id=directive.active_investigation_id,
        status=status,
        selected_evidence_ids=selected,
        working_note=conclusion,
        resolved_aspects=None,
        remaining_aspects=None,
        obligation_supports=supports,
        review_outcome=review_outcome,
        residual_uncertainty=residual_uncertainty,
        closure_reason=closure_reason,
        runtime=runtime,
    )
    changed = isinstance(command.update, dict) and bool(command.update.get("investigations"))
    return _outcome_command(
        runtime=runtime,
        tool_name=tool_name,
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed else "REJECTED",
        reason_code=None if changed else "INVESTIGATION_CLOSE_REJECTED",
        message_for_model="调查已关闭。" if changed else _message_text(command),
        base_command=command,
        state_changed=changed,
        retryable_by_model=not changed,
        recovery_action=None if changed else "REPAIR_CLOSE_ONCE",
        evidence_ids=selected,
    )


@tool
async def close_current_regimen_investigation(
    status: Literal["answered", "insufficient"],
    conclusion: Annotated[str, Field(min_length=1, max_length=800)],
    review_outcome: AdaptiveReviewOutcome,
    residual_uncertainty: Annotated[str | None, Field(max_length=800)] = None,
    closure_reason: Annotated[str | None, Field(max_length=800)] = None,
    runtime: ToolRuntime = None,
) -> Command:
    """关闭当前逐项用药审查，并给出 appropriate/adjust/avoid；派生状态由 controller 计算。"""
    if runtime is None:
        raise RuntimeError("close_current_regimen_investigation 缺少 ToolRuntime")
    return await _close_investigation(
        runtime=runtime,
        tool_name="close_current_regimen_investigation",
        status=status,
        conclusion=conclusion,
        residual_uncertainty=residual_uncertainty,
        closure_reason=closure_reason,
        review_outcome=review_outcome,
    )


@tool
async def close_active_investigation(
    status: Literal["answered", "insufficient"],
    conclusion: Annotated[str, Field(min_length=1, max_length=800)],
    residual_uncertainty: Annotated[str | None, Field(max_length=800)] = None,
    closure_reason: Annotated[str | None, Field(max_length=800)] = None,
    runtime: ToolRuntime = None,
) -> Command:
    """关闭当前非逐项用药调查；Evidence 并集和 resolved/remaining 由 controller 计算。"""
    if runtime is None:
        raise RuntimeError("close_active_investigation 缺少 ToolRuntime")
    return await _close_investigation(
        runtime=runtime,
        tool_name="close_active_investigation",
        status=status,
        conclusion=conclusion,
        residual_uncertainty=residual_uncertainty,
        closure_reason=closure_reason,
        review_outcome=None,
    )


@tool
async def audit_coverage(
    entries: Annotated[list[BoundedAuditEntry], Field(min_length=1)],
    rationale: Annotated[str, Field(min_length=1, max_length=1200)],
    proposed_items: list[BoundedAgendaItemDraft] | None = None,
    runtime: ToolRuntime = None,
) -> Command:
    """审计固定六维覆盖；gap 的重开目标和 material_gap_found 由 controller 推导。"""
    if runtime is None:
        raise RuntimeError("audit_coverage 缺少 ToolRuntime")
    arguments = {
        "entries": [value.model_dump(mode="json") for value in entries],
        "rationale": rationale,
        "proposed_items": [value.model_dump(mode="json") for value in (proposed_items or [])],
    }
    if rejected := _rejection_if_unreachable(
        runtime=runtime,
        tool_name="audit_coverage",
        arguments=arguments,
    ):
        return rejected
    directive = _directive(runtime)
    dimensions = [value.dimension for value in entries]
    if set(dimensions) != set(ADAPTIVE_AUDIT_DIMENSIONS) or len(dimensions) != len(set(dimensions)):
        return _outcome_command(
            runtime=runtime,
            tool_name="audit_coverage",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="AUDIT_DIMENSION_SET_INVALID",
            message_for_model="coverage audit 必须恰好包含六个固定维度且每个一次。",
            retryable_by_model=True,
            recovery_action="REPAIR_AUDIT_ONCE",
        )
    unknown_aliases = sorted(
        {
            alias
            for entry in entries
            for alias in entry.investigation_aliases
            if alias.upper() not in directive.investigation_aliases
        }
    )
    if unknown_aliases:
        return _outcome_command(
            runtime=runtime,
            tool_name="audit_coverage",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_INVESTIGATION_ALIAS",
            message_for_model="未知 Investigation alias：" + "、".join(unknown_aliases),
            retryable_by_model=True,
            recovery_action="REPAIR_ALIAS_ONCE",
        )
    normalized_entries = [
        AdaptiveCoverageAuditEntry(
            dimension=value.dimension,
            status=value.status,
            investigation_ids=[directive.investigation_aliases[alias.upper()] for alias in value.investigation_aliases],
            rationale=value.rationale,
        )
        for value in entries
    ]
    gap_ids = list(
        dict.fromkeys(
            investigation_id
            for entry in normalized_entries
            if entry.status == "gap"
            for investigation_id in entry.investigation_ids
        )
    )
    drafts, unknown_evidence = _agenda_drafts(proposed_items or [], directive)
    if unknown_evidence:
        return _outcome_command(
            runtime=runtime,
            tool_name="audit_coverage",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="UNKNOWN_PARENT_EVIDENCE_ALIAS",
            message_for_model="未知 parent Evidence alias：" + "、".join(unknown_evidence),
            retryable_by_model=True,
        )
    material_gap = bool(gap_ids or drafts)
    if any(value.status == "gap" for value in normalized_entries) and not material_gap:
        return _outcome_command(
            runtime=runtime,
            tool_name="audit_coverage",
            arguments=arguments,
            semantic_outcome="INVALID_ARGUMENT",
            reason_code="GAP_WITHOUT_RECOVERY_TARGET",
            message_for_model="gap 必须引用需重开的 I alias，或提出新的实质调查。",
            retryable_by_model=True,
        )
    command = await submit_coverage_gap_assessment.coroutine(
        material_gap_found=material_gap,
        rationale=rationale,
        coverage_audit=normalized_entries,
        unsupported_investigation_ids=gap_ids,
        unsupported_obligations=None,
        proposed_items=drafts,
        runtime=runtime,
    )
    changed = isinstance(command.update, dict) and bool(command.update.get("adaptive_gap_assessments"))
    return _outcome_command(
        runtime=runtime,
        tool_name="audit_coverage",
        arguments=arguments,
        semantic_outcome="SUCCESS" if changed else "REJECTED",
        reason_code=None if changed else "AUDIT_CONTRACT_REJECTED",
        message_for_model="全局覆盖审计已保存。" if changed else _message_text(command),
        base_command=command,
        state_changed=changed,
        retryable_by_model=not changed,
        recovery_action=None if changed else "REPAIR_AUDIT_ONCE",
    )


BOUNDED_TOOLS = [
    propose_initial_agenda,
    extend_investigation_agenda_bounded,
    search_active_obligation,
    read_active_evidence,
    open_active_evidence,
    open_active_atlas_document,
    record_active_obligation_support,
    close_current_regimen_investigation,
    close_active_investigation,
    audit_coverage,
]
