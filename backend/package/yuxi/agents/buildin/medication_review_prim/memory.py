from __future__ import annotations

from typing import Any

from yuxi.agents.buildin.medication_review_lite.models import PlanAnchor

from .models import (
    ExperimentProfile,
    InvestigationItem,
    PatientModifier,
    QueryRecord,
)


def _as_anchor(value: PlanAnchor | dict[str, Any]) -> PlanAnchor:
    return (
        value
        if isinstance(value, PlanAnchor)
        else PlanAnchor.model_validate(value)
    )


def _as_modifier(
    value: PatientModifier | dict[str, Any],
) -> PatientModifier:
    return (
        value
        if isinstance(value, PatientModifier)
        else PatientModifier.model_validate(value)
    )


def _as_query(value: QueryRecord | dict[str, Any]) -> QueryRecord:
    return (
        value
        if isinstance(value, QueryRecord)
        else QueryRecord.model_validate(value)
    )


def _as_investigation(
    value: InvestigationItem | dict[str, Any],
) -> InvestigationItem:
    return (
        value
        if isinstance(value, InvestigationItem)
        else InvestigationItem.model_validate(value)
    )


def anchors_from_state(state: dict[str, Any]) -> list[PlanAnchor]:
    values: list[PlanAnchor] = []
    for raw in state.get("plan_anchors") or []:
        try:
            values.append(_as_anchor(raw))
        except Exception:  # noqa: BLE001 - corrupted state stays trace-local
            continue
    return sorted(
        values,
        key=lambda value: (value.source_start, value.element_id),
    )


def modifiers_from_state(state: dict[str, Any]) -> list[PatientModifier]:
    values: list[PatientModifier] = []
    for raw in state.get("patient_modifiers") or []:
        try:
            values.append(_as_modifier(raw))
        except Exception:  # noqa: BLE001
            continue
    return sorted(
        values,
        key=lambda value: (value.source_start, value.modifier_id),
    )


def queries_from_state(state: dict[str, Any]) -> list[QueryRecord]:
    values: list[QueryRecord] = []
    for raw in state.get("query_records") or []:
        try:
            values.append(_as_query(raw))
        except Exception:  # noqa: BLE001
            continue
    return sorted(
        values,
        key=lambda value: (value.started_at, value.query_id),
    )


def investigations_from_state(
    state: dict[str, Any],
) -> list[InvestigationItem]:
    values: list[InvestigationItem] = []
    for raw in state.get("investigations") or []:
        try:
            values.append(_as_investigation(raw))
        except Exception:  # noqa: BLE001
            continue
    return sorted(
        values,
        key=lambda value: (value.created_at, value.investigation_id),
    )


def uninvestigated_plan_ids(
    anchors: list[PlanAnchor],
    investigations: list[InvestigationItem],
) -> list[str]:
    focused = {
        element_id
        for investigation in investigations
        for element_id in investigation.focus_plan_ids
    }
    return [
        anchor.element_id
        for anchor in anchors
        if anchor.element_id not in focused
    ]


def _node_sections(
    *,
    anchors: list[PlanAnchor],
    modifiers: list[PatientModifier],
) -> list[str]:
    sections: list[str] = []
    if anchors:
        sections.append(
            "【明确治疗方案要素】\n"
            + "\n".join(
                f"- [{anchor.element_id}] {anchor.label}；原文：{anchor.source_span}"
                for anchor in anchors
            )
        )
    if modifiers:
        sections.append(
            "【患者事实线索】\n"
            + "\n".join(
                f"- [{modifier.modifier_id}] {modifier.source_span}"
                for modifier in modifiers
            )
        )
    return sections


def build_case_node_memory(
    *,
    profile: ExperimentProfile,
    anchors: list[PlanAnchor],
    modifiers: list[PatientModifier],
) -> str:
    """Render only stable case nodes, without investigation or query history."""
    if profile == "b1":
        return ""
    visible_modifiers = modifiers if profile in {"m2", "m3", "full"} else []
    return "\n\n".join(
        _node_sections(
            anchors=anchors,
            modifiers=visible_modifiers,
        )
    )


def _query_line(record: QueryRecord) -> str:
    scope = (
        f"文档内({record.file_id})"
        if record.retrieval_scope == "document"
        else "全库"
    )
    evidence = "、".join(record.evidence_ids) if record.evidence_ids else "无"
    return (
        f"- [{record.query_id}] {scope}；{record.query_text}；"
        f"结果={record.status}；候选={evidence}"
    )


def _investigation_block(
    investigation: InvestigationItem,
    *,
    query_by_id: dict[str, QueryRecord],
) -> str:
    focus = [
        *investigation.focus_plan_ids,
        *investigation.focus_modifier_ids,
    ]
    evidence = (
        "、".join(investigation.candidate_evidence_ids)
        if investigation.candidate_evidence_ids
        else "无"
    )
    selected = (
        "、".join(investigation.selected_evidence_ids)
        if investigation.selected_evidence_ids
        else "无"
    )
    documents = (
        "、".join(investigation.candidate_file_ids)
        if investigation.candidate_file_ids
        else "无"
    )
    query_lines = [
        _query_line(query_by_id[query_id])
        for query_id in investigation.query_ids[-3:]
        if query_id in query_by_id
    ]
    lines = [
        f"[{investigation.investigation_id}] {investigation.status}",
        f"问题：{investigation.question}",
        "病例对象：" + ("、".join(focus) if focus else "未绑定"),
        f"候选 Evidence：{evidence}",
        f"已选 Evidence：{selected}",
        f"候选文档 file_id：{documents}",
    ]
    if investigation.working_note:
        lines.append(f"当前记录：{investigation.working_note}")
    if query_lines:
        lines.append("最近查询：\n" + "\n".join(query_lines))
    return "\n".join(lines)


def build_investigation_memory(
    *,
    profile: ExperimentProfile,
    anchors: list[PlanAnchor],
    modifiers: list[PatientModifier],
    queries: list[QueryRecord],
    investigations: list[InvestigationItem],
) -> str:
    if profile == "b1":
        return ""

    case_node_memory = build_case_node_memory(
        profile=profile,
        anchors=anchors,
        modifiers=modifiers,
    )
    sections = [case_node_memory] if case_node_memory else []

    if profile in {"m1", "m2"}:
        if queries:
            sections.append(
                "【最近查询】\n"
                + "\n".join(_query_line(record) for record in queries[-5:])
            )
        sections.append(
            "提示：以上只记录病例节点和查询历史；搜索返回片段不代表"
            "已经回答任何临床问题。"
        )
        return "\n\n".join(sections)

    query_by_id = {record.query_id: record for record in queries}
    open_items = [
        value for value in investigations if value.status == "open"
    ]
    closed_items = [
        value for value in investigations if value.status != "open"
    ]
    if open_items:
        sections.append(
            "【当前待解决的证据问题】\n"
            + "\n\n".join(
                _investigation_block(value, query_by_id=query_by_id)
                for value in open_items
            )
        )
    if closed_items:
        sections.append(
            "【已处理的证据问题】\n"
            + "\n".join(
                f"- [{value.investigation_id}] {value.status}；"
                f"{value.question}；采用证据："
                f"{'、'.join(value.selected_evidence_ids) or '无'}"
                for value in closed_items
            )
        )

    missing_plan_ids = uninvestigated_plan_ids(anchors, investigations)
    if missing_plan_ids:
        anchor_by_id = {value.element_id: value for value in anchors}
        sections.append(
            "【尚未关联任何调查的方案要素】\n"
            + "\n".join(
                f"- [{element_id}] {anchor_by_id[element_id].label}"
                for element_id in missing_plan_ids
            )
        )

    attached_queries = {
        query_id
        for investigation in investigations
        for query_id in investigation.query_ids
    }
    ungrouped = [
        record
        for record in queries[-5:]
        if record.query_id not in attached_queries
    ]
    if ungrouped:
        sections.append(
            "【未关联调查的最近查询】\n"
            + "\n".join(_query_line(record) for record in ungrouped)
        )

    sections.append(
        "提示：Investigation 只是 Agent 正在回答的证据问题，不表示医学"
        "关系已经成立。候选 Evidence 也不等于可用证据。阅读结果后，请"
        "使用 update_investigation 记录当前缺口、选择 Evidence，并明确"
        "继续调查、可以回答、证据不足或与病例无关。"
    )
    return "\n\n".join(sections)
