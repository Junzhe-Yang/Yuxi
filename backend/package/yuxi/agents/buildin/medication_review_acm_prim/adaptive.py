from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from yuxi.agents.buildin.medication_review_prim.memory import (
    investigations_from_state,
    queries_from_state,
)

from .context import MedicationReviewAcmPrimContext, adaptive_coverage_enabled
from .models import (
    AdaptiveCheckpointRecord,
    AdaptiveCoverageReport,
    AdaptiveGapAssessment,
    AdaptiveInvestigationAgenda,
    AdaptiveInvestigationMeta,
    AdaptiveObligationSupport,
    AdaptiveProbeRecord,
    AdaptiveRecoveryRequirement,
)

SUCCESSFUL_SEARCH_STATUSES = {"success", "success_empty"}
ADAPTIVE_AUDIT_DIMENSIONS = (
    "indication_and_expected_benefit",
    "dose_route_frequency_duration_titration",
    "patient_specific_safety_contraindication_interaction",
    "monitoring_followup_and_stop_rules",
    "alternatives_and_missing_therapy",
    "cross_regimen_and_long_term_management",
)
ADAPTIVE_COVERAGE_INSTRUCTIONS = (
    "调查数量和搜索数量都不是完成目标。首次建议程时，一次列出所有可能实质改变最终"
    "判断、彼此有区别且值得检索的问题。每个 PlanAnchor 必须有独立的 "
    "current_regimen_review 并判定 appropriate/adjust/avoid；adjust/avoid 后必须追加"
    "关联同一 PlanAnchor 的 improvement_plan。现用药审查既要找问题，也要保留有直接"
    "证据支持的合理治疗。每项调查列出可由直接证据回答的最小 evidence_obligations；"
    "适应性、剂量/疗程、患者特异风险、监测/停药、替代/缺失治疗和非药物干预不能被"
    "一句笼统结论跳过。每个义务必须有定向真实检索和 obligation_supports，或在实际"
    "尝试后明确保留语料不足边界。临床问题看似能回答不等于证据义务已经覆盖。"
)
ADAPTIVE_QUERY_INSTRUCTIONS = (
    "当前 search 只处理一个 evidence obligation。query_text 使用空格分隔的中文关键词，"
    "最多 6 个概念块：一个核心主体、一个待查属性，最多再加一个真正改变检索方向的患者"
    "条件。不要复制 obligation 全句，不要同时混入适应性、剂量/疗程、安全性、监测、"
    "替代方案或长期管理中的多个属性轴，也不要枚举预计答案、多个候选药或无关条件。"
    "source_discovery/global 用于发现新来源；当前调查已有可信 candidate file_id 且仍缺"
    "直接证据时，优先 within_document_localization/document。相邻原文使用 "
    "open_review_evidence；文档内零新增后的 pending recovery 仍必须由 "
    "source_discovery/global 处理。"
)
ADAPTIVE_REVIEW_INSTRUCTIONS = (
    "关闭前核对当前 Investigation 的全部 evidence obligations。answered 仍要求每项义务"
    "完成定向成功 probe、绑定该义务 probe 返回或相邻打开的真实 Evidence，并且没有"
    "pending recovery；insufficient 仍要求实际尝试所有可执行义务并记录残余不确定性和"
    "停止理由。不得因一条候选 Evidence 看似足够就跳过其它义务。"
)
ADAPTIVE_AUDIT_INSTRUCTIONS = (
    "全部调查关闭后才能提交结构化全局缺口审计。六个维度必须逐项 covered、"
    "not_applicable 或 gap；gap 必须重开旧调查或追加实质性新调查。"
)

ADAPTIVE_QUERY_MAX_CONCEPTS = 6
_ADAPTIVE_QUERY_CONCEPT_SEPARATOR = re.compile(r"[\s,，、/|;；]+")
_ADAPTIVE_QUERY_AXES: dict[str, tuple[str, ...]] = {
    "适应性/疗效": ("适应证", "适应症", "疗效", "获益", "有效性", "推荐地位"),
    "剂量/疗程": (
        "剂量",
        "用量",
        "起始量",
        "目标量",
        "加量",
        "减量",
        "滴定",
        "频次",
        "疗程",
        "使用时间",
        "用药时间",
        "停药",
        "停用",
    ),
    "安全性": ("安全性", "禁忌", "慎用", "风险", "不良反应", "副作用", "相互作用", "毒性"),
    "监测/随访": ("监测", "随访", "复诊", "复查", "筛查", "观察指标"),
    "替代/缺失治疗": ("替代", "备选", "换药", "加用", "药物选择", "治疗选择", "非药物"),
    "长期管理": ("长期管理", "二级预防", "复发预防", "长期预防"),
}


def adaptive_agenda_from_state(
    state: dict[str, Any],
) -> AdaptiveInvestigationAgenda | None:
    raw = state.get("adaptive_agenda")
    if raw is None:
        return None
    try:
        return raw if isinstance(raw, AdaptiveInvestigationAgenda) else AdaptiveInvestigationAgenda.model_validate(raw)
    except Exception:  # noqa: BLE001 - malformed state stays trace-local
        return None


def adaptive_meta_from_state(
    state: dict[str, Any],
) -> list[AdaptiveInvestigationMeta]:
    result: list[AdaptiveInvestigationMeta] = []
    for raw in state.get("adaptive_investigation_meta") or []:
        try:
            result.append(
                raw if isinstance(raw, AdaptiveInvestigationMeta) else AdaptiveInvestigationMeta.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return result


def adaptive_probe_records_from_state(
    state: dict[str, Any],
) -> list[AdaptiveProbeRecord]:
    result: list[AdaptiveProbeRecord] = []
    for raw in state.get("adaptive_probe_records") or []:
        try:
            result.append(raw if isinstance(raw, AdaptiveProbeRecord) else AdaptiveProbeRecord.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    return result


def adaptive_recoveries_from_state(
    state: dict[str, Any],
) -> list[AdaptiveRecoveryRequirement]:
    result: list[AdaptiveRecoveryRequirement] = []
    for raw in state.get("adaptive_recovery_requirements") or []:
        try:
            result.append(
                raw if isinstance(raw, AdaptiveRecoveryRequirement) else AdaptiveRecoveryRequirement.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return result


def adaptive_gap_assessments_from_state(
    state: dict[str, Any],
) -> list[AdaptiveGapAssessment]:
    result: list[AdaptiveGapAssessment] = []
    for raw in state.get("adaptive_gap_assessments") or []:
        try:
            result.append(raw if isinstance(raw, AdaptiveGapAssessment) else AdaptiveGapAssessment.model_validate(raw))
        except Exception:  # noqa: BLE001
            continue
    return result


def adaptive_checkpoint_records_from_state(
    state: dict[str, Any],
) -> list[AdaptiveCheckpointRecord]:
    result: list[AdaptiveCheckpointRecord] = []
    for raw in state.get("adaptive_checkpoint_records") or []:
        try:
            result.append(
                raw if isinstance(raw, AdaptiveCheckpointRecord) else AdaptiveCheckpointRecord.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return result


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def adaptive_query_contract_violations(query_text: str) -> list[str]:
    """Return deterministic query-shape errors without judging medical content."""
    normalized = query_text.strip()
    concepts = [value for value in _ADAPTIVE_QUERY_CONCEPT_SEPARATOR.split(normalized) if value]
    violations: list[str] = []
    if len(concepts) > ADAPTIVE_QUERY_MAX_CONCEPTS:
        violations.append(
            f"query_text 最多 {ADAPTIVE_QUERY_MAX_CONCEPTS} 个空格分隔的概念块，当前为 {len(concepts)} 个"
        )

    detected_axes = [
        axis
        for axis, keywords in _ADAPTIVE_QUERY_AXES.items()
        if any(keyword.casefold() in normalized.casefold() for keyword in keywords)
    ]
    if len(detected_axes) > 1:
        violations.append("query_text 必须只保留单一待查属性；当前同时包含：" + "、".join(detected_axes))
    return violations


def adaptive_obligation_support_map(
    meta: AdaptiveInvestigationMeta | None,
) -> dict[str, AdaptiveObligationSupport]:
    if meta is None:
        return {}
    return {
        normalize_text(value.obligation): value
        for value in meta.obligation_supports
        if normalize_text(value.obligation)
    }


def adaptive_successful_obligation_probes(
    state: dict[str, Any],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for value in adaptive_probe_records_from_state(state):
        if value.status not in SUCCESSFUL_SEARCH_STATUSES:
            continue
        normalized = normalize_text(value.uncovered_aspect)
        if normalized:
            result.setdefault(value.investigation_id, set()).add(normalized)
    return result


def adaptive_attempted_obligation_probes(
    state: dict[str, Any],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for value in adaptive_probe_records_from_state(state):
        normalized = normalize_text(value.uncovered_aspect)
        if normalized:
            result.setdefault(value.investigation_id, set()).add(normalized)
    return result


def adaptive_route_key(
    *,
    investigation_id: str,
    uncovered_aspect: str,
    retrieval_scope: str,
    file_id: str | None,
) -> str:
    payload = {
        "investigation_id": investigation_id.strip().upper(),
        "uncovered_aspect": normalize_text(uncovered_aspect),
        "retrieval_scope": retrieval_scope,
        "file_id": (file_id or "").strip(),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _plan_ids(state: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for raw in state.get("plan_anchors") or []:
        value = raw.get("element_id") if isinstance(raw, dict) else getattr(raw, "element_id", None)
        if value:
            values.append(str(value))
    return list(dict.fromkeys(values))


def build_adaptive_state_fingerprint(state: dict[str, Any]) -> str:
    agenda = adaptive_agenda_from_state(state)
    investigations = investigations_from_state(state)
    queries = queries_from_state(state)
    meta = adaptive_meta_from_state(state)
    recoveries = adaptive_recoveries_from_state(state)
    payload = {
        "agenda": agenda.model_dump(mode="json") if agenda else None,
        "investigations": [
            {
                "id": value.investigation_id,
                "status": value.status,
                "query_ids": value.query_ids,
                "candidate_evidence_ids": value.candidate_evidence_ids,
                "candidate_file_ids": value.candidate_file_ids,
                "selected_evidence_ids": value.selected_evidence_ids,
                "working_note": value.working_note,
            }
            for value in investigations
        ],
        "meta": [value.model_dump(mode="json") for value in sorted(meta, key=lambda item: item.investigation_id)],
        "queries": [
            {
                "id": value.query_id,
                "investigation_id": value.investigation_id,
                "status": value.status,
                "scope": value.retrieval_scope,
                "file_id": value.file_id,
                "evidence_ids": value.evidence_ids,
                "new_evidence_ids": value.new_evidence_ids,
            }
            for value in queries
        ],
        "recoveries": [
            value.model_dump(mode="json") for value in sorted(recoveries, key=lambda item: item.recovery_id)
        ],
        "evidence_ids": sorted(
            str(value)
            for value in (state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {})
        ),
        "open_records": [
            {
                "record_id": (value.get("record_id") if isinstance(value, dict) else getattr(value, "record_id", None)),
                "status": (value.get("status") if isinstance(value, dict) else getattr(value, "status", None)),
                "evidence_ids": (
                    value.get("evidence_ids") if isinstance(value, dict) else getattr(value, "evidence_ids", [])
                ),
                "new_evidence_ids": (
                    value.get("new_evidence_ids") if isinstance(value, dict) else getattr(value, "new_evidence_ids", [])
                ),
            }
            for value in state.get("open_records") or []
        ],
        "atlas_document_open_ids": [
            (value.get("record_id") if isinstance(value, dict) else getattr(value, "record_id", None))
            for value in state.get("atlas_document_open_records") or []
        ],
        "plan_ids": _plan_ids(state),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_adaptive_coverage_report(
    state: dict[str, Any],
    context: MedicationReviewAcmPrimContext,
) -> AdaptiveCoverageReport:
    executed_search_calls = int(state.get("search_count") or 0)
    successful_queries = [value for value in queries_from_state(state) if value.status in SUCCESSFUL_SEARCH_STATUSES]
    fingerprint = build_adaptive_state_fingerprint(state)
    if not adaptive_coverage_enabled(context):
        return AdaptiveCoverageReport(
            status="disabled",
            state_fingerprint=fingerprint,
            executed_search_calls=executed_search_calls,
            successful_search_calls=len(successful_queries),
            maximum_search_calls=context.max_search_calls,
        )

    agenda = adaptive_agenda_from_state(state)
    agenda_ids = [value.investigation_id for value in agenda.items] if agenda else []
    investigation_by_id = {value.investigation_id: value for value in investigations_from_state(state)}
    active_items = [
        value
        for value in (agenda.items if agenda else [])
        if investigation_by_id.get(value.investigation_id) is None
        or investigation_by_id[value.investigation_id].status != "dismissed"
    ]
    meta_by_id = {value.investigation_id: value for value in adaptive_meta_from_state(state)}
    covered_plan_ids = list(dict.fromkeys(plan_id for item in active_items for plan_id in item.focus_plan_ids))
    plan_ids = _plan_ids(state)
    current_regimen_items = [value for value in active_items if value.investigation_kind == "current_regimen_review"]
    current_regimen_reviewed_plan_ids = list(
        dict.fromkeys(plan_id for item in current_regimen_items for plan_id in item.focus_plan_ids)
    )
    missing_current_regimen_review_plan_ids = [
        value for value in plan_ids if value not in current_regimen_reviewed_plan_ids
    ]
    uncovered_plan_ids = list(missing_current_regimen_review_plan_ids)
    action_required_plan_ids = list(
        dict.fromkeys(
            plan_id
            for item in current_regimen_items
            if (meta_by_id.get(item.investigation_id) is not None)
            and meta_by_id[item.investigation_id].review_outcome in {"adjust", "avoid"}
            for plan_id in item.focus_plan_ids
        )
    )
    improvement_plan_covered_plan_ids = list(
        dict.fromkeys(
            plan_id
            for item in active_items
            if item.investigation_kind == "improvement_plan"
            for plan_id in item.focus_plan_ids
        )
    )
    missing_improvement_plan_ids = [
        value for value in action_required_plan_ids if value not in improvement_plan_covered_plan_ids
    ]
    attempted_by_investigation = {value.investigation_id for value in adaptive_probe_records_from_state(state)}
    successful_by_investigation = {value.investigation_id for value in successful_queries if value.investigation_id}
    unprobed = [
        value
        for value in agenda_ids
        if value not in successful_by_investigation
        and not (
            investigation_by_id.get(value) is not None
            and investigation_by_id[value].status == "insufficient"
            and value in attempted_by_investigation
        )
    ]
    open_ids = [
        value for value in agenda_ids if value not in investigation_by_id or investigation_by_id[value].status == "open"
    ]
    pending_recoveries = [value for value in adaptive_recoveries_from_state(state) if value.status == "pending"]
    pending_recovery_ids = [value.recovery_id for value in pending_recoveries]
    pending_investigation_ids = list(dict.fromkeys(value.investigation_id for value in pending_recoveries))
    probed_obligations = adaptive_successful_obligation_probes(state)
    attempted_obligations = adaptive_attempted_obligation_probes(state)
    unprobed_obligations: dict[str, list[str]] = {}
    unsupported_obligations: dict[str, list[str]] = {}
    items_without_obligations: list[str] = []
    total_obligation_count = 0
    supported_obligation_count = 0
    for item in active_items:
        obligations = list(dict.fromkeys(value.strip() for value in item.evidence_obligations if value.strip()))
        if not obligations:
            items_without_obligations.append(item.investigation_id)
            continue
        total_obligation_count += len(obligations)
        support_map = adaptive_obligation_support_map(meta_by_id.get(item.investigation_id))
        supported = [value for value in obligations if normalize_text(value) in support_map]
        supported_obligation_count += len(supported)
        investigation = investigation_by_id.get(item.investigation_id)
        obligation_probe_basis = (
            attempted_obligations.get(item.investigation_id, set())
            if investigation is not None and investigation.status == "insufficient"
            else probed_obligations.get(item.investigation_id, set())
        )
        not_probed = [value for value in obligations if normalize_text(value) not in obligation_probe_basis]
        if not_probed:
            unprobed_obligations[item.investigation_id] = not_probed
        missing_support = [value for value in obligations if normalize_text(value) not in support_map]
        item_meta = meta_by_id.get(item.investigation_id)
        if investigation is not None and investigation.status == "insufficient":
            acknowledged = {
                normalize_text(value)
                for value in (item_meta.remaining_aspects if item_meta else [])
                if normalize_text(value)
            }
            missing_support = [value for value in missing_support if normalize_text(value) not in acknowledged]
        if missing_support:
            unsupported_obligations[item.investigation_id] = missing_support
    obligation_priority_ids = [
        value.investigation_id for value in active_items if value.investigation_id in unprobed_obligations
    ]
    current_regimen_unprobed_ids = [
        value.investigation_id for value in current_regimen_items if value.investigation_id in unprobed
    ]
    eligible_ids = (
        pending_investigation_ids
        if pending_investigation_ids
        else current_regimen_unprobed_ids
        if current_regimen_unprobed_ids
        else unprobed
        if unprobed
        else obligation_priority_ids
        if obligation_priority_ids
        else open_ids
    )

    assessments = adaptive_gap_assessments_from_state(state)
    latest = assessments[-1] if assessments else None
    if latest is None:
        gap_status = "missing"
    elif latest.state_fingerprint == fingerprint:
        gap_status = "current"
    else:
        gap_status = "stale"

    reasons: list[str] = []
    if agenda is None or not agenda_ids:
        reasons.append("agenda_missing")
    if items_without_obligations:
        reasons.append("evidence_obligations_missing")
    if missing_current_regimen_review_plan_ids:
        reasons.append("current_regimen_review_missing")
    if missing_improvement_plan_ids:
        reasons.append("improvement_plan_missing")
    if unprobed:
        reasons.append("investigations_unprobed")
    if unprobed_obligations:
        reasons.append("evidence_obligations_unprobed")
    if open_ids:
        reasons.append("investigations_open")
    if unsupported_obligations:
        reasons.append("evidence_obligations_unsupported")
    if pending_recovery_ids:
        reasons.append("global_recovery_pending")
    if gap_status == "missing":
        reasons.append("gap_assessment_missing")
    elif gap_status == "stale":
        reasons.append("gap_assessment_stale")
    elif latest is not None and (latest.material_gap_found or latest.proposed_investigation_ids):
        reasons.append("material_gap_reported")

    return AdaptiveCoverageReport(
        status="completed" if not reasons else "incomplete",
        incomplete_reasons=reasons,
        agenda_created=agenda is not None,
        agenda_investigation_ids=agenda_ids,
        covered_plan_ids=covered_plan_ids,
        uncovered_plan_ids=uncovered_plan_ids,
        current_regimen_reviewed_plan_ids=current_regimen_reviewed_plan_ids,
        missing_current_regimen_review_plan_ids=missing_current_regimen_review_plan_ids,
        action_required_plan_ids=action_required_plan_ids,
        improvement_plan_covered_plan_ids=improvement_plan_covered_plan_ids,
        missing_improvement_plan_ids=missing_improvement_plan_ids,
        unprobed_investigation_ids=unprobed,
        open_investigation_ids=open_ids,
        pending_recovery_ids=pending_recovery_ids,
        eligible_investigation_ids=eligible_ids,
        unprobed_evidence_obligations=unprobed_obligations,
        unsupported_evidence_obligations=unsupported_obligations,
        total_evidence_obligation_count=total_obligation_count,
        supported_evidence_obligation_count=supported_obligation_count,
        gap_assessment_status=gap_status,
        latest_gap_assessment_id=(latest.assessment_id if latest else None),
        state_fingerprint=fingerprint,
        actual_investigation_count=len(agenda_ids),
        executed_search_calls=executed_search_calls,
        successful_search_calls=len(successful_queries),
        maximum_search_calls=context.max_search_calls,
    )


def adaptive_memory_phase(report: AdaptiveCoverageReport) -> str:
    if (
        not report.agenda_created
        or report.missing_current_regimen_review_plan_ids
        or report.missing_improvement_plan_ids
    ):
        return "agenda"
    if report.pending_recovery_ids or report.unprobed_investigation_ids or report.unprobed_evidence_obligations:
        return "search"
    if report.open_investigation_ids or report.unsupported_evidence_obligations:
        return "review"
    if report.status != "completed":
        return "audit"
    return "complete"


def build_adaptive_contract_memory(report: AdaptiveCoverageReport) -> str:
    if report.status == "disabled":
        return ""
    lines = [
        "【ACM 自适应调查覆盖】",
        (
            f"调查 {report.actual_investigation_count} 项；"
            f"证据义务已支持 {report.supported_evidence_obligation_count}/"
            f"{report.total_evidence_obligation_count}；"
            f"搜索已执行 {report.executed_search_calls} 次。"
            "调查数和搜索数不是完成目标。"
        ),
    ]
    if not report.agenda_created:
        lines.append("尚未建立调查议程。")
    if report.missing_current_regimen_review_plan_ids:
        lines.append("缺 current_regimen_review：" + "、".join(report.missing_current_regimen_review_plan_ids))
    if report.missing_improvement_plan_ids:
        lines.append("缺 improvement_plan：" + "、".join(report.missing_improvement_plan_ids))
    if report.agenda_created:
        lines.append(
            "未首次检索调查="
            f"{len(report.unprobed_investigation_ids)}；"
            "未定向检索义务="
            f"{sum(len(values) for values in report.unprobed_evidence_obligations.values())}；"
            "未绑定直接 Evidence 义务="
            f"{sum(len(values) for values in report.unsupported_evidence_obligations.values())}；"
            f"开放调查={len(report.open_investigation_ids)}；"
            f"待全库恢复={len(report.pending_recovery_ids)}；"
            f"全局缺口审计={report.gap_assessment_status}。"
        )
    if report.status == "completed":
        lines.append("覆盖合同已完成；可以生成最终答案。")
    else:
        lines.append("当前未完成原因：" + "、".join(report.incomplete_reasons))
    return "\n".join(lines)


def build_adaptive_investigation_memory(
    state: dict[str, Any],
    report: AdaptiveCoverageReport,
) -> str:
    """Render a phase-specific adaptive view without changing coverage state."""
    contract = build_adaptive_contract_memory(report)
    if report.status == "disabled":
        return contract

    phase = adaptive_memory_phase(report)
    lines = ["【ACM 自适应覆盖协议】", ADAPTIVE_COVERAGE_INSTRUCTIONS]
    if phase == "agenda":
        lines.extend(
            [
                "当前阶段：建立或补充调查议程",
                "先覆盖每个 PlanAnchor 的 current_regimen_review；发现 adjust/avoid 后补充 "
                "improvement_plan。evidence_obligations 必须是彼此不同、可由直接证据回答的最小问题。",
            ]
        )
    elif phase == "search":
        lines.extend(["当前阶段：定向检索", ADAPTIVE_QUERY_INSTRUCTIONS])
    elif phase == "review":
        lines.extend(["当前阶段：证据核对与关闭", ADAPTIVE_REVIEW_INSTRUCTIONS])
    elif phase == "audit":
        lines.extend(["当前阶段：全局缺口审计", ADAPTIVE_AUDIT_INSTRUCTIONS])
    else:
        lines.append("当前阶段：覆盖合同已完成")

    agenda = adaptive_agenda_from_state(state)
    if agenda is None:
        return "\n".join([*lines, contract])

    investigations = {value.investigation_id: value for value in investigations_from_state(state)}
    meta = {value.investigation_id: value for value in adaptive_meta_from_state(state)}
    probed_obligations = adaptive_successful_obligation_probes(state)
    pending_recoveries = [value for value in adaptive_recoveries_from_state(state) if value.status == "pending"]
    item_by_id = {value.investigation_id: value for value in agenda.items}
    active_id = next(
        iter(
            [
                *report.eligible_investigation_ids,
                *report.open_investigation_ids,
                *report.unsupported_evidence_obligations,
            ]
        ),
        None,
    )
    active_item = item_by_id.get(active_id) if active_id else None

    def obligation_statuses(item) -> list[str]:
        item_meta = meta.get(item.investigation_id)
        support_map = adaptive_obligation_support_map(item_meta)
        result: list[str] = []
        for obligation in item.evidence_obligations:
            normalized_obligation = normalize_text(obligation)
            support = support_map.get(normalized_obligation)
            if support is not None:
                result.append(f"已支持：{obligation} -> {','.join(support.evidence_ids)}")
            elif normalized_obligation in probed_obligations.get(item.investigation_id, set()):
                result.append(f"已检索但待直接证据：{obligation}")
            else:
                result.append(f"待定向检索：{obligation}")
        return result

    def append_full_item(item) -> None:
        investigation = investigations.get(item.investigation_id)
        item_meta = meta.get(item.investigation_id)
        status = investigation.status if investigation else "missing"
        selected = (
            "、".join(investigation.selected_evidence_ids)
            if investigation and investigation.selected_evidence_ids
            else "无"
        )
        candidate_files = (
            "、".join(investigation.candidate_file_ids) if investigation and investigation.candidate_file_ids else "无"
        )
        remaining = "；".join(item_meta.remaining_aspects) if item_meta and item_meta.remaining_aspects else "无"
        lines.extend(
            [
                f"[{item.investigation_id}] kind={item.investigation_kind or 'legacy'}；"
                f"status={status}；review_outcome={item_meta.review_outcome if item_meta else '未记录'}",
                f"问题={item.question}；意义={item.why_it_matters}；scope={item.distinct_scope}",
                f"remaining={remaining}；selected Evidence={selected}；candidate file_id={candidate_files}",
                "证据义务：",
                *(f"- {value}" for value in obligation_statuses(item)),
            ]
        )

    if phase == "search" and active_item is not None:
        active_recovery = next(
            (value for value in pending_recoveries if value.investigation_id == active_item.investigation_id),
            None,
        )
        target_obligation = (
            active_recovery.uncovered_aspect
            if active_recovery is not None
            else next(
                iter(report.unprobed_evidence_obligations.get(active_item.investigation_id, [])),
                next(
                    iter(report.unsupported_evidence_obligations.get(active_item.investigation_id, [])),
                    active_item.evidence_obligations[0] if active_item.evidence_obligations else "",
                ),
            )
        )
        investigation = investigations.get(active_item.investigation_id)
        candidate_files = (
            "、".join(investigation.candidate_file_ids) if investigation and investigation.candidate_file_ids else "无"
        )
        lines.extend(
            [
                "【当前检索焦点】",
                f"investigation={active_item.investigation_id}；kind={active_item.investigation_kind or 'legacy'}",
                f"问题={active_item.question}",
                f"当前必须处理的 evidence obligation={target_obligation}",
                f"candidate file_id={candidate_files}",
                "本轮不要展开或改写其它 obligation；完成本轮工具结果后，控制器会继续调度。",
            ]
        )
        if active_recovery is not None:
            lines.extend(
                [
                    "当前强制全库恢复任务：",
                    f"- [{active_recovery.recovery_id}] 原 file_id={active_recovery.file_id or '无'}；"
                    f"未覆盖方面={active_recovery.uncovered_aspect}",
                ]
            )
        recent_probes = [
            value
            for value in adaptive_probe_records_from_state(state)
            if value.investigation_id == active_item.investigation_id
            and normalize_text(value.uncovered_aspect) == normalize_text(target_obligation)
        ][-3:]
        if recent_probes:
            lines.append("当前 obligation 的最近路线：")
            lines.extend(
                f"- [{value.query_id}] intent={value.retrieval_intent}；"
                f"status={value.status}；redundant={value.redundant}"
                for value in recent_probes
            )
    elif phase == "review" and active_item is not None:
        lines.append("【当前待核对调查】")
        append_full_item(active_item)
    elif phase in {"agenda", "audit"}:
        lines.append(f"当前议程：{agenda.agenda_id} revision={agenda.revision}")
        for item in agenda.items:
            append_full_item(item)

    if phase in {"search", "review", "complete"}:
        lines.append("【其他调查状态】")
        for item in agenda.items:
            if active_item is not None and item.investigation_id == active_item.investigation_id:
                continue
            investigation = investigations.get(item.investigation_id)
            support_count = len(adaptive_obligation_support_map(meta.get(item.investigation_id)))
            lines.append(
                f"- [{item.investigation_id}] kind={item.investigation_kind or 'legacy'}；"
                f"status={investigation.status if investigation else 'missing'}；"
                f"义务支持={support_count}/{len(item.evidence_obligations)}"
            )

    assessments = adaptive_gap_assessments_from_state(state)
    if assessments and phase in {"audit", "complete"}:
        latest = assessments[-1]
        lines.append(
            f"最近全局缺口审计 [{latest.assessment_id}]："
            f"material_gap_found={latest.material_gap_found}；rationale={latest.rationale}"
        )
    if phase == "audit":
        lines.append("coverage_audit 必须逐项覆盖：" + "、".join(ADAPTIVE_AUDIT_DIMENSIONS))
    return "\n".join([*lines, contract])
