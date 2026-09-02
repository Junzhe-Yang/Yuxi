from __future__ import annotations

import hashlib
import json
from typing import Any

from yuxi.agents.buildin.medication_review_prim.memory import (
    investigations_from_state,
    queries_from_state,
)

from .context import (
    V7_MINIMUM_SEARCH_CALLS,
    MedicationReviewAcmPrimContext,
    v7_required_investigation_count,
    v7_retrieval_depths,
)
from .models import (
    V7CheckpointRecord,
    V7ContractReport,
    V7InvestigationAgenda,
    V7ProbeRecord,
)

SUCCESSFUL_SEARCH_STATUSES = {"success", "success_empty"}


def v7_contract_enabled(context: MedicationReviewAcmPrimContext) -> bool:
    return context.v7_experiment_arm != "a0"


def v7_trace_enabled(context: MedicationReviewAcmPrimContext) -> bool:
    return v7_contract_enabled(context) or context.v7_retrieval_depth != "top10"


def configure_v7_retrieval(context: MedicationReviewAcmPrimContext) -> None:
    fetch_k, visible_k = v7_retrieval_depths(context.v7_retrieval_depth)
    setattr(context, "_prim_retrieval_fetch_k", fetch_k)
    setattr(context, "_prim_retrieval_visible_k", visible_k)
    if context.v7_retrieval_depth == "top10":
        if hasattr(context, "_prim_retrieval_diagnostic_state_key"):
            delattr(context, "_prim_retrieval_diagnostic_state_key")
    else:
        setattr(
            context,
            "_prim_retrieval_diagnostic_state_key",
            "v7_retrieval_records",
        )


def agenda_from_state(state: dict[str, Any]) -> V7InvestigationAgenda | None:
    raw = state.get("v7_agenda")
    if raw is None:
        return None
    try:
        return (
            raw
            if isinstance(raw, V7InvestigationAgenda)
            else V7InvestigationAgenda.model_validate(raw)
        )
    except Exception:  # noqa: BLE001 - malformed state stays trace-local
        return None


def probe_records_from_state(state: dict[str, Any]) -> list[V7ProbeRecord]:
    result: list[V7ProbeRecord] = []
    for raw in state.get("v7_probe_records") or []:
        try:
            result.append(
                raw
                if isinstance(raw, V7ProbeRecord)
                else V7ProbeRecord.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return result


def checkpoint_records_from_state(
    state: dict[str, Any],
) -> list[V7CheckpointRecord]:
    result: list[V7CheckpointRecord] = []
    for raw in state.get("v7_checkpoint_records") or []:
        try:
            result.append(
                raw
                if isinstance(raw, V7CheckpointRecord)
                else V7CheckpointRecord.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return result


def _plan_ids(state: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for raw in state.get("plan_anchors") or []:
        value = (
            raw.get("element_id")
            if isinstance(raw, dict)
            else getattr(raw, "element_id", None)
        )
        if value:
            values.append(str(value))
    return list(dict.fromkeys(values))


def build_v7_contract_report(
    state: dict[str, Any],
    context: MedicationReviewAcmPrimContext,
) -> V7ContractReport:
    fetch_k, visible_k = v7_retrieval_depths(context.v7_retrieval_depth)
    arm = context.v7_experiment_arm
    executed_search_calls = int(state.get("search_count") or 0)
    successful_search_calls = sum(
        record.status in SUCCESSFUL_SEARCH_STATUSES
        for record in queries_from_state(state)
    )
    checkpoints = checkpoint_records_from_state(state)
    if arm == "a0":
        return V7ContractReport(
            status="disabled",
            experiment_arm=arm,
            retrieval_depth=context.v7_retrieval_depth,
            maximum_search_calls=context.max_search_calls,
            executed_search_calls=executed_search_calls,
            successful_search_calls=successful_search_calls,
            premature_final_attempts=len(checkpoints),
            fetch_k=fetch_k,
            visible_k=visible_k,
        )

    required_count = v7_required_investigation_count(arm)
    agenda = agenda_from_state(state)
    agenda_ids = [value.investigation_id for value in agenda.items] if agenda else []
    covered_plan_ids = list(
        dict.fromkeys(
            element_id
            for item in (agenda.items if agenda else [])
            for element_id in item.focus_plan_ids
        )
    )
    plan_ids = _plan_ids(state)
    missing_plan_ids = (
        [value for value in plan_ids if value not in covered_plan_ids]
        if required_count
        else []
    )
    successful_probes = [
        value
        for value in probe_records_from_state(state)
        if value.status in SUCCESSFUL_SEARCH_STATUSES
    ]
    initial_completed = list(
        dict.fromkeys(
            value.investigation_id
            for value in successful_probes
            if value.probe_pass == "initial_probe"
        )
    )
    complementary_completed = list(
        dict.fromkeys(
            value.investigation_id
            for value in successful_probes
            if value.probe_pass == "complementary_probe"
        )
    )
    missing_initial = [
        value for value in agenda_ids if value not in initial_completed
    ]
    missing_complementary = [
        value for value in agenda_ids if value not in complementary_completed
    ]
    investigation_by_id = {
        value.investigation_id: value
        for value in investigations_from_state(state)
    }
    open_required = [
        value
        for value in agenda_ids
        if value not in investigation_by_id
        or investigation_by_id[value].status == "open"
    ]

    reasons: list[str] = []
    if required_count and agenda is None:
        reasons.append("agenda_missing")
    elif required_count and len(agenda_ids) != required_count:
        reasons.append("agenda_size_mismatch")
    if missing_plan_ids:
        reasons.append("plan_anchors_uncovered")
    if missing_initial:
        reasons.append("initial_probe_incomplete")
    if missing_complementary:
        reasons.append("complementary_probe_incomplete")
    if successful_search_calls < V7_MINIMUM_SEARCH_CALLS:
        reasons.append("minimum_search_calls_not_met")
    if open_required:
        reasons.append("required_investigations_open")

    return V7ContractReport(
        status="completed" if not reasons else "incomplete",
        experiment_arm=arm,
        retrieval_depth=context.v7_retrieval_depth,
        minimum_search_calls=V7_MINIMUM_SEARCH_CALLS,
        maximum_search_calls=context.max_search_calls,
        required_investigation_count=required_count,
        agenda_created=agenda is not None,
        agenda_investigation_ids=agenda_ids,
        covered_plan_ids=covered_plan_ids,
        missing_plan_ids=missing_plan_ids,
        executed_search_calls=executed_search_calls,
        successful_search_calls=successful_search_calls,
        initial_probe_completed_ids=initial_completed,
        complementary_probe_completed_ids=complementary_completed,
        missing_initial_probe_ids=missing_initial,
        missing_complementary_probe_ids=missing_complementary,
        open_required_investigation_ids=open_required,
        incomplete_reasons=reasons,
        premature_final_attempts=len(checkpoints),
        fetch_k=fetch_k,
        visible_k=visible_k,
    )


def v7_contract_fingerprint(report: V7ContractReport) -> str:
    payload = {
        "reasons": report.incomplete_reasons,
        "successful": report.successful_search_calls,
        "missing_initial": report.missing_initial_probe_ids,
        "missing_complementary": report.missing_complementary_probe_ids,
        "open": report.open_required_investigation_ids,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_v7_contract_memory(report: V7ContractReport) -> str:
    if report.status == "disabled":
        if report.visible_k == 25:
            return (
                "【V7 检索深度设置】\n"
                "本实验每次 search_review_kb 会向你展示 25 个候选块；"
                "基础说明中的 Top-10 在本实验中由该设置替代。"
            )
        if report.fetch_k == 25:
            return (
                "【V7 检索深度设置】\n"
                "本实验后台保存 Top-25 排名，但你仍只会看到前 10 个候选块；"
                "第 11—25 名仅用于离线诊断。"
            )
        return ""
    lines = [
        "【V7 调查努力合同】",
        (
            f"有效搜索 {report.successful_search_calls}/至少"
            f"{report.minimum_search_calls}；运行保护上限"
            f" {report.maximum_search_calls}。最低次数不是停止目标；"
            "达到后如仍有证据缺口，可以继续自主检索。"
        ),
    ]
    if report.visible_k == 25:
        lines.append(
            "本实验每次 search_review_kb 会向你展示 25 个候选块；"
            "基础说明中的 Top-10 在本实验中由该设置替代。"
        )
    elif report.fetch_k == 25:
        lines.append(
            "本实验后台保存 Top-25 排名，但你仍只会看到前 10 个候选块；"
            "第 11—25 名仅用于离线诊断。"
        )
    if report.required_investigation_count:
        if not report.agenda_created:
            lines.append(
                "本实验的固定议程规则替代基础说明中“可在首次搜索时创建调查”的做法。"
                f"尚未建立议程：先调用 set_investigation_agenda，恰好建立 "
                f"{report.required_investigation_count} 个不同调查，并覆盖全部 PlanAnchor。"
            )
        else:
            lines.extend(
                [
                    "必需调查：" + "、".join(report.agenda_investigation_ids),
                    "待初始探查："
                    + ("、".join(report.missing_initial_probe_ids) or "无"),
                    "待互补探查："
                    + (
                        "、".join(report.missing_complementary_probe_ids)
                        or "无"
                    ),
                    "仍开放："
                    + (
                        "、".join(report.open_required_investigation_ids)
                        or "无"
                    ),
                ]
            )
            if report.missing_initial_probe_ids:
                lines.append("先让所有调查各完成一次 initial_probe，再进入第二轮。")
            elif report.missing_complementary_probe_ids:
                lines.append(
                    "现在执行 complementary_probe；查询应针对首轮仍缺少的另一个证据方面，"
                    "不要只做同义改写。"
                )
            else:
                lines.append(
                    "两轮必需探查已完成。可按缺口执行 adaptive_probe；"
                    "最终回答前请显式关闭全部必需调查。"
                )
    else:
        lines.append(
            "本组不规定调查结构；请自主选择尚值得核查的方面。"
            "基础说明中的“较大预算不需要用完”只在完成至少 6 次有效搜索后适用。"
        )
    return "\n".join(lines)
