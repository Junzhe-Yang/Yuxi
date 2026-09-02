from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yuxi.agents.buildin.medication_review_acm_prim.adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
    build_adaptive_coverage_report,
    build_adaptive_investigation_memory,
)
from yuxi.agents.buildin.medication_review_acm_prim.memory import (
    build_adaptive_atlas_document_memory,
)
from yuxi.agents.buildin.medication_review_lite.models import EvidenceItem
from yuxi.agents.buildin.medication_review_prim.memory import (
    anchors_from_state,
    build_case_node_memory,
    modifiers_from_state,
)

from .context import MedicationReviewAcmBoundedContext
from .models import ActionDirective

PROMPT_VERSION = "acm-bounded-adaptive-v1"


@dataclass(frozen=True)
class BoundedPrompt:
    text: str
    case_memory: str
    ledger_memory: str
    atlas_memory: str
    evidence_memory: str


def _evidence_store(state: dict[str, Any]) -> dict[str, EvidenceItem]:
    values: dict[str, EvidenceItem] = {}
    for evidence_id, raw in (state.get("evidence_store") or {}).items():
        try:
            values[str(evidence_id).upper()] = (
                raw if isinstance(raw, EvidenceItem) else EvidenceItem.model_validate(raw)
            )
        except Exception:  # noqa: BLE001
            continue
    return values


def _evidence_memory(
    state: dict[str, Any],
    directive: ActionDirective,
) -> str:
    store = _evidence_store(state)
    blocks: list[str] = []
    exact = directive.phase in {
        "REVIEW_ACTIVE_OBLIGATION",
        "CLOSE_ACTIVE_INVESTIGATION",
        "DRAFT_FINAL",
    }
    for alias, evidence_id in directive.evidence_aliases.items():
        item = store.get(evidence_id)
        if item is None:
            continue
        source = item.source_document or "未知来源"
        location = ", ".join(
            value
            for value in (
                f"file_id={item.file_id}" if item.file_id else "",
                f"chunk_id={item.chunk_id}" if item.chunk_id is not None else "",
                f"chunk_index={item.chunk_index}" if item.chunk_index is not None else "",
            )
            if value
        )
        header = (
            f"[{alias} -> {evidence_id}] 来源={source}"
            + (f"；{location}" if location else "")
            + f"；hash={item.content_hash}"
        )
        if exact:
            blocks.append(f"{header}\n原文：\n{item.raw_text}")
        else:
            digest = " ".join(item.raw_text.split())[:500]
            blocks.append(f"{header}\n摘要：{digest}")
    if not blocks:
        return ""
    fidelity = (
        "以下是当前判断直接需要的 Evidence 精确原文；不得用历史自述替代。"
        if exact
        else "以下仅用于最终组织的 Evidence 确定性摘要；原文仍完整保存在 Evidence Store。"
    )
    return "【当前 Evidence】\n" + fidelity + "\n\n" + "\n\n".join(blocks)


def _directive_memory(directive: ActionDirective) -> str:
    lines = [
        "【当前 ActionDirective】",
        f"directive_id={directive.directive_id}",
        f"phase={directive.phase}",
        f"state_version={directive.state_version}",
        "allowed_actions=" + (", ".join(directive.allowed_actions) or "none"),
    ]
    if directive.active_investigation_id:
        lines.append(f"active_investigation={directive.active_investigation_id}")
    if directive.active_obligation:
        lines.append(f"active_obligation={directive.active_obligation}")
    if directive.active_recovery_id:
        lines.append(f"active_recovery={directive.active_recovery_id}")
    if directive.bound_retrieval_scope:
        lines.append(
            f"bound_route={directive.bound_retrieval_scope}; "
            f"intent={directive.bound_retrieval_intent}; "
            f"file_id={directive.bound_file_id or 'none'}"
        )
    if directive.evidence_aliases:
        lines.append(
            "evidence_aliases=" + ", ".join(f"{key}:{value}" for key, value in directive.evidence_aliases.items())
        )
    if directive.file_aliases:
        lines.append("file_aliases=" + ", ".join(f"{key}:{value}" for key, value in directive.file_aliases.items()))
    if directive.investigation_aliases and directive.phase == "AUDIT_COVERAGE":
        lines.append(
            "investigation_aliases="
            + ", ".join(f"{key}:{value}" for key, value in directive.investigation_aliases.items())
        )
    return "\n".join(lines)


def _phase_instruction(directive: ActionDirective) -> str:
    if directive.phase == "PROPOSE_INITIAL_AGENDA":
        return (
            "一次提出所有可能实质改变最终判断、彼此有区别且值得检索的调查；数量没有上限或固定值。"
            "每个 PlanAnchor 都要有独立 current_regimen_review。每项必须填写临床意义、区别范围、"
            "类型和可由直接证据回答的最小 evidence_obligations。直接调用 propose_initial_agenda。"
        )
    if directive.phase == "EXTEND_AGENDA":
        return (
            "只补充控制器指出的缺失 current_regimen_review、improvement_plan 或审计新缺口；"
            "不要改写既有议程。直接调用 extend_investigation_agenda_bounded。"
        )
    if directive.phase == "SEARCH_ACTIVE_OBLIGATION":
        return (
            "只为 active_obligation 生成一个短中文检索式并调用 search_active_obligation。"
            "query 使用 2—6 个语料关键词：一个核心主体、一个待查属性，必要时加一个患者限定；"
            "不要复制义务整句、解释原因、枚举答案或混合多个属性轴。route、Investigation、"
            "obligation 和 file_id 已由控制器绑定。若 Atlas 标题能明显改善检索式，可先打开一个未读文档。"
        )
    if directive.phase == "REVIEW_ACTIVE_OBLIGATION":
        return (
            "只判断当前义务。Evidence 是候选，不因命中就自动成立。选择当前 E alias，"
            "给出 SUPPORTED、CONTRADICTED 或 INSUFFICIENT 和简短医学理由；"
            "需要相邻原文时先调用 open_active_evidence。不要讨论其它义务。"
        )
    if directive.phase == "CLOSE_ACTIVE_INVESTIGATION":
        return (
            "综合当前调查的全部义务判断后关闭调查。resolved/remaining、Evidence 并集和 provenance"
            "由控制器计算，不要复述这些字段。若证据不足，明确 residual_uncertainty 和停止理由。"
        )
    if directive.phase == "AUDIT_COVERAGE":
        return (
            "逐项审计六个固定维度，每个维度恰好一次 covered/not_applicable/gap。"
            "引用 I alias，不手写 Investigation ID。gap 必须引用需重开的既有调查，或提出新的实质调查；"
            "material_gap_found 由控制器派生。维度：" + "、".join(ADAPTIVE_AUDIT_DIMENSIONS)
        )
    if directive.phase == "FAIL_EXPLICIT":
        return (
            "同一状态下的模型语义修复已经耗尽。不要继续调查、伪造完成状态或给出临床结论；"
            "仅简短说明当前 phase、未完成范围和最后失败原因，并明确建议查看 Trace 后重试。"
        )
    return (
        "生成最终审查正文第②至第⑤部分；不要输出第①和第⑥部分，程序会补齐。"
        "逐项覆盖每个 PE，保留合理、不合理、需调整、证据不足和改进方案；"
        "只引用已存在的 [EV-...]，不得引用 E alias、Atlas 或病例节点。不要讨论内部流程。"
    )


def build_bounded_prompt(
    *,
    state: dict[str, Any],
    context: MedicationReviewAcmBoundedContext,
    directive: ActionDirective,
) -> BoundedPrompt:
    case_memory = build_case_node_memory(
        profile="full",
        anchors=anchors_from_state(state),
        modifiers=modifiers_from_state(state),
    )
    report = build_adaptive_coverage_report(state, context)
    ledger_memory = build_adaptive_investigation_memory(state, report)
    atlas_memory = ""
    atlas = getattr(context, "_acm_prim_atlas", None)
    if (
        directive.phase
        in {
            "PROPOSE_INITIAL_AGENDA",
            "EXTEND_AGENDA",
            "SEARCH_ACTIVE_OBLIGATION",
        }
        and atlas is not None
    ):
        atlas_memory = build_adaptive_atlas_document_memory(atlas.compact_view())
        if directive.atlas_document_aliases:
            atlas_memory += "\n未打开文档 alias：" + ", ".join(
                f"{key}:{value}" for key, value in directive.atlas_document_aliases.items()
            )
    evidence_memory = _evidence_memory(state, directive)
    sections = [
        "你是老年患者完整治疗方案的循证审查 Agent。本输出仅用于方法研究，不构成诊疗结论。",
        context.system_prompt.strip(),
        (
            "运行约束：调查数和搜索数都不是完成目标；未检索义务、pending recovery、"
            "未完成改进方案和全局审计会由控制器阻止结束。只执行当前 directive，"
            "不要输出思维过程、计划自述或反复声明将调用工具。"
        ),
        _phase_instruction(directive),
        _directive_memory(directive),
        case_memory,
        ledger_memory,
        atlas_memory,
        evidence_memory,
    ]
    return BoundedPrompt(
        text="\n\n".join(value for value in sections if value.strip()),
        case_memory=case_memory,
        ledger_memory=ledger_memory,
        atlas_memory=atlas_memory,
        evidence_memory=evidence_memory,
    )
