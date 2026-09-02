from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_io import StructuredOutputError, invoke_json_schema
from .models import (
    CasePlanExtractionDraft,
    PatientCase,
    PlanExtractionAudit,
    PlanVerificationResult,
    TreatmentPlanElement,
)
from .plan_validation import (
    PlanValidationError,
    apply_plan_repairs,
    canonicalize_patient_facts,
    canonicalize_plan_elements,
    issue_dicts,
    reconcile_grounded_plan_references,
    validate_plan_inventory,
)
from .planning import assign_stable_ids, validate_grounding
from .prompt import (
    PLAN_EXTRACTION_PROMPT,
    PLAN_EXTRACTION_SYSTEM_PROMPT,
    PLAN_VERIFICATION_PROMPT,
    PLAN_VERIFICATION_SYSTEM_PROMPT,
)


class CasePlanExtractionError(ValueError):
    def __init__(self, message: str, audit: PlanExtractionAudit | None = None):
        super().__init__(message)
        self.audit = audit or PlanExtractionAudit()


@dataclass(frozen=True)
class CasePlanExtractionResult:
    patient_case: PatientCase
    patient_facts: list[dict[str, Any]]
    plan_elements: list[TreatmentPlanElement]
    audit: PlanExtractionAudit
    warnings: list[str]
    degraded: bool = False


def _literal_in_raw(raw_question: str, value: str) -> bool:
    return " ".join(value.split()).casefold() in " ".join(
        raw_question.split()
    ).casefold()


def _drop_ungrounded_clinical_risks(
    *,
    raw_question: str,
    draft: CasePlanExtractionDraft,
) -> tuple[CasePlanExtractionDraft, list[str]]:
    patient_case = draft.patient_case
    grounded_risks = [
        value
        for value in patient_case.clinical_risks
        if _literal_in_raw(raw_question, value)
    ]
    dropped_risks = [
        value
        for value in patient_case.clinical_risks
        if value not in grounded_risks
    ]
    warnings: list[str] = []
    if dropped_risks:
        warnings.append(
            "patient_case.clinical_risks 中的归纳性表述未在原文逐字出现，"
            f"已移除并仅保留带 source_span 的 patient_facts：{dropped_risks}"
        )
    if not warnings:
        return draft, []
    return (
        draft.model_copy(
            update={
                "patient_case": patient_case.model_copy(
                    update={
                        "clinical_risks": grounded_risks,
                    }
                )
            }
        ),
        warnings,
    )


def _direct_payload(raw_question: str) -> CasePlanExtractionDraft | None:
    try:
        value = json.loads(raw_question)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if "patient_case" not in value or "plan_elements" not in value:
        return None
    return CasePlanExtractionDraft.model_validate(value)


def _verification_has_findings(result: PlanVerificationResult) -> bool:
    return any(
        [
            result.missing_explicit_spans,
            result.duplicated_element_ids,
            result.hallucinated_element_ids,
            result.incorrect_element_types,
            result.incorrect_relationships,
        ]
    )


async def extract_case_and_plan(
    *,
    raw_question: str,
    model: Any | None,
    system_prompt: str,
    plan_repair_limit: int = 1,
    technical_retry_limit: int = 0,
    retain_raw_output: bool = True,
) -> CasePlanExtractionResult:
    warnings: list[str] = []
    degraded = False
    audit = PlanExtractionAudit()
    direct = _direct_payload(raw_question)

    if direct is not None:
        draft = direct
        warnings.append("使用结构化病例与治疗方案输入")
    else:
        if model is None:
            raise CasePlanExtractionError("自由文本病例需要配置解析模型", audit)
        try:
            extraction_result = await invoke_json_schema(
                model=model,
                stage="extract_case_and_plan",
                system_prompt=PLAN_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=PLAN_EXTRACTION_PROMPT.format(
                    system_prompt=system_prompt.strip(),
                    raw_question=raw_question,
                ),
                output_model=CasePlanExtractionDraft,
                repair_limit=1,
                technical_retry_limit=technical_retry_limit,
                retain_raw_output=retain_raw_output,
            )
        except StructuredOutputError as exc:
            audit = audit.model_copy(update={"extraction": exc.audit})
            raise CasePlanExtractionError(str(exc), audit) from exc
        draft = CasePlanExtractionDraft.model_validate(extraction_result.value)
        audit = audit.model_copy(update={"extraction": extraction_result.audit})

    draft, grounding_warnings = _drop_ungrounded_clinical_risks(
        raw_question=raw_question,
        draft=draft,
    )
    warnings.extend(grounding_warnings)
    reconciled_case, reconciliation_events = reconcile_grounded_plan_references(
        raw_question=raw_question,
        patient_case=draft.patient_case,
        drafts=draft.plan_elements,
    )
    if reconciliation_events:
        draft = draft.model_copy(update={"patient_case": reconciled_case})
        audit = audit.model_copy(
            update={
                "runtime_revisions": [
                    *audit.runtime_revisions,
                    *reconciliation_events,
                ]
            }
        )
        warnings.append(
            "解析模型在患者实体清单中漏掉了方案已引用的原文实体；"
            f"已按原文确定性补齐 {len(reconciliation_events)} 项并继续"
        )
    grounding_errors = validate_grounding(raw_question, draft.patient_case)
    if grounding_errors:
        raise CasePlanExtractionError(
            "患者事实未通过病例原文校验：" + "；".join(grounding_errors),
            audit,
        )
    patient_case = assign_stable_ids(raw_question, draft.patient_case)

    try:
        patient_facts = canonicalize_patient_facts(raw_question, draft.patient_facts)
        plan_elements = canonicalize_plan_elements(
            raw_question=raw_question,
            case=patient_case,
            drafts=draft.plan_elements,
        )
    except PlanValidationError as exc:
        raise CasePlanExtractionError(f"治疗方案规范化失败：{exc}", audit) from exc

    deterministic_issues = validate_plan_inventory(
        raw_question=raw_question,
        case=patient_case,
        elements=plan_elements,
    )
    audit = audit.model_copy(update={"deterministic_issues": deterministic_issues})

    if model is None:
        verifier_result = PlanVerificationResult()
        warnings.append("结构化输入未执行独立 LLM 方案核查")
    else:
        try:
            verification_result = await invoke_json_schema(
                model=model,
                stage="verify_plan_inventory",
                system_prompt=PLAN_VERIFICATION_SYSTEM_PROMPT,
                user_prompt=PLAN_VERIFICATION_PROMPT.format(
                    raw_question=raw_question,
                    patient_case=json.dumps(
                        patient_case.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    plan_elements=json.dumps(
                        [item.model_dump(mode="json") for item in plan_elements],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    deterministic_issues=json.dumps(
                        issue_dicts(deterministic_issues),
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                output_model=PlanVerificationResult,
                repair_limit=1,
                technical_retry_limit=technical_retry_limit,
                retain_raw_output=retain_raw_output,
            )
        except StructuredOutputError as exc:
            audit = audit.model_copy(update={"verification": exc.audit})
            verifier_result = PlanVerificationResult()
            degraded = True
            warnings.append(f"独立方案核查失败，保留已通过确定性校验的方案清单：{exc}")
        else:
            verifier_result = PlanVerificationResult.model_validate(
                verification_result.value
            )
            audit = audit.model_copy(
                update={
                    "verification": verification_result.audit,
                    "verifier_result": verifier_result,
                }
            )

    operations = verifier_result.repair_operations
    if operations:
        if plan_repair_limit < 1:
            degraded = True
            warnings.append("方案核查提出修复，但当前配置禁止初始方案修复；保留原清单")
        else:
            original_elements = plan_elements
            try:
                repaired_elements = apply_plan_repairs(
                    raw_question=raw_question,
                    case=patient_case,
                    elements=plan_elements,
                    operations=operations,
                )
            except PlanValidationError as exc:
                degraded = True
                warnings.append(f"方案核查修复操作无效，已保留原清单：{exc}")
            else:
                repair_issues = validate_plan_inventory(
                    raw_question=raw_question,
                    case=patient_case,
                    elements=repaired_elements,
                )
                if any(item.severity == "error" for item in repair_issues):
                    plan_elements = original_elements
                    degraded = True
                    warnings.append("方案核查修复后未通过确定性校验，已回退原清单")
                else:
                    plan_elements = repaired_elements
                    audit = audit.model_copy(
                        update={"applied_repair_operations": operations}
                    )
                    warnings.append("治疗方案清单经过一次独立核查修复")
    elif _verification_has_findings(verifier_result):
        degraded = True
        warnings.append("方案核查报告了未解决问题但没有合法修复操作；保留原清单并继续")

    final_issues = validate_plan_inventory(
        raw_question=raw_question,
        case=patient_case,
        elements=plan_elements,
    )
    final_errors = [item for item in final_issues if item.severity == "error"]
    if final_errors:
        audit = audit.model_copy(update={"deterministic_issues": final_issues})
        raise CasePlanExtractionError(
            "治疗方案清单修复后仍未通过校验："
            + "；".join(item.message for item in final_errors),
            audit,
        )
    warnings.extend(item.message for item in final_issues if item.severity == "warning")
    return CasePlanExtractionResult(
        patient_case=patient_case,
        patient_facts=[item.model_dump(mode="json") for item in patient_facts],
        plan_elements=plan_elements,
        audit=audit,
        warnings=warnings,
        degraded=degraded,
    )
