from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from .models import (
    Diagnosis,
    LabValue,
    Medication,
    PatientCase,
    PatientCaseInput,
    QueryBundle,
    ReviewSlot,
)
from .prompt import EXTRACTION_PROMPT, JSON_FALLBACK_SUFFIX, REPAIR_PROMPT

MAX_QUERY_TEXT_LENGTH = 500


class CaseExtractionError(ValueError):
    pass


class PlanValidationError(ValueError):
    pass


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    start = text.find("{")
    if start < 0:
        raise CaseExtractionError("模型输出中没有 JSON 对象")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise CaseExtractionError(f"模型输出不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise CaseExtractionError("病例抽取结果必须是 JSON 对象")
    return parsed


def _strip_generated_ids(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(payload, ensure_ascii=False))
    for key in ("case_id", "raw_question_hash", "schema_version"):
        cleaned.pop(key, None)
    for item in cleaned.get("medications") or []:
        if isinstance(item, dict):
            item.pop("medication_id", None)
    for item in cleaned.get("diagnoses") or []:
        if isinstance(item, dict):
            item.pop("diagnosis_id", None)
    for key in ("renal_function", "hepatic_function"):
        item = cleaned.get(key)
        if isinstance(item, dict):
            item.pop("lab_id", None)
    for item in cleaned.get("other_labs") or []:
        if isinstance(item, dict):
            item.pop("lab_id", None)
    return cleaned


def _literal_in_raw(raw_question: str, value: str | int | None) -> bool:
    if value is None:
        return True
    needle = _normalized_text(str(value)).casefold()
    return not needle or needle in _normalized_text(raw_question).casefold()


def _numeric_value_in_raw(raw_question: str, value: str | int | None) -> bool:
    if value is None:
        return True
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", str(value))
    if not numbers:
        return True
    raw_numbers: set[Decimal] = set()
    for item in re.findall(r"[+-]?\d+(?:\.\d+)?", raw_question):
        try:
            raw_numbers.add(Decimal(item))
        except InvalidOperation:
            continue
    try:
        return all(Decimal(item) in raw_numbers for item in numbers)
    except InvalidOperation:
        return False


def validate_grounding(raw_question: str, extracted: PatientCaseInput) -> list[str]:
    errors: list[str] = []

    if extracted.age is not None and not _literal_in_raw(raw_question, extracted.age):
        errors.append(f"年龄 {extracted.age} 未在原文出现")

    for diagnosis in extracted.diagnoses:
        if not _literal_in_raw(raw_question, diagnosis.source_mention):
            errors.append(f"疾病原始提及未在原文出现：{diagnosis.source_mention}")

    for medication in extracted.medications:
        if not _literal_in_raw(raw_question, medication.source_mention):
            errors.append(f"药物原始提及未在原文出现：{medication.source_mention}")
            continue
        if medication.dose and not _numeric_value_in_raw(
            raw_question,
            medication.dose,
        ):
            errors.append(
                f"药物 {medication.source_mention} 的 dose 数值未在原文出现："
                f"{medication.dose}"
            )

    labs = [
        extracted.renal_function,
        extracted.hepatic_function,
        *extracted.other_labs,
    ]
    for lab in labs:
        if lab is None:
            continue
        if lab.source_mention and not _literal_in_raw(
            raw_question,
            lab.source_mention,
        ):
            errors.append(f"检验值原始提及未在原文出现：{lab.source_mention}")
        if lab.value and not _numeric_value_in_raw(raw_question, lab.value):
            errors.append(f"检验值 {lab.indicator} 的数值未在原文出现：{lab.value}")

    for value in extracted.allergies:
        if value and not _literal_in_raw(raw_question, value):
            errors.append(f"过敏信息未在原文出现：{value}")

    return errors


def _validate_grounding_d0(
    raw_question: str,
    extracted: PatientCaseInput,
) -> list[str]:
    """Keep the frozen D0 literal-grounding behaviour separate from PEA-RAG v2."""
    errors: list[str] = []
    if extracted.age is not None and not _literal_in_raw(raw_question, extracted.age):
        errors.append(f"年龄 {extracted.age} 未在原文出现")
    for diagnosis in extracted.diagnoses:
        if not _literal_in_raw(raw_question, diagnosis.source_mention):
            errors.append(f"疾病原始提及未在原文出现：{diagnosis.source_mention}")
    for medication in extracted.medications:
        if not _literal_in_raw(raw_question, medication.source_mention):
            errors.append(f"药物原始提及未在原文出现：{medication.source_mention}")
            continue
        for field_name in (
            "dose",
            "dose_unit",
            "route",
            "frequency",
            "duration",
            "indication",
        ):
            value = getattr(medication, field_name)
            if value and not _literal_in_raw(raw_question, value):
                errors.append(
                    f"药物 {medication.source_mention} 的 {field_name} "
                    f"未在原文出现：{value}"
                )
    labs = [
        extracted.renal_function,
        extracted.hepatic_function,
        *extracted.other_labs,
    ]
    for lab in labs:
        if lab is None:
            continue
        for field_name in (
            "indicator",
            "value",
            "unit",
            "measured_at",
            "source_mention",
        ):
            value = getattr(lab, field_name)
            if value and not _literal_in_raw(raw_question, value):
                errors.append(
                    f"检验值 {lab.indicator} 的 {field_name} 未在原文出现：{value}"
                )
    for value in [*extracted.clinical_risks, *extracted.allergies]:
        if value and not _literal_in_raw(raw_question, value):
            errors.append(f"风险或过敏信息未在原文出现：{value}")
    return errors


def assign_stable_ids(raw_question: str, extracted: PatientCaseInput) -> PatientCase:
    medications = list(enumerate(extracted.medications))
    medications.sort(
        key=lambda item: (
            raw_question.casefold().find(item[1].source_mention.casefold())
            if item[1].source_mention.casefold() in raw_question.casefold()
            else len(raw_question),
            item[0],
        )
    )
    stable_medications = [
        medication.model_copy(update={"medication_id": f"M{position:03d}"})
        for position, (_, medication) in enumerate(medications, start=1)
    ]

    diagnoses = list(enumerate(extracted.diagnoses))
    diagnoses.sort(
        key=lambda item: (
            raw_question.casefold().find(item[1].source_mention.casefold())
            if item[1].source_mention.casefold() in raw_question.casefold()
            else len(raw_question),
            item[0],
        )
    )
    stable_diagnoses = [
        diagnosis.model_copy(update={"diagnosis_id": f"D{position:03d}"})
        for position, (_, diagnosis) in enumerate(diagnoses, start=1)
    ]

    renal = extracted.renal_function.model_copy(update={"lab_id": "R001"}) if extracted.renal_function else None
    hepatic = extracted.hepatic_function.model_copy(update={"lab_id": "H001"}) if extracted.hepatic_function else None
    other_labs = [
        lab.model_copy(update={"lab_id": f"L{position:03d}"})
        for position, lab in enumerate(extracted.other_labs, start=1)
    ]
    normalized = _normalized_text(raw_question)
    question_hash = _sha256(normalized)
    base_fields = extracted.model_dump(
        exclude={"medications", "diagnoses", "renal_function", "hepatic_function", "other_labs"}
    )
    return PatientCase(
        **base_fields,
        case_id=f"CASE-{question_hash[:16]}",
        raw_question_hash=question_hash,
        medications=stable_medications,
        diagnoses=stable_diagnoses,
        renal_function=renal,
        hepatic_function=hepatic,
        other_labs=other_labs,
    )


async def _invoke_json_model(model: Any, prompt: str) -> PatientCaseInput:
    target_schema = json.dumps(
        PatientCaseInput.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fallback_instructions = JSON_FALLBACK_SUFFIX.format(target_schema=target_schema)
    response = await model.ainvoke(
        [
            SystemMessage(content="只抽取病例原文事实并返回 JSON。"),
            HumanMessage(content=f"{prompt}\n\n{fallback_instructions}"),
        ]
    )
    return PatientCaseInput.model_validate(_parse_json_object(_message_text(response)))


async def extract_patient_case(
    raw_question: str,
    model: Any | None,
    system_prompt: str,
) -> tuple[PatientCase, str, list[str]]:
    warnings: list[str] = []
    try:
        direct_payload = json.loads(raw_question)
    except json.JSONDecodeError:
        direct_payload = None

    if isinstance(direct_payload, dict) and (
        "medications" in direct_payload or "diagnoses" in direct_payload or "renal_function" in direct_payload
    ):
        extracted = PatientCaseInput.model_validate(_strip_generated_ids(direct_payload))
        grounding_errors = _validate_grounding_d0(raw_question, extracted)
        if grounding_errors:
            raise CaseExtractionError("结构化病例未通过原文校验：" + "；".join(grounding_errors))
        return assign_stable_ids(raw_question, extracted), "structured_input", warnings

    if model is None:
        raise CaseExtractionError("自由文本病例需要配置解析模型")

    prompt = EXTRACTION_PROMPT.format(system_prompt=system_prompt.strip(), raw_question=raw_question)
    extraction_mode = "structured_output"
    first_error: Exception | None = None
    try:
        structured_model = model.with_structured_output(PatientCaseInput)
        result = await structured_model.ainvoke(
            [
                SystemMessage(content="只抽取病例原文事实。"),
                HumanMessage(content=prompt),
            ]
        )
        extracted = result if isinstance(result, PatientCaseInput) else PatientCaseInput.model_validate(result)
    except Exception as exc:  # noqa: BLE001 - provider compatibility fallback
        first_error = exc
        extraction_mode = "json_fallback"
        warnings.append(f"结构化输出不可用，已改用 JSON 模式：{type(exc).__name__}")
        try:
            extracted = await _invoke_json_model(model, prompt)
        except Exception as fallback_exc:  # noqa: BLE001
            first_error = fallback_exc
            extracted = None

    validation_error = ""
    if extracted is not None:
        grounding_errors = _validate_grounding_d0(raw_question, extracted)
        validation_error = "；".join(grounding_errors)
        if not grounding_errors:
            return assign_stable_ids(raw_question, extracted), extraction_mode, warnings
    elif first_error:
        validation_error = str(first_error)

    repair_prompt = REPAIR_PROMPT.format(
        validation_error=validation_error or "输出不符合病例 schema",
        raw_question=raw_question,
    )
    try:
        repaired = await _invoke_json_model(model, repair_prompt)
    except Exception as exc:  # noqa: BLE001 - provider and schema errors share one repair boundary
        raise CaseExtractionError(f"病例抽取修复失败：{exc}") from exc

    grounding_errors = _validate_grounding_d0(raw_question, repaired)
    if grounding_errors:
        raise CaseExtractionError("病例抽取修复后仍未通过原文校验：" + "；".join(grounding_errors))
    warnings.append("病例抽取经过一次修复")
    return assign_stable_ids(raw_question, repaired), f"{extraction_mode}_repaired", warnings


def _medication_name(medication: Medication) -> str:
    primary = medication.generic_name or medication.normalized_name or medication.source_mention
    if primary.casefold() == medication.source_mention.casefold():
        return primary
    return f"{primary}（原文：{medication.source_mention}）"


def _diagnosis_name(diagnosis: Diagnosis) -> str:
    return diagnosis.name or diagnosis.source_mention


def _age_label(case: PatientCase) -> str:
    return f"{case.age}岁老年" if case.age is not None else "老年"


def _patient_label(case: PatientCase, medication: Medication | None = None) -> str:
    indication = medication.indication if medication else None
    if not indication:
        active = next((item for item in case.diagnoses if item.status == "active"), None)
        indication = _diagnosis_name(active) if active else None
    if indication:
        return f"{_age_label(case)}{indication}患者"
    return f"{_age_label(case)}患者"


def _medication_details(medication: Medication) -> str:
    details: list[str] = []
    if medication.dose:
        dose = medication.dose
        if medication.dose_unit and medication.dose_unit.casefold() not in dose.casefold():
            dose = f"{dose} {medication.dose_unit}"
        details.append(dose)
    for value in (medication.frequency, medication.route, medication.duration):
        if value:
            details.append(value)
    return "、".join(details)


def _lab_text(lab: LabValue) -> str:
    value = f"{lab.indicator} {lab.value}"
    if lab.unit:
        value += f" {lab.unit}"
    return value


def _diagnosis_with_status(diagnosis: Diagnosis) -> str:
    name = _diagnosis_name(diagnosis)
    if diagnosis.status == "history":
        return f"既往有{name}病史"
    if diagnosis.status == "suspected":
        return f"疑似{name}"
    if diagnosis.status == "unknown":
        return f"疾病状态未明确的{name}"
    return f"患有{name}"


def _validate_bundle(bundle: QueryBundle, entity_names: dict[str, str]) -> QueryBundle:
    errors: list[str] = []
    text = bundle.query_text.strip()
    if not text:
        errors.append("query_text 为空")
    if len(text) > MAX_QUERY_TEXT_LENGTH:
        errors.append(f"query_text 超过 {MAX_QUERY_TEXT_LENGTH} 字符")
    if len(bundle.core_entity_ids) > 6:
        errors.append("核心实体超过六个")
    if any(prefix in text for prefix in ("请检索", "关键词：", "关键词 ")):
        errors.append("查询包含无意义检索指令或关键词前缀")
    for entity_id in bundle.core_entity_ids:
        name = entity_names.get(entity_id)
        if name and name not in text:
            errors.append(f"查询缺少核心实体 {entity_id}")
    if bundle.template_id == "medication_profile" and ("肾功能" in text or "肝功能" in text):
        errors.append("药物整体查询混入器官功能关系")
    return bundle.model_copy(
        update={
            "query_text": text,
            "validation_status": "invalid" if errors else "valid",
            "validation_errors": errors,
        }
    )


def build_review_plan(case: PatientCase) -> tuple[list[ReviewSlot], list[QueryBundle]]:
    medications = [item for item in case.medications if item.status in {"current", "planned"}]
    diagnoses = list(case.diagnoses)
    active_diagnoses = [item for item in diagnoses if item.status == "active"]
    slots: list[ReviewSlot] = []

    for medication in medications:
        medication_id = str(medication.medication_id)
        slots.append(
            ReviewSlot(
                slot_id=f"MP:{medication_id}",
                slot_type="medication_profile",
                subject_ids=[medication_id],
                patient_constraints={"age": case.age},
                required_attributes=["contraindications", "adverse_effects", "monitoring"],
                generation_reason="每个当前或计划药物均需一般老年用药安全性检索",
            )
        )
        for organ in ("RENAL", "HEPATIC"):
            lab = case.renal_function if organ == "RENAL" else case.hepatic_function
            slots.append(
                ReviewSlot(
                    slot_id=f"OF:{organ}:{medication_id}",
                    slot_type="organ_function",
                    subject_ids=[medication_id],
                    patient_constraints={
                        "organ": organ.lower(),
                        "status": "known" if lab else "unknown",
                        "lab": lab.model_dump() if lab else None,
                    },
                    required_attributes=["dose_adjustment", "interval_adjustment", "avoidance", "monitoring"],
                    generation_reason=f"每个药物均独立检查{organ.lower()}功能调整要求",
                )
            )

    medication_pairs = list(combinations(medications, 2))
    for left, right in medication_pairs:
        left_id, right_id = str(left.medication_id), str(right.medication_id)
        slots.append(
            ReviewSlot(
                slot_id=f"DD:{left_id}:{right_id}",
                slot_type="drug_drug",
                subject_ids=[left_id],
                target_ids=[right_id],
                patient_constraints={"age": case.age},
                required_attributes=["interaction", "contraindicated_combination", "duplication", "cumulative_risk"],
                generation_reason="枚举所有当前或计划药物的无序对",
            )
        )

    for medication in medications:
        for diagnosis in diagnoses:
            medication_id, diagnosis_id = str(medication.medication_id), str(diagnosis.diagnosis_id)
            slots.append(
                ReviewSlot(
                    slot_id=f"DX:{medication_id}:{diagnosis_id}",
                    slot_type="drug_disease",
                    subject_ids=[medication_id],
                    target_ids=[diagnosis_id],
                    patient_constraints={"age": case.age, "diagnosis_status": diagnosis.status},
                    required_attributes=["contraindication", "caution", "disease_worsening", "monitoring"],
                    generation_reason="枚举每个当前或计划药物与每个保留疾病",
                )
            )

    for diagnosis in active_diagnoses:
        diagnosis_id = str(diagnosis.diagnosis_id)
        slots.append(
            ReviewSlot(
                slot_id=f"PO:{diagnosis_id}",
                slot_type="prescribing_omission",
                target_ids=[diagnosis_id],
                patient_constraints={"age": case.age, "diagnosis_status": diagnosis.status},
                required_attributes=["necessary_treatment", "first_line_treatment", "preventive_treatment"],
                generation_reason="每个活动疾病均检查标准治疗要求",
            )
        )

    duplication_status = "queryable" if len(medications) >= 2 else "not_applicable"
    slots.append(
        ReviewSlot(
            slot_id="DUP:CASE",
            slot_type="duplication",
            subject_ids=[str(item.medication_id) for item in medications],
            patient_constraints={"age": case.age},
            required_attributes=["duplicate_therapy"],
            generation_reason="病例级重复用药由全部药物对查询共同覆盖",
            applicability_status=duplication_status,
        )
    )
    burden_status = "queryable" if medications else "not_applicable"
    slots.append(
        ReviewSlot(
            slot_id="BURDEN:CASE",
            slot_type="cumulative_burden",
            subject_ids=[str(item.medication_id) for item in medications],
            patient_constraints={"age": case.age},
            required_attributes=[
                "anticholinergic_burden",
                "sedation_fall",
                "hypotension",
                "qt_prolongation",
                "bleeding",
            ],
            generation_reason="病例级多药累积风险按最多六种药物分组检索",
            applicability_status=burden_status,
        )
    )

    entity_names = {
        **{str(item.medication_id): _medication_name(item) for item in medications},
        **{str(item.diagnosis_id): _diagnosis_name(item) for item in diagnoses},
    }
    bundles: list[QueryBundle] = []

    for medication in medications:
        medication_id = str(medication.medication_id)
        name = _medication_name(medication)
        details = _medication_details(medication)
        detail_text = f"（{details}）" if details else ""
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:MP:{medication_id}",
                slot_ids=[f"MP:{medication_id}"],
                query_text=(
                    f"{_patient_label(case, medication)}使用{name}{detail_text}时，"
                    "主要禁忌、慎用、不良反应和监测要求是什么？"
                ),
                template_id="medication_profile",
                expected_evidence_types=["contraindication", "adverse_effect", "monitoring"],
                core_entity_ids=[medication_id],
            )
        )

        if case.renal_function:
            renal_query = (
                f"{_lab_text(case.renal_function)}的{_patient_label(case, medication)}使用{name}时，"
                "是否需要减量、延长给药间隔或避免使用？"
            )
        else:
            renal_query = (
                f"{_patient_label(case, medication)}使用{name}时，"
                "哪些肾功能条件需要调整剂量、给药间隔或避免使用？"
            )
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:OF:RENAL:{medication_id}",
                slot_ids=[f"OF:RENAL:{medication_id}"],
                query_text=renal_query,
                template_id="organ_renal",
                expected_evidence_types=["dose_adjustment", "interval_adjustment", "avoidance"],
                core_entity_ids=[medication_id],
            )
        )

        if case.hepatic_function:
            hepatic_query = (
                f"{_lab_text(case.hepatic_function)}的{_patient_label(case, medication)}使用{name}时，"
                "是否存在禁忌、剂量调整或额外监测要求？"
            )
        else:
            hepatic_query = (
                f"{_patient_label(case, medication)}使用{name}时，"
                "哪些肝功能条件需要避免使用、调整剂量或加强监测？"
            )
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:OF:HEPATIC:{medication_id}",
                slot_ids=[f"OF:HEPATIC:{medication_id}"],
                query_text=hepatic_query,
                template_id="organ_hepatic",
                expected_evidence_types=["contraindication", "dose_adjustment", "monitoring"],
                core_entity_ids=[medication_id],
            )
        )

    for left, right in medication_pairs:
        left_id, right_id = str(left.medication_id), str(right.medication_id)
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:DD:{left_id}:{right_id}",
                slot_ids=[f"DD:{left_id}:{right_id}", "DUP:CASE"],
                query_text=(
                    f"老年患者同时使用{_medication_name(left)}和{_medication_name(right)}时，"
                    "是否存在禁忌联用、药物相互作用、重复用药或累积不良反应？"
                ),
                template_id="drug_drug",
                expected_evidence_types=["interaction", "contraindicated_combination", "duplication"],
                core_entity_ids=[left_id, right_id],
            )
        )

    for medication in medications:
        for diagnosis in diagnoses:
            medication_id, diagnosis_id = str(medication.medication_id), str(diagnosis.diagnosis_id)
            bundles.append(
                QueryBundle(
                    bundle_id=f"QB:DX:{medication_id}:{diagnosis_id}",
                    slot_ids=[f"DX:{medication_id}:{diagnosis_id}"],
                    query_text=(
                        f"{_diagnosis_with_status(diagnosis)}的{_age_label(case)}患者使用{_medication_name(medication)}时，"
                        "是否存在禁忌、慎用、疾病加重、剂量调整或监测要求？"
                    ),
                    template_id="drug_disease",
                    expected_evidence_types=["contraindication", "caution", "disease_worsening", "monitoring"],
                    core_entity_ids=[medication_id, diagnosis_id],
                )
            )

    for diagnosis in active_diagnoses:
        diagnosis_id = str(diagnosis.diagnosis_id)
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:PO:{diagnosis_id}",
                slot_ids=[f"PO:{diagnosis_id}"],
                query_text=(
                    f"{_age_label(case)}{_diagnosis_name(diagnosis)}患者的标准治疗中，"
                    "哪些药物属于必要治疗，哪些情况需要调整治疗方案？"
                ),
                template_id="prescribing_omission",
                expected_evidence_types=["necessary_treatment", "first_line_treatment", "preventive_treatment"],
                core_entity_ids=[diagnosis_id],
            )
        )

    for group_index in range(0, len(medications), 6):
        group = medications[group_index : group_index + 6]
        if not group:
            continue
        group_number = group_index // 6 + 1
        bundles.append(
            QueryBundle(
                bundle_id=f"QB:BURDEN:CASE:{group_number:03d}",
                slot_ids=["BURDEN:CASE"],
                query_text=(
                    f"老年患者同时使用{'、'.join(_medication_name(item) for item in group)}时，"
                    "是否存在抗胆碱负荷、镇静跌倒、低血压、QT 延长、出血或其它累积用药风险？"
                ),
                template_id="cumulative_burden",
                expected_evidence_types=["cumulative_adverse_effect"],
                core_entity_ids=[str(item.medication_id) for item in group],
            )
        )

    bundles = [_validate_bundle(bundle, entity_names) for bundle in bundles]
    coverage: dict[str, list[str]] = {slot.slot_id: [] for slot in slots}
    for bundle in bundles:
        for slot_id in bundle.slot_ids:
            if slot_id not in coverage:
                raise PlanValidationError(f"QueryBundle 引用了不存在的槽位：{slot_id}")
            coverage[slot_id].append(bundle.bundle_id)
    slots = [slot.model_copy(update={"covered_by_bundle_ids": coverage[slot.slot_id]}) for slot in slots]
    missing_coverage = [
        slot.slot_id
        for slot in slots
        if slot.applicability_status == "queryable" and not slot.covered_by_bundle_ids
    ]
    if missing_coverage:
        raise PlanValidationError(f"可查询槽位没有 QueryBundle 覆盖：{missing_coverage}")
    return slots, bundles
