from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import (
    Diagnosis,
    Medication,
    PatientCase,
    PatientCaseInput,
    PatientFact,
    PatientFactDraft,
    PlanRepairOperation,
    PlanValidationIssue,
    TreatmentPlanElement,
    TreatmentPlanElementDraft,
)

CONCRETE_VALUE = re.compile(
    r"(?:[<>≤≥]\s*)?\d+(?:\.\d+)?\s*"
    r"(?:mg|g|μg|ml|mL|次|周|月|天|日|岁|小时|h|%|ml/min)"
    r"|\d+[A-Z]{1,5}",
    re.IGNORECASE,
)
CONCRETE_VALUE_PARTS = re.compile(
    r"(?P<comparator>[<>≤≥]?)\s*(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|g|μg|ml|mL|次|周|月|天|日|岁|小时|h|%|ml/min|[A-Z]{1,5})",
    re.IGNORECASE,
)


class PlanValidationError(ValueError):
    pass


def concrete_value_supported(value: str, raw_question: str) -> bool:
    compact = re.sub(r"\s+", "", value).casefold()
    raw_compact = re.sub(r"\s+", "", raw_question).casefold()
    if compact in raw_compact:
        return True
    parsed = CONCRETE_VALUE_PARTS.fullmatch(value.strip())
    if parsed is None:
        return False
    try:
        number = Decimal(parsed.group("number"))
    except InvalidOperation:
        return False
    comparator = parsed.group("comparator")
    unit = parsed.group("unit").casefold()
    for match in CONCRETE_VALUE.finditer(raw_question):
        source = CONCRETE_VALUE_PARTS.fullmatch(match.group(0).strip())
        if source is None:
            continue
        try:
            source_number = Decimal(source.group("number"))
        except InvalidOperation:
            continue
        if (
            source_number == number
            and source.group("comparator") == comparator
            and source.group("unit").casefold() == unit
        ):
            return True
    return False


def _find_source_span(raw_question: str, source_span: str, start_at: int = 0) -> tuple[int, int]:
    if not source_span:
        raise PlanValidationError("source_span 不能为空")
    start = raw_question.find(source_span, start_at)
    if start < 0 and start_at:
        start = raw_question.find(source_span)
    if start < 0:
        raise PlanValidationError(f"source_span 未在病例原文逐字出现：{source_span}")
    return start, start + len(source_span)


def canonicalize_patient_facts(
    raw_question: str,
    drafts: list[PatientFactDraft],
) -> list[PatientFact]:
    facts: list[PatientFact] = []
    next_search: dict[str, int] = {}
    for index, draft in enumerate(drafts, start=1):
        start_at = next_search.get(draft.source_span, 0)
        start, end = _find_source_span(raw_question, draft.source_span, start_at)
        next_search[draft.source_span] = end
        facts.append(
            PatientFact(
                **draft.model_dump(),
                fact_id=f"PF{index:03d}",
                source_start=start,
                source_end=end,
            )
        )
    facts.sort(key=lambda item: (item.source_start, item.source_end, item.fact_id))
    return [
        item.model_copy(update={"fact_id": f"PF{index:03d}"})
        for index, item in enumerate(facts, start=1)
    ]


def _entity_lookup(case: PatientCase) -> tuple[dict[str, str], dict[str, str]]:
    medication_lookup: dict[str, str] = {}
    for medication in case.medications:
        medication_id = str(medication.medication_id)
        for value in (
            medication.source_mention,
            medication.generic_name,
            medication.normalized_name,
        ):
            if value:
                medication_lookup.setdefault(value.strip().casefold(), medication_id)

    diagnosis_lookup: dict[str, str] = {}
    for diagnosis in case.diagnoses:
        diagnosis_id = str(diagnosis.diagnosis_id)
        for value in (diagnosis.source_mention, diagnosis.name):
            if value:
                diagnosis_lookup.setdefault(value.strip().casefold(), diagnosis_id)
    return medication_lookup, diagnosis_lookup


def _resolve_entity_mention(
    mention: str,
    lookup: dict[str, str],
) -> str | None:
    normalized = mention.strip().casefold()
    exact = lookup.get(normalized)
    if exact is not None:
        return exact
    compact = re.sub(r"\s+", "", normalized)
    candidates = {
        entity_id
        for source_mention, entity_id in lookup.items()
        if compact
        and (
            compact in re.sub(r"\s+", "", source_mention)
            or re.sub(r"\s+", "", source_mention) in compact
        )
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def reconcile_grounded_plan_references(
    *,
    raw_question: str,
    patient_case: PatientCaseInput,
    drafts: list[TreatmentPlanElementDraft],
) -> tuple[PatientCaseInput, list[dict[str, Any]]]:
    """Recover entities omitted from one side of the model's redundant output.

    Plan references are allowed to repair the patient entity inventory only when
    the referenced mention occurs verbatim in the source case. Semantic guesses
    and ungrounded aliases remain hard failures in ``canonicalize_plan_elements``.
    """
    medications = list(patient_case.medications)
    diagnoses = list(patient_case.diagnoses)
    revisions: list[dict[str, Any]] = []

    for draft in drafts:
        temporary_case = PatientCase(
            **patient_case.model_dump(
                exclude={"medications", "diagnoses"},
            ),
            case_id="entity-reconciliation",
            raw_question_hash="",
            medications=[
                item.model_copy(update={"medication_id": f"TMPM{index:03d}"})
                for index, item in enumerate(medications, start=1)
            ],
            diagnoses=[
                item.model_copy(update={"diagnosis_id": f"TMPD{index:03d}"})
                for index, item in enumerate(diagnoses, start=1)
            ],
        )
        medication_lookup, diagnosis_lookup = _entity_lookup(temporary_case)

        for mention in draft.medication_mentions:
            if _resolve_entity_mention(mention, medication_lookup) is not None:
                continue
            if mention not in raw_question:
                continue
            medications.append(
                Medication(
                    source_mention=mention,
                    generic_name=mention,
                    normalization_source="input",
                    status="unknown",
                )
            )
            revisions.append(
                {
                    "stage": "entity_reconciliation",
                    "action": "add_grounded_medication",
                    "draft_id": draft.draft_id,
                    "source_mention": mention,
                    "reason": "方案要素引用已落回原文，但患者药物清单漏项",
                }
            )
            temporary_case = temporary_case.model_copy(
                update={
                    "medications": [
                        item.model_copy(
                            update={"medication_id": f"TMPM{index:03d}"}
                        )
                        for index, item in enumerate(medications, start=1)
                    ]
                }
            )
            medication_lookup, _ = _entity_lookup(temporary_case)

        for mention in draft.target_diagnosis_mentions:
            if _resolve_entity_mention(mention, diagnosis_lookup) is not None:
                continue
            if mention not in raw_question:
                continue
            diagnoses.append(
                Diagnosis(
                    name=mention,
                    source_mention=mention,
                    status="unknown",
                )
            )
            revisions.append(
                {
                    "stage": "entity_reconciliation",
                    "action": "add_grounded_diagnosis",
                    "draft_id": draft.draft_id,
                    "source_mention": mention,
                    "reason": "方案要素引用已落回原文，但患者疾病清单漏项",
                }
            )
            temporary_case = temporary_case.model_copy(
                update={
                    "diagnoses": [
                        item.model_copy(
                            update={"diagnosis_id": f"TMPD{index:03d}"}
                        )
                        for index, item in enumerate(diagnoses, start=1)
                    ]
                }
            )
            _, diagnosis_lookup = _entity_lookup(temporary_case)

    return (
        patient_case.model_copy(
            update={
                "medications": medications,
                "diagnoses": diagnoses,
            }
        ),
        revisions,
    )


def canonicalize_plan_elements(
    *,
    raw_question: str,
    case: PatientCase,
    drafts: list[TreatmentPlanElementDraft],
) -> list[TreatmentPlanElement]:
    draft_ids = [item.draft_id for item in drafts]
    if not draft_ids or len(draft_ids) != len(set(draft_ids)):
        if not draft_ids:
            raise PlanValidationError("病例中没有抽取到任何治疗方案要素")
        raise PlanValidationError("TreatmentPlanElementDraft.draft_id 必须唯一")

    positioned: list[tuple[int, int, int, TreatmentPlanElementDraft]] = []
    next_search: dict[str, int] = {}
    for original_index, draft in enumerate(drafts):
        start_at = next_search.get(draft.source_span, 0)
        start, end = _find_source_span(raw_question, draft.source_span, start_at)
        next_search[draft.source_span] = end
        positioned.append((start, end, original_index, draft))
    positioned.sort(key=lambda item: (item[0], item[1], item[2]))

    draft_to_element = {
        draft.draft_id: f"PE{index:03d}"
        for index, (_start, _end, _original_index, draft) in enumerate(positioned, start=1)
    }
    medication_lookup, diagnosis_lookup = _entity_lookup(case)
    result: list[TreatmentPlanElement] = []

    for start, end, _original_index, draft in positioned:
        medication_ids: list[str] = []
        for mention in draft.medication_mentions:
            medication_id = _resolve_entity_mention(
                mention,
                medication_lookup,
            )
            if medication_id is None:
                raise PlanValidationError(
                    f"方案要素 {draft.draft_id} 引用了无法解析的药物提及：{mention}"
                )
            if medication_id not in medication_ids:
                medication_ids.append(medication_id)

        diagnosis_ids: list[str] = []
        for mention in draft.target_diagnosis_mentions:
            diagnosis_id = _resolve_entity_mention(
                mention,
                diagnosis_lookup,
            )
            if diagnosis_id is None:
                raise PlanValidationError(
                    f"方案要素 {draft.draft_id} 引用了无法解析的疾病提及：{mention}"
                )
            if diagnosis_id not in diagnosis_ids:
                diagnosis_ids.append(diagnosis_id)

        parent_element_id = None
        if draft.parent_draft_id:
            parent_element_id = draft_to_element.get(draft.parent_draft_id)
            if parent_element_id is None:
                raise PlanValidationError(
                    f"方案要素 {draft.draft_id} 引用了不存在的 parent_draft_id："
                    f"{draft.parent_draft_id}"
                )

        component_element_ids: list[str] = []
        for component_id in draft.component_draft_ids:
            resolved = draft_to_element.get(component_id)
            if resolved is None:
                raise PlanValidationError(
                    f"方案要素 {draft.draft_id} 引用了不存在的 component_draft_id：{component_id}"
                )
            if resolved not in component_element_ids:
                component_element_ids.append(resolved)

        result.append(
            TreatmentPlanElement(
                element_id=draft_to_element[draft.draft_id],
                element_type=draft.element_type,
                source_span=draft.source_span,
                source_start=start,
                source_end=end,
                normalized_summary=draft.normalized_summary,
                parent_element_id=parent_element_id,
                component_element_ids=component_element_ids,
                target_diagnosis_ids=diagnosis_ids,
                medication_ids=medication_ids,
                attributes=draft.attributes,
            )
        )
    return result


def validate_plan_inventory(
    *,
    raw_question: str,
    case: PatientCase,
    elements: list[TreatmentPlanElement],
) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    element_ids = {item.element_id for item in elements}
    seen_semantics: set[tuple[str, str, str]] = set()
    source_type_index: dict[str, set[str]] = {}

    for element in elements:
        source_type_index.setdefault(element.source_span, set()).add(
            element.element_type
        )
        if raw_question[element.source_start : element.source_end] != element.source_span:
            issues.append(
                PlanValidationIssue(
                    issue_type="invalid_source_position",
                    message="source_start/source_end 与病例原文不一致",
                    element_id=element.element_id,
                    source_span=element.source_span,
                )
            )
        normalized_payload = (
            element.normalized_summary
            + " "
            + str(element.attributes)
        )
        unsupported_values = [
            match.group(0).strip()
            for match in CONCRETE_VALUE.finditer(normalized_payload)
            if match.group(0).strip()
            and not concrete_value_supported(
                match.group(0).strip(),
                raw_question,
            )
        ]
        if unsupported_values:
            issues.append(
                PlanValidationIssue(
                    issue_type="ungrounded_concrete_value",
                    message=(
                        "方案要素归一化内容引入了病例原文中不存在的具体数值："
                        f"{unsupported_values}"
                    ),
                    element_id=element.element_id,
                )
            )
        semantic_key = (
            element.source_span,
            element.element_type,
            re.sub(r"\s+", "", element.normalized_summary).casefold(),
        )
        if semantic_key in seen_semantics:
            issues.append(
                PlanValidationIssue(
                    issue_type="duplicate_element",
                    message="同一原文和语义生成了重复方案要素",
                    element_id=element.element_id,
                    source_span=element.source_span,
                )
            )
        seen_semantics.add(semantic_key)

        references = [
            value
            for value in [element.parent_element_id, *element.component_element_ids]
            if value is not None
        ]
        invalid_references = [value for value in references if value not in element_ids]
        if invalid_references:
            issues.append(
                PlanValidationIssue(
                    issue_type="invalid_element_reference",
                    message=f"引用了不存在的方案要素：{invalid_references}",
                    element_id=element.element_id,
                )
            )
        if element.element_id in references:
            issues.append(
                PlanValidationIssue(
                    issue_type="self_referencing_element",
                    message="方案要素不能把自身作为父要素或组成要素",
                    element_id=element.element_id,
                )
            )
        if (
            element.element_type == "medication_order"
            and len(element.medication_ids) != 1
        ):
            issues.append(
                PlanValidationIssue(
                    issue_type="invalid_medication_order_binding",
                    message="每个 medication_order 必须恰好绑定一个明确药物",
                    element_id=element.element_id,
                )
            )
        if element.element_type == "combination_regimen":
            component_count = len(
                set(element.component_element_ids)
                | set(element.medication_ids)
            )
            if component_count < 2:
                issues.append(
                    PlanValidationIssue(
                        issue_type="invalid_combination_regimen",
                        message="combination_regimen 必须连接至少两个药物或组成要素",
                        element_id=element.element_id,
                    )
                )

    for source_span, element_types in source_type_index.items():
        if {
            "combination_regimen",
            "treatment_intent",
        } <= element_types:
            issues.append(
                PlanValidationIssue(
                    issue_type="duplicated_regimen_intent",
                    message=(
                        "同一原文片段不能同时机械生成 combination_regimen "
                        "和 treatment_intent；治疗意图应优先作为组合方案属性"
                    ),
                    source_span=source_span,
                )
            )

    relationship_graph = {
        element.element_id: [
            value
            for value in [
                element.parent_element_id,
                *element.component_element_ids,
            ]
            if value is not None
        ]
        for element in elements
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def has_cycle(element_id: str) -> bool:
        if element_id in visiting:
            return True
        if element_id in visited:
            return False
        visiting.add(element_id)
        cycle = any(
            has_cycle(linked_id)
            for linked_id in relationship_graph.get(element_id, [])
            if linked_id in relationship_graph
        )
        visiting.remove(element_id)
        visited.add(element_id)
        return cycle

    if any(has_cycle(element_id) for element_id in relationship_graph):
        issues.append(
            PlanValidationIssue(
                issue_type="cyclic_element_relationship",
                message="方案要素的 parent/component 关系存在环",
            )
        )

    medication_elements: dict[str, list[str]] = {}
    for element in elements:
        if element.element_type != "medication_order":
            continue
        for medication_id in element.medication_ids:
            medication_elements.setdefault(medication_id, []).append(element.element_id)
    for medication in case.medications:
        if medication.status not in {"current", "planned"}:
            continue
        medication_id = str(medication.medication_id)
        linked = medication_elements.get(medication_id, [])
        if not linked:
            issues.append(
                PlanValidationIssue(
                    issue_type="missing_medication_order",
                    message=f"当前或计划药物 {medication.source_mention} 没有 medication_order",
                    source_span=medication.source_mention,
                )
            )
        elif len(linked) > 1:
            issues.append(
                PlanValidationIssue(
                    issue_type="duplicate_medication_order",
                    message=f"药物 {medication.source_mention} 对应多个 medication_order：{linked}",
                    source_span=medication.source_mention,
                )
            )

    return issues


def _renumber_elements(elements: list[TreatmentPlanElement]) -> list[TreatmentPlanElement]:
    ordered = sorted(
        elements,
        key=lambda item: (item.source_start, item.source_end, item.element_id),
    )
    id_map = {item.element_id: f"PE{index:03d}" for index, item in enumerate(ordered, start=1)}
    return [
        item.model_copy(
            update={
                "element_id": id_map[item.element_id],
                "parent_element_id": (
                    id_map.get(item.parent_element_id) if item.parent_element_id else None
                ),
                "component_element_ids": [
                    id_map[value] for value in item.component_element_ids if value in id_map
                ],
            }
        )
        for item in ordered
    ]


def apply_plan_repairs(
    *,
    raw_question: str,
    case: PatientCase,
    elements: list[TreatmentPlanElement],
    operations: list[PlanRepairOperation],
) -> list[TreatmentPlanElement]:
    updated = {item.element_id: item for item in elements}
    for operation in operations:
        if operation.operation == "add_element":
            if operation.element is None:
                raise PlanValidationError("add_element 缺少 element")
            parent = operation.element.parent_draft_id
            components = list(operation.element.component_draft_ids)
            invalid = [
                value
                for value in [parent, *components]
                if value is not None and value not in updated
            ]
            if invalid:
                raise PlanValidationError(
                    f"新增要素引用了不存在的既有要素：{invalid}"
                )
            standalone = operation.element.model_copy(
                update={"parent_draft_id": None, "component_draft_ids": []}
            )
            new_elements = canonicalize_plan_elements(
                raw_question=raw_question,
                case=case,
                drafts=[standalone],
            )
            candidate = new_elements[0].model_copy(
                update={
                    "element_id": f"PE_TMP_{len(updated) + 1:03d}",
                    "parent_element_id": parent,
                    "component_element_ids": components,
                }
            )
            updated[candidate.element_id] = candidate
            continue

        target_id = operation.target_element_id
        if not target_id or target_id not in updated:
            raise PlanValidationError(
                f"{operation.operation} 引用了不存在的 target_element_id：{target_id}"
            )
        if operation.operation in {"remove_duplicate", "remove_hallucinated"}:
            updated.pop(target_id)
        elif operation.operation == "update_element_type":
            if operation.replacement_type is None:
                raise PlanValidationError("update_element_type 缺少 replacement_type")
            updated[target_id] = updated[target_id].model_copy(
                update={"element_type": operation.replacement_type}
            )
        elif operation.operation == "update_attributes":
            updated[target_id] = updated[target_id].model_copy(
                update={"attributes": operation.replacement_attributes}
            )
        elif operation.operation == "update_relationship":
            invalid = [
                value
                for value in operation.replacement_component_element_ids
                if value not in updated
            ]
            if invalid:
                raise PlanValidationError(f"修复关系引用了不存在的要素：{invalid}")
            parent = operation.replacement_parent_element_id
            if parent is not None and parent not in updated:
                raise PlanValidationError(f"修复关系引用了不存在的父要素：{parent}")
            updated[target_id] = updated[target_id].model_copy(
                update={
                    "parent_element_id": parent,
                    "component_element_ids": operation.replacement_component_element_ids,
                }
            )
    return _renumber_elements(list(updated.values()))


def apply_runtime_plan_revision(
    *,
    raw_question: str,
    case: PatientCase,
    elements: list[TreatmentPlanElement],
    operations: list[PlanRepairOperation],
) -> list[TreatmentPlanElement]:
    updated = {item.element_id: item for item in elements}
    numeric_ids = [
        int(item.element_id[2:])
        for item in elements
        if item.element_id.startswith("PE") and item.element_id[2:].isdigit()
    ]
    next_id = max(numeric_ids, default=0) + 1
    semantic_groups: dict[tuple[str, str, str], list[str]] = {}
    for item in elements:
        key = (
            item.source_span,
            item.element_type,
            re.sub(r"\s+", "", item.normalized_summary).casefold(),
        )
        semantic_groups.setdefault(key, []).append(item.element_id)
    duplicate_ids = {
        element_id
        for ids in semantic_groups.values()
        if len(ids) > 1
        for element_id in ids
    }

    for operation in operations:
        if operation.operation == "add_element":
            draft = operation.element
            if draft is None:
                raise PlanValidationError("add_element 缺少 element")
            parent = draft.parent_draft_id
            components = list(draft.component_draft_ids)
            invalid = [
                value
                for value in [parent, *components]
                if value is not None and value not in updated
            ]
            if invalid:
                raise PlanValidationError(
                    f"运行时新增要素引用了不存在的既有要素：{invalid}"
                )
            standalone = draft.model_copy(
                update={"parent_draft_id": None, "component_draft_ids": []}
            )
            candidate = canonicalize_plan_elements(
                raw_question=raw_question,
                case=case,
                drafts=[standalone],
            )[0].model_copy(
                update={
                    "element_id": f"PE{next_id:03d}",
                    "parent_element_id": parent,
                    "component_element_ids": components,
                }
            )
            next_id += 1
            updated[candidate.element_id] = candidate
            continue

        target_id = operation.target_element_id
        if not target_id or target_id not in updated:
            raise PlanValidationError(
                f"{operation.operation} 引用了不存在的 target_element_id：{target_id}"
            )
        if operation.operation in {"remove_duplicate", "remove_hallucinated"}:
            if operation.operation == "remove_hallucinated":
                raise PlanValidationError(
                    "运行时不能删除已通过初始 grounding 的要素；"
                    "疑似误抽取必须在初始独立核查阶段修复"
                )
            if target_id not in duplicate_ids:
                raise PlanValidationError(
                    f"不能删除未被确定性校验识别为重复的要素：{target_id}"
                )
            referenced_by = [
                item.element_id
                for item in updated.values()
                if item.parent_element_id == target_id
                or target_id in item.component_element_ids
            ]
            if referenced_by:
                raise PlanValidationError(
                    f"不能删除仍被其它要素引用的 {target_id}：{referenced_by}"
                )
            updated.pop(target_id)
        elif operation.operation == "update_element_type":
            if operation.replacement_type is None:
                raise PlanValidationError("update_element_type 缺少 replacement_type")
            updated[target_id] = updated[target_id].model_copy(
                update={"element_type": operation.replacement_type}
            )
        elif operation.operation == "update_attributes":
            updated[target_id] = updated[target_id].model_copy(
                update={"attributes": operation.replacement_attributes}
            )
        elif operation.operation == "update_relationship":
            references = [
                value
                for value in [
                    operation.replacement_parent_element_id,
                    *operation.replacement_component_element_ids,
                ]
                if value is not None
            ]
            invalid = [value for value in references if value not in updated]
            if invalid:
                raise PlanValidationError(
                    f"运行时关系修订引用了不存在的要素：{invalid}"
                )
            updated[target_id] = updated[target_id].model_copy(
                update={
                    "parent_element_id": operation.replacement_parent_element_id,
                    "component_element_ids": operation.replacement_component_element_ids,
                }
            )

    ordered = sorted(
        updated.values(),
        key=lambda item: (item.source_start, item.source_end, item.element_id),
    )
    issues = validate_plan_inventory(
        raw_question=raw_question,
        case=case,
        elements=ordered,
    )
    errors = [item for item in issues if item.severity == "error"]
    if errors:
        raise PlanValidationError(
            "运行时方案修订未通过校验：" + "；".join(item.message for item in errors)
        )
    return ordered


def issue_dicts(issues: list[PlanValidationIssue]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in issues]
