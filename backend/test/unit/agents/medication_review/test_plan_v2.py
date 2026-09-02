from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review.extraction import (
    CasePlanExtractionError,
    _drop_ungrounded_clinical_risks,
    extract_case_and_plan,
)
from yuxi.agents.buildin.medication_review.models import (
    CasePlanExtractionDraft,
    Medication,
    PatientCaseInput,
    PlanVerificationResult,
    TreatmentPlanElementDraft,
)
from yuxi.agents.buildin.medication_review.plan_validation import (
    canonicalize_plan_elements,
    validate_plan_inventory,
)
from yuxi.agents.buildin.medication_review.planning import assign_stable_ids
from yuxi.agents.buildin.medication_review.planning import validate_grounding


RAW_HRZE_CASE = (
    "72岁男性，疑诊肺结核。原方案为异烟肼片300mg qd、利福平胶囊450mg qd、"
    "吡嗪酰胺片1.0g qd、乙胺丁醇片750mg qd，采用HRZE四联诊断性治疗，"
    "计划3个月后评估疗效。"
)


def _patient_case_input() -> PatientCaseInput:
    return PatientCaseInput(
        age=72,
        diagnoses=[
            {
                "name": "肺结核",
                "source_mention": "肺结核",
                "status": "suspected",
            }
        ],
        medications=[
            Medication(source_mention="异烟肼片", status="planned"),
            Medication(source_mention="利福平胶囊", status="planned"),
            Medication(source_mention="吡嗪酰胺片", status="planned"),
            Medication(source_mention="乙胺丁醇片", status="planned"),
        ],
    )


def _plan_drafts() -> list[TreatmentPlanElementDraft]:
    medication_spans = [
        ("E1", "异烟肼片300mg qd", "异烟肼片"),
        ("E2", "利福平胶囊450mg qd", "利福平胶囊"),
        ("E3", "吡嗪酰胺片1.0g qd", "吡嗪酰胺片"),
        ("E4", "乙胺丁醇片750mg qd", "乙胺丁醇片"),
    ]
    drafts = [
        TreatmentPlanElementDraft(
            draft_id=draft_id,
            element_type="medication_order",
            source_span=span,
            normalized_summary=span,
            medication_mentions=[mention],
            attributes={"dose_frequency": span.removeprefix(mention)},
        )
        for draft_id, span, mention in medication_spans
    ]
    drafts.extend(
        [
            TreatmentPlanElementDraft(
                draft_id="E5",
                element_type="combination_regimen",
                source_span="HRZE四联诊断性治疗",
                normalized_summary="HRZE 四联诊断性治疗",
                component_draft_ids=["E1", "E2", "E3", "E4"],
                medication_mentions=[item[2] for item in medication_spans],
                target_diagnosis_mentions=["肺结核"],
                attributes={"treatment_intent": "diagnostic"},
            ),
            TreatmentPlanElementDraft(
                draft_id="E6",
                element_type="evaluation_timing",
                source_span="计划3个月后评估疗效",
                normalized_summary="治疗后3个月评估疗效",
                target_diagnosis_mentions=["肺结核"],
                attributes={"timing": "3个月后", "evaluation": "疗效"},
            ),
        ]
    )
    return drafts


def test_grounding_does_not_make_semantic_status_decisions() -> None:
    raw_question = (
        "疾病甲。药物乙 5mg 口服每日一次。"
    )
    patient_case = PatientCaseInput(
        diagnoses=[
            {
                "name": "疾病甲",
                "source_mention": "疾病甲",
                "status": "active",
            },
        ],
        medications=[
            Medication(
                source_mention="药物乙",
                dose="5mg",
                route="口服",
                frequency="每日一次",
                status="current",
            )
        ],
    )

    assert validate_grounding(raw_question, patient_case) == []


def test_hrze_plan_is_canonicalized_as_six_independently_judgeable_elements():
    patient_case = assign_stable_ids(RAW_HRZE_CASE, _patient_case_input())

    elements = canonicalize_plan_elements(
        raw_question=RAW_HRZE_CASE,
        case=patient_case,
        drafts=_plan_drafts(),
    )
    issues = validate_plan_inventory(
        raw_question=RAW_HRZE_CASE,
        case=patient_case,
        elements=elements,
    )

    assert len(elements) == 6
    assert [item.element_type for item in elements] == [
        "medication_order",
        "medication_order",
        "medication_order",
        "medication_order",
        "combination_regimen",
        "evaluation_timing",
    ]
    assert elements[4].attributes["treatment_intent"] == "diagnostic"
    assert elements[4].component_element_ids == ["PE001", "PE002", "PE003", "PE004"]
    assert [item for item in issues if item.severity == "error"] == []


@pytest.mark.asyncio
async def test_free_text_extraction_uses_explicit_json_schema_and_independent_verifier():
    draft = CasePlanExtractionDraft(
        patient_case=_patient_case_input(),
        plan_elements=_plan_drafts(),
    )
    prompts: list[str] = []

    class JsonOnlyModel:
        async def ainvoke(self, messages):
            prompts.append(messages[-1].content)
            if len(prompts) == 1:
                return AIMessage(content=draft.model_dump_json())
            return AIMessage(content=PlanVerificationResult().model_dump_json())

    result = await extract_case_and_plan(
        raw_question=RAW_HRZE_CASE,
        model=JsonOnlyModel(),
        system_prompt="保持完整治疗方案要素",
    )

    assert len(result.plan_elements) == 6
    assert len(prompts) == 2
    assert all("目标 JSON Schema" in prompt for prompt in prompts)
    assert '"plan_elements"' in prompts[0]
    assert '"repair_operations"' in prompts[1]
    assert result.audit.extraction is not None
    assert result.audit.verification is not None


@pytest.mark.asyncio
async def test_verifier_format_failure_keeps_grounded_extracted_plan():
    draft = CasePlanExtractionDraft(
        patient_case=_patient_case_input(),
        plan_elements=_plan_drafts(),
    )
    calls = 0

    class FailingVerifierModel:
        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content=draft.model_dump_json())
            return AIMessage(content="not-json")

    result = await extract_case_and_plan(
        raw_question=RAW_HRZE_CASE,
        model=FailingVerifierModel(),
        system_prompt="保持完整治疗方案要素",
    )

    assert len(result.plan_elements) == 6
    assert result.degraded
    assert any("保留已通过确定性校验" in item for item in result.warnings)
    assert result.audit.verification is not None


@pytest.mark.asyncio
async def test_grounded_plan_reference_recovers_entity_omitted_from_patient_case():
    raw_question = "处方：盐酸特拉唑嗪片 2mg 口服 每晚一次。"
    draft = CasePlanExtractionDraft(
        patient_case=PatientCaseInput(),
        plan_elements=[
            TreatmentPlanElementDraft(
                draft_id="med_order_1",
                element_type="medication_order",
                source_span="盐酸特拉唑嗪片 2mg 口服 每晚一次",
                normalized_summary="盐酸特拉唑嗪片 2mg 口服 每晚一次",
                medication_mentions=["盐酸特拉唑嗪片"],
                attributes={
                    "dose": "2mg",
                    "route": "口服",
                    "frequency": "每晚一次",
                },
            )
        ],
    )
    calls = 0

    class EntityOmittingModel:
        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AIMessage(content=draft.model_dump_json())
            return AIMessage(content=PlanVerificationResult().model_dump_json())

    result = await extract_case_and_plan(
        raw_question=raw_question,
        model=EntityOmittingModel(),
        system_prompt="完整审查治疗方案",
    )

    assert [item.source_mention for item in result.patient_case.medications] == [
        "盐酸特拉唑嗪片"
    ]
    assert result.plan_elements[0].medication_ids == ["M001"]
    assert result.audit.runtime_revisions[0]["action"] == "add_grounded_medication"
    assert any("确定性补齐" in item for item in result.warnings)


@pytest.mark.asyncio
async def test_ungrounded_plan_reference_is_not_recovered():
    raw_question = "处方：药物甲 5mg 口服 每日一次。"
    draft = CasePlanExtractionDraft(
        patient_case=PatientCaseInput(),
        plan_elements=[
            TreatmentPlanElementDraft(
                draft_id="med_order_1",
                element_type="medication_order",
                source_span="药物甲 5mg 口服 每日一次",
                normalized_summary="药物甲 5mg 口服 每日一次",
                medication_mentions=["药物乙"],
            )
        ],
    )

    class HallucinatingModel:
        async def ainvoke(self, _messages):
            return AIMessage(content=draft.model_dump_json())

    with pytest.raises(CasePlanExtractionError, match="无法解析的药物提及：药物乙"):
        await extract_case_and_plan(
            raw_question=raw_question,
            model=HallucinatingModel(),
            system_prompt="完整审查治疗方案",
        )


def test_inferred_risk_summary_is_dropped_and_short_drug_mention_still_binds():
    raw_question = (
        "立位血压110/70mmHg（伴头晕）。"
        "盐酸特拉唑嗪片 2mg 口服 每晚一次。"
    )
    draft = CasePlanExtractionDraft(
        patient_case=PatientCaseInput(
            medications=[
                Medication(
                    source_mention="盐酸特拉唑嗪片 2mg 口服 每晚一次",
                    generic_name="盐酸特拉唑嗪",
                    normalized_name="盐酸特拉唑嗪",
                    dose="2",
                    dose_unit="mg",
                    route="口服",
                    frequency="每晚一次",
                    status="current",
                )
            ],
            clinical_risks=["体位性低血压风险"],
        ),
        patient_facts=[
            {
                "fact_type": "clinical_risk",
                "source_span": "立位血压110/70mmHg（伴头晕）",
                "normalized_summary": "体位性低血压伴头晕",
            }
        ],
        plan_elements=[
            TreatmentPlanElementDraft(
                draft_id="MO1",
                element_type="medication_order",
                source_span="盐酸特拉唑嗪片 2mg 口服 每晚一次",
                normalized_summary="盐酸特拉唑嗪片 2mg 口服 每晚一次",
                medication_mentions=["盐酸特拉唑嗪片"],
                attributes={
                    "dose": "2mg",
                    "route": "口服",
                    "frequency": "每晚一次",
                },
            )
        ],
    )

    sanitized, warnings = _drop_ungrounded_clinical_risks(
        raw_question=raw_question,
        draft=draft,
    )
    assert sanitized.patient_case.clinical_risks == []
    assert warnings and "已移除" in warnings[0]
    assert validate_grounding(raw_question, sanitized.patient_case) == []

    patient_case = assign_stable_ids(
        raw_question,
        sanitized.patient_case,
    )
    elements = canonicalize_plan_elements(
        raw_question=raw_question,
        case=patient_case,
        drafts=sanitized.plan_elements,
    )

    assert elements[0].medication_ids == ["M001"]


def test_grounding_allows_semantic_normalization_but_rejects_new_numbers():
    raw_question = "药物甲 1g po qd。"
    normalized = PatientCaseInput(
        medications=[
            Medication(
                source_mention="药物甲",
                dose="1.0",
                dose_unit="g",
                route="口服",
                frequency="每日一次",
                status="planned",
            )
        ]
    )
    hallucinated = normalized.model_copy(
        update={
            "medications": [
                normalized.medications[0].model_copy(update={"dose": "1.5"})
            ]
        }
    )

    assert validate_grounding(raw_question, normalized) == []
    assert "dose 数值未在原文出现" in validate_grounding(
        raw_question,
        hallucinated,
    )[0]
