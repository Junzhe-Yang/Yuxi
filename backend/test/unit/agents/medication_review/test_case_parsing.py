from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review.planning import (
    CaseExtractionError,
    assign_stable_ids,
    extract_patient_case,
    validate_grounding,
)
from yuxi.agents.buildin.medication_review.models import Diagnosis, Medication, PatientCaseInput


def _structured_case() -> dict:
    return {
        "age": 72,
        "sex": "男",
        "diagnoses": [
            {
                "name": "慢性肾脏病3期",
                "source_mention": "慢性肾脏病3期",
                "status": "active",
            },
            {
                "name": "肺结核",
                "source_mention": "肺结核",
                "status": "active",
            },
        ],
        "medications": [
            {
                "source_mention": "吡嗪酰胺",
                "generic_name": "吡嗪酰胺",
                "normalization_source": "input",
                "dose": "1.5",
                "dose_unit": "g",
                "frequency": "qd",
                "status": "current",
            }
        ],
        "renal_function": {
            "indicator": "eGFR",
            "value": "35",
            "unit": "mL/min",
            "source_mention": "eGFR 35 mL/min",
        },
    }


@pytest.mark.asyncio
async def test_structured_json_input_bypasses_model_and_assigns_stable_ids():
    raw_question = json.dumps(_structured_case(), ensure_ascii=False)

    patient_case, extraction_mode, warnings = await extract_patient_case(raw_question, None, "")

    assert extraction_mode == "structured_input"
    assert warnings == []
    assert patient_case.medications[0].medication_id == "M001"
    assert [item.diagnosis_id for item in patient_case.diagnoses] == ["D001", "D002"]
    assert patient_case.renal_function.lab_id == "R001"
    assert patient_case.case_id.startswith("CASE-")


def test_stable_ids_follow_first_source_position_not_model_output_order():
    raw_question = "患者先使用链霉素，后加用异烟肼，诊断肺结核。"
    extracted = PatientCaseInput(
        diagnoses=[Diagnosis(name="肺结核", source_mention="肺结核", status="active")],
        medications=[
            Medication(source_mention="异烟肼", status="current"),
            Medication(source_mention="链霉素", status="current"),
        ],
    )

    patient_case = assign_stable_ids(raw_question, extracted)

    assert [(item.medication_id, item.source_mention) for item in patient_case.medications] == [
        ("M001", "链霉素"),
        ("M002", "异烟肼"),
    ]


def test_grounding_rejects_invented_drug_and_metric_conversion():
    raw_question = "患者使用链霉素，eGFR 35 mL/min。"
    extracted = PatientCaseInput(
        medications=[Medication(source_mention="异烟肼", status="current")],
        renal_function={
            "indicator": "CrCl",
            "value": "35",
            "unit": "mL/min",
        },
    )

    errors = validate_grounding(raw_question, extracted)

    assert any("异烟肼" in error for error in errors)
    assert any("CrCl" in error for error in errors)


@pytest.mark.asyncio
async def test_invalid_structured_case_stops_without_partial_plan():
    payload = _structured_case()
    payload["medications"][0]["source_mention"] = "原文不存在的药物"

    with pytest.raises(CaseExtractionError):
        await extract_patient_case(json.dumps(payload, ensure_ascii=False), None, "")


@pytest.mark.asyncio
async def test_json_fallback_prompt_contains_complete_patient_case_schema():
    raw_question = (
        "72岁男性，诊断慢性肾脏病3期和肺结核，当前使用吡嗪酰胺1.5 g qd，"
        "eGFR 35 mL/min。"
    )
    captured_prompts: list[str] = []

    class NoStructuredOutputModel:
        def with_structured_output(self, _schema):
            raise NotImplementedError("gateway does not support structured output")

        async def ainvoke(self, messages):
            captured_prompts.append(messages[-1].content)
            return AIMessage(content=json.dumps(_structured_case(), ensure_ascii=False))

    patient_case, extraction_mode, warnings = await extract_patient_case(
        raw_question,
        NoStructuredOutputModel(),
        "",
    )

    fallback_prompt = captured_prompts[0]
    assert extraction_mode == "json_fallback"
    assert patient_case.medications[0].source_mention == "吡嗪酰胺"
    assert warnings == ["结构化输出不可用，已改用 JSON 模式：NotImplementedError"]
    assert "目标 JSON Schema" in fallback_prompt
    assert '"additionalProperties":false' in fallback_prompt
    assert '"medications"' in fallback_prompt
    assert '"source_mention"' in fallback_prompt
    assert '"dosage"' not in fallback_prompt
