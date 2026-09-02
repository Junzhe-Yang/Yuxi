from __future__ import annotations

from yuxi.agents.buildin.medication_review.models import Diagnosis, LabValue, Medication, PatientCase
from yuxi.agents.buildin.medication_review.planning import build_review_plan


def _case() -> PatientCase:
    return PatientCase(
        case_id="CASE-test",
        raw_question_hash="hash",
        age=72,
        diagnoses=[
            Diagnosis(
                diagnosis_id="D001",
                name="慢性肾脏病3期",
                source_mention="慢性肾脏病3期",
                status="active",
            )
        ],
        medications=[
            Medication(
                medication_id="M001",
                source_mention="吡嗪酰胺",
                generic_name="吡嗪酰胺",
                normalization_source="input",
                dose="1.5",
                dose_unit="g",
                frequency="qd",
                route="po",
                status="current",
            )
        ],
        renal_function=LabValue(indicator="eGFR", value="35", unit="mL/min"),
    )


def test_queries_are_short_natural_language_atomic_propositions():
    _slots, bundles = build_review_plan(_case())

    assert all(item.query_style == "atomic_clinical_proposition_v1" for item in bundles)
    assert all(item.retrieval_view == "dense" for item in bundles)
    assert all(item.query_text.endswith("？") for item in bundles)
    assert all("请检索" not in item.query_text and "关键词" not in item.query_text for item in bundles)
    assert all(item.validation_status == "valid" for item in bundles)


def test_renal_query_preserves_exact_metric_and_does_not_convert_to_crcl():
    _slots, bundles = build_review_plan(_case())

    renal = next(item for item in bundles if item.template_id == "organ_renal")

    assert "eGFR 35 mL/min" in renal.query_text
    assert "CrCl" not in renal.query_text
    assert renal.query_text == (
        "eGFR 35 mL/min的72岁老年慢性肾脏病3期患者使用吡嗪酰胺时，"
        "是否需要减量、延长给药间隔或避免使用？"
    )


def test_medication_profile_keeps_dose_but_excludes_organ_relation():
    _slots, bundles = build_review_plan(_case())

    profile = next(item for item in bundles if item.template_id == "medication_profile")

    assert "1.5 g、qd、po" in profile.query_text
    assert "肾功能" not in profile.query_text
    assert "肝功能" not in profile.query_text


def test_query_does_not_include_gold_document_or_unreasonable_conclusion():
    _slots, bundles = build_review_plan(_case())

    combined = "\n".join(item.query_text for item in bundles)

    assert "金标准" not in combined
    assert "老年肺结核诊断与治疗专家共识" not in combined
    assert "不合理" not in combined

