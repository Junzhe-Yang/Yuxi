from __future__ import annotations

from yuxi.agents.buildin.medication_review.models import (
    Diagnosis,
    LabValue,
    Medication,
    PatientCase,
)
from yuxi.agents.buildin.medication_review.planning import build_review_plan


def _case(medication_count: int = 3) -> PatientCase:
    medications = [
        Medication(
            medication_id=f"M{index:03d}",
            source_mention=f"药物{index}",
            normalized_name=f"药物{index}",
            normalization_source="input",
            status="current",
        )
        for index in range(1, medication_count + 1)
    ]
    return PatientCase(
        case_id="CASE-test",
        raw_question_hash="hash",
        age=70,
        diagnoses=[
            Diagnosis(diagnosis_id="D001", name="活动疾病", source_mention="活动疾病", status="active"),
            Diagnosis(diagnosis_id="D002", name="既往疾病", source_mention="既往疾病", status="history"),
        ],
        medications=medications,
        renal_function=LabValue(indicator="eGFR", value="35", unit="mL/min"),
    )


def test_relation_counts_follow_frozen_formula_and_all_queryable_slots_are_covered():
    patient_case = _case(3)

    slots, bundles = build_review_plan(patient_case)

    slot_counts: dict[str, int] = {}
    for slot in slots:
        slot_counts[slot.slot_type] = slot_counts.get(slot.slot_type, 0) + 1
        if slot.applicability_status == "queryable":
            assert slot.covered_by_bundle_ids
    assert slot_counts == {
        "medication_profile": 3,
        "organ_function": 6,
        "drug_drug": 3,
        "drug_disease": 6,
        "prescribing_omission": 1,
        "duplication": 1,
        "cumulative_burden": 1,
    }
    assert len(bundles) == 3 + 6 + 3 + 6 + 1 + 1


def test_drug_pairs_are_unordered_and_duplication_reuses_pair_queries():
    slots, bundles = build_review_plan(_case(3))

    pair_bundles = [item for item in bundles if item.template_id == "drug_drug"]

    assert [item.bundle_id for item in pair_bundles] == [
        "QB:DD:M001:M002",
        "QB:DD:M001:M003",
        "QB:DD:M002:M003",
    ]
    assert all("DUP:CASE" in item.slot_ids for item in pair_bundles)
    duplication = next(item for item in slots if item.slot_id == "DUP:CASE")
    assert duplication.covered_by_bundle_ids == [item.bundle_id for item in pair_bundles]


def test_organ_queries_are_independent_from_medication_profile():
    _slots, bundles = build_review_plan(_case(1))

    profile = next(item for item in bundles if item.template_id == "medication_profile")
    renal = next(item for item in bundles if item.template_id == "organ_renal")
    hepatic = next(item for item in bundles if item.template_id == "organ_hepatic")

    assert profile.slot_ids == ["MP:M001"]
    assert renal.slot_ids == ["OF:RENAL:M001"]
    assert hepatic.slot_ids == ["OF:HEPATIC:M001"]


def test_burden_queries_group_at_most_six_medications():
    _slots, bundles = build_review_plan(_case(8))

    burden = [item for item in bundles if item.template_id == "cumulative_burden"]

    assert len(burden) == 2
    assert len(burden[0].core_entity_ids) == 6
    assert len(burden[1].core_entity_ids) == 2

