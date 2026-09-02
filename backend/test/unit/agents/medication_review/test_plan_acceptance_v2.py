from __future__ import annotations

import hashlib

import pytest

from yuxi.agents.buildin.medication_review.models import (
    Medication,
    PatientCase,
    TreatmentPlanElementDraft,
)
from yuxi.agents.buildin.medication_review.plan_validation import (
    canonicalize_plan_elements,
    validate_plan_inventory,
)


def _case(raw_question: str, medication_mentions: list[str]) -> PatientCase:
    return PatientCase(
        case_id="CASE_ACCEPTANCE",
        raw_question_hash=hashlib.sha256(raw_question.encode()).hexdigest(),
        medications=[
            Medication(
                medication_id=f"MED{index:03d}",
                source_mention=mention,
                status="planned",
            )
            for index, mention in enumerate(medication_mentions, start=1)
        ],
    )


ACCEPTANCE_CASES = [
    pytest.param(
        "给予阿司匹林100mg qd口服。",
        ["阿司匹林"],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="medication_order",
                source_span="阿司匹林100mg qd口服",
                normalized_summary="阿司匹林100mg qd口服",
                medication_mentions=["阿司匹林"],
                attributes={"dose": "100mg", "frequency": "qd", "route": "口服"},
            )
        ],
        ["medication_order"],
        id="single-medication-order",
    ),
    pytest.param(
        "给予异烟肼和利福平联合治疗。",
        ["异烟肼", "利福平"],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="medication_order",
                source_span="异烟肼",
                normalized_summary="异烟肼",
                medication_mentions=["异烟肼"],
            ),
            TreatmentPlanElementDraft(
                draft_id="D2",
                element_type="medication_order",
                source_span="利福平",
                normalized_summary="利福平",
                medication_mentions=["利福平"],
            ),
            TreatmentPlanElementDraft(
                draft_id="D3",
                element_type="combination_regimen",
                source_span="异烟肼和利福平联合治疗",
                normalized_summary="异烟肼和利福平联合治疗",
                component_draft_ids=["D1", "D2"],
                medication_mentions=["异烟肼", "利福平"],
            ),
        ],
        ["medication_order", "combination_regimen", "medication_order"],
        id="combination-regimen",
    ),
    pytest.param(
        "采用诊断性抗结核治疗。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="treatment_intent",
                source_span="诊断性抗结核治疗",
                normalized_summary="诊断性抗结核治疗",
            )
        ],
        ["treatment_intent"],
        id="treatment-intent",
    ),
    pytest.param(
        "先进入强化期治疗。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="treatment_phase",
                source_span="强化期治疗",
                normalized_summary="强化期治疗",
            )
        ],
        ["treatment_phase"],
        id="treatment-phase",
    ),
    pytest.param(
        "整体疗程为6个月。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="duration_or_schedule",
                source_span="整体疗程为6个月",
                normalized_summary="整体疗程为6个月",
                attributes={"duration": "6个月"},
            )
        ],
        ["duration_or_schedule"],
        id="duration",
    ),
    pytest.param(
        "计划治疗3个月后评估疗效。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="evaluation_timing",
                source_span="治疗3个月后评估疗效",
                normalized_summary="治疗3个月后评估疗效",
                attributes={"timing": "3个月后"},
            )
        ],
        ["evaluation_timing"],
        id="evaluation-timing",
    ),
    pytest.param(
        "每2周监测一次肝功能。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="monitoring_plan",
                source_span="每2周监测一次肝功能",
                normalized_summary="每2周监测一次肝功能",
                attributes={"interval": "每2周"},
            )
        ],
        ["monitoring_plan"],
        id="monitoring-plan",
    ),
    pytest.param(
        "治疗后每月门诊随访。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="follow_up_plan",
                source_span="每月门诊随访",
                normalized_summary="每月门诊随访",
                attributes={"interval": "每月"},
            )
        ],
        ["follow_up_plan"],
        id="follow-up-plan",
    ),
    pytest.param(
        "若出现严重皮疹则停药。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="switch_stop_escalation_rule",
                source_span="若出现严重皮疹则停药",
                normalized_summary="若出现严重皮疹则停药",
            )
        ],
        ["switch_stop_escalation_rule"],
        id="stop-rule",
    ),
    pytest.param(
        "建议进行低盐饮食和康复锻炼。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="nonpharmacologic_plan",
                source_span="低盐饮食和康复锻炼",
                normalized_summary="低盐饮食和康复锻炼",
            )
        ],
        ["nonpharmacologic_plan"],
        id="nonpharmacologic-plan",
    ),
    pytest.param(
        "安排多学科会诊。",
        [],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="other_explicit_plan",
                source_span="多学科会诊",
                normalized_summary="多学科会诊",
            )
        ],
        ["other_explicit_plan"],
        id="other-explicit-plan",
    ),
    pytest.param(
        "给予甲氨蝶呤10mg每周一次，并每月复查血常规。",
        ["甲氨蝶呤"],
        [
            TreatmentPlanElementDraft(
                draft_id="D1",
                element_type="medication_order",
                source_span="甲氨蝶呤10mg每周一次",
                normalized_summary="甲氨蝶呤10mg每周一次",
                medication_mentions=["甲氨蝶呤"],
                attributes={"dose": "10mg", "frequency": "每周一次"},
            ),
            TreatmentPlanElementDraft(
                draft_id="D2",
                element_type="monitoring_plan",
                source_span="每月复查血常规",
                normalized_summary="每月复查血常规",
                attributes={"interval": "每月"},
            ),
        ],
        ["medication_order", "monitoring_plan"],
        id="medication-with-independent-monitoring",
    ),
]


@pytest.mark.parametrize(
    ("raw_question", "medication_mentions", "drafts", "expected_types"),
    ACCEPTANCE_CASES,
)
def test_twelve_representative_plan_inventories_pass_deterministic_validation(
    raw_question: str,
    medication_mentions: list[str],
    drafts: list[TreatmentPlanElementDraft],
    expected_types: list[str],
):
    case = _case(raw_question, medication_mentions)

    elements = canonicalize_plan_elements(
        raw_question=raw_question,
        case=case,
        drafts=drafts,
    )
    errors = [
        issue
        for issue in validate_plan_inventory(
            raw_question=raw_question,
            case=case,
            elements=elements,
        )
        if issue.severity == "error"
    ]

    assert [element.element_type for element in elements] == expected_types
    assert errors == []
