from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review.models import (
    ClinicalFindingDraft,
    ElementReviewDraft,
    EvidenceItemV3,
    NumericFact,
    PatientCase,
    ReviewSynthesisDraft,
    SourceCitationDraft,
    SupportedStatementDraft,
    TreatmentPlanElement,
)
from yuxi.agents.buildin.medication_review.rendering import render_review_v3
from yuxi.agents.buildin.medication_review.review_synthesis import (
    _canonicalize_review,
    synthesize_review,
)


def _case() -> PatientCase:
    return PatientCase(case_id="CASE-1", raw_question_hash="hash", age=72)


def _element() -> TreatmentPlanElement:
    return TreatmentPlanElement(
        element_id="PE001",
        element_type="medication_order",
        source_span="药物甲 1片 每日一次",
        source_start=0,
        source_end=11,
        normalized_summary="药物甲 1片 每日一次",
    )


def _evidence() -> EvidenceItemV3:
    return EvidenceItemV3(
        evidence_id="EV001",
        content_hash="hash-1",
        raw_text="当指标低于30ml/min时，应调整给药间隔。",
        source_document="专家共识.md",
        chunk_index=18,
    )


def test_invalid_citation_degrades_only_affected_element():
    draft = ReviewSynthesisDraft(
        findings=[
            ClinicalFindingDraft(
                linked_element_ids=["PE001"],
                assessment="concern",
                statement="当前方案需要调整。",
                citations=[
                    SourceCitationDraft(
                        evidence_id="EV001",
                        source_span="原文中不存在的句子",
                    )
                ],
            )
        ],
        element_reviews=[
            ElementReviewDraft(
                element_id="PE001",
                overall_disposition="modify",
                evidence_basis="direct_support",
                summary="需要调整。",
            )
        ],
    )

    review, events, _warnings = _canonicalize_review(
        draft=draft,
        patient_case=_case(),
        patient_facts=[],
        elements=[_element()],
        evidence=[_evidence()],
        claims=[],
        synthesis_mode="direct_chunks",
    )

    assert review.element_reviews[0].overall_disposition == "uncertain"
    assert review.element_reviews[0].evidence_basis == "insufficient"
    assert any(item.action == "dropped" for item in events)
    assert any(item.action == "degraded" for item in events)


def test_grounded_numeric_recommendation_accepts_spacing_difference_and_renders_six_sections():
    citation = SourceCitationDraft(
        evidence_id="EV001",
        source_span="当指标低于30ml/min时，应调整给药间隔。",
    )
    draft = ReviewSynthesisDraft(
        findings=[
            ClinicalFindingDraft(
                linked_element_ids=["PE001"],
                assessment="concern",
                statement="来源提示阈值条件下需要调整。",
                citations=[citation],
            )
        ],
        element_reviews=[
            ElementReviewDraft(
                element_id="PE001",
                overall_disposition="modify",
                evidence_basis="direct_support",
                summary="患者需结合指标复核。",
                recommendation=SupportedStatementDraft(
                    text="指标低于阈值时调整给药间隔。",
                    citations=[citation],
                    source_scope="retrieved_corpus",
                    numeric_facts=[
                        NumericFact(value="30", unit="ml/min", context="指标阈值")
                    ],
                ),
            )
        ],
    )

    review, events, _warnings = _canonicalize_review(
        draft=draft,
        patient_case=_case(),
        patient_facts=[],
        elements=[_element()],
        evidence=[_evidence()],
        claims=[],
        synthesis_mode="direct_chunks",
    )
    answer = render_review_v3(
        review=review,
        plan_elements=[_element()],
        evidence_items=[_evidence()],
        claims=[],
    )

    recommendation = review.element_reviews[0].recommendation
    assert recommendation is not None
    assert recommendation.numeric_facts[0].value == "30"
    assert not any("数值" in item.reason for item in events)
    for heading in (
        "①【原方案要素清单】",
        "②【逐项判断】",
        "③【正面判断汇总】",
        "④【负面与不确定判断汇总】",
        "⑤【综合建议】",
        "⑥【依据清单】",
    ):
        assert heading in answer


@pytest.mark.asyncio
async def test_final_synthesis_provider_failure_is_fatal_not_formal_partial_answer():
    class UnavailableModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("provider unavailable")

    result = await synthesize_review(
        model=UnavailableModel(),
        raw_case_text="药物甲 1片 每日一次",
        patient_case=_case(),
        patient_facts=[],
        elements=[_element()],
        questions=[],
        evidence=[_evidence()],
        claims=[],
        unresolved_questions=[],
        synthesis_mode="direct_chunks",
        system_prompt="",
    )

    assert result.fatal
    assert result.audit["failure_kind"] == "provider"
