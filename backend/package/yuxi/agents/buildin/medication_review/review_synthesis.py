from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .llm_io import StructuredOutputError, invoke_json_schema
from .models import (
    ClinicalFinding,
    ClinicalFindingDraft,
    ElementReview,
    ElementReviewDraft,
    EvidenceClaim,
    EvidenceItemV3,
    LocalValidationEvent,
    NumericFact,
    PatientCase,
    ReviewQuestion,
    ReviewSynthesis,
    ReviewSynthesisDraft,
    SourceCitationDraft,
    SupportedStatement,
    SupportedStatementDraft,
    SynthesisMode,
    TreatmentPlanElement,
)
from .plan_validation import CONCRETE_VALUE, concrete_value_supported
from .prompt import REVIEW_SYNTHESIS_PROMPT, REVIEW_SYNTHESIS_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReviewSynthesisResult:
    review: ReviewSynthesis
    audit: dict[str, Any]
    validation_events: list[LocalValidationEvent]
    warnings: list[str]
    degraded: bool = False
    fatal: bool = False


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _compact_normalized(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _span_grounded(span: str, raw_text: str) -> bool:
    return bool(span.strip()) and _normalized(span) in _normalized(raw_text)


def _validate_citations(
    *,
    citations: list[SourceCitationDraft],
    synthesis_mode: SynthesisMode,
    evidence_by_id: dict[str, EvidenceItemV3],
    claims_by_id: dict[str, EvidenceClaim],
    object_type: str,
    object_id: str | None,
) -> tuple[list[SourceCitationDraft], list[LocalValidationEvent]]:
    valid: list[SourceCitationDraft] = []
    events: list[LocalValidationEvent] = []
    seen: set[tuple[str, str | None, str]] = set()
    for citation in citations:
        item = evidence_by_id.get(citation.evidence_id)
        reason: str | None = None
        if item is None:
            reason = "citation 引用了未知 Evidence"
        elif not _span_grounded(citation.source_span, item.raw_text):
            reason = "citation source_span 未落回 Evidence 原文"
        elif synthesis_mode == "direct_chunks" and citation.claim_id is not None:
            reason = "direct_chunks 模式不得引用 Claim"
        elif synthesis_mode == "claims":
            claim = claims_by_id.get(citation.claim_id or "")
            if claim is None:
                reason = "claims 模式必须引用真实 Claim"
            elif claim.evidence_id != citation.evidence_id:
                reason = "Claim 与 Evidence 不一致"
            elif not _span_grounded(citation.source_span, claim.source_span):
                reason = "citation source_span 未落回 Claim 原文"
        if reason is not None:
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type=object_type,
                    object_id=object_id,
                    action="dropped",
                    reason=reason,
                )
            )
            continue
        key = (
            citation.evidence_id,
            citation.claim_id,
            _normalized(citation.source_span),
        )
        if key not in seen:
            valid.append(citation)
            seen.add(key)
    return valid, events


def _statement_ids(
    citations: list[SourceCitationDraft],
) -> tuple[list[str], list[str]]:
    evidence_ids = list(dict.fromkeys(item.evidence_id for item in citations))
    claim_ids = list(
        dict.fromkeys(
            item.claim_id for item in citations if item.claim_id is not None
        )
    )
    return evidence_ids, claim_ids


def _fact_is_grounded(
    fact: NumericFact,
    citations: list[SourceCitationDraft],
) -> bool:
    needle = _compact_normalized(f"{fact.value}{fact.unit or ''}")
    if not needle:
        return False
    return any(needle in _compact_normalized(item.source_span) for item in citations)


def _unsupported_concrete_values(
    text: str,
    citations: list[SourceCitationDraft],
) -> list[str]:
    source = "\n".join(item.source_span for item in citations)
    return [
        match.group(0).strip()
        for match in CONCRETE_VALUE.finditer(text)
        if not concrete_value_supported(match.group(0).strip(), source)
    ]


def _normalize_supported_statement(
    *,
    draft: SupportedStatementDraft | None,
    synthesis_mode: SynthesisMode,
    evidence_by_id: dict[str, EvidenceItemV3],
    claims_by_id: dict[str, EvidenceClaim],
    object_type: str,
    object_id: str | None,
    fallback_text: str,
) -> tuple[SupportedStatement | None, list[LocalValidationEvent]]:
    if draft is None:
        return None, []
    citations, events = _validate_citations(
        citations=draft.citations,
        synthesis_mode=synthesis_mode,
        evidence_by_id=evidence_by_id,
        claims_by_id=claims_by_id,
        object_type=object_type,
        object_id=object_id,
    )
    grounded_terms = [
        term
        for term in dict.fromkeys(draft.grounded_terms)
        if term.strip()
        and any(_normalized(term) in _normalized(item.source_span) for item in citations)
    ]
    numeric_facts = [
        fact for fact in draft.numeric_facts if _fact_is_grounded(fact, citations)
    ]
    source_scope = draft.source_scope
    text = draft.text.strip()
    unsupported_values = _unsupported_concrete_values(text, citations)
    if unsupported_values:
        source_scope = "general_review"
        text = fallback_text
        citations = []
        grounded_terms = []
        numeric_facts = []
        events.append(
            LocalValidationEvent(
                stage="review_synthesis",
                object_type=object_type,
                object_id=object_id,
                action="degraded",
                reason=(
                    "具体参数未落回合法 citation，已降级为一般复核建议："
                    f"{unsupported_values}"
                ),
            )
        )
    if source_scope == "retrieved_corpus" and not citations:
        source_scope = "general_review"
        text = fallback_text
        grounded_terms = []
        numeric_facts = []
        events.append(
            LocalValidationEvent(
                stage="review_synthesis",
                object_type=object_type,
                object_id=object_id,
                action="degraded",
                reason="来源型建议失去全部合法 citation，已降级为一般复核建议",
            )
        )
    elif source_scope == "general_review":
        grounded_terms = []
        numeric_facts = []
    evidence_ids, claim_ids = _statement_ids(citations)
    return (
        SupportedStatement(
            text=text or fallback_text,
            citations=citations,
            source_scope=source_scope,
            grounded_terms=grounded_terms,
            numeric_facts=numeric_facts,
            evidence_ids=evidence_ids,
            claim_ids=claim_ids,
        ),
        events,
    )


def _fallback_review(
    *,
    case_id: str,
    elements: list[TreatmentPlanElement],
    unresolved_items: list[str],
) -> ReviewSynthesis:
    return ReviewSynthesis(
        case_id=case_id,
        element_reviews=[
            ElementReview(
                element_id=element.element_id,
                overall_disposition="uncertain",
                evidence_basis="insufficient",
                summary="当前证据或结构化综合结果不足，无法形成可靠的患者级判断。",
                missing_information=["需要进一步复核患者资料和适用指南证据"],
            )
            for element in elements
        ],
        unresolved_items=list(dict.fromkeys(unresolved_items)),
    )


def _canonicalize_review(
    *,
    draft: ReviewSynthesisDraft,
    patient_case: PatientCase,
    patient_facts: list[dict[str, Any]],
    elements: list[TreatmentPlanElement],
    evidence: list[EvidenceItemV3],
    claims: list[EvidenceClaim],
    synthesis_mode: SynthesisMode,
) -> tuple[ReviewSynthesis, list[LocalValidationEvent], list[str]]:
    element_ids = {item.element_id for item in elements}
    fact_ids = {
        str(item.get("fact_id")) for item in patient_facts if item.get("fact_id")
    }
    evidence_by_id = {item.evidence_id: item for item in evidence}
    claims_by_id = {item.claim_id: item for item in claims}
    events: list[LocalValidationEvent] = []
    warnings: list[str] = []
    findings: list[ClinicalFinding] = []

    for source in draft.findings:
        linked_elements = [
            value for value in dict.fromkeys(source.linked_element_ids)
            if value in element_ids
        ]
        linked_facts = [
            value for value in dict.fromkeys(source.linked_patient_fact_ids)
            if value in fact_ids
        ]
        finding_id = f"F{len(findings) + 1:03d}"
        if not linked_elements and not linked_facts:
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="finding",
                    object_id=finding_id,
                    action="dropped",
                    reason="finding 没有合法方案要素或患者事实关联",
                )
            )
            continue
        citations, citation_events = _validate_citations(
            citations=source.citations,
            synthesis_mode=synthesis_mode,
            evidence_by_id=evidence_by_id,
            claims_by_id=claims_by_id,
            object_type="finding",
            object_id=finding_id,
        )
        events.extend(citation_events)
        assessment = source.assessment
        statement = source.statement
        unsupported_values = _unsupported_concrete_values(statement, citations)
        if unsupported_values:
            assessment = "uncertain"
            statement = "该方面包含未通过来源校验的具体参数，需要结合原始来源进一步复核。"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="finding",
                    object_id=finding_id,
                    action="degraded",
                    reason=f"finding 具体参数未落回 citation：{unsupported_values}",
                )
            )
        if assessment == "concern" and not citations:
            assessment = "uncertain"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="finding",
                    object_id=finding_id,
                    action="degraded",
                    reason="负向 finding 失去全部合法 citation",
                )
            )
        evidence_ids, claim_ids = _statement_ids(citations)
        findings.append(
            ClinicalFinding(
                **source.model_dump(
                    exclude={
                        "linked_element_ids",
                        "linked_patient_fact_ids",
                        "assessment",
                        "statement",
                        "citations",
                    }
                ),
                finding_id=finding_id,
                linked_element_ids=linked_elements,
                linked_patient_fact_ids=linked_facts,
                assessment=assessment,
                statement=statement,
                citations=citations,
                evidence_ids=evidence_ids,
                claim_ids=claim_ids,
            )
        )

    drafts_by_element: dict[str, ElementReviewDraft] = {}
    for source in draft.element_reviews:
        if source.element_id not in element_ids:
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=source.element_id,
                    action="dropped",
                    reason="ElementReview 引用了未知方案要素",
                )
            )
            continue
        if source.element_id in drafts_by_element:
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=source.element_id,
                    action="dropped",
                    reason="重复 ElementReview，保留第一条",
                )
            )
            continue
        drafts_by_element[source.element_id] = source

    element_reviews: list[ElementReview] = []
    for element in elements:
        source = drafts_by_element.get(element.element_id)
        linked_findings = [
            item for item in findings if element.element_id in item.linked_element_ids
        ]
        if source is None:
            element_reviews.append(
                ElementReview(
                    element_id=element.element_id,
                    overall_disposition="uncertain",
                    evidence_basis="insufficient",
                    summary="结构化综合未返回该方案要素，程序已补充不确定结果。",
                    missing_information=["需要重新综合该方案要素"],
                    finding_ids=[item.finding_id for item in linked_findings],
                    evidence_ids=list(
                        dict.fromkeys(
                            evidence_id
                            for item in linked_findings
                            for evidence_id in item.evidence_ids
                        )
                    ),
                )
            )
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=element.element_id,
                    action="repaired",
                    reason="缺失 ElementReview，已补充 uncertain + insufficient",
                )
            )
            continue

        recommendation, recommendation_events = _normalize_supported_statement(
            draft=source.recommendation,
            synthesis_mode=synthesis_mode,
            evidence_by_id=evidence_by_id,
            claims_by_id=claims_by_id,
            object_type="recommendation",
            object_id=element.element_id,
            fallback_text="建议由相关临床医师或药师结合完整资料复核是否需要调整方案。",
        )
        monitoring, monitoring_events = _normalize_supported_statement(
            draft=source.monitoring,
            synthesis_mode=synthesis_mode,
            evidence_by_id=evidence_by_id,
            claims_by_id=claims_by_id,
            object_type="monitoring",
            object_id=element.element_id,
            fallback_text="建议根据患者情况制定并复核监测计划。",
        )
        events.extend(recommendation_events)
        events.extend(monitoring_events)
        evidence_ids = list(
            dict.fromkeys(
                [
                    *[
                        evidence_id
                        for item in linked_findings
                        for evidence_id in item.evidence_ids
                    ],
                    *(recommendation.evidence_ids if recommendation else []),
                    *(monitoring.evidence_ids if monitoring else []),
                ]
            )
        )
        disposition = source.overall_disposition
        basis = source.evidence_basis
        if basis in {"direct_support", "mixed_evidence"} and not evidence_ids:
            basis = "insufficient"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=element.element_id,
                    action="degraded",
                    reason=f"{source.evidence_basis} 没有合法 citation",
                )
            )
        if basis == "no_material_conflict_found" and not evidence:
            basis = "insufficient"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=element.element_id,
                    action="degraded",
                    reason="没有任何最终 Evidence，不能声称未发现实质冲突",
                )
            )
        if basis == "insufficient" and disposition != "uncertain":
            disposition = "uncertain"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=element.element_id,
                    action="degraded",
                    reason="证据基础为 insufficient，明确处置已降级为 uncertain",
                )
            )
        if disposition in {"avoid", "modify"} and not any(
            item.assessment == "concern" and item.evidence_ids
            for item in linked_findings
        ):
            disposition = "uncertain"
            basis = "insufficient"
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="element_review",
                    object_id=element.element_id,
                    action="degraded",
                    reason="明确调整/避免处置没有带来源的 concern finding",
                )
            )
        if disposition in {"avoid", "modify"} and recommendation is None:
            recommendation = SupportedStatement(
                text=(
                    "当前检索证据支持需要调整或避免该方案要素，但未提供可直接采用的"
                    "具体替代方案；建议由相关临床医师或药师结合完整资料制定替代方案。"
                ),
                source_scope="general_review",
            )
            events.append(
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="recommendation",
                    object_id=element.element_id,
                    action="repaired",
                    reason="明确调整/避免处置缺少替代建议，已补充来源不足说明",
                )
            )
        element_reviews.append(
            ElementReview(
                **source.model_dump(
                    exclude={
                        "overall_disposition",
                        "evidence_basis",
                        "recommendation",
                        "monitoring",
                    }
                ),
                overall_disposition=disposition,
                evidence_basis=basis,
                recommendation=recommendation,
                monitoring=monitoring,
                finding_ids=[item.finding_id for item in linked_findings],
                evidence_ids=evidence_ids,
            )
        )

    integrated: list[SupportedStatement] = []
    for index, source in enumerate(draft.integrated_recommendations, start=1):
        statement, statement_events = _normalize_supported_statement(
            draft=source,
            synthesis_mode=synthesis_mode,
            evidence_by_id=evidence_by_id,
            claims_by_id=claims_by_id,
            object_type="integrated_recommendation",
            object_id=f"IR{index:03d}",
            fallback_text="建议由相关临床医师或药师结合完整资料复核整体治疗方案。",
        )
        events.extend(statement_events)
        if statement is not None:
            integrated.append(statement)

    used_evidence_ids = list(
        dict.fromkeys(
            [
                *[
                    evidence_id
                    for item in findings
                    for evidence_id in item.evidence_ids
                ],
                *[
                    evidence_id
                    for item in element_reviews
                    for evidence_id in item.evidence_ids
                ],
                *[
                    evidence_id
                    for item in integrated
                    for evidence_id in item.evidence_ids
                ],
            ]
        )
    )
    used_claim_ids = list(
        dict.fromkeys(
            [
                *[claim_id for item in findings for claim_id in item.claim_ids],
                *[
                    claim_id
                    for item in element_reviews
                    for statement in (item.recommendation, item.monitoring)
                    if statement is not None
                    for claim_id in statement.claim_ids
                ],
                *[claim_id for item in integrated for claim_id in item.claim_ids],
            ]
        )
    )
    return (
        ReviewSynthesis(
            case_id=patient_case.case_id,
            findings=findings,
            element_reviews=element_reviews,
            integrated_recommendations=integrated,
            unresolved_items=list(dict.fromkeys(draft.unresolved_items)),
            used_evidence_ids=used_evidence_ids,
            used_claim_ids=used_claim_ids,
        ),
        events,
        warnings,
    )


async def synthesize_review(
    *,
    model: Any,
    raw_case_text: str,
    patient_case: PatientCase,
    patient_facts: list[dict[str, Any]],
    elements: list[TreatmentPlanElement],
    questions: list[ReviewQuestion],
    evidence: list[EvidenceItemV3],
    claims: list[EvidenceClaim],
    unresolved_questions: list[str],
    synthesis_mode: SynthesisMode,
    system_prompt: str,
    technical_retry_limit: int = 0,
    retain_raw_output: bool = False,
) -> ReviewSynthesisResult:
    if synthesis_mode == "claims":
        evidence_input: dict[str, Any] = {
            "mode": "claims",
            "claims": [item.model_dump(mode="json") for item in claims],
        }
    else:
        evidence_input = {
            "mode": "direct_chunks",
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
    try:
        result = await invoke_json_schema(
            model=model,
            stage="review_synthesis",
            system_prompt=REVIEW_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=REVIEW_SYNTHESIS_PROMPT.format(
                synthesis_mode=synthesis_mode,
                system_prompt=system_prompt.strip(),
                raw_case_text=raw_case_text,
                patient_case=json.dumps(
                    patient_case.model_dump(mode="json"),
                    ensure_ascii=False,
                ),
                patient_facts=json.dumps(patient_facts, ensure_ascii=False),
                plan_elements=json.dumps(
                    [item.model_dump(mode="json") for item in elements],
                    ensure_ascii=False,
                ),
                review_questions=json.dumps(
                    [item.model_dump(mode="json") for item in questions],
                    ensure_ascii=False,
                ),
                unresolved_questions=json.dumps(
                    unresolved_questions,
                    ensure_ascii=False,
                ),
                evidence_input=json.dumps(evidence_input, ensure_ascii=False),
            ),
            output_model=ReviewSynthesisDraft,
            repair_limit=1,
            technical_retry_limit=technical_retry_limit,
            retain_raw_output=retain_raw_output,
        )
        review, events, warnings = _canonicalize_review(
            draft=ReviewSynthesisDraft.model_validate(result.value),
            patient_case=patient_case,
            patient_facts=patient_facts,
            elements=elements,
            evidence=evidence,
            claims=claims,
            synthesis_mode=synthesis_mode,
        )
        return ReviewSynthesisResult(
            review=review,
            audit=result.audit.model_dump(mode="json"),
            validation_events=events,
            warnings=warnings,
            degraded=any(item.action in {"degraded", "repaired"} for item in events),
        )
    except StructuredOutputError as exc:
        fatal = exc.audit.failure_kind == "provider"
        review = _fallback_review(
            case_id=patient_case.case_id,
            elements=elements,
            unresolved_items=[
                *unresolved_questions,
                "结构化综合失败，已生成逐要素不确定结果",
            ],
        )
        return ReviewSynthesisResult(
            review=review,
            audit=exc.audit.model_dump(mode="json"),
            validation_events=[
                LocalValidationEvent(
                    stage="review_synthesis",
                    object_type="review",
                    action="degraded",
                    reason="结构化综合失败，已执行整结构内的逐要素局部回退",
                )
            ],
            warnings=[str(exc)],
            degraded=True,
            fatal=fatal,
        )
