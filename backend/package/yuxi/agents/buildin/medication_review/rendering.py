from __future__ import annotations

from .models import (
    EvidenceClaim,
    EvidenceItem,
    EvidenceItemV3,
    FinalReview,
    ReviewSynthesis,
    SourceCitationDraft,
    TreatmentPlanElement,
)


JUDGEMENT_LABELS = {
    "appropriate": "合理",
    "appropriate_with_monitoring": "合理（需监测）",
    "needs_adjustment": "需调整",
    "inappropriate": "不合理",
    "insufficient_evidence": "证据不足",
}


def _evidence_ref(evidence_id: str, evidence: dict[str, EvidenceItem]) -> str:
    item = evidence.get(evidence_id)
    if item is None:
        return f"【{evidence_id}】"
    source = item.source_document or "未知文档"
    chunk = f" · #{item.chunk_index}" if item.chunk_index is not None else ""
    return f"【依据：{source}{chunk} · {evidence_id}】"


def render_final_review(
    final_review: FinalReview,
    evidence_items: list[EvidenceItem],
) -> str:
    evidence = {item.evidence_id: item for item in evidence_items}
    judgement_index = {
        item.element_id: item for item in final_review.element_judgements
    }
    lines = ["①【原方案要素清单】（完整治疗方案拆解）"]
    for index, element in enumerate(final_review.plan_elements, 1):
        lines.append(f"{index}. {element.normalized_summary}（{element.element_id}）")

    lines.extend(["", "②【逐项判断】（每个要素：合理 / 不合理 / 需调整 + 证据）"])
    for index, element in enumerate(final_review.plan_elements, 1):
        judgement = judgement_index[element.element_id]
        evidence_ids = list(
            dict.fromkeys(
                [
                    *judgement.support_evidence_ids,
                    *judgement.challenge_evidence_ids,
                    *judgement.condition_evidence_ids,
                    *judgement.recommendation_evidence_ids,
                ]
            )
        )
        evidence_text = (
            "；".join(_evidence_ref(value, evidence) for value in evidence_ids)
            if evidence_ids
            else "当前知识库未提供足够的患者适用证据"
        )
        lines.extend(
            [
                "",
                f"■ 要素{index} {element.normalized_summary}",
                f"- 判断：{JUDGEMENT_LABELS[judgement.judgement]}",
                f"- 患者适用性：{judgement.patient_applicability_summary}",
                f"- 依据：{evidence_text}",
                f"- 说明：{judgement.rationale}",
            ]
        )
        if judgement.clinical_risk:
            lines.append(f"- 临床风险：{judgement.clinical_risk}")
        if judgement.recommended_action:
            lines.append(f"- 建议修正方案：{judgement.recommended_action}")
        if judgement.monitoring_requirement:
            lines.append(f"- 监测要求：{judgement.monitoring_requirement}")
        if judgement.missing_information:
            lines.append(
                f"- 尚缺信息：{'；'.join(judgement.missing_information)}"
            )

    lines.extend(["", "③【正面判断汇总】（合理项 + 依据）"])
    if final_review.positive_element_ids:
        for element_id in final_review.positive_element_ids:
            element = next(
                item for item in final_review.plan_elements if item.element_id == element_id
            )
            judgement = judgement_index[element_id]
            ids = [
                *judgement.support_evidence_ids,
                *judgement.condition_evidence_ids,
            ]
            lines.append(
                f"- {element.normalized_summary}："
                f"{JUDGEMENT_LABELS[judgement.judgement]}；"
                f"{'；'.join(_evidence_ref(value, evidence) for value in dict.fromkeys(ids))}"
            )
    else:
        lines.append("- 无。")

    lines.extend(["", "④【负面判断汇总】（不合理/需调整项 + 依据）"])
    if final_review.negative_element_ids:
        for element_id in final_review.negative_element_ids:
            element = next(
                item for item in final_review.plan_elements if item.element_id == element_id
            )
            judgement = judgement_index[element_id]
            ids = [
                *judgement.challenge_evidence_ids,
                *judgement.recommendation_evidence_ids,
            ]
            lines.append(
                f"- {element.normalized_summary}："
                f"{JUDGEMENT_LABELS[judgement.judgement]}；"
                f"{judgement.recommended_action or '当前知识库未提供有据的替代方案'}；"
                f"{'；'.join(_evidence_ref(value, evidence) for value in dict.fromkeys(ids))}"
            )
    else:
        lines.append("- 无。")
    for finding in final_review.cross_element_findings:
        ids = "；".join(_evidence_ref(value, evidence) for value in finding.evidence_ids)
        lines.append(f"- {finding.judgement}：{finding.rationale}；{ids}")
    if final_review.insufficient_element_ids:
        lines.append("- 证据不足项目：")
        for element_id in final_review.insufficient_element_ids:
            element = next(
                item
                for item in final_review.plan_elements
                if item.element_id == element_id
            )
            judgement = judgement_index[element_id]
            reason = (
                "；".join(judgement.missing_information)
                or "当前知识库证据、患者信息或检索预算不足"
            )
            lines.append(f"  - {element.normalized_summary}：{reason}")

    lines.extend(["", "⑤【综合建议】"])
    if final_review.integrated_recommendations:
        for index, recommendation in enumerate(
            final_review.integrated_recommendations, 1
        ):
            refs = "；".join(
                _evidence_ref(value, evidence)
                for value in recommendation.evidence_ids
            )
            scope = (
                refs
                if recommendation.source_scope == "retrieved_corpus"
                else "⚠️【当前检索语料未提供直接依据，建议临床核定】"
            )
            lines.append(f"{index}. {recommendation.text} {scope}".rstrip())
    else:
        lines.append("1. 当前没有形成额外的综合建议。")
    if final_review.unresolved_items:
        lines.append(
            f"- 尚未解决：{'；'.join(final_review.unresolved_items)}"
        )

    lines.extend(["", "⑥【依据清单】"])
    if final_review.used_evidence_ids:
        supported_objects: dict[str, list[str]] = {}
        for judgement in final_review.element_judgements:
            for evidence_id in [
                *judgement.support_evidence_ids,
                *judgement.challenge_evidence_ids,
                *judgement.condition_evidence_ids,
                *judgement.recommendation_evidence_ids,
            ]:
                supported_objects.setdefault(evidence_id, []).append(
                    judgement.element_id
                )
        for finding in final_review.cross_element_findings:
            for evidence_id in finding.evidence_ids:
                supported_objects.setdefault(evidence_id, []).append(
                    finding.finding_id
                )
        for recommendation in final_review.integrated_recommendations:
            for evidence_id in recommendation.evidence_ids:
                supported_objects.setdefault(evidence_id, []).append(
                    recommendation.recommendation_id
                )
        for evidence_id in final_review.used_evidence_ids:
            item = evidence.get(evidence_id)
            if item is None:
                continue
            excerpt = " ".join(item.raw_text.split())
            if len(excerpt) > 240:
                excerpt = excerpt[:240] + "…"
            version = (
                item.raw_metadata.get("document_version")
                or item.raw_metadata.get("version")
            )
            version_text = f"；版本：{version}" if version else ""
            objects = "、".join(
                dict.fromkeys(supported_objects.get(evidence_id, []))
            )
            object_text = objects or "仅作为未采用的上下文证据"
            lines.append(
                f"- {_evidence_ref(evidence_id, evidence)}{version_text}；"
                f"支持对象：{object_text}；原文：{excerpt}"
            )
    else:
        lines.append("- 本次未形成可用于最终判断的活跃知识库证据。")
    return "\n".join(lines)


DISPOSITION_LABELS = {
    "continue": "合理，可继续",
    "continue_with_monitoring": "合理，但需监测",
    "modify": "需调整",
    "avoid": "不合理，应避免",
    "uncertain": "证据不足或仍不确定",
}

EVIDENCE_BASIS_LABELS = {
    "direct_support": "有直接来源支持",
    "mixed_evidence": "正负或条件性证据并存",
    "no_material_conflict_found": "已检索但未发现实质冲突，缺少直接正向推荐",
    "insufficient": "证据或患者信息不足",
}


def _citation_ref(
    citation: SourceCitationDraft,
    evidence: dict[str, EvidenceItemV3],
) -> str:
    item = evidence.get(citation.evidence_id)
    if item is None:
        return f"【{citation.evidence_id}】"
    source = item.source_document or "未知文档"
    chunk = f" · #{item.chunk_index}" if item.chunk_index is not None else ""
    claim = f" · {citation.claim_id}" if citation.claim_id else ""
    span = " ".join(citation.source_span.split())
    return f"【依据：{source}{chunk}{claim}】{span}"


def render_review_v3(
    *,
    review: ReviewSynthesis,
    plan_elements: list[TreatmentPlanElement],
    evidence_items: list[EvidenceItemV3],
    claims: list[EvidenceClaim],
) -> str:
    del claims
    evidence = {item.evidence_id: item for item in evidence_items}
    findings_by_id = {item.finding_id: item for item in review.findings}
    reviews_by_id = {item.element_id: item for item in review.element_reviews}
    lines = ["①【原方案要素清单】（完整治疗方案拆解）"]
    for index, element in enumerate(plan_elements, start=1):
        lines.append(f"{index}. {element.normalized_summary}（{element.element_id}）")

    lines.extend(["", "②【逐项判断】（每个要素：合理 / 不合理 / 需调整 + 证据）"])
    for index, element in enumerate(plan_elements, start=1):
        element_review = reviews_by_id[element.element_id]
        element_findings = [
            findings_by_id[value]
            for value in element_review.finding_ids
            if value in findings_by_id
        ]
        lines.extend(
            [
                "",
                f"■ 要素{index} {element.normalized_summary}",
                f"- 判断：{DISPOSITION_LABELS[element_review.overall_disposition]}",
                f"- 判断基础：{EVIDENCE_BASIS_LABELS[element_review.evidence_basis]}",
                f"- 说明：{element_review.summary}",
            ]
        )
        for finding in element_findings:
            label = {
                "appropriate": "合理方面",
                "concern": "风险/问题",
                "uncertain": "不确定方面",
            }[finding.assessment]
            refs = "；".join(
                _citation_ref(value, evidence) for value in finding.citations
            )
            lines.append(
                f"- {label}：{finding.statement}"
                + (f"；{refs}" if refs else "；当前无直接来源")
            )
        if element_review.recommendation:
            recommendation = element_review.recommendation
            refs = "；".join(
                _citation_ref(value, evidence)
                for value in recommendation.citations
            )
            scope = (
                refs
                if recommendation.source_scope == "retrieved_corpus"
                else "⚠️【一般复核建议，当前检索语料无直接具体依据】"
            )
            lines.append(f"- 建议修正方案：{recommendation.text}；{scope}")
        if element_review.monitoring:
            monitoring = element_review.monitoring
            refs = "；".join(
                _citation_ref(value, evidence) for value in monitoring.citations
            )
            scope = (
                refs
                if monitoring.source_scope == "retrieved_corpus"
                else "⚠️【一般监测复核建议，当前检索语料无直接具体依据】"
            )
            lines.append(f"- 监测要求：{monitoring.text}；{scope}")
        if element_review.missing_information:
            lines.append(
                f"- 尚缺信息：{'；'.join(element_review.missing_information)}"
            )

    positive = [item for item in review.findings if item.assessment == "appropriate"]
    lines.extend(["", "③【正面判断汇总】（合理项 + 依据）"])
    if positive:
        for finding in positive:
            refs = "；".join(
                _citation_ref(value, evidence) for value in finding.citations
            )
            lines.append(
                f"- {finding.statement}"
                + (f"；{refs}" if refs else "；未取得直接正向来源")
            )
    else:
        lines.append("- 当前未形成明确的正面 finding。")

    negative = [
        item for item in review.findings if item.assessment in {"concern", "uncertain"}
    ]
    lines.extend(["", "④【负面与不确定判断汇总】（风险/需调整/证据不足 + 依据）"])
    if negative:
        for finding in negative:
            label = "风险/问题" if finding.assessment == "concern" else "不确定"
            refs = "；".join(
                _citation_ref(value, evidence) for value in finding.citations
            )
            lines.append(
                f"- {label}：{finding.statement}"
                + (f"；{refs}" if refs else "；当前无直接来源")
            )
    else:
        lines.append("- 当前未形成负面或不确定 finding。")

    lines.extend(["", "⑤【综合建议】"])
    if review.integrated_recommendations:
        for index, recommendation in enumerate(
            review.integrated_recommendations,
            start=1,
        ):
            refs = "；".join(
                _citation_ref(value, evidence)
                for value in recommendation.citations
            )
            scope = (
                refs
                if recommendation.source_scope == "retrieved_corpus"
                else "⚠️【一般复核建议，当前检索语料无直接具体依据】"
            )
            lines.append(f"{index}. {recommendation.text}；{scope}")
    else:
        lines.append("1. 建议结合逐项判断，由相关临床医师或药师复核整体方案。")
    if review.unresolved_items:
        lines.append(f"- 尚未解决：{'；'.join(review.unresolved_items)}")

    lines.extend(["", "⑥【依据清单】"])
    used = [evidence[value] for value in review.used_evidence_ids if value in evidence]
    if not used:
        lines.append("- 本次没有形成被最终判断采用的知识库证据。")
    else:
        for item in used:
            source = item.source_document or "未知文档"
            chunk = f" · #{item.chunk_index}" if item.chunk_index is not None else ""
            spans = list(
                dict.fromkeys(
                    citation.source_span
                    for finding in review.findings
                    for citation in finding.citations
                    if citation.evidence_id == item.evidence_id
                )
            )
            excerpt = "；".join(" ".join(value.split()) for value in spans)
            if not excerpt:
                excerpt = " ".join(item.raw_text.split())[:240]
            lines.append(f"- 【依据：{source}{chunk} · {item.evidence_id}】{excerpt}")
    return "\n".join(lines)
