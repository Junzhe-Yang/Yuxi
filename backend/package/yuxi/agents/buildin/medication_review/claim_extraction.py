from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_io import StructuredOutputError, invoke_json_schema
from .models import (
    EvidenceClaim,
    EvidenceClaimBatchDraft,
    EvidenceClaimDraft,
    EvidenceItemV3,
    ReviewQuestion,
    TreatmentPlanElement,
)
from .prompt import CLAIM_EXTRACTION_PROMPT, CLAIM_EXTRACTION_SYSTEM_PROMPT


@dataclass(frozen=True)
class ClaimExtractionResult:
    claims: list[EvidenceClaim]
    audit: dict[str, Any]
    warnings: list[str]
    degraded: bool = False


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _canonicalize_claims(
    *,
    drafts: list[EvidenceClaimDraft],
    evidence: list[EvidenceItemV3],
    elements: list[TreatmentPlanElement],
    questions: list[ReviewQuestion],
) -> tuple[list[EvidenceClaim], list[str]]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    element_ids = {item.element_id for item in elements}
    question_ids = {item.question_id for item in questions}
    claims: list[EvidenceClaim] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for draft in drafts:
        item = evidence_by_id.get(draft.evidence_id)
        if item is None:
            warnings.append(f"删除引用未知 Evidence 的 Claim：{draft.evidence_id}")
            continue
        if not draft.source_span or _normalized(draft.source_span) not in _normalized(item.raw_text):
            warnings.append(f"删除 source_span 未落回原文的 Claim：{draft.evidence_id}")
            continue
        linked_elements = [
            value for value in dict.fromkeys(draft.linked_element_ids)
            if value in element_ids
        ]
        linked_questions = [
            value for value in dict.fromkeys(draft.linked_question_ids)
            if value in question_ids
        ]
        if len(linked_elements) != len(set(draft.linked_element_ids)):
            warnings.append(f"{draft.evidence_id} Claim 的无效 element_id 已删除")
        if len(linked_questions) != len(set(draft.linked_question_ids)):
            warnings.append(f"{draft.evidence_id} Claim 的无效 question_id 已删除")
        key = (
            draft.evidence_id,
            _normalized(draft.source_span),
            _normalized(draft.statement),
        )
        if key in seen:
            warnings.append(f"合并重复 Claim：{draft.evidence_id}")
            continue
        seen.add(key)
        claims.append(
            EvidenceClaim(
                **draft.model_dump(
                    exclude={"linked_element_ids", "linked_question_ids"}
                ),
                claim_id=f"CL{len(claims) + 1:03d}",
                linked_element_ids=linked_elements,
                linked_question_ids=linked_questions,
            )
        )
    return claims, warnings


async def extract_claims(
    *,
    model: Any,
    evidence: list[EvidenceItemV3],
    elements: list[TreatmentPlanElement],
    questions: list[ReviewQuestion],
    technical_retry_limit: int = 0,
    retain_raw_output: bool = False,
) -> ClaimExtractionResult:
    if not evidence:
        return ClaimExtractionResult(
            claims=[],
            audit={"stage": "extract_claims", "skipped": "no_selected_evidence"},
            warnings=["没有最终 Evidence，Claim 抽取已跳过"],
        )
    try:
        result = await invoke_json_schema(
            model=model,
            stage="extract_claims",
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=CLAIM_EXTRACTION_PROMPT.format(
                plan_elements=json.dumps(
                    [item.model_dump(mode="json") for item in elements],
                    ensure_ascii=False,
                ),
                review_questions=json.dumps(
                    [item.model_dump(mode="json") for item in questions],
                    ensure_ascii=False,
                ),
                evidence=json.dumps(
                    [item.model_dump(mode="json") for item in evidence],
                    ensure_ascii=False,
                ),
            ),
            output_model=EvidenceClaimBatchDraft,
            repair_limit=1,
            technical_retry_limit=technical_retry_limit,
            retain_raw_output=retain_raw_output,
        )
        drafts = EvidenceClaimBatchDraft.model_validate(result.value).claims
        claims, warnings = _canonicalize_claims(
            drafts=drafts,
            evidence=evidence,
            elements=elements,
            questions=questions,
        )
        return ClaimExtractionResult(
            claims=claims,
            audit=result.audit.model_dump(mode="json"),
            warnings=warnings,
        )
    except StructuredOutputError as exc:
        return ClaimExtractionResult(
            claims=[],
            audit=exc.audit.model_dump(mode="json"),
            warnings=[f"Claim 抽取失败，已使用空 Claim 集继续综合：{exc}"],
            degraded=True,
        )
