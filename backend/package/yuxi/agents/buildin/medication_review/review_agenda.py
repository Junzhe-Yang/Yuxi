from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm_io import StructuredOutputError, invoke_json_schema
from .models import (
    ReviewAgendaDraft,
    ReviewQuestion,
    ReviewQuestionDraft,
    TreatmentPlanElement,
)
from .prompt import REVIEW_AGENDA_PROMPT, REVIEW_AGENDA_SYSTEM_PROMPT


@dataclass(frozen=True)
class ReviewAgendaResult:
    questions: list[ReviewQuestion]
    audit: dict[str, Any]
    warnings: list[str]


def _normalize_question(value: str) -> str:
    return " ".join(value.split()).casefold()


def _fallback_questions(
    elements: list[TreatmentPlanElement],
    max_questions: int,
) -> list[ReviewQuestionDraft]:
    return [
        ReviewQuestionDraft(
            question_text=(
                f"在当前患者条件下，方案要素“{element.normalized_summary}”"
                "有哪些适用条件、潜在限制或需要有来源支持的调整建议？"
            ),
            linked_element_ids=[element.element_id],
            priority="medium",
            reason="动态议程模型不可用时，根据显式方案要素生成的通用调查提示",
        )
        for element in elements[:max_questions]
    ]


def _canonicalize_questions(
    *,
    drafts: list[ReviewQuestionDraft],
    valid_element_ids: set[str],
    valid_fact_ids: set[str],
    max_questions: int,
) -> tuple[list[ReviewQuestion], list[str]]:
    questions: list[ReviewQuestion] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for draft in drafts:
        text = " ".join(draft.question_text.split()).strip()
        normalized = _normalize_question(text)
        if not text or normalized in seen:
            warnings.append("删除了空白或重复的动态审查问题")
            continue
        seen.add(normalized)
        element_ids = [
            value for value in dict.fromkeys(draft.linked_element_ids)
            if value in valid_element_ids
        ]
        fact_ids = [
            value for value in dict.fromkeys(draft.linked_patient_fact_ids)
            if value in valid_fact_ids
        ]
        if len(element_ids) != len(set(draft.linked_element_ids)):
            warnings.append(f"问题“{text}”包含无效方案要素引用，已局部删除")
        if len(fact_ids) != len(set(draft.linked_patient_fact_ids)):
            warnings.append(f"问题“{text}”包含无效患者事实引用，已局部删除")
        questions.append(
            ReviewQuestion(
                **draft.model_dump(
                    exclude={"question_text", "linked_element_ids", "linked_patient_fact_ids"}
                ),
                question_id=f"RQ{len(questions) + 1:03d}",
                question_text=text,
                linked_element_ids=element_ids,
                linked_patient_fact_ids=fact_ids,
            )
        )
        if len(questions) >= max_questions:
            break
    return questions, warnings


async def build_review_agenda(
    *,
    model: Any,
    raw_case_text: str,
    patient_facts: list[dict[str, Any]],
    elements: list[TreatmentPlanElement],
    system_prompt: str,
    max_questions: int,
    technical_retry_limit: int = 0,
    retain_raw_output: bool = False,
) -> ReviewAgendaResult:
    warnings: list[str] = []
    audit: dict[str, Any] = {}
    try:
        result = await invoke_json_schema(
            model=model,
            stage="build_review_agenda",
            system_prompt=REVIEW_AGENDA_SYSTEM_PROMPT,
            user_prompt=REVIEW_AGENDA_PROMPT.format(
                max_questions=max_questions,
                system_prompt=system_prompt.strip(),
                raw_case_text=raw_case_text,
                patient_facts=json.dumps(patient_facts, ensure_ascii=False),
                plan_elements=json.dumps(
                    [item.model_dump(mode="json") for item in elements],
                    ensure_ascii=False,
                ),
            ),
            output_model=ReviewAgendaDraft,
            repair_limit=1,
            technical_retry_limit=technical_retry_limit,
            retain_raw_output=retain_raw_output,
        )
        drafts = ReviewAgendaDraft.model_validate(result.value).questions
        audit = result.audit.model_dump(mode="json")
    except StructuredOutputError as exc:
        drafts = _fallback_questions(elements, max_questions)
        audit = exc.audit.model_dump(mode="json")
        warnings.append(f"动态审查议程生成失败，已使用通用方案要素提示：{exc}")

    questions, validation_warnings = _canonicalize_questions(
        drafts=drafts,
        valid_element_ids={item.element_id for item in elements},
        valid_fact_ids={
            str(item.get("fact_id")) for item in patient_facts if item.get("fact_id")
        },
        max_questions=max_questions,
    )
    warnings.extend(validation_warnings)
    if not questions and elements:
        questions, fallback_warnings = _canonicalize_questions(
            drafts=_fallback_questions(elements, max_questions),
            valid_element_ids={item.element_id for item in elements},
            valid_fact_ids=set(),
            max_questions=max_questions,
        )
        warnings.extend(fallback_warnings)
        warnings.append("动态议程为空，已使用通用方案要素提示")
    return ReviewAgendaResult(questions=questions, audit=audit, warnings=warnings)
