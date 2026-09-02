from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain.agents.middleware import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

from yuxi.agents.buildin.medication_review_lite.context import (
    MedicationReviewLiteContext,
)
from yuxi.agents.buildin.medication_review_lite.harness import (
    ReviewHarnessMiddleware,
    _coverage,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    AnchorExtractionAudit,
    EvidenceItem,
    PlanAnchor,
)


def _anchor(element_id: str, span: str, start: int) -> PlanAnchor:
    return PlanAnchor(
        element_id=element_id,
        source_span=span,
        source_start=start,
        source_end=start + len(span),
        label=span,
        kind="medication_order",
    )


def _state() -> dict:
    evidence = EvidenceItem(
        evidence_id="EV-0123456789ABCDEF",
        content_hash="0" * 64,
        raw_text="依据原文",
        source_document="共识.md",
        chunk_index=3,
    )
    return {
        "messages": [HumanMessage(content="方案甲和方案乙")],
        "review_run_id": "run-1",
        "experiment_profile": "m3",
        "raw_case_text": "方案甲和方案乙",
        "raw_question_hash": "hash",
        "plan_anchors": [
            _anchor("PE001", "方案甲", 0),
            _anchor("PE002", "方案乙", 3),
        ],
        "anchor_extraction": AnchorExtractionAudit(
            status="success",
            started_at="2026-01-01T00:00:00Z",
        ),
        "evidence_store": {evidence.evidence_id: evidence},
        "search_records": [],
        "open_records": [],
        "knowledge_base_snapshot": {"name": "知识库"},
        "search_count": 1,
        "open_count": 0,
        "technical_attempts": 1,
        "warnings": [],
    }


def test_element_mentioned_only_in_summary_is_still_missing_from_items() -> None:
    anchors = [
        _anchor("PE001", "方案甲", 0),
        _anchor("PE002", "方案乙", 3),
    ]
    answer = (
        "②【逐项判断】\n"
        "■ 【PE001】方案甲\n判断：合理。\n\n"
        "③【正面判断汇总】\nPE002 合理。\n\n"
        "④【负面判断汇总】\n无。\n\n"
        "⑤【综合建议】\n继续。"
    )

    report = _coverage(
        answer_body=answer,
        anchors=anchors,
        evidence_store={},
    )

    assert report.missing_after_patch == ["PE002"]
    assert report.unknown_element_ids == []


@pytest.mark.asyncio
async def test_m3_patches_only_missing_anchor_and_returns_one_final_message() -> None:
    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile="m3",
    )
    state = _state()
    runtime = SimpleNamespace(context=context)
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=runtime,
        tools=[],
    )
    responses = [
        ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "②【逐项判断】\n"
                        "■ 【PE001】方案甲\n判断：合理。依据：[EV-0123456789ABCDEF]\n\n"
                        "③【正面判断汇总】\n方案甲合理。\n\n"
                        "④【负面判断汇总】\n方案乙待审查。\n\n"
                        "⑤【综合建议】\n继续核定。"
                    )
                )
            ]
        ),
        ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "■ 【PE002】方案乙\n"
                        "判断：需调整。依据：[EV-0123456789ABCDEF]\n"
                        "说明：按来源调整。"
                    ),
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                )
            ]
        ),
    ]
    calls = 0

    async def handler(_request):
        nonlocal calls
        value = responses[calls]
        calls += 1
        return value

    middleware = ReviewHarnessMiddleware(model=object())
    result = await middleware.awrap_model_call(request, handler)

    assert isinstance(result, ExtendedModelResponse)
    assert calls == 2
    assert len(result.model_response.result) == 1
    final = result.model_response.result[0]
    assert isinstance(final, AIMessage)
    assert final.content.count("①【原方案要素清单】") == 1
    assert final.content.count("■ 【PE002】") == 1
    assert final.content.count("⑥【依据清单】") == 1
    trace = final.additional_kwargs["medication_review_trace"]
    assert trace["schema_version"] == "4.0"
    assert trace["run_status"] == "completed"
    assert trace["coverage_report"]["patch_attempted"] is True
    assert trace["coverage_report"]["patch_succeeded"] is True


@pytest.mark.asyncio
async def test_unknown_evidence_keeps_finding_and_marks_partial() -> None:
    context = MedicationReviewLiteContext(
        knowledges=["知识库"],
        experiment_profile="m2",
    )
    state = _state()
    state["experiment_profile"] = "m2"
    runtime = SimpleNamespace(context=context)
    request = ModelRequest(
        model=object(),
        messages=state["messages"],
        state=state,
        runtime=runtime,
        tools=[],
    )

    async def handler(_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "②【逐项判断】\n"
                        "■ 【PE001】方案甲\n判断：合理。[EV-FFFFFFFFFFFFFFFF]\n"
                        "■ 【PE002】方案乙\n判断：合理。\n\n"
                        "③【正面判断汇总】\n均合理。\n\n"
                        "④【负面判断汇总】\n无。\n\n"
                        "⑤【综合建议】\n继续。"
                    )
                )
            ]
        )

    result = await ReviewHarnessMiddleware(model=object()).awrap_model_call(
        request,
        handler,
    )
    final = result.model_response.result[0]
    trace = final.additional_kwargs["medication_review_trace"]

    assert "[EV-FFFFFFFFFFFFFFFF]" in final.content
    assert trace["run_status"] == "partial"
    assert trace["coverage_report"]["unknown_evidence_ids"] == [
        "EV-FFFFFFFFFFFFFFFF"
    ]
