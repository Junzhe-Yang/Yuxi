from __future__ import annotations

from dataclasses import dataclass

import pytest

from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    extract_plan_anchors,
    validate_anchor_drafts,
)
from yuxi.agents.buildin.medication_review_lite.models import PlanAnchorDraft


@dataclass
class FakeResponse:
    content: str
    usage_metadata: dict[str, int] | None = None


class FakeModel:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        del messages
        self.calls += 1
        return self.responses.pop(0)


class FailingModel:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        del messages
        self.calls += 1
        raise RuntimeError("provider unavailable")


def test_anchor_validation_uses_original_source_order_and_span() -> None:
    raw = "方案：药物甲 1片 每日一次。\n监测：两周后复查。"
    drafts = [
        PlanAnchorDraft(
            source_span="两周后复查",
            label="复查计划",
            kind="explicit_monitoring_or_followup",
        ),
        PlanAnchorDraft(
            source_span="药物甲 1片 每日一次",
            label="药物甲医嘱",
            kind="medication_order",
        ),
    ]

    anchors, dropped = validate_anchor_drafts(raw, drafts)

    assert dropped == []
    assert [item.element_id for item in anchors] == ["PE001", "PE002"]
    assert [item.source_span for item in anchors] == [
        "药物甲 1片 每日一次",
        "两周后复查",
    ]
    assert raw[anchors[0].source_start : anchors[0].source_end] == (
        anchors[0].source_span
    )


def test_anchor_validation_only_repairs_whitespace_differences() -> None:
    raw = "药物乙  2片\n每日一次"
    drafts = [
        PlanAnchorDraft(
            source_span="药物乙 2片 每日一次",
            label="药物乙医嘱",
            kind="medication_order",
        ),
        PlanAnchorDraft(
            source_span="药物丙 1片每日一次",
            label="不存在的医嘱",
            kind="medication_order",
        ),
    ]

    anchors, dropped = validate_anchor_drafts(raw, drafts)

    assert len(anchors) == 1
    assert anchors[0].source_span == raw
    assert any(
        item.get("reason") == "source_span_not_found" for item in dropped
    )


def test_duplicate_anchor_draft_is_deduplicated_without_degrading() -> None:
    raw = "方案甲。后文再次提到方案甲。"
    draft = PlanAnchorDraft(
        source_span="方案甲",
        label="方案甲",
        kind="explicit_regimen_or_other",
    )

    anchors, dropped = validate_anchor_drafts(raw, [draft, draft])

    assert [item.element_id for item in anchors] == ["PE001"]
    assert anchors[0].source_start == 0
    assert dropped == []


@pytest.mark.asyncio
async def test_anchor_extraction_repairs_invalid_json_once() -> None:
    raw = "药物甲 1片 每日一次"
    model = FakeModel(
        [
            FakeResponse(content="不是JSON"),
            FakeResponse(
                content=(
                    '{"anchors":[{"source_span":"药物甲 1片 每日一次",'
                    '"label":"药物甲医嘱","kind":"medication_order"}]}'
                ),
                usage_metadata={"input_tokens": 10, "output_tokens": 5},
            ),
        ]
    )

    anchors, audit = await extract_plan_anchors(
        model=model,
        raw_text=raw,
        technical_retry_limit=0,
    )

    assert model.calls == 2
    assert [item.element_id for item in anchors] == ["PE001"]
    assert audit.status == "repaired"
    assert audit.repair_raw_output is not None


@pytest.mark.asyncio
async def test_anchor_extraction_failure_returns_degraded_result() -> None:
    model = FakeModel(
        [
            FakeResponse(content="bad"),
            FakeResponse(content="still bad"),
        ]
    )

    anchors, audit = await extract_plan_anchors(
        model=model,
        raw_text="任意治疗方案",
        technical_retry_limit=0,
    )

    assert anchors == []
    assert audit.status == "failed"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_provider_failure_does_not_trigger_json_repair_call() -> None:
    model = FailingModel()

    anchors, audit = await extract_plan_anchors(
        model=model,
        raw_text="任意治疗方案",
        technical_retry_limit=0,
    )

    assert anchors == []
    assert audit.status == "failed"
    assert audit.error_type == "RuntimeError"
    assert model.calls == 1
