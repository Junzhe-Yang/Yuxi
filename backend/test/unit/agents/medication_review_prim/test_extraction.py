from __future__ import annotations

from yuxi.agents.buildin.medication_review_prim.extraction import (
    _schema_instructions,
    validate_modifier_drafts,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    PatientModifierDraft,
)


def test_modifier_validation_keeps_grounded_spans_and_stable_order() -> None:
    raw = "患者肾功能减退，近期出现头晕。"
    modifiers, dropped = validate_modifier_drafts(
        raw,
        [
            PatientModifierDraft(source_span="近期出现头晕"),
            PatientModifierDraft(source_span="肾功能减退"),
            PatientModifierDraft(source_span="肾功能减退"),
            PatientModifierDraft(source_span="未提供的事实"),
        ],
    )

    assert [value.modifier_id for value in modifiers] == ["PM001", "PM002"]
    assert [value.source_span for value in modifiers] == [
        "肾功能减退",
        "近期出现头晕",
    ]
    assert dropped == [
        {
            "source_span": "未提供的事实",
            "reason": "source_span_not_found",
        }
    ]


def test_modifier_fallback_prompt_contains_complete_target_schema() -> None:
    instructions = _schema_instructions()

    assert "目标 JSON Schema" in instructions
    assert '"modifiers"' in instructions
    assert '"source_span"' in instructions
