from __future__ import annotations

from yuxi.agents.buildin.medication_review_lite.models import PlanAnchor
from yuxi.agents.buildin.medication_review_prim.memory import (
    build_investigation_memory,
    uninvestigated_plan_ids,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    InvestigationItem,
    PatientModifier,
    QueryRecord,
    merge_investigations,
)


def _anchor(element_id: str = "PE001") -> PlanAnchor:
    return PlanAnchor(
        element_id=element_id,
        source_span="方案甲",
        source_start=0,
        source_end=3,
        label="方案甲",
        kind="explicit_regimen_or_other",
    )


def _modifier() -> PatientModifier:
    return PatientModifier(
        modifier_id="PM001",
        source_span="肾功能减退",
        source_start=4,
        source_end=9,
    )


def _query(investigation_id: str | None = None) -> QueryRecord:
    return QueryRecord(
        query_id="Q-ONE",
        tool_call_id="call-1",
        investigation_id=investigation_id,
        query_text="方案甲 肾功能 剂量调整",
        reason="核验患者特异性适用条件",
        focus_plan_ids=["PE001"],
        focus_modifier_ids=["PM001"],
        started_at="2026-01-01T00:00:00Z",
        elapsed_ms=1,
        status="success",
        evidence_ids=["EV-0123456789ABCDEF"],
    )


def _investigation() -> InvestigationItem:
    return InvestigationItem(
        investigation_id="INV-ONE",
        question="肾功能减退是否改变方案甲？",
        focus_plan_ids=["PE001"],
        focus_modifier_ids=["PM001"],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        query_ids=["Q-ONE"],
        candidate_evidence_ids=["EV-0123456789ABCDEF"],
    )


def test_investigation_reducer_keeps_candidates_but_latest_judgment() -> None:
    first = _investigation()
    second = first.model_copy(
        update={
            "status": "answered",
            "query_ids": ["Q-TWO"],
            "candidate_evidence_ids": ["EV-SECOND"],
            "selected_evidence_ids": ["EV-SECOND"],
            "working_note": "第二条证据足以形成有边界回答。",
            "updated_at": "2026-01-01T00:00:01Z",
        }
    )

    merged = merge_investigations([first], [second])

    assert len(merged) == 1
    assert merged[0].query_ids == ["Q-ONE", "Q-TWO"]
    assert merged[0].candidate_evidence_ids == [
        "EV-0123456789ABCDEF",
        "EV-SECOND",
    ]
    assert merged[0].selected_evidence_ids == ["EV-SECOND"]
    assert merged[0].status == "answered"


def test_m3_memory_exposes_open_question_and_uninvestigated_plan() -> None:
    second_anchor = _anchor("PE002").model_copy(
        update={
            "source_span": "方案乙",
            "label": "方案乙",
            "source_start": 4,
            "source_end": 7,
        }
    )
    investigation = _investigation()
    memory = build_investigation_memory(
        profile="m3",
        anchors=[_anchor(), second_anchor],
        modifiers=[_modifier()],
        queries=[_query("INV-ONE")],
        investigations=[investigation],
    )

    assert "当前待解决的证据问题" in memory
    assert "肾功能减退是否改变方案甲" in memory
    assert "候选 Evidence" in memory
    assert "尚未关联任何调查的方案要素" in memory
    assert "PE002" in memory
    assert "候选 Evidence 也不等于可用证据" in memory
    assert uninvestigated_plan_ids(
        [_anchor(), second_anchor],
        [investigation],
    ) == ["PE002"]
