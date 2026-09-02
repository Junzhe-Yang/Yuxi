from __future__ import annotations

from yuxi.agents.buildin.medication_review.evidence_board import (
    merge_evidence,
    select_evidence,
)
from yuxi.agents.buildin.medication_review.models import (
    EvidenceCandidate,
    EvidenceOccurrenceV3,
)


def _candidate(content_hash: str, query_id: str, rank: int, text: str) -> EvidenceCandidate:
    return EvidenceCandidate(
        content_hash=content_hash,
        raw_text=text,
        source_document="共识.md",
        occurrences=[EvidenceOccurrenceV3(query_id=query_id, rank=rank)],
    )


def test_merge_assigns_short_ids_and_preserves_cross_query_occurrences():
    first = merge_evidence(
        existing=[],
        candidates=[_candidate("same", "Q001", 1, "同一个来源片段")],
    )
    second = merge_evidence(
        existing=first.evidence,
        candidates=[
            _candidate("same", "Q002", 2, "同一个来源片段"),
            _candidate("new", "Q002", 1, "另一个来源片段"),
        ],
    )

    assert [item.evidence_id for item in second.evidence] == ["EV001", "EV002"]
    assert [item.query_id for item in second.evidence[0].occurrences] == ["Q001", "Q002"]
    assert second.duplicate_ratio == 0.5
    assert second.new_evidence_ids == ["EV002"]


def test_selection_honours_priority_then_round_robins_queries():
    merged = merge_evidence(
        existing=[],
        candidates=[
            _candidate("a", "Q001", 1, "证据A"),
            _candidate("b", "Q001", 2, "证据B"),
            _candidate("c", "Q002", 1, "证据C"),
        ],
    )

    audit = select_evidence(
        evidence=merged.evidence,
        priority_evidence_ids=["EV002"],
        max_evidence=3,
        max_tokens=1000,
    )

    assert audit.selected_evidence_ids == ["EV002", "EV001", "EV003"]
    assert audit.skipped_evidence_ids == []
