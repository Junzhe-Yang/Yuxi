from __future__ import annotations

from yuxi.agents.buildin.medication_review_lite.evidence import (
    build_query_centered_excerpt,
    content_identity_hash,
    display_text,
    evidence_id_from_hash,
    extract_evidence_ids,
    extract_item_element_ids,
)
from yuxi.agents.buildin.medication_review_lite.models import (
    EvidenceItem,
    EvidenceOccurrence,
    merge_evidence_store,
)


def _occurrence(record_id: str, excerpt: str) -> EvidenceOccurrence:
    return EvidenceOccurrence(
        record_id=record_id,
        tool_call_id=record_id,
        source_method="search",
        query_text="方案要素 风险",
        reason="核验",
        shown_excerpt=excerpt,
        rank=1,
    )


def test_stable_evidence_id_does_not_depend_on_query_order() -> None:
    left = content_identity_hash(
        db_id="db",
        raw_text="同一片段",
        file_id="file",
        chunk_id="chunk",
        chunk_index=2,
    )
    right = content_identity_hash(
        db_id="db",
        raw_text="同一片段",
        file_id="file",
        chunk_id="chunk",
        chunk_index=2,
    )

    assert left == right
    assert evidence_id_from_hash(left) == evidence_id_from_hash(right)
    assert evidence_id_from_hash(left).startswith("EV-")


def test_different_chunks_from_same_document_have_different_ids() -> None:
    left = content_identity_hash(
        db_id="db",
        raw_text="片段一",
        file_id="file",
        chunk_id="chunk-1",
        chunk_index=1,
    )
    right = content_identity_hash(
        db_id="db",
        raw_text="片段二",
        file_id="file",
        chunk_id="chunk-2",
        chunk_index=2,
    )

    assert evidence_id_from_hash(left) != evidence_id_from_hash(right)


def test_parallel_evidence_merge_preserves_both_occurrences() -> None:
    item = EvidenceItem(
        evidence_id="EV-0123456789ABCDEF",
        content_hash="0" * 64,
        raw_text="完整原文",
        occurrences=[_occurrence("SEARCH-A", "片段A")],
    )
    other = item.model_copy(
        update={"occurrences": [_occurrence("SEARCH-B", "片段B")]}
    )

    merged = merge_evidence_store(
        {item.evidence_id: item},
        {other.evidence_id: other},
    )

    assert merged[item.evidence_id].raw_text == "完整原文"
    assert {
        value.record_id
        for value in merged[item.evidence_id].occurrences
    } == {"SEARCH-A", "SEARCH-B"}


def test_query_centered_excerpt_finds_evidence_after_old_prefix_cutoff() -> None:
    raw = "无关背景。" * 200 + "关键方案证据：需要调整给药间隔。" + "其它内容。" * 100

    excerpt = build_query_centered_excerpt(
        raw_text=raw,
        focus_text="调整给药间隔",
        target_chars=600,
    )

    assert "关键方案证据：需要调整给药间隔" in excerpt.text
    assert excerpt.start > 260
    assert raw.endswith("其它内容。" * 100)


def test_display_cleanup_does_not_mutate_saved_raw_text() -> None:
    raw = "<p>剂量&nbsp;说明</p>\n<table><tr><td>$x$</td></tr></table>"

    shown = display_text(raw)

    assert "<p>" not in shown
    assert "剂量" in shown
    assert raw == "<p>剂量&nbsp;说明</p>\n<table><tr><td>$x$</td></tr></table>"


def test_structural_id_parsing_accepts_old_and_new_evidence_ids() -> None:
    answer = (
        "②【逐项判断】\n"
        "■ 【PE001】方案一\n依据：[EV001] [EV-0123456789ABCDEF]\n"
        "③【正面判断汇总】"
    )

    item_ids, degraded = extract_item_element_ids(answer)

    assert item_ids == ["PE001"]
    assert degraded is False
    assert extract_evidence_ids(answer) == [
        "EV001",
        "EV-0123456789ABCDEF",
    ]
