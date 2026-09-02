from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models import EvidenceItem, PlanAnchor

EVIDENCE_ID_PATTERN = re.compile(
    r"\b(?:EV\d{3}|EV-[0-9A-Fa-f]{16})\b"
)
ELEMENT_ID_PATTERN = re.compile(r"\bPE\d{3}\b")
ITEM_ELEMENT_PATTERN = re.compile(
    r"^\s*■\s*【(PE\d{3})】",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class ExcerptView:
    text: str
    start: int
    end: int
    fallback: bool


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return str(value)


def content_identity_hash(
    *,
    db_id: str,
    raw_text: str,
    file_id: str | None,
    chunk_id: str | None,
    chunk_index: int | str | None,
) -> str:
    identity = "\x1f".join(
        [
            db_id,
            file_id or "",
            chunk_id or "",
            str(chunk_index) if chunk_index is not None else "",
            hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def evidence_id_from_hash(content_hash: str) -> str:
    return f"EV-{content_hash[:16].upper()}"


def display_text(raw_text: str) -> str:
    value = html.unescape(raw_text)
    value = re.sub(
        r"</?(?:table|thead|tbody|tr|p|div|li|ul|ol|h[1-6]|br)\b[^>]*>",
        "\n",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", "", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _searchable_characters(text: str) -> str:
    return "".join(char.casefold() for char in text if char.isalnum())


def _ngrams(text: str, size: int) -> Counter[str]:
    compact = _searchable_characters(text)
    if len(compact) < size:
        return Counter({compact: 1}) if compact else Counter()
    return Counter(
        compact[index : index + size]
        for index in range(len(compact) - size + 1)
    )


def _window_score(window: str, focus_text: str) -> float:
    if not focus_text.strip():
        return 0.0
    score = 0.0
    for size, weight in ((2, 1.0), (3, 1.5)):
        focus = _ngrams(focus_text, size)
        candidate = _ngrams(window, size)
        denominator = sum(focus.values())
        if denominator:
            overlap = sum((focus & candidate).values())
            score += weight * overlap / denominator
    terms = [
        value
        for value in re.split(r"[^\w\u3400-\u9fff]+", focus_text.casefold())
        if len(value) >= 2
    ]
    score += sum(0.2 for term in set(terms) if term in window.casefold())
    return score


def build_query_centered_excerpt(
    *,
    raw_text: str,
    focus_text: str,
    target_chars: int,
) -> ExcerptView:
    text = display_text(raw_text)
    if len(text) <= target_chars:
        return ExcerptView(
            text=text,
            start=0,
            end=len(text),
            fallback=False,
        )

    step = max(target_chars // 2, 1)
    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        windows.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start += step

    scored = [
        (_window_score(window, focus_text), start, end, window)
        for start, end, window in windows
    ]
    score, start, end, window = max(
        scored,
        key=lambda value: (value[0], -value[1]),
    )
    fallback = score <= 0
    if fallback:
        start, end, window = windows[0]
    return ExcerptView(
        text=window.strip(),
        start=start,
        end=end,
        fallback=fallback,
    )


def format_evidence_card(
    *,
    item: EvidenceItem,
    excerpt: ExcerptView,
    rank: int | None,
    score: float | None,
    distance: float | None,
    include_file_id: bool = False,
) -> str:
    source = item.source_document or "<未知来源>"
    position = (
        f"chunk {item.chunk_index}"
        if item.chunk_index is not None
        else "chunk <未知>"
    )
    metrics: list[str] = []
    if rank is not None:
        metrics.append(f"rank={rank}")
    if score is not None:
        metrics.append(f"score={score:g}")
    if distance is not None:
        metrics.append(f"distance={distance:g}")
    metric_text = "，".join(metrics) if metrics else "<未提供>"
    file_line = (
        f"文档ID：{item.file_id or '<未知>'}\n" if include_file_id else ""
    )
    return (
        f"[{item.evidence_id}]\n"
        f"来源：{source}\n"
        f"{file_line}"
        f"位置：{position}\n"
        f"排名/分数：{metric_text}\n"
        f"内容：\n{excerpt.text}"
    )


def extract_evidence_ids(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0).upper()
            for match in EVIDENCE_ID_PATTERN.finditer(text)
        )
    )


def extract_all_element_ids(text: str) -> list[str]:
    return [match.group(0) for match in ELEMENT_ID_PATTERN.finditer(text)]


def _review_item_region(text: str) -> tuple[str, bool]:
    start_match = re.search(r"②\s*【逐项判断】", text)
    end_match = re.search(r"③\s*【正面判断汇总】", text)
    if start_match and end_match and end_match.start() > start_match.end():
        return text[start_match.end() : end_match.start()], False
    return text, True


def extract_item_element_ids(text: str) -> tuple[list[str], bool]:
    region, degraded = _review_item_region(text)
    return [
        match.group(1) for match in ITEM_ELEMENT_PATTERN.finditer(region)
    ], degraded


def strip_program_owned_sections(text: str) -> str:
    value = text.strip()
    section_two = re.search(r"②\s*【逐项判断】", value)
    if re.match(r"^\s*①\s*【原方案要素清单】", value) and section_two:
        value = value[section_two.start() :]
    section_six = re.search(r"⑥\s*【依据清单】", value)
    if section_six:
        value = value[: section_six.start()]
    return value.strip()


def insert_coverage_patch(answer_body: str, patch_text: str) -> str:
    marker = re.search(r"③\s*【正面判断汇总】", answer_body)
    patch = patch_text.strip()
    if marker:
        return (
            answer_body[: marker.start()].rstrip()
            + "\n\n"
            + patch
            + "\n\n"
            + answer_body[marker.start() :].lstrip()
        )
    return answer_body.rstrip() + "\n\n【覆盖补写】\n" + patch


def build_plan_section(anchors: list[PlanAnchor]) -> str:
    lines = ["①【原方案要素清单】"]
    if not anchors:
        lines.append("- 本实验组未启用或未成功获得方案锚点；请以第②部分的原文审查为准。")
        return "\n".join(lines)
    for index, anchor in enumerate(anchors, start=1):
        lines.append(
            f"{index}. 【{anchor.element_id}】{anchor.source_span}"
        )
    return "\n".join(lines)


def build_evidence_section(
    *,
    cited_evidence_ids: list[str],
    unknown_evidence_ids: list[str],
    evidence_store: dict[str, EvidenceItem],
) -> str:
    lines = ["⑥【依据清单】"]
    valid = [
        evidence_id
        for evidence_id in cited_evidence_ids
        if evidence_id in evidence_store
    ]
    if not valid:
        lines.append("- 本次回答未形成可解析的有效 Evidence 引用。")
    for evidence_id in valid:
        item = evidence_store[evidence_id]
        source = item.source_document or "<未知来源>"
        position = (
            f"chunk {item.chunk_index}"
            if item.chunk_index is not None
            else "chunk <未知>"
        )
        lines.append(f"- [{evidence_id}] {source}，{position}")
    if unknown_evidence_ids:
        lines.append(
            "- 未解析引用："
            + "、".join(f"[{value}]" for value in unknown_evidence_ids)
        )
    return "\n".join(lines)
