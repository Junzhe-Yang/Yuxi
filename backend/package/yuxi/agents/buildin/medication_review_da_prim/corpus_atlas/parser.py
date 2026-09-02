from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SETEXT_HEADING_RE = re.compile(r"^\s*(=+|-+)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
MARKDOWN_DECORATION_RE = re.compile(r"[`*_~]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class ParsedSection:
    heading_path: list[str]
    heading_level: int
    body_lines: list[str] = field(default_factory=list)
    table_titles: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    document_title: str
    lead_text: str
    heading_titles: list[str]
    table_titles: list[str]
    sections: list[ParsedSection]


def normalize_markdown(markdown: str) -> str:
    return markdown.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _clean_text(value: str) -> str:
    value = MARKDOWN_LINK_RE.sub(r"\1", value)
    value = HTML_TAG_RE.sub(" ", value)
    value = MARKDOWN_DECORATION_RE.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _lead_text(lines: list[str], max_chars: int) -> str:
    parts: list[str] = []
    for line in lines:
        cleaned = _clean_text(line)
        if not cleaned or TABLE_SEPARATOR_RE.match(line):
            continue
        parts.append(cleaned)
        joined = " ".join(parts)
        if len(joined) >= max_chars:
            return joined[:max_chars].rstrip()
    return " ".join(parts)[:max_chars].rstrip()


def _table_titles(lines: list[str]) -> list[str]:
    titles: list[str] = []
    for index in range(1, len(lines)):
        if not TABLE_SEPARATOR_RE.match(lines[index]):
            continue
        previous = index - 2
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0 or "|" in lines[previous]:
            continue
        title = _clean_text(lines[previous])
        if title and title not in titles:
            titles.append(title)
    return titles


def _heading_at(lines: list[str], index: int) -> tuple[int, str, int] | None:
    match = ATX_HEADING_RE.match(lines[index])
    if match:
        return len(match.group(1)), _clean_text(match.group(2)), 1
    if index + 1 < len(lines) and SETEXT_HEADING_RE.match(lines[index + 1]):
        title = _clean_text(lines[index])
        if title:
            level = 1 if lines[index + 1].lstrip().startswith("=") else 2
            return level, title, 2
    return None


def parse_markdown_document(
    *,
    file_id: str,
    file_name: str,
    markdown: str,
    lead_chars: int = 600,
) -> ParsedDocument:
    del file_id
    lines = normalize_markdown(markdown).split("\n")
    headings: list[str] = []
    sections: list[ParsedSection] = []
    preamble: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    current: ParsedSection | None = None
    index = 0
    while index < len(lines):
        heading = _heading_at(lines, index)
        if heading is None:
            if current is None:
                preamble.append(lines[index])
            else:
                current.body_lines.append(lines[index])
            index += 1
            continue
        level, title, consumed = heading
        if current is not None:
            current.table_titles = _table_titles(current.body_lines)
            sections.append(current)
        headings.append(title)
        heading_stack = [item for item in heading_stack if item[0] < level]
        heading_stack.append((level, title))
        current = ParsedSection(
            heading_path=[value for _, value in heading_stack],
            heading_level=level,
        )
        index += consumed

    if current is not None:
        current.table_titles = _table_titles(current.body_lines)
        sections.append(current)

    fallback_title = Path(file_name).stem.strip() or file_name.strip() or "未命名文档"
    document_title = next(
        (section.heading_path[-1] for section in sections if section.heading_level == 1 and section.heading_path),
        fallback_title,
    )
    if not sections:
        sections = [
            ParsedSection(
                heading_path=[document_title],
                heading_level=1,
                body_lines=preamble,
                table_titles=_table_titles(preamble),
            )
        ]
    document_lead = _lead_text(preamble, lead_chars)
    if not document_lead:
        document_lead = _lead_text(sections[0].body_lines, lead_chars)
    table_titles = list(dict.fromkeys(title for section in sections for title in section.table_titles))
    return ParsedDocument(
        document_title=document_title,
        lead_text=document_lead,
        heading_titles=list(dict.fromkeys(headings)),
        table_titles=table_titles,
        sections=sections,
    )


def stable_section_id(
    *,
    file_id: str,
    heading_path: list[str],
    ordinal: int,
) -> str:
    source = f"{file_id}\0{' / '.join(heading_path)}\0{ordinal}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"SEC-{digest.upper()}"


def section_lead_text(section: ParsedSection, max_chars: int = 600) -> str:
    return _lead_text(section.body_lines, max_chars)
