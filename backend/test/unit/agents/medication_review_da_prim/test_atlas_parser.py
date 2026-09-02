from yuxi.agents.buildin.medication_review_da_prim.corpus_atlas.parser import (
    parse_markdown_document,
    stable_section_id,
)


def test_parser_preserves_heading_paths_and_table_titles() -> None:
    parsed = parse_markdown_document(
        file_id="file-1",
        file_name="fallback.md",
        markdown="""
# 老年治疗共识

导言内容。

## 肾功能调整

表1 剂量调整

| 药物 | 剂量 |
| --- | --- |
| 甲 | 1 |

### 监测

监测肾功能。
""",
    )

    assert parsed.document_title == "老年治疗共识"
    assert parsed.heading_titles == ["老年治疗共识", "肾功能调整", "监测"]
    assert parsed.table_titles == ["表1 剂量调整"]
    assert parsed.sections[1].heading_path == ["老年治疗共识", "肾功能调整"]
    assert parsed.sections[2].heading_path == [
        "老年治疗共识",
        "肾功能调整",
        "监测",
    ]


def test_parser_creates_pseudo_section_for_headingless_document() -> None:
    parsed = parse_markdown_document(
        file_id="file-2",
        file_name="无标题文档.md",
        markdown="第一段。\n\n第二段。",
    )

    assert parsed.document_title == "无标题文档"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading_path == ["无标题文档"]
    assert parsed.lead_text.startswith("第一段")


def test_section_id_is_stable_and_ordinal_sensitive() -> None:
    first = stable_section_id(
        file_id="file-1",
        heading_path=["文档", "章节"],
        ordinal=1,
    )
    repeated = stable_section_id(
        file_id="file-1",
        heading_path=["文档", "章节"],
        ordinal=1,
    )
    second = stable_section_id(
        file_id="file-1",
        heading_path=["文档", "章节"],
        ordinal=2,
    )

    assert first == repeated
    assert first != second
