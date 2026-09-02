from __future__ import annotations

from .models import CaseRouteRecord, RetrievalOpportunity


def build_atlas_memory(
    *,
    case_route: CaseRouteRecord,
    opportunities: list[RetrievalOpportunity],
    include_opportunities: bool,
) -> str:
    sections_by_file: dict[str, list[str]] = {}
    for section in case_route.map_sections:
        heading = " / ".join(section.heading_path)
        if heading:
            sections_by_file.setdefault(section.file_id, []).append(heading)

    lines = [
        "【Corpus Atlas 语料地图】",
        "以下文档和章节仅是检索导航，不表示临床关系或合理性结论。",
    ]
    for document in case_route.ranked_documents[:6]:
        headings = list(dict.fromkeys(sections_by_file.get(document.file_id, [])))
        view_types = []
        for view_id in document.source_view_ids:
            if view_id == "VIEW-CASE":
                view_types.append("完整病例")
            elif view_id == "VIEW-REGIMEN":
                view_types.append("完整方案")
            elif view_id.startswith("VIEW-PE"):
                view_types.append("方案要素")
            elif view_id.startswith("VIEW-PM"):
                view_types.append("患者事实")
        detail = f"；匹配章节：{'；'.join(headings[:2])}" if headings else ""
        provenance = f"；主要视图：{'、'.join(dict.fromkeys(view_types))}" if view_types else ""
        lines.append(f"{document.rank}. {document.document_title}（{document.file_name}）" f"{detail}{provenance}")

    if include_opportunities:
        lines.extend(
            [
                "",
                "【语料诱导的调查机会】",
                "这些机会只表示多个病例对象共同指向同一章节；请自行判断是否值得调查。",
                "若决定调查，可在 search_review_kb 的 opportunity_id 中传入对应 OP ID；也可完全忽略。",
            ]
        )
        if not opportunities:
            lines.append("当前没有达到质量条件的调查机会。")
        for opportunity in opportunities:
            node_text = "＋".join(value.source_span for value in opportunity.node_matches)
            lines.append(
                f"[{opportunity.opportunity_id}] {node_text} -> "
                f"{opportunity.file_name} / {' / '.join(opportunity.heading_path)}"
            )
    return "\n".join(lines)
