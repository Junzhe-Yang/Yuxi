from __future__ import annotations

import hashlib

ATLAS_NAVIGATION_PROMPT_VERSION = "acm-atlas-navigation-v1"
ATLAS_NAVIGATION_INSTRUCTIONS = (
    "这是知识库的离线导航地图，不是病例结论，也不是 Evidence。"
    "请结合当前调查自行选择值得查看的文档。先调用 "
    "open_atlas_document(doc_id, reason) 阅读该文档的详细治疗主题，再根据主题形成查询，"
    "并用同一 file_id 执行 document 范围的 Milvus 检索。"
    "只有 Milvus 返回并登记到 Evidence Store 的原文才能作为答案依据。"
    "不要把 Atlas 主题或其离线来源标记直接当作证据。"
)
ATLAS_DOCUMENT_LINE_TEMPLATE = "- {title}（doc_id={doc_id}）：{scope_summary}"
ATLAS_DOCUMENT_OPEN_INSTRUCTIONS = (
    "以下主题只是离线导航提示，不是 Evidence。请先阅读主题并形成查询，"
    "再调用 search_review_kb，设置 retrieval_scope='document' 且 "
    "file_id='{doc_id}'，重新检索原文。"
)
ADAPTIVE_ATLAS_NAVIGATION_PROMPT_VERSION = "acm-atlas-navigation-adaptive-v2"
ADAPTIVE_ATLAS_NAVIGATION_INSTRUCTIONS = (
    "这是知识库的离线导航地图，不是病例结论，也不是 Evidence。"
    "寻找未知来源、替代方案或跨文档证据时，直接使用 source_discovery/global；"
    "source_discovery 命中正确文档不表示具体证据义务已经覆盖。已经建立具体来源、"
    "但当前 evidence obligation 尚未绑定直接原文时，应优先调用 "
    "open_atlas_document(doc_id, reason)，再使用 within_document_localization/document。"
    "Atlas 概览本身不会强制后续限定到某篇文档。只有 Milvus 返回并登记到 "
    "Evidence Store 的原文才能作为答案依据。"
)
ADAPTIVE_ATLAS_DOCUMENT_OPEN_INSTRUCTIONS = (
    "以下主题只是离线导航提示，不是 Evidence。若当前 evidence obligation 需要在这篇"
    "已知来源内定位直接原文，"
    "可调用 search_review_kb，设置 retrieval_intent='within_document_localization'、"
    "retrieval_scope='document' 且 file_id='{doc_id}'；若仍在寻找新来源或替代方案，"
    "应使用 retrieval_intent='source_discovery' 和 retrieval_scope='global'。"
)
ATLAS_NAVIGATION_PROMPT_HASH = hashlib.sha256(
    (
        ATLAS_NAVIGATION_INSTRUCTIONS + "\n" + ATLAS_DOCUMENT_LINE_TEMPLATE + "\n" + ATLAS_DOCUMENT_OPEN_INSTRUCTIONS
    ).encode("utf-8")
).hexdigest()
ADAPTIVE_ATLAS_NAVIGATION_PROMPT_HASH = hashlib.sha256(
    (
        ADAPTIVE_ATLAS_NAVIGATION_INSTRUCTIONS
        + "\n"
        + ATLAS_DOCUMENT_LINE_TEMPLATE
        + "\n"
        + ADAPTIVE_ATLAS_DOCUMENT_OPEN_INSTRUCTIONS
    ).encode("utf-8")
).hexdigest()


def build_atlas_document_memory(documents: list[dict[str, str]]) -> str:
    if not documents:
        return ""
    lines = [
        "【Corpus Atlas 文档概览】",
        ATLAS_NAVIGATION_INSTRUCTIONS,
    ]
    lines.extend(ATLAS_DOCUMENT_LINE_TEMPLATE.format(**value) for value in documents)
    return "\n".join(lines)


def build_adaptive_atlas_document_memory(
    documents: list[dict[str, str]],
) -> str:
    if not documents:
        return ""
    lines = [
        "【Corpus Atlas 文档概览】",
        ADAPTIVE_ATLAS_NAVIGATION_INSTRUCTIONS,
    ]
    lines.extend(ATLAS_DOCUMENT_LINE_TEMPLATE.format(**value) for value in documents)
    return "\n".join(lines)
