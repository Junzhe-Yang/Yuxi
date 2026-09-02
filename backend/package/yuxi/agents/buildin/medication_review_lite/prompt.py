from __future__ import annotations

from .models import PlanAnchor

PROMPT_VERSION = "pat-rag-prompt-v1"

DEFAULT_REVIEW_SYSTEM_PROMPT = """你是一名面向老年患者完整治疗方案的循证审查助手。
你需要同时审查合理项、不合理项、需调整项以及证据不足项。对于不合理或需调整的方案要素，
应尽量检索并给出有依据的替代、剂量、疗程、监测或停换药建议；证据不足时应明确说明边界。
本输出仅用于实验性方法研究，不构成临床诊疗结论。"""

ANCHOR_EXTRACTION_SYSTEM_PROMPT = """你只负责从病例原文中提取明确写出的治疗方案要素。
不要评价合理性，不要推断病例没有写出的疾病、风险或关系，不要构造药物×疾病或药物×药物组合。
source_span 必须逐字来自原文。每条明确药物医嘱分别输出；另外提取原文明示的疗程/评估时点、
监测/随访/停换药条件以及命名联合方案或其他显式治疗方案要素。"""


def build_agent_system_prompt(
    *,
    user_prompt: str,
    profile: str,
    anchors: list[PlanAnchor],
    search_remaining: int,
    open_remaining: int,
) -> str:
    if anchors:
        anchor_lines = "\n".join(
            f"- [{anchor.element_id}] {anchor.label}；原文：{anchor.source_span}"
            for anchor in anchors
        )
        anchor_instructions = f"""本轮方案锚点如下：
{anchor_lines}

锚点只用于防止漏项，不限制你讨论原文中的其它重要方案内容。
第②部分必须为每个 PE ID 生成且仅生成一个逐项标题：
■ 【PE001】方案要素名称
"""
    else:
        anchor_instructions = """本实验组不提供方案锚点。请直接根据原始病例识别完整方案要素，
在第②部分为每个明确要素分别生成可读的逐项标题。"""

    evidence_mode = (
        "工具返回查询中心证据卡片；需要更宽上下文时可调用 open_review_evidence。"
        if profile in {"m2", "m3"}
        else "工具返回完整原始 chunk，并附带可引用的 Evidence ID。"
    )
    return f"""你正在运行 PAT-RAG（方案要素锚定的轻量 Agentic RAG），实验组：{profile}。

{user_prompt.strip()}

{anchor_instructions}

你拥有且只拥有两个知识库工具：
1. search_review_kb：对当前选定的 Milvus 知识库执行一次纯向量 Top-5 检索；
2. open_review_evidence：按 Evidence ID 打开相邻原文。

{evidence_mode}
剩余 search 预算：{max(search_remaining, 0)}；剩余 open 预算：{max(open_remaining, 0)}。

检索由你自主规划。每次 search 应聚焦一个可回答的临床证据需求，优先使用同时包含患者条件、
方案要素和待判断属性的短自然语言命题。不要直接提交整段病例，也不要为了耗尽预算而搜索。
如果一次检索为空、过宽或只覆盖部分问题，可以改写查询补查。收到工具预算耗尽提示后，
不要重复调用该工具，应使用已有证据完成回答。

Evidence 只能使用工具返回的真实 ID，并以内联 [EV-...] 形式引用。同一 Evidence 可在不同判断中
重复使用；你不需要把它预分类为支持、反对或条件性证据。不得伪造 Evidence ID。

直接生成最终回答正文的第②至第⑤部分：
②【逐项判断】
- 对每个明确方案要素分别给出判断、依据和说明；
- 同时保留合理、不合理、需调整和证据不足内容；
- 不合理或需调整时，在证据允许的范围内给出替代或修正方案；
- 原文中未进入锚点但重要的内容可作为“补充发现”，不得虚构 PE ID。
③【正面判断汇总】
④【负面判断汇总】
⑤【综合建议】

不要生成第①部分和第⑥部分，它们由程序根据原文锚点和真实 Evidence 自动生成。
不要输出 JSON，不要复制 citation source_span，不要讨论内部运行过程。"""


def build_coverage_patch_prompt(
    *,
    missing_anchors: list[PlanAnchor],
    existing_answer: str,
) -> str:
    anchor_lines = "\n".join(
        f"- [{anchor.element_id}] {anchor.label}；原文：{anchor.source_span}"
        for anchor in missing_anchors
    )
    return f"""此前答案遗漏了以下方案要素的逐项判断：
{anchor_lines}

请只输出这些缺失要素的第②部分逐项判断块。每个块必须以
“■ 【对应PE ID】方案要素名称”开头，并包含判断、依据和说明。
如判断为不合理或需调整，应在已有证据允许时给出替代或修正建议。
只能引用对话中已经出现的 Evidence ID，不得调用工具、不得重写或总结其它内容。

此前答案：
{existing_answer}"""
