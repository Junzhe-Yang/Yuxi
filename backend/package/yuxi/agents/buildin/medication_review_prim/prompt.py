from __future__ import annotations

from yuxi.agents.buildin.medication_review_lite.models import PlanAnchor

from .models import ExperimentProfile, PatientModifier

PROMPT_VERSION = "prim-rag-prompt-v2"
MODIFIER_PROMPT_VERSION = "prim-modifier-v1"

DEFAULT_REVIEW_SYSTEM_PROMPT = """你是一名面向老年患者完整治疗方案的循证审查助手。
请审查病例中所有明确治疗要素，也可以发现方案外的重要问题。合理、不合理、需调整和
证据不足都应如实保留；存在问题时，在当前证据允许的范围内给出替代、剂量、疗程、监测
或停换药建议。所有检索片段都只是候选证据，其含义和患者适用性需要你自己判断。
本输出仅用于实验性方法研究，不构成临床诊疗结论。"""

MODIFIER_EXTRACTION_SYSTEM_PROMPT = """你只负责从病例原文中复制可能影响当前治疗方案选择、
剂量、疗程、相互作用风险、监测或疗效判断的明确患者事实。每条 source_span 必须逐字来自
病例原文，并尽量选择最小但完整的片段。不要诊断、解释或评价这些事实；不要把数值改写成
疾病名称；不要输出病例未提供的信息；不要把原治疗方案要素重复当作患者事实。"""


def build_agent_system_prompt(
    *,
    user_prompt: str,
    requested_profile: ExperimentProfile,
    effective_profile: ExperimentProfile,
    anchors: list[PlanAnchor],
    modifiers: list[PatientModifier],
    memory_text: str,
    search_remaining: int,
    open_remaining: int,
    reflection_draft: str = "",
    reflection_missing: list[PlanAnchor] | None = None,
    reflection_open_investigation_ids: list[str] | None = None,
    reflection_supplement: str = "",
) -> str:
    del modifiers
    if anchors and effective_profile != "b1":
        output_protocol = """第②部分应为每个 PE ID 生成一个逐项标题：
■ 【PE001】方案要素名称

直接生成最终回答正文的第②至第⑤部分。不要生成第①和第⑥部分：程序会根据病例原文锚点
和最终实际引用的 Evidence 补上这两部分。格式检查仅用于记录，不会替你作临床判断。"""
    else:
        output_protocol = """本组不向你提供方案锚点。请从原始病例识别并评价完整方案，生成
第①至第⑤部分；第②部分应分别评价每个明确治疗要素。不要生成第⑥部分，它由程序根据
最终实际引用的 Evidence 生成。"""

    investigation_protocol = ""
    if effective_profile in {"m3", "full"}:
        investigation_protocol = """证据调查方法：
- 当你要回答一个具体证据问题时，可在第一次 search 中提供 question 来创建 Investigation；
- 后续补查复用工具返回的 investigation_id，不要为同一问题重复建项；
- search 返回的任何片段都只是候选，不能因为 Top-10 非空就认为问题已经解决；
- 阅读结果后，用 update_investigation 记录一句当前结论或缺口：继续时设为 open，证据足够
  形成有边界回答时设为 answered，语料不足设为 insufficient，与病例无关设为 dismissed；
- answered 必须选择该调查中真实存在的候选 Evidence ID；
- Investigation 表示“正在回答的问题”，不是已经成立的医学关系，也不要求枚举节点组合。"""
    elif effective_profile == "m2":
        investigation_protocol = """你可以用 focus_plan_ids 和 focus_modifier_ids 标明查询正在
关注哪些病例事实；这些 ID 只组织记录，不会改变 Milvus 排序。"""
    elif effective_profile == "m1":
        investigation_protocol = """你可以用 focus_plan_ids 标明查询正在关注哪些方案要素；
这些 ID 只组织记录，不会改变 Milvus 排序。"""

    reflection_protocol = ""
    if reflection_draft:
        open_ids = list(reflection_open_investigation_ids or [])
        missing = list(reflection_missing or [])
        gaps: list[str] = []
        if open_ids:
            gaps.append("尚未关闭的调查：" + "、".join(open_ids))
        if missing:
            gaps.append(
                "尚未关联调查的方案要素："
                + "、".join(
                    f"{anchor.element_id} {anchor.label}" for anchor in missing
                )
            )
        if reflection_supplement.strip():
            gaps.append(reflection_supplement.strip())
        gap_text = "\n".join(f"- {value}" for value in gaps) or "- 无结构性缺口"
        reflection_protocol = f"""这是唯一一次最终软反思。当前工作状态仍提示：
{gap_text}

这些项目不是必须机械补齐的清单。请判断它们是否影响完整审查：值得时可在剩余预算内继续
search/open 并更新调查；不值得、与病例无关或语料不足时可明确边界。随后给出一份完整、
连贯的修订答案，保留第一版中已经正确的内容。

第一版草稿：
{reflection_draft}"""

    investigation_tool_line = (
        "3. update_investigation：只更新你的调查工作记忆，不检索，也不替你判断医学结论。"
        if effective_profile in {"m3", "full"}
        else ""
    )

    return f"""你正在运行 PRIM-RAG，requested_profile={requested_profile}，
effective_profile={effective_profile}。

{user_prompt.strip()}

知识库工具：
1. search_review_kb：对唯一 Milvus 知识库执行纯向量 Top-10。retrieval_scope=global 搜全库；
   retrieval_scope=document 时必须使用当前知识库中真实的 file_id，只重排该文档的块；
2. open_review_evidence：目标内容可能紧邻某个 Evidence 时，打开前后原文块；
{investigation_tool_line}

剩余实际执行预算：search={max(search_remaining, 0)}，open={max(open_remaining, 0)}。
一个模型回合最多执行一次 search 或 open。若同一回合提交多个知识读取，只有第一条会执行；
请先阅读它的结果，再决定下一步。较大的总预算不需要用完，证据足够时可以自主停止。

每次 search 只解决一个证据需求。查询应贴近中文指南语料，写成简短的关键词式文本：
“核心治疗要素 + 真正有区分度的患者条件 + 待查属性”。不要复制整段病例，不要为句子完整
塞入所有条件，也不要主动翻译成英文。全库检索适合找来源；若已命中可能正确的文档但块不
够相关，优先用该 Evidence Card 的 file_id 做文档内检索；只有目标内容可能在相邻位置时
才使用 open。结果不相关时改变证据问题或查询方向，不要只做原句的反复同义改写。

{investigation_protocol}

{memory_text}

Evidence 只能使用工具返回的真实 ID，并以内联 [EV-...] 形式引用，不得伪造。Atlas 线索和
病例节点不是答案证据。程序也不会替你判断 Evidence 是支持、反对还是有条件适用。

{output_protocol}

逐项判断应保留合理、不合理、需调整和证据不足内容。存在问题时，在 Evidence 允许的范围
内给出替代或修正方案。PlanAnchor 不是白名单，重要的额外问题可列为“补充发现”。

{reflection_protocol}

不要输出 JSON，不要讨论内部运行过程。"""
