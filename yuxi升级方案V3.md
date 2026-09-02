前两轮设计中，把“可追溯、可评价、可控”错误地实现成了多层语义审批：一个模型检索，另一个模型抽取Claim，第三个模型综合，程序再按另一套规则否决，最后渲染器又混合校验前后的文本。这样的架构会系统性压制LLM已经完成的正确推理。

我完整检查了 `Yuxi_re1.zip` 中约7,400行MedicationReviewAgent代码、最新失败分析文档和本次运行的完整trace。结论很明确：

> **PEA-RAG V2不值得继续修补，应当退出主实验。新版应回到Yuxi原生Agent的模型驱动检索循环，只在其前后增加非常薄的处方要素锚定、Evidence记忆和输出覆盖检查。**

Yuxi原生ChatbotAgent使用`create_agent`构造模型驱动的工具循环，由模型决定查什么、如何改写、是否继续查、是否打开原文以及何时回答。 新方法应继承这一核心，不再另建一个与LLM对抗的临床规则引擎。

---

# 一、这次失败的根本原因

当前系统的核心问题可以概括为：

> **语义理解权被分散给了过多组件，而检索Agent本人反而没有看到完整证据。**

本次运行已经成功检索到关键内容：

> 已有体位性低血压或血压过低的老年人应禁用α₁受体阻滞剂；α₁受体阻滞剂与其他降压药合用会增强降压作用；用药期间应监测立卧位血压。

这条证据几乎可以直接完成对特拉唑嗪的患者特异性判断。最终系统却将其降为“不确定”，原因发生在检索之后。

更严重的是，检索Agent运行时并没有真正看到上述句子。

当前`search_evidence`工具返回给Agent的主要内容只有：

```json
{
  "executed": 3,
  "new_evidence_ids": ["EV006", "EV007"],
  "remaining_query_budget": 5
}
```

Evidence全文保存在隐藏状态中。下一次模型调用时，`graph.py::_decision_context()`只向Agent展示每条Evidence开头的260个字符：

```python
"excerpt": " ".join(item.raw_text.split())[:260]
```

本例EV007规范化后约有1,520个字符，其中：

* “体位性低血压是老年人……”约位于第390字符；
* “已有体位性低血压……应禁用”约位于第438字符；
* “与其他降压药物合用……”约位于第501字符。

Agent看到的260字符仅包含前面的剂量、肾肝功能和高选择性α₁受体阻滞剂介绍，直接禁忌证全部被截掉。Claim抽取器稍后读取了全文，Synthesis也读取了Claim；于是出现了真正的“左右脑分裂”：

```text
检索Agent：
只知道拿到了EV007，但看不到关键句

Claim抽取器：
看到了关键句并生成CL005、CL006、CL007、CL008

Synthesis：
基于Claim形成avoid结论

程序校验器：
因为LaTeX形式与重新复制的source_span不一致，删除CL005引用

最终渲染：
判断写成不确定，未校验的summary仍写“典型禁忌证，应避免”
```

所以当前结果并不能说明LLM不会利用证据。代码首先不让检索Agent看到证据，随后又否定了后处理模型正确抽取的证据。

---

# 二、当前全流程中有问题的环节

## 1. 病例与方案抽取过重

当前方案抽取包括：

* 约10种治疗方案要素类型；
* 父子关系和组合方案关系；
* source span严格匹配；
* 时间表达绑定；
* 组合方案和治疗意图去重；
* 关系循环检查；
* 第一次LLM抽取；
* 第二次独立LLM核查；
* 修复操作；
* 再次确定性验证。

对于包含联合治疗阶段、方案级疗程和明确复评时点的病例，方案级要素确实有价值。但对大量普通处方病例，真正需要稳定锚定的对象只有：

* 明确药物医嘱；
* 原文明确提出的疗程或时点；
* 原文明确提出的监测、随访或停换药安排；
* 明确命名的联合方案。

现有抽取器在本例中识别出三个药物，结果基本正确；第二个Verifier没有带来新增信息，却额外消耗2,965 tokens。更加关键的是，原病例全文始终存在，方案抽取结果本来只应作为“防漏提示”，不应成为Agent允许讨论内容的边界。

新版不需要独立Verifier。保留一次结构化抽取和一次仅在JSON非法时触发的修复即可。

## 2. 动态议程增加了第二个规划大脑

本例只有三个药物和八次查询预算，动态议程生成了五个问题：

* 特拉唑嗪与氨氯地平的叠加降压风险；
* 氨氯地平和体位性低血压；
* BPH联合治疗疗程和监测；
* 特拉唑嗪起始剂量和监测；
* 夜尿病因、去氨加压素和生活方式干预。

前三个高优先级问题已经存在重叠。最后一个问题还在证据出现前主动扩展到去氨加压素，最终第三轮的三条查询全部用于夜尿分支，其中一条因预算不足被跳过。

动态议程消耗3,453 tokens，却没有真正控制完成性。当前只要一个问题关联的查询成功返回片段，问题状态即可变为`searched`，不代表：

* 返回内容相关；
* 证据适用于患者；
* 问题已经获得答案；
* Agent在最终输出中使用了证据。

因此，动态议程同时产生三类副作用：

1. 与Agent自身规划重复；
2. 在检索前预设答案和替代治疗方向；
3. 把有限预算平均分配给大量问题。

新版应删除`review_agenda.py`。Agent直接看到病例、方案锚点、已有Evidence和剩余预算，自行决定下一步调查重点。

## 3. Agent工具结果没有证据内容

这是最严重的执行缺陷。

当前检索工具只返回Evidence ID、数量和预算。下一轮上下文又只附上固定前260字符。Agent无法判断：

* 结果是否回答了问题；
* 是否应继续改写查询；
* 是否应打开原文；
* 某个Evidence是否比另一个更关键；
* 是否已经获得停药、监测或替代建议。

Agent的“自主检索”因而缺少ReAct中的Observation。形式上存在循环，实质上模型依赖自己的既有判断不断发起查询。

新版工具必须直接返回可读Evidence Card，例如：

```json
{
  "query_id": "Q002",
  "results": [
    {
      "evidence_id": "EV007",
      "source": "老年人BPH/LUTS药物治疗共识（2015）",
      "chunk_index": 12,
      "excerpt": "体位性低血压是老年人应用α1受体阻滞剂的不良反应。已有体位性低血压或血压过低的老年人应禁用……与其他降压药合用，降压作用增强……用药期间建议监测立卧位血压。"
    }
  ],
  "remaining_search": 5,
  "remaining_open": 2
}
```

全文仍保存在Evidence Store，Agent可以按ID打开更大窗口。

## 4. 固定前缀截断必须删除

不能继续使用：

```python
raw_text[:260]
```

新版应生成“查询中心窗口”，过程可以完全确定性实现，不需要增加LLM：

1. 清理HTML、LaTeX标签和重复空白；
2. 按句子、段落或表格行切分；
3. 从查询和病例锚点中提取药物、疾病、症状和主要动作词；
4. 找出词汇重合最高的片段；
5. 返回该片段及其前后相邻内容，总长约600至1,000字符；
6. 无法定位时才返回开头窗口。

本例查询中明确包含“特拉唑嗪”“体位性低血压”“降压药”，因此会直接定位到EV007的禁忌证段落。

## 5. 搜索预算耗尽会提前杀死Agent

当前`tools.py`中：

```python
if executed_count >= context.max_search_calls:
    route = "prepare_evidence"
```

这会绕过Agent下一轮决策。即使还有两次`open_evidence_source`预算，Agent也无法再使用；同时Agent没有机会调用`finish_retrieval`指定重要Evidence和未解决问题。

本次运行中：

* 最大搜索次数为8；
* 实际执行8次；
* open预算为2；
* 实际open次数为0；
* Agent曾明确认为一条内容被截断，需要打开原文。

新版应采用工具级可用性控制：

```text
搜索预算耗尽：
search工具不再可用
open工具仍然可用
Agent仍可直接生成最终回答

open预算也耗尽：
Agent仍可使用已有证据回答

模型自然输出最终答案：
结束循环
```

不再需要`finish_retrieval`工具。原生Yuxi Agent在模型不产生工具调用时自然结束，这一机制已经足够。

## 6. Evidence选择器没有临床判断能力

当前没有Agent优先Evidence时，`select_evidence()`按查询轮询：

```text
所有查询的rank 1
→ 所有查询的rank 2
→ 所有查询的rank 3
```

本例选入15条Evidence，其中包括：

* 无CKD患者的CKD指南；
* 未诊断代谢综合征患者的代谢综合征共识；
* 当前未使用M受体拮抗剂的急性尿潴留规则；
* 夜尿和去氨加压素支线；
* 多个弱相关或OCR残片。

它没有使用相关性、患者适用性、问题优先级、互补性或Agent判断。

新版不再需要全局Evidence选择节点。Agent在检索时直接看到结果并引用所需Evidence；最终依据清单根据Agent实际引用的Evidence ID生成。换言之：

> Evidence由Agent通过引用完成选择，而不是由一个不理解医学语义的round-robin算法替Agent选择。

若上下文过长，只在Agent消息中保留：

* 最近一次检索返回的可读卡片；
* 已有Evidence的短索引；
* Agent已引用或打开过的Evidence；
* 完整原文留在后端存储中。

## 7. Claim抽取是高成本的二次抄写

Claim抽取消耗27,728 tokens，是本次开销最高的单一阶段。其主要流程是：

```text
Evidence全文
→ LLM重新表达statement
→ LLM重新逐字复制source_span
→ 程序检查source_span是否是Evidence子串
→ 生成Claim ID
```

Claim原子化在离线评价中有价值，但不适合成为运行时回答的必经层。程序已经拥有Evidence原文，重新复制source span只会引入：

* LaTeX格式差异；
* OCR空格差异；
* 标点差异；
* 模型轻微改写；
* 额外token和延迟。

本次三条Claim因此被删除。

新版删除运行时`claim_extraction.py`。需要分析Citation蕴含或原子事实时，在实验结束后由离线评价器处理，不参与答案生成。

## 8. Synthesis重复复制原文并重新裁决

在claims模式下，程序已经知道：

```text
CL005
→ EV007
→ 规范source_span
```

Synthesis仍然要再次输出：

```json
{
  "evidence_id": "EV007",
  "claim_id": "CL005",
  "source_span": "已有体位性低血压……"
}
```

模型把LaTeX形式的`$ \alpha_{1} $`写成普通`α1`后，字符串校验认为source span未落回原文，于是删除Citation。直接禁忌证随之消失，原本的`avoid`被降为`uncertain`。

这是一种典型的反模式：

> 程序已经拥有确定映射，却让LLM重新转录，再用转录误差否定原映射。

新版中Agent只引用`[EV007]`。来源名称、chunk、Evidence全文和依据清单全部由程序根据ID生成。LLM不再输出source span。

## 9. 数值校验错误地否定患者事实

当前程序把Finding或建议中的所有数值都要求落回知识库Citation，没有区分：

* 原病例和原处方中的2 mg、5 mg、150/90 mmHg；
* 模型新增的替代剂量、阈值或疗程。

患者当前使用5 mg是病例事实，不需要指南再次证明“患者正在使用5 mg”。这导致多个Finding被替换为无信息的固定句。

同时，正则`\d+[A-Z]{1,5}`把“α1A”中的`1A`识别成具体临床参数，进一步降级替代建议。

新版运行时不做自由文本数值正则否决。系统提示中要求：

> 新增的具体替代剂量、频次、阈值和疗程应引用Evidence；没有直接来源时给出一般复核方向。

是否遵守由离线评价指标“无来源具体建议率”衡量。运行时不应因为正则误判删除整段临床判断。

## 10. 证据门槛存在不对称

当前逻辑会把没有Citation的负向Finding从`concern`降为`uncertain`，却允许没有Citation的`appropriate`继续保留。因此最终输出中出现：

> 特拉唑嗪适应证合理；未取得直接正向来源。

同时，`no_material_conflict_found`检查的是整个病例是否有任何Evidence：

```python
if basis == "no_material_conflict_found" and not evidence:
```

并未检查当前方案要素是否获得相关Evidence。于是非那雄胺没有合法Evidence，仍然被判断为“合理但需监测”。

新版不设置程序化的正负证据门槛。Agent根据病例和已读Evidence形成结论；离线Judge对合理项、不合理项、Citation蕴含和患者适用性统一评分。

## 11. 校验前后的字段被混合渲染

本地校验修改：

* Finding；
* disposition；
* evidence basis；
* recommendation；
* monitoring。

`ElementReview.summary`却直接沿用校验前文本。最终出现：

```text
判断：证据不足或仍不确定

说明：属于α1受体阻滞剂典型禁忌证，应避免使用并换药
```

氨氯地平也出现类似冲突：

```text
判断：不确定

说明：5mg偏高，建议调低起始剂量
```

这已经证明多层校验不能保证一致性，反而会制造程序性矛盾。

新版由同一个Agent一次性形成最终语义文本。程序不重写其医学结论；若结构不完整，只做一次定向补写。

## 12. 六段渲染重复放大错误

当前逐项判断完整展开Finding，正面汇总和负面汇总又逐条复制同样的Finding。降级建议被替换为相同固定句后也没有去重。

新版仍可保留你们金标准的六段结构，但要求：

* 第①部分由程序根据方案锚点生成；
* 第②部分由Agent逐项完整评价；
* 第③、④部分仅作一行式汇总，不复制全文；
* 第⑤部分给出综合建议；
* 第⑥部分由程序根据实际引用的Evidence ID生成。

---

# 三、新版设计：薄Harness下的单一语义Agent

建议名称可以暂定为：

> **Plan-Anchored Thin-Agent RAG，PAT-RAG**
> 中文：**方案要素锚定的轻量Agentic RAG**

它的核心原则是：

> 方案锚点负责防漏；
> Agent负责检索和临床解释；
> Evidence Store负责记忆和追踪；
> Harness只负责预算、ID和覆盖；
> 离线评价器负责判断答案对不对。

## 1. 新流程

```text
原病例
  ↓
轻量方案锚点抽取
  ↓
Yuxi式自主Agent循环
  ├─ search_review_kb
  ├─ open_review_evidence
  └─ 直接生成最终回答
  ↓
结构与方案覆盖检查
  ├─ 完整：追加依据清单
  └─ 漏项：同一模型执行一次定向补写
  ↓
最终答案与trace
```

检索没有变成固定流程。Agent仍然自主决定：

* 是否需要检索；
* 先检查哪种药物或患者风险；
* 如何构造查询；
* 是否连续执行多轮检索；
* 是否打开原文；
* 何时信息已经足够；
* 如何处理正负证据；
* 最终如何回答。

固定部分只有：

* 最大查询和open预算；
* 方案锚点ID；
* Evidence ID；
* 每个显式要素至少出现一次。

## 2. 最小状态模型

```python
class MedicationReviewLiteState(TypedDict):
    messages: list
    raw_case_text: str

    plan_anchors: list[PlanAnchor]

    evidence_store: dict[str, EvidenceItem]
    search_records: list[SearchRecord]
    open_records: list[OpenRecord]

    search_count: int
    open_count: int
    technical_attempts: int

    final_answer: str | None
    coverage_warnings: list[str]
```

当前以下运行时对象可以删除：

* ReviewAgenda；
* ReviewQuestion状态；
* EvidenceClaim；
* role_hints；
* ReviewSynthesisDraft；
* SourceCitationDraft；
* LocalValidationEvent；
* ElementLedger；
* round-robin EvidenceSelection；
* finish_retrieval状态。

## 3. 轻量方案锚点

```python
class PlanAnchor(BaseModel):
    element_id: str
    source_span: str
    label: str

    kind: Literal[
        "medication_order",
        "explicit_duration_or_timing",
        "explicit_monitoring_or_followup",
        "explicit_regimen_or_other",
    ]
```

抽取原则：

* 每种明确药物医嘱一个锚点；
* 原文明确提出的方案级疗程或评价时点一个锚点；
* 原文明确提出的监测、随访、停换药条件一个锚点；
* 命名联合方案可建立一个方案锚点；
* 不预先生成药物×疾病、所有药物对或器官功能关系；
* 原病例全文始终交给Agent；
* 锚点遗漏不会禁止Agent讨论原文中的其它内容。

只进行以下程序校验：

* source span是否出现在病例原文；
* ID是否唯一；
* 至少包含已明确列出的药物医嘱；
* JSON非法时修复一次。

取消第二个LLM Verifier。

## 4. 检索工具

建议把搜索Schema缩减为：

```python
class SearchReviewKBInput(BaseModel):
    query_text: str
    reason: str
    focus_element_ids: list[str] = []
```

`focus_element_ids`只用于trace，不参与Evidence资格判断。删除：

* evidence_role；
* linked_question_ids；
* required_dimensions；
* polarity；
* applicability；
* review target状态。

工具结果直接向Agent返回证据卡片。Evidence Store后台保存完整原文和来源。

## 5. Evidence不再预分类

同一Evidence可以同时说明：

* 药物适应证；
* 患者特异禁忌；
* 联用风险；
* 监测要求；
* 调整原则。

无需强迫它拥有唯一`support`、`challenge`或`conditional`角色。Agent可以在不同Finding中多次引用同一Evidence。

## 6. 最终回答

最快落地版本让Agent直接生成六段式文本，内联引用：

```text
【PE001】……
证据：[EV007]
```

程序只检查：

* 每个PE ID是否至少出现一次；
* 是否引用不存在的Evidence ID；
* 是否有重复或未知PE ID。

第⑥部分依据清单自动生成：

```text
[EV007] 老年人良性前列腺增生症/下尿路症状药物治疗共识（2015），chunk 12
```

LLM不再复制source span。

如果某个要素漏写，只执行一次局部补写：

```text
你此前的答案遗漏PE002。
请只补充PE002的评价，不重写其它内容。
可用证据为……
```

这项补写只保证覆盖，不判断医学正确性。

---

# 四、建议直接复用Yuxi原生Agent

不建议继续维护当前约900行的自定义`graph.py`。更合适的实现是：

```text
extract_anchors
→ create_agent(...)
→ coverage_check
→ optional_patch
→ append_evidence_list
```

内部Agent继续使用Yuxi的`create_agent`模式。原生ChatbotAgent已经提供：

* 模型工具循环；
* 对话和checkpoint；
* ToolMessage；
* 多轮调用；
* 最终AIMessage；
* 各类中间件。

新Agent只增加：

* `ReviewAnchorMiddleware`：把方案锚点和简短任务提示注入系统上下文；
* `ReviewEvidenceTool`：包装现有Milvus检索并分配EV ID；
* `ToolCallLimitMiddleware`：限制检索和open预算；
* `CoverageCheck`：最终方案要素覆盖检查。

这使方法与baseline具有天然可比性，因为两者共享同一Agent循环和知识库工具边界。

---

# 五、当前代码如何处理

建议冻结当前版本，标记为：

```text
pea-rag-v2-overconstrained-prototype
```

不再继续修复Claim和Synthesis链。

可以保留：

* `retrieval.py`中的Milvus调用、超时和错误分类；
* `evidence_board.merge_evidence()`的去重和稳定EV ID；
* `extraction.py`和`llm_io.py`中的JSON fallback；
* `trace.py`；
* 批处理导出；
* 每题独立thread；
* 技术重试和调用记录。

应退出主路径：

* `review_agenda.py`；
* `claim_extraction.py`；
* `review_synthesis.py`；
* round-robin `select_evidence()`；
* `finish_retrieval`；
* `SourceCitationDraft.source_span`；
* 语义性的LocalValidationEvent；
* 自由文本数值正则；
* 六段式重复渲染。

建议新增：

```text
anchor_extraction.py
evidence_excerpt.py
review_tools_lite.py
coverage_check.py
graph_lite.py
```

---

# 六、失败病例在新版中的执行方式

对当前BPH病例，新版应出现类似轨迹：

1. 抽取三个锚点：特拉唑嗪、非那雄胺、氨氯地平。
2. Agent看到体位性低血压和既往晕厥骨折，优先查询特拉唑嗪。
3. 工具把EV007的禁忌证段落直接返回。
4. Agent可以继续检索非那雄胺的适应证和剂量，以及氨氯地平与当前血压状态。
5. 若需要更多上下文，可打开EV007。
6. Agent直接生成回答。

合理的输出边界大致是：

* 特拉唑嗪：BPH治疗方向有其一般依据，但患者已有症状性体位性低血压并发生过晕厥骨折，直接来源提示该情况下应避免α₁受体阻滞剂；与其他降压药联用还会增强降压作用。患者特异风险具有决定性意义，应调整方案并监测立卧位血压。[EV007]
* 非那雄胺：5 mg每日一次是否有直接剂量依据、患者是否满足前列腺体积或PSA条件，应由检索证据决定。缺少相关信息时表达为条件性合理。
* 氨氯地平：可以讨论总体降压负担和体位性症状；没有直接来源时不得凭空指定2.5 mg替代剂量。
* 不使用CKD指南作为无CKD患者的主要依据。
* 如果无法找到具体替代方案，建议写成“一般性换药或减量评估”，而非编造具体剂量。

即使Agent出现某个Citation不足，答案仍会保留并在离线评价中被扣分，不会整例退化成“证据不足”。

---

# 七、建议的增量实验

## 第一阶段：不重新检索，先证明后处理链是否有害

直接使用本次trace中已有的15条入选Evidence：

* R0：当前V2输出；
* R1：同一模型直接读取15条chunk，一次性回答；
* R2：R1加三个方案锚点；
* R3：R2加一次覆盖补写。

这项实验几乎不需要改检索器。如果R1或R2明显优于R0，就可以直接证明：

> Claim抽取、Synthesis和语义校验链产生了净信息损失。

还可以增加一个更强的回放：

* R4：把全部31条Evidence交给模型，但由模型自主选择引用；
* 对比R2，判断当前round-robin选择是否进一步损害结果。

## 第二阶段：在线Agent实验

建议仅保留四组：

| 组别            | 方案锚点 | 自主检索 | Evidence可读卡片 | 覆盖补写 |
| ------------- | ---: | ---: | -----------: | ---: |
| B0 原生Yuxi     |    否 |    是 |       原生工具结果 |    否 |
| B1 统一输出Prompt |    否 |    是 |       原生工具结果 |    否 |
| M1            |    是 |    是 |       原生工具结果 |    否 |
| M2            |    是 |    是 |            是 |    否 |
| M3 完整新版       |    是 |    是 |            是 |    是 |

所有组都必须：

* 使用同一模型；
* 使用同一纯向量知识库；
* 使用同一Top-5；
* 最大实际子查询数相同；
* 最大open次数相同；
* 使用相同六段式输出要求；
* 每题独立thread；
* 生成完整最终答案。

M1检验软方案锚点是否减少漏项；M2检验证据真正向Agent可见是否改善判断；M3检验一次覆盖补写是否进一步提高完整性。

## 第三阶段：全数据集

先在20至30个病例上调试，不再继续向Harness增加医学规则。开发阶段只能调整：

* Evidence卡片长度；
* 最大查询次数；
* open预算；
* Prompt长度；
* 是否启用一次覆盖补写。

确认M2或M3有稳定增益后，再运行全部数据集。

---

# 八、推荐指标

你现在的金标准已经包含正确和不适当部分，可以报告：

* Finding级Micro-F1和Macro-F1；
* 合理项Precision、Recall、F1；
* 不合理项Precision、Recall、F1；
* 方案要素覆盖率；
* 属性完整率；
* 完整Finding率；
* 完整病例正确率；
* Citation ID有效率；
* Citation蕴含率；
* Citation患者适用率；
* 无来源具体替代药、剂量、阈值和疗程比例；
* 最终答案生成成功率；
* 整例失败率；
* 检索子查询数；
* open次数；
* tokens、延迟和费用。

尤其建议增加两个指标。

### Evidence利用率

[
\text{Evidence Utilization}
===========================

\frac{\text{最终被引用的唯一Evidence数}}
{\text{检索到的唯一Evidence数}}
]

当前V2检索31条，最终只保留3条，其中还有一条CKD证据不适用于患者，Evidence利用效率很低。

### 检索后事实存活率

[
\text{Post-retrieval Survival}
==============================

\frac{\text{最终答案正确保留的已检索金标准事实数}}
{\text{检索结果中已经存在的金标准事实数}}
]

这能直接量化当前系统的核心问题：证据已经查到，却在Claim、Synthesis、验证或渲染阶段丢失。

---

# 九、新版的硬性验收条件

在进行正式批量实验前，至少满足：

1. EV007关键禁忌证直接出现在Agent可见的工具结果中。
2. Agent看到的是查询中心窗口，不再是固定前260字符。
3. 搜索预算耗尽后，open和最终回答仍然可用。
4. 不再运行动态议程、Claim抽取和第二个Synthesis模型。
5. 不再要求LLM复制source span。
6. 单个Citation错误不会删除Finding或导致整例失败。
7. 每个明确方案锚点至少评价一次。
8. 具体Evidence ID必须存在，但程序不判断它属于support或challenge。
9. 最终答案不存在校验前后状态混排。
10. 本例总token较当前73,735至少下降60%。
11. Harness中不出现疾病、药物类别或临床规则的特殊分支。
12. 所有语义评价放到离线评估环节，不在运行时阻断模型。

---

# 十、对当前项目的最终建议

这次不应继续“修正P0后再观察P1”。Citation canonicalization、数值正则和summary同步当然都能修，但修完仍然保留：

* 双重方案抽取；
* 动态议程；
* Agent盲读Evidence；
* round-robin选择；
* Claim层；
* Synthesis层；
* 本地语义校验；
* 六段重复渲染。

这条链会继续产生新的边界冲突。

最稳妥的收敛路线是：

> **回到Yuxi原生Agent，增加方案锚点和Evidence记忆，删除所有运行时语义裁判。**

这不是放弃严谨性。严谨性应体现在：

* 完整trace；
* 明确实验预算；
* 稳定Evidence ID；
* 金标准Finding评价；
* Citation蕴含和患者适用性评价；
* 失败案例分析；
* 专家抽样复核。

运行时系统负责完成任务，评价系统负责判断它完成得如何。把评价器塞进生成循环，最终只会让系统学会通过契约，却无法保证临床结论更正确。

我还尝试运行压缩包中的MedicationReviewAgent单元测试。当前分析容器缺少`langchain_core`和`tomli`，13个测试模块在收集阶段终止，因此我没有把“测试是否通过”作为上述结论依据；结论来自完整代码静态审查和真实运行trace。

相关材料：

[下载Yuxi_re1完整源码](sandbox:/mnt/data/Yuxi_re1.zip)

[查看2026-07-28失败案例分析](sandbox:/mnt/data/Yuxi_re1_extracted/Yuxi/docs/vibe/2026-07-28-pea-rag-v2-full-trace-analysis.md)

[查看本次完整运行JSON](sandbox:/mnt/data/Yuxi_re1_extracted/Yuxi/outputs/0728-re1/79b4a83c-a522-4d01-8d4b-cf1cf5720803.json)

[下载PEA-RAG V3轻量Agent重构报告](sandbox:/mnt/data/2026-07-28-pea-rag-v3-thin-agent-redesign.md)
