## 一、总体评估：审计结论成立，当前问题属于架构级需求偏移

当前实现并非“处方关系覆盖方法尚未调好”，而是把原始任务从“完整治疗方案合理性审查”缩减成了“药物—疾病等关系的固定批量检索”。它缺少完整方案要素、正向支持证据、证据适用性判断、缺口驱动补检、最终回答和完整性验证，因此不能作为完整方法组与Yuxi的自主多轮ChatbotAgent比较。当前49次程序化查询和45个唯一片段只能证明检索接口运行成功，无法证明这些片段能够支持完整治疗方案判断。

原先提出的“患者特异处方关系覆盖图”也确实不足以单独构成下一版方法。它最多解决：

> 系统应该考虑哪些药物关系？

而你的最终任务还要求回答：

> 原方案中究竟有哪些明确要素？
> 每个药物、联合方案、治疗方向、疗程、评估时点和监测安排是否合理？
> 合理判断有什么正向依据？
> 不合理判断有什么风险依据和修改依据？
> 检索证据是否适用于当前患者？
> 最终答案是否逐项完整表达？

因此，下一版不能继续采用：

```text
病例解析
→ 确定性关系枚举
→ 固定查询
→ 检索结束
```

也不能简单在现有`finalize`后追加一次自由回答。后者仍然无法判断45个片段中哪些相关、哪些适用、哪些支持合理性、哪些支持调整建议，也无法保证每个方案要素恰好评价一次。

我建议将新方法正式调整为：

> **治疗方案要素—证据—回答三层覆盖引导的有界自主Agentic RAG**

暂可简称为：

> **PEA-RAG：Plan–Evidence–Answer Coverage-Guided Agentic RAG**

这里的“有界自主”具有明确含义：

- Agent自主选择下一步审查哪个要素；
- Agent自主生成查询；
- Agent自主决定查询支持、风险、患者条件还是修改建议；
- Agent自主决定是否打开原文、是否修改判断、是否补充检索；
- 状态图只负责限制预算、检查证据契约和阻止不完整终止；
- 系统不预先固定完整查询序列，也不强制每个槽位执行一条查询。

这仍然是Agentic retrieval，与Yuxi baseline的模型驱动工具循环具有可比性。Yuxi baseline本身也是由模型决定是否检索、如何改写查询、是否继续检索和何时停止，而非固定RAG流程。

近期SEMA-RAG同样将医学问题解释、证据探索和证据裁决分开，并基于具体证据缺口开展后续检索；其结果显示，大部分收益来自证据充分性反馈，而非简单增加固定检索轮数。该论文也明确承认，其充分性标准尚未针对开放式答案的生成完整性进行设计。这正是你们需要增加“治疗方案要素级完整性门控”的原因。

------

# 二、下一版方法必须同时控制三种覆盖

对第(i)个治疗方案要素，完整审查应同时满足：

# [ C_{\mathrm{complete}}(i)

C_{\mathrm{plan}}(i)
\land
C_{\mathrm{evidence}}(i)
\land
C_{\mathrm{answer}}(i)
]

其中：

- (C_{\mathrm{plan}})：原方案中的该要素是否被正确识别；
- (C_{\mathrm{evidence}})：是否取得足以支持判断的患者适用证据；
- (C_{\mathrm{answer}})：该要素及其必需属性是否完整进入最终答案。

当前原型主要尝试了部分关系层面的检索覆盖，没有实现上述三个覆盖中的任何一个完整闭环。

例如HRZE示例中：

```text
要素1：异烟肼及其剂量、频次
要素2：利福平及其剂量、频次
要素3：吡嗪酰胺及其剂量、频次
要素4：乙胺丁醇及其剂量、频次
要素5：HRZE联合方案及诊断性治疗方向
要素6：3个月后的疗效评估时点
```

药物—疾病关系枚举最多较稳定地覆盖前四项的一部分。它不能自然产生“HRZE方案方向”和“3个月评估时点”两个输出对象，也不能保证对四种药物分别获得正向方案依据。

因此，必须将“方案要素”和“临床关系”分成两层。

------

# 三、任务表示：采用“方案要素＋审查目标”两层模型

## 1. 第一层：显式治疗方案要素

这一层与金标准中的“原方案要素清单”直接对齐。只抽取病例中明确提出的治疗方案，不在这一层推断药物相互作用或处方遗漏。

建议定义：

```python
class TreatmentPlanElement(BaseModel):
    element_id: str

    element_type: Literal[
        "medication_order",
        "combination_regimen",
        "treatment_intent",
        "treatment_phase",
        "duration_or_schedule",
        "evaluation_timing",
        "monitoring_plan",
        "follow_up_plan",
        "switch_stop_escalation_rule",
        "nonpharmacologic_plan",
        "other_explicit_plan"
    ]

    source_span: str
    normalized_summary: str

    parent_element_id: str | None = None
    component_element_ids: list[str] = []
    target_diagnosis_ids: list[str] = []
    medication_ids: list[str] = []

    attributes: dict = {}
```

各类型的操作定义如下。

| 类型                          | 何时建立独立要素                                   |
| ----------------------------- | -------------------------------------------------- |
| `medication_order`            | 每种明确药物及其剂量、途径、频次、疗程作为一个要素 |
| `combination_regimen`         | 文本明确把多种药物视为一个联合方案或命名方案       |
| `treatment_intent`            | 诊断性、经验性、根治性、维持性、预防性等治疗方向   |
| `treatment_phase`             | 强化期、继续期、诱导期、维持期等阶段               |
| `duration_or_schedule`        | 方案级总疗程或阶段疗程，不能仅归属于单一药物       |
| `evaluation_timing`           | “治疗后3个月复评”等明确评价时点                    |
| `monitoring_plan`             | 实验室、症状、影像或药物毒性监测安排               |
| `follow_up_plan`              | 复查、随访及其时间安排                             |
| `switch_stop_escalation_rule` | 停药、换药、升级或降阶治疗条件                     |
| `other_explicit_plan`         | 上述分类无法稳定覆盖，但在原方案中明确出现的内容   |

两个重要原则：

第一，药物剂量、频次和单药疗程通常作为`medication_order`的属性，不必机械拆成多个要素；只有方案级疗程或独立评价时点才单独建立要素。

第二，所有要素必须带有病例原文中的`source_span`。没有原文片段的对象不能作为显式方案要素。

## 2. 第二层：派生审查目标

药物相互作用、药物—疾病冲突、器官功能适配、重复用药和处方遗漏属于系统需要调查的**派生审查目标**，不是原方案要素。

```python
class ReviewTarget(BaseModel):
    target_id: str

    target_type: Literal[
        "indication_support",
        "geriatric_appropriateness",
        "dose_frequency",
        "duration",
        "drug_disease",
        "drug_drug",
        "organ_function",
        "duplication",
        "cumulative_burden",
        "monitoring_requirement",
        "prescribing_omission",
        "regimen_consistency",
        "timing_consistency",
        "other"
    ]

    linked_element_ids: list[str]
    linked_patient_fact_ids: list[str] = []

    generation_source: Literal[
        "deterministic",
        "agent",
        "evidence_triggered"
    ]

    generation_reason: str
    priority: Literal["high", "medium", "low"]
```

当前七类关系枚举代码可以在这里继续复用，但用途需要改变：

```text
旧用途：
ReviewSlot → 固定生成查询 → 全部执行

新用途：
ReviewTarget → 作为Agent的审查提示和证据缺口对象
```

也就是说，现有关系枚举负责提醒Agent：

- 哪些药物对可能需要检查；
- 哪些药物与疾病可能需要核验；
- 哪些药物涉及器官功能；
- 是否存在重复或累积负担。

它不再自动生成49条查询。

## 3. 显式要素和派生发现的输出关系

最终答案中应有两类结果：

```python
class ElementJudgement(BaseModel):
    element_id: str
    judgement: Literal[
        "appropriate",
        "appropriate_with_monitoring",
        "needs_adjustment",
        "inappropriate",
        "insufficient_evidence"
    ]
    ...

class CrossElementFinding(BaseModel):
    finding_id: str
    linked_element_ids: list[str]
    finding_type: str
    judgement: str
    ...
```

这样可以保证：

- 每个原方案要素恰好评价一次；
- 一个药物—药物相互作用不会被错误计为新的原方案要素；
- 跨药物问题可以只呈现一次，同时链接到两种药物；
- 处方遗漏可以作为派生发现出现，不会污染原方案要素清单。

------

# 四、方案要素抽取不能只做单次LLM解析

当前解析基础可以保留，包括JSON Schema fallback、Pydantic校验、`source_mention` grounding和稳定ID。需要扩展的是Schema和完整性检查。

建议采用“抽取—核查—一次修复”的三步流程。

## 1. 第一次抽取

提示词应明确：

```text
只抽取病例中明确给出的治疗方案要素。
不得在该阶段推断药物相互作用、处方遗漏或替代方案。
每个要素必须复制对应的最小原文片段。
多药联合方案、治疗意图、治疗阶段、总疗程、
评价时点、监测和随访必须作为独立对象检查。
```

## 2. 确定性检查

程序至少执行：

- 所有`source_span`必须是原始病例的子串；
- 所有药物名称均有对应`medication_order`；
- 明确剂量、频次和给药途径必须绑定到某一药物；
- 所有时间表达，如“3个月后”“每2周”“疗程6个月”，必须绑定到药物属性或独立时间要素；
- `component_element_ids`必须真实存在；
- 同一原文片段不能无理由生成两个语义重复要素；
- 方案简称或组合表达若存在，必须检查是否形成`combination_regimen`。

## 3. 独立完整性核查

第二次LLM调用只做核查，不重新自由抽取：

```json
{
  "missing_explicit_spans": [],
  "duplicated_element_ids": [],
  "hallucinated_element_ids": [],
  "incorrect_element_types": [],
  "repair_operations": []
}
```

只允许修复一次。

需要保存：

- 原始抽取输出；
- Pydantic校验错误；
- 核查输出；
- 修复后对象。

不能只保存最终结果，否则发生漏抽时无法知道是模型没有识别、Schema过窄还是修复逻辑删除了对象。

------

# 五、核心Agent设计：使用“完成约束下的自主工具循环”

## 1. 状态图

下一版建议采用：

```text
parse_case_and_plan
        ↓
verify_plan_inventory
        ↓
initialize_review_state
        ↓
      agent
   ↙    ↓     ↘
search open  submit/revise
   ↓    ↓       ↓
assess_evidence
   ↓
update_ledger
   └────────→ agent
                 ↓
          request_finalize
                 ↓
          completion_guard
          ↙              ↘
    gap_report             generate_answer
        ↓                        ↓
      agent               verify_answer
                                 ↓
                              finalize
```

这里不存在固定的：

```text
build_all_queries
→ execute_all_queries
```

Agent每一轮根据当前状态决定下一步动作。

## 2. Agent每轮能够看到什么

每轮模型输入不应包含全部45个原始片段，而应包含：

- 患者结构化信息；
- 显式方案要素；
- 派生审查目标摘要；
- 每个要素当前判断状态；
- 当前已有正向、负向、条件性和建议证据；
- 未满足的证据要求；
- 最近一次检索结果；
- 剩余查询、打开原文和token预算；
- 上一次终止请求被拒绝的具体原因。

## 3. Agent可调用的工具

### `search_evidence`

```python
search_evidence(
    query_text: str,
    target_element_ids: list[str],
    target_review_ids: list[str],
    evidence_role: Literal[
        "support",
        "challenge",
        "condition",
        "recommendation",
        "gap"
    ],
    search_reason: str
)
```

关键点：

- 查询文本由Agent自由生成；
- 一个查询可以服务多个方案要素；
- Agent决定证据角色；
- 不要求每个要素各执行一次查询；
- 工具内部继续使用现有Milvus Retriever；
- 主实验建议暂时使用当前表现较好的纯向量检索，避免把BM25或图谱变量再次混入方法贡献。

### `open_evidence_source`

```python
open_evidence_source(
    evidence_id: str,
    reason: str,
    window_before: int = 1,
    window_after: int = 1
)
```

适用于：

- 检索片段缺少表头；
- 剂量规则与脚注分离；
- 例外条件位于相邻段落；
- 只召回推荐结论，未召回适用条件；
- 需要确认原文上下文。

### `submit_element_judgement`

```python
submit_element_judgement(
    element_id: str,
    judgement: str,
    support_evidence_ids: list[str],
    challenge_evidence_ids: list[str],
    condition_evidence_ids: list[str],
    recommendation_evidence_ids: list[str],
    rationale: str,
    missing_information: list[str]
)
```

该工具必须执行程序校验：

- Evidence ID必须真实存在；
- Evidence必须已经被判断为与该要素相关；
- 不允许把“不适用患者”的证据作为直接依据；
- 判断类型和证据角色必须满足下述证据契约。

### `submit_cross_element_finding`

用于：

- 药物—药物相互作用；
- 药物—疾病冲突；
- 重复用药；
- 累积风险；
- 处方遗漏；
- 方案内部不一致。

### `revise_plan_inventory`

Agent可以在检索过程中发现原解析遗漏了方案对象，但修订受到严格限制：

- 新要素必须给出病例原文`source_span`；
- 不能加入从指南中推断出的治疗方案；
- 删除要素只能因为重复或抽取幻觉；
- 所有修订都需进入trace；
- 默认最多一次修订。

### `request_finalize`

Agent不能直接结束。它只能请求终止，由完成性门控检查。

------

# 六、检索后必须自动执行证据评估

当前“Retriever返回了dict”不能继续被定义为证据成功。

每次`search_evidence`或`open_evidence_source`之后，自动对候选证据进行批量结构化评估。该评估属于检索后的证据处理，不限制Agent的查询自主性。

```python
class EvidenceAssessment(BaseModel):
    evidence_id: str

    matched_element_ids: list[str]
    matched_review_ids: list[str]

    relevance: Literal[
        "direct",
        "partial",
        "background",
        "irrelevant"
    ]

    applicability: Literal[
        "applicable",
        "partially_applicable",
        "not_applicable",
        "uncertain"
    ]

    polarity: Literal[
        "supports_appropriate",
        "supports_inappropriate",
        "conditional",
        "supports_recommendation",
        "neutral"
    ]

    covered_dimensions: list[str]

    matched_patient_conditions: list[str]
    mismatched_patient_conditions: list[str]
    missing_patient_conditions: list[str]

    contains_numeric_threshold: bool
    contains_duration_rule: bool
    contains_exception: bool
    contains_monitoring: bool
    contains_recommendation: bool

    supporting_span: str
```

评估提示词必须要求：

```text
仅依据当前证据文本判断。
文本提到目标药物并不自动表示其适用于当前患者。
必须检查年龄、疾病、适应证、剂量、疗程、肾肝功能、
合并用药、治疗阶段、给药途径和例外条件。
无法确认时输出uncertain，不得补充外部知识。
```

近期药物—疾病信息抽取研究专门指出，仅识别关系存在无法满足临床使用需求，因为治疗关系经常依赖具体人群、疾病状态和其他适用条件。([ACL Anthology](https://aclanthology.org/people/yuji-matsumoto/))

这里可以先使用与Agent相同的模型，但采用独立Prompt和JSON Schema。后续消融再比较：

- Agent自行判断证据；
- 独立证据评估器判断证据。

CrossDDI采用“LLM抽取证据＋独立裁决”的验证优先设计，并要求药物相互作用判断绑定明确证据。这为证据与结论分离提供了直接的方法学依据。([ACL Anthology](https://aclanthology.org/2026.bionlp-1.73/))

------

# 七、证据账本及证据契约

## 1. 证据账本

```python
class ElementLedgerEntry(BaseModel):
    element_id: str

    judgement: Literal[
        "unresolved",
        "appropriate",
        "appropriate_with_monitoring",
        "needs_adjustment",
        "inappropriate",
        "insufficient_evidence"
    ]

    support_evidence_ids: list[str] = []
    challenge_evidence_ids: list[str] = []
    condition_evidence_ids: list[str] = []
    recommendation_evidence_ids: list[str] = []
    rejected_evidence_ids: list[str] = []

    reviewed_dimensions: list[str] = []
    missing_dimensions: list[str] = []

    quantitative_issue: str | None = None
    qualitative_issue: str | None = None
    clinical_risk: str | None = None
    recommended_action: str | None = None
    monitoring_requirement: str | None = None

    missing_patient_information: list[str] = []
    evidence_gap: str | None = None
```

同一条证据可以同时支持：

- HRZE联合方案；
- 四种组成药物的方案角色；
- 治疗意图。

它应当在Evidence Store中只保存一次，再以多对多关系绑定多个要素。这样可以避免为四种药物机械执行四次内容相同的正向查询。

## 2. 不同判断的最低证据契约

| 判断                          | 最低要求                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| `appropriate`                 | 至少一条患者适用的正向支持证据；已对关键风险维度进行针对性检查；不存在未解决的直接反对证据 |
| `appropriate_with_monitoring` | 正向支持证据＋风险或条件证据＋具体监测依据                   |
| `needs_adjustment`            | 直接问题证据＋患者条件适用性＋有来源的调整建议               |
| `inappropriate`               | 直接负向证据＋适用条件核验＋停用、替代或其他修正依据         |
| `insufficient_evidence`       | 明确说明缺失的患者变量、语料证据或冲突信息，不得生成确定性结论 |

“没有检索到负向证据”不能单独支持`appropriate`。

对“不合理”或“需调整”项目，如果知识库中没有具体替代方案，系统应输出：

> 已检索到原方案存在问题的依据，但当前知识库未提供足以支持具体替代方案的证据，建议由临床药师或相关专科医师进一步评估。

不能让模型凭参数知识生成具体替代药，再附上无关引用。

## 3. 不同要素类型的必要维度

| 要素类型      | 完成判断前应检查的维度                                       |
| ------------- | ------------------------------------------------------------ |
| 药物条目      | 适应证或方案角色、剂量频次、疗程、老年风险、器官功能、必要监测 |
| 联合方案      | 方案组成、治疗意图、治疗阶段、患者适用性、组合风险           |
| 治疗意图      | 当前疾病状态下的方向依据及其限制                             |
| 疗程          | 推荐疗程、当前疗程、适用条件、调整建议                       |
| 评价时点      | 推荐评估时间、评价内容、当前时点是否匹配                     |
| 监测计划      | 监测项目、频率、阈值和处理动作                               |
| 停药/换药规则 | 触发条件、替代动作和风险                                     |
| 随访计划      | 推荐复查时机和复查内容                                       |

这些是**证据要求**，不是固定检索模板。Agent可以用一个查询覆盖多个维度，也可以在发现缺口后再执行具体查询。

------

# 八、完成性门控：Agent可以自主结束，但不能不完整结束

`request_finalize`后运行：

```python
def evaluate_completion(state) -> GapReport:
    gaps = []

    for element in state.plan_elements:
        ledger = state.ledger[element.element_id]

        if ledger.judgement == "unresolved":
            gaps.append(...)

        if ledger.judgement == "appropriate":
            require_applicable_support_evidence(...)

        if ledger.judgement == "appropriate_with_monitoring":
            require_support_and_monitoring_evidence(...)

        if ledger.judgement in {"needs_adjustment", "inappropriate"}:
            require_problem_evidence(...)
            require_recommendation_evidence_or_explicit_gap(...)

        require_valid_source_mapping(...)
        require_element_appears_once(...)

    check_unresolved_cross_element_findings(...)
    check_invalid_evidence_ids(...)
    check_remaining_budget(...)

    return GapReport(gaps=gaps)
```

如果存在缺口且还有预算，将具体缺口返回Agent：

```json
{
  "finalization_approved": false,
  "gaps": [
    {
      "element_id": "PE05",
      "gap_type": "missing_support",
      "description": "尚无证据支持该联合方案及其治疗方向"
    },
    {
      "element_id": "PE06",
      "gap_type": "missing_timing_evidence",
      "description": "已有疾病治疗证据，但没有支持3个月复评时点的证据"
    },
    {
      "element_id": "PE03",
      "gap_type": "recommendation_gap",
      "description": "已判断需调整，但缺少调整方案的来源"
    }
  ]
}
```

Agent据此自主选择：

- 重新改写查询；
- 打开某一文档；
- 调查特定患者条件；
- 修改原判断；
- 将项目标为证据不足；
- 再次申请结束。

如果查询预算耗尽，系统不能继续循环。未解决项目统一转为`insufficient_evidence`，随后仍生成完整答案。

## 推荐停止条件

初始开发参数可以设为：

```text
最大向量检索子查询：8
最大打开原文次数：2
最大Agent决策步数：18
每次检索Top-K：5
技术重试：1次
方案修复：1次
答案修复：1次
最终活跃证据token：不超过12000
```

正式数值应在开发集上冻结。

另增加两个早停条件：

```text
连续两次语义检索没有满足任何新的证据要求
```

或：

```text
新增证据与已有证据的重复率超过80%，且没有新增患者条件或建议
```

技术错误和语义失败必须分开：

- timeout、embedding error、backend error：程序自动重试一次；
- success_empty、结果无关、患者条件不匹配：交给Agent生成新查询；
- 技术重试不计为新的Agent检索决策，但应单独计入资源消耗。

------

# 九、最终回答不再自由生成六个部分

建议先生成机器可验证JSON，再程序化渲染六段答案。

```python
class FinalReview(BaseModel):
    case_id: str

    plan_elements: list[TreatmentPlanElement]
    element_judgements: list[ElementJudgement]
    cross_element_findings: list[CrossElementFinding]

    positive_element_ids: list[str]
    negative_element_ids: list[str]

    integrated_recommendations: list[Recommendation]
    unresolved_items: list[str]

    used_evidence_ids: list[str]
```

六段输出的生成方式应为：

### ① 原方案要素清单

完全由`plan_elements`程序化生成。

### ② 逐项判断

按照`element_id`顺序，每个显式要素恰好输出一次。

跨要素问题另设一个小节，例如：

```text
跨要素或方案级问题
- CF01：药物A与药物B……
- CF02：某疾病可能存在处方遗漏……
```

### ③ 正面判断汇总

由判断标签程序化筛选，不让LLM重新总结并遗漏。

### ④ 负面判断汇总

同样程序化筛选。

### ⑤ 综合建议

允许LLM综合，但每项具体修改建议必须带有：

```text
evidence_ids
```

或：

```text
source_scope = "not_supported_by_retrieved_corpus"
```

### ⑥ 依据清单

由Evidence Store程序化映射：

```text
证据ID
文档名称
版本
章节或chunk
原文片段
支持的方案要素或发现
```

## 回答验证

生成后检查：

- 所有显式要素是否恰好出现一次；
- 是否出现病例中不存在的方案要素；
- 判断标签与正负汇总是否一致；
- 每项判断是否有合法Evidence ID；
- `appropriate`项目是否有正向支持；
- `needs_adjustment`和`inappropriate`是否有建议证据或明确证据缺口；
- 综合建议是否超出证据；
- 依据清单是否与实际Evidence Store一致。

只允许一次定向修复，而且只补写缺失字段，不重新自由生成整篇答案。

------

# 十、与Yuxi代码的具体整合

建议保留现有目录，新增模块：

```text
backend/package/yuxi/agents/buildin/medication_review/
├── context.py
├── models.py
├── prompt.py
├── extraction.py
├── plan_validation.py
├── review_targets.py
├── retrieval.py
├── evidence_assessment.py
├── ledger.py
├── tools.py
├── completion_guard.py
├── answer_generation.py
├── rendering.py
├── trace.py
└── graph.py
```

## `models.py`

新增：

- `TreatmentPlanElement`
- `ReviewTarget`
- `EvidenceAssessment`
- `ElementLedgerEntry`
- `ElementJudgement`
- `CrossElementFinding`
- `GapReport`
- `FinalReview`
- `MedicationReviewTraceV2`

旧证据、检索记录和错误状态模型可以扩展复用。

## `planning.py`

不再负责固定生成查询。

它只负责：

- grounding；
- 稳定ID；
- 方案要素基础结构；
- 当前七类关系转换为`ReviewTarget`；
- 默认审查维度；
- 预算预估。

建议重命名为：

```text
review_targets.py
```

或保留文件名但删除固定QueryBundle生成逻辑。

## `retrieval.py`

保留现有：

- 用户知识库权限；
- 直接Retriever调用；
- async Embedding；
- timeout；
- 后端错误区分；
- evidence去重；
- score、distance和occurrence保存。

改为接收Agent动态生成的查询：

```python
async def retrieve_for_agent(
    query_text: str,
    target_element_ids: list[str],
    evidence_role: str,
    ...
)
```

初始主实验保持：

```text
search_mode=vector
use_reranker=False
top_k=5
```

向量检索、BM25和LightRAG的差异已经在现有实验中获得初步结果，新方法应先固定检索后端，避免把方法增益与检索器变化混合。

## `graph.py`

当前：

```text
parse_case
→ build_review_plan
→ retrieve_bundles
→ finalize
```

替换为：

```text
parse_case_and_plan
→ verify_plan
→ initialize_state
→ agent
↔ tools
→ completion_guard
→ generate_answer
→ verify_answer
→ finalize
```

## `context.py`

新增可配置字段：

```python
max_search_calls: int = 8
max_open_calls: int = 2
max_agent_steps: int = 18

retrieval_top_k: int = 5
max_active_evidence_per_element: int = 4
max_final_evidence_tokens: int = 12000

technical_retry_limit: int = 1
plan_repair_limit: int = 1
answer_repair_limit: int = 1

use_applicability_assessor: bool = True
allow_gap_search: bool = True
enforce_completion_guard: bool = True
enable_answer_repair: bool = True
```

`system_prompt`应当真正参与Agent决策和最终回答，不再只参与病例抽取。

## `batch_yuxi_rag.py`

更新语义：

```text
answer = 最终六段式合理性回答
trace = MedicationReviewTraceV2
```

运行状态改为：

- `completed`：最终答案已生成并通过结构验证；
- `partial`：最终答案已生成，但存在预算耗尽或证据不足项目；
- `failed`：病例解析失败、后端致命错误或未生成最终答案。

## 前端

论文实验阶段不需要优先实现复杂trace页面。

最低要求是：

- `AIMessage.content`显示最终临床审查答案；
- 完整trace继续保存在metadata；
- 批量脚本能够导出trace。

------

# 十一、保持Agentic性质的具体执行示例

假设病例中有六个显式要素：

```text
PE01—PE04：四种药物条目
PE05：联合治疗方案和治疗方向
PE06：评价时点
```

系统初始化的是：

```text
需要判断的要素和证据要求
```

而不是49条检索命令。

Agent可能采用如下轨迹：

### 第一步

Agent认为PE05及PE01—PE04首先需要方案正向证据，调用：

```json
{
  "query_text": "目标疾病 老年患者 四药联合方案 组成 治疗阶段 推荐方案",
  "target_element_ids": ["PE01", "PE02", "PE03", "PE04", "PE05"],
  "evidence_role": "support",
  "search_reason": "核验联合方案方向及各药物在方案中的作用"
}
```

一条证据可以同时绑定五个要素。

### 第二步

证据评估发现：

- 支持PE05和四种药物的方案角色；
- 没有患者特异剂量、器官功能和监测信息。

Agent自主生成患者条件查询。

### 第三步

Agent发现PE06仍无时点依据，执行针对评价时机的查询。

### 第四步

某种药物已经获得风险证据，但缺少处理建议。完成性门控返回`recommendation_gap`，Agent再检索调整或监测方案。

### 第五步

Agent请求结束。门控检查六个要素、跨要素发现和证据契约，批准后生成答案。

不同病例可能：

- 三次检索完成；
- 八次检索后因证据不足终止；
- 打开一篇指南的完整上下文；
- 修改某个错误的初始判断。

系统没有固定检索路线。

------

# 十二、实验设计：先实现完整方法，再通过配置做消融

不要再构建一个“只完成关系图”的A1。第一次能够进入正式实验的方法版本就应是完整端到端系统。

建议实验组如下。

## 1. 主对照组

### B0：Yuxi ChatbotAgent baseline

- 当前自由ReAct工具循环；
- 同一Agent模型；
- 同一向量知识库；
- 同一Top-K；
- 同一检索和打开文档上限；
- 同一六段式输出要求；
- 最终答案由模型生成。

### B1：ChatbotAgent＋方案要素清单

在B0基础上，将系统抽取出的方案要素清单放入Prompt，但：

- 不使用证据账本；
- 不使用完成性门控；
- 不向模型返回结构化证据缺口；
- 仍由原生Agent自主检索和停止。

该组仍然是完整端到端Agent，不是半成品。它用于回答：

> 改善是否仅来自将病例结构化为方案要素？

### M-Full：完整新方法

包含：

- 完整方案要素抽取；
- 派生审查目标；
- 自主检索工具循环；
- 证据适用性判断；
- 证据账本；
- 证据契约；
- 缺口反馈；
- 完成性门控；
- 结构化回答；
- 完整性验证。

### D0：当前确定性关系检索原型

只用于诊断：

- 查询数量；
- 召回片段数量；
- 关系覆盖；
- 运行成本。

不进入端到端答案性能主表。

## 2. 消融实验

先实现完整`M-Full`，再通过配置开关消融，避免维护多个功能残缺分支。

| 消融                 | 改动                         | 仍保留的能力                         |
| -------------------- | ---------------------------- | ------------------------------------ |
| `M-soft-guard`       | 终止门控只提示，不阻止结束   | 仍生成最终答案和trace                |
| `M-no-gap-search`    | 首次请求结束后不允许语义补检 | 未满足项标为证据不足，仍生成完整答案 |
| `M-no-applicability` | 不执行独立患者适用性评估     | Agent仍自行判断并生成答案            |
| `M-no-answer-repair` | 不执行最终定向修复           | 仍输出结构化答案                     |
| `M-no-plan-revision` | Agent不能修订初始要素清单    | 其他流程完整                         |

这些配置都能产生最终答案，不需要重新制造检索-only半成品。

------

# 十三、实验公平性

## 1. 固定主要变量

B0、B1和M-Full应统一：

- Agent模型；
- 知识库；
- Embedding模型；
- 向量检索模式；
- 单次Top-K；
- 最大向量子查询数；
- 最大打开文档次数；
- 最终证据token上限；
- 最终答案格式；
- 每题独立thread；
- 运行日期和模型版本。

## 2. 预算以“子查询数”而非“Agent轮次”匹配

M-Full可能一次Agent决策生成一个服务多个要素的查询；baseline可能多次调用`query_kb`。

应分别记录：

```text
Agent模型决策轮次
知识库子查询数
打开原文次数
检索片段数
唯一证据数
进入最终答案的证据数
```

主预算应匹配：

```text
知识库子查询数
＋打开原文次数
＋最终证据token
```

不能仅匹配Agent消息轮数。

## 3. 两种比较方式

### 自然运行比较

按各自默认行为运行，反映系统实际使用效果。

### 预算匹配比较

两组使用相同：

```text
最大8次知识库子查询
最大2次打开原文
单次Top-5
最终证据不超过12000 tokens
```

具体上限可以根据开发集中baseline查询次数的第90百分位确定，然后冻结。

------

# 十四、正式实验前必须做的三个诊断上限实验

## 1. Oracle-Plan

向M-Full提供金标准中的原方案要素清单，但不给判断和证据。

它回答：

> 如果方案要素抽取完全正确，后续Agent检索和判断能达到什么水平？

若Oracle-Plan显著高于正常M-Full，主要瓶颈在方案抽取。

## 2. Oracle-Evidence

向相同回答生成模块直接提供金标准所依据的完整证据。

它回答：

> 在检索完全充分时，答案生成和完整性验证的上限是多少？

若仍出现“5点只回答4点”，问题在证据账本、输出契约或Judge。

## 3. Oracle-Ledger

向渲染模块提供金标准结构化判断和Evidence ID。

它回答：

> 六段式渲染、引用映射和评价脚本是否正确？

如果该组仍无法获得接近满分的结果，优先修复评价器或答案格式，不应继续调检索。

------

# 十五、数据集和评价指标

## 1. 方案要素抽取

设金标准方案要素集合为(G_P)，系统抽取集合为(S_P)。

# [ \text{Plan Precision}

\frac{|S_P\cap G_P|}{|S_P|}
]

# [ \text{Plan Recall}

\frac{|S_P\cap G_P|}{|G_P|}
]

同时报告：

- 类型准确率；
- source span grounding率；
- 全要素完整病例率；
- 虚构要素率；
- 重复要素率。

需要区分：

```text
Gold Element Coverage：
相对于金标准的真实覆盖

Internal Element Coverage：
相对于系统自己抽取结果的输出覆盖
```

系统可能对自己漏抽后的要素清单实现100%内部完整性，因此二者不能混为一谈。

## 2. 证据层

### 证据角色覆盖率

# [ \text{Evidence Role Coverage}

\frac{
\text{已满足的必需证据角色数}
}{
\text{全部必需证据角色数}
}
]

角色包括：

- support；
- challenge；
- condition；
- recommendation。

### 患者适用证据精确率

# [ \text{Applicability Precision@K}

\frac{
\text{前K条中适用于当前患者的证据数}
}{
K
}
]

### 每次检索新增收益

# [ \text{Search Yield}

\frac{
\text{本次查询新满足的证据要求数}
}{
\text{知识库子查询数}
}
]

### 缺口修复率

# [ \text{Gap Closure Rate}

\frac{
\text{后续Agent检索成功关闭的缺口数}
}{
\text{完成门控发现的全部缺口数}
}
]

### 提前结束率

# [ \text{Premature Finalization Rate}

\frac{
\text{被完成门控拒绝的终止请求数}
}{
\text{全部终止请求数}
}
]

该指标可以直接评价baseline常见的过早停止问题是否得到改善。

## 3. 最终判断

对四种主要判断计算：

- Macro-F1；
- Micro-F1；
- 各标签Precision、Recall、F1；
- 完整病例准确率；
- 证据不足率。

系统额外标签`insufficient_evidence`可以视为弃答。需要同时报告：

- 总体任务F1，其中弃答按未正确判断处理；
- 非弃答样本准确率；
- 系统覆盖率。

## 4. 回答完整性

# [ \text{Element Judgement Coverage}

\frac{
\text{恰好评价一次的金标准方案要素数}
}{
\text{全部金标准方案要素数}
}
]

# [ \text{Attribute Completeness}

\frac{
\text{正确覆盖的必需属性数}
}{
\text{全部金标准必需属性数}
}
]

# [ \text{Full Element Rate}

\frac{
\text{判断和全部必需属性均正确的要素数}
}{
\text{全部金标准要素数}
}
]

另报告：

- 正向依据覆盖率；
- 调整建议有据率；
- 无依据临床论断率；
- 引用蕴含率；
- 引用患者适用率；
- 六段式结构通过率。

## 5. 资源指标

报告：

- Agent决策轮次；
- 知识库子查询数；
- 打开原文次数；
- 候选片段数；
- 唯一证据数；
- 最终使用证据数；
- Agent模型调用数；
- 证据评估模型调用数；
- 输入输出token；
- 延迟；
- API成本；
- 技术失败和重试次数。

------

# 十六、推荐的实施与验收顺序

## 阶段0：冻结当前原型

将当前版本标记为：

```text
deterministic-relation-retrieval-prototype
```

保留：

- 抽取fallback；
- grounding；
- 稳定ID；
- Milvus直接Retriever；
- 错误状态；
- evidence ID；
- trace；
- 批处理。

停止在该分支继续添加最终回答。

## 阶段1：建立12例方案要素验收集

至少覆盖：

- 单药方案；
- 多药联合方案；
- 诊断性或经验性治疗方向；
- 方案级疗程；
- 评价时点；
- 监测计划；
- 随访；
- 停药或换药条件；
- 药物—疾病问题；
- 药物—药物问题；
- 合理项目；
- 不合理项目。

HRZE示例必须成为固定回归测试：

```text
4个药物要素
＋1个联合方案/治疗方向要素
＋1个评价时点要素
```

验收标准：

- 所有显式要素有稳定ID；
- source span grounding 100%；
- 不得将联合方案和评价时点塞入某个药物属性；
- 不得生成病例未明确提出的方案要素。

## 阶段2：完成Agent自主工具循环

通过人工构造的工具结果测试：

1. 首轮检索无关；
2. Agent观察到无关结果；
3. Agent改写查询；
4. 第二轮获得相关证据；
5. Agent提交判断；
6. 结束请求因缺少建议证据被拒绝；
7. Agent补查建议；
8. 最终完成。

这一测试必须通过后，才能称为Agentic retrieval。

## 阶段3：完成证据评估和账本

验收：

- 非空结果不再自动计为充分；
- 患者条件不匹配证据进入`rejected_evidence`；
- 同一证据能够绑定多个要素；
- 正向、负向和建议证据可以区分；
- 技术失败和语义失败有不同路径。

## 阶段4：完成最终回答和验证

验收：

- 网页显示临床审查答案；
- 批处理`answer`为同一答案；
- 六段结构齐全；
- 每个显式要素恰好一次；
- 无效Evidence ID为零；
- 正负汇总和逐项判断一致；
- 缺少建议来源时明确标识；
- `completed`只在答案验证后设置。

## 阶段5：开发集实验

建议使用此前确定的60例开发集：

- Beers相关24例；
- 单病种指南24例；
- 复杂病例12例。

开发集只用于：

- Schema和Prompt；
- Top-K；
- 查询和打开文档预算；
- 完成门控规则；
- 证据选择上限；
- 一次修复规则。

## 阶段6：锁定测试

在剩余数据上一次性运行：

- B0；
- B1；
- M-Full；
- 必要的配置消融。

全部40个复杂病例应单独报告；复杂病例可重复运行3次评价稳定性。

------

# 十七、建议保留的最低实验矩阵

| 组别                  | 完整方案要素 | 自主查询 | 证据账本 | 缺口补检 | 患者适用性 | 最终六段答案 |
| --------------------- | ------------ | -------- | -------- | -------- | ---------- | ------------ |
| B0 Yuxi baseline      | 否           | 是       | 否       | 自由式   | 隐式       | 是           |
| B1 baseline＋要素清单 | 是           | 是       | 否       | 自由式   | 隐式       | 是           |
| M-Full                | 是           | 是       | 是       | 是       | 显式       | 是           |
| M-no-gap              | 是           | 是       | 是       | 否       | 显式       | 是           |
| M-no-applicability    | 是           | 是       | 是       | 是       | 隐式       | 是           |
| M-soft-guard          | 是           | 是       | 是       | 是       | 显式       | 是           |
| D0当前原型            | 部分         | 否       | 否       | 否       | 否         | 否           |

D0只进入流程和成本分析，不与其他组比较最终答案F1。

------

# 十八、需要立即避免的实现方式

下面几种方案不应继续采用：

1. 不要把所有药物×疾病、所有药物对机械转成检索查询。
2. 不要把合理项的证据需求等价为“没有召回负面内容”。
3. 不要把Retriever非空结果等价为成功证据。
4. 不要把所有唯一片段直接放入最终模型上下文。
5. 不要在没有证据账本的情况下直接让模型自由总结45个片段。
6. 不要让Agent通过删除或合并方案要素绕过完成性门控。
7. 不要把技术超时和语义检索不足统一处理。
8. 不要以检索统计报告作为`answer`。
9. 不要把当前固定关系检索原型命名为A1或Agentic retrieval。
10. 不要在新方法中同时更换向量模型、增加BM25、启用图谱和改变Agent流程，否则无法归因。

------

# 十九、最小但完整的新方法版本

真正值得实现和运行的第一个版本，应至少包含：

```text
完整治疗方案要素抽取
＋方案要素完整性核查
＋当前七类关系作为候选审查目标
＋Agent自主生成查询和选择检索顺序
＋Agent自主打开原文
＋证据相关性、适用性和极性判断
＋治疗方案要素级证据账本
＋基于证据契约的终止门控
＋不充分时的Agent自主补检
＋结构化最终判断
＋六段式程序化渲染
＋逐要素完整性和引用验证
```

这一版本已经不是简单“建立关系覆盖图”，其方法核心可以概括为：

> 方案要素决定必须评价什么；
> 证据契约决定判断需要什么依据；
> Agent自主决定如何检索这些依据；
> 完成性门控决定何时能够结束；
> 程序化答案契约保证所有要素进入最终结果。

这既保留了Yuxi baseline最重要的自主查询、多轮检索和动态停止能力，也修复了自由ReAct在老年多重用药场景中容易漏项、过早终止、证据与方案要素难以对齐的问题。当前向量检索表现优于通用图谱检索的负结果也可以自然纳入论文：新方法的增益来自临床任务表示、证据适用性和Agent检索控制，而不依赖更复杂的检索后端。