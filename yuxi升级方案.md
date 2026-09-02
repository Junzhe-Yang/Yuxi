# 一、建议采用的最终技术路线

结合你目前已经完成的系统、数据集和实验，最适合继续推进的方案是：

> **在Yuxi 0.6.3基础上实现“处方关系覆盖引导的证据充分性Agentic RAG”**，保留现有Milvus向量知识库，将通用ReAct工具循环改造成一个有界、可观测的临床审查状态图。

推荐的完整流程为：

```text
病例结构化
→ 生成处方审查关系槽位
→ 将关系槽位组织为检索任务束
→ 批量候选召回
→ 患者适用条件重排
→ 覆盖与多样性感知的证据选择
→ 建立证据账本
→ 仅针对证据缺口执行第二轮检索
→ 逐关系生成结构化结论
→ 完整性与证据一致性回查
```

这一路线对你现有工作的改动是可控的：

* 不重建完整知识图谱；
* 不更换Yuxi的数据库、会话接口和模型管理；
* 不需要训练新的生成模型；
* 可以继续使用当前表现最好的向量检索；
* 主要增加一个自定义Agent状态图、一组结构化数据模型和一个批量检索服务；
* 每个新增组件都能够独立消融和评价。

方法贡献也会从“把RAG接入Agent”变为：

1. 将老年多重用药审查形式化为患者特异的关系覆盖任务；
2. 通过正向与风险向证据检索、患者条件匹配和证据缺口反馈，控制复杂处方的查全与查准；
3. 建立从临床关系规划、证据召回、证据适用性到答案完整性的分层评价体系。

Beers标准本身就包含一般PIM、药物—疾病、慎用药物、药物—药物以及肾功能剂量调整等不同类别；STOPP/START还同时覆盖不当用药和潜在处方遗漏。因此，把老年多重用药审查表示为多个关系类型，比将整个病例压缩成一个开放式问答查询更符合任务结构。([PMC][1])

---

# 二、不要直接修改现有ChatbotAgent，先建立独立实验分支

根据你提供的源码级分析，Yuxi 0.6.3的`ChatbotAgent`使用LangChain/LangGraph的`create_agent`构造通用工具循环，模型自行决定是否检索、如何改写查询、是否继续检索以及何时停止。`query_kb`只接收知识库名称、检索文本和可选文件名，也不能在单次调用中动态暴露完整的top-k、混合权重、重排和多查询参数。

因此，建议保留现有ChatbotAgent作为论文基线，另外建立一个自定义Agent：

```text
现有 ChatbotAgent
    └── 保持不动，用作 B0/B1/B2 基线

新增 MedicationReviewAgent
    └── 实现关系规划、批量检索、证据账本和完整性检查
```

建议的目录结构是：

```text
backend/package/yuxi/agents/buildin/medication_review/
├── __init__.py
├── graph.py
├── state.py
├── schemas.py
├── prompts.py
├── retrieval.py
├── reranking.py
├── adjudication.py
└── logging.py
```

如果暂时不想注册新的Yuxi Agent，可以先在容器内建立一个独立实验脚本：

```text
scripts/medication_review_experiment/
├── run_pipeline.py
├── yuxi_retriever.py
├── schemas.py
├── prompts.py
├── evaluate.py
└── configs/
```

先用60例开发集证明方法有效，再封装成Yuxi自定义Agent。这样能够避免在框架集成阶段花费大量时间，却最后发现某个方法组件没有增益。

---

# 三、第一步：把病例和标准答案转换为稳定的结构化数据

## 1. 病例输入结构

建议为每个病例建立下面的Pydantic模型。临床字段缺失时必须明确保存为`None`或`unknown`，不能让模型自行补充。

```python
from typing import Literal
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    diagnosis_id: str
    name: str
    aliases: list[str] = []
    status: Literal["active", "history", "suspected", "unknown"] = "active"


class LabValue(BaseModel):
    name: str
    value: float | None = None
    unit: str | None = None
    reference_time: str | None = None


class Medication(BaseModel):
    medication_id: str
    generic_name: str
    aliases: list[str] = []
    drug_class: str | None = None

    dose_value: float | None = None
    dose_unit: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration_days: float | None = None

    indication: str | None = None
    prn: bool | None = None


class PatientCase(BaseModel):
    case_id: str
    age: int | None = None
    sex: str | None = None

    diagnoses: list[Diagnosis]
    medications: list[Medication]

    renal_function: list[LabValue] = []
    hepatic_function: list[LabValue] = []
    other_labs: list[LabValue] = []

    clinical_risks: list[str] = []
    allergies: list[str] = []
    care_setting: str | None = None
    goals_of_care: str | None = None

    missing_information: list[str] = []
```

推荐处理方式：

1. 优先从你现有数据字段中确定性读取；
2. 只有原始问题是自由文本时，才调用LLM结构化；
3. 对药物名称建立一个项目内的通用名—商品名—缩写映射表；
4. 对剂量单位、频次、疗程和肾功能指标做程序化归一；
5. 结构化结果中出现病例原文不存在的药物或诊断时，自动拒绝并重试。

## 2. 将标准答案转换为固定金标准关系项

你现在已经完善了合理和不合理两方面的标准答案，这使得评价可以从“错误要点召回”升级为关系级分类。

建议每个金标准项目采用：

```python
class GoldAttribute(BaseModel):
    name: Literal[
        "indication",
        "quantitative_issue",
        "qualitative_issue",
        "patient_condition",
        "clinical_risk",
        "recommended_action",
        "monitoring",
        "exception"
    ]
    value: str
    required: bool = True


class GoldFinding(BaseModel):
    finding_id: str
    case_id: str

    slot_type: Literal[
        "medication_profile",
        "drug_disease",
        "drug_drug",
        "organ_function",
        "duplication",
        "cumulative_burden",
        "prescribing_omission"
    ]

    subject_ids: list[str]
    target_ids: list[str]

    label: Literal[
        "appropriate",
        "inappropriate",
        "conditional"
    ]

    core_finding: str
    attributes: list[GoldAttribute]

    source_documents: list[str] = []
    source_spans: list[str] = []
```

这里最重要的是固定`finding_id`。后续LLM-as-judge只判断系统是否覆盖该`finding_id`及其属性，不再临时重新分解参考答案。

一个标准答案中的内容可以表示为：

```json
{
  "finding_id": "C012_F03",
  "slot_type": "medication_profile",
  "subject_ids": ["M02"],
  "target_ids": [],
  "label": "inappropriate",
  "core_finding": "该治疗方案的疗程设置不合理",
  "attributes": [
    {
      "name": "quantitative_issue",
      "value": "当前疗程超过指南推荐上限",
      "required": true
    },
    {
      "name": "qualitative_issue",
      "value": "该治疗不适合长期持续使用",
      "required": true
    },
    {
      "name": "recommended_action",
      "value": "重新评估继续治疗的必要性并调整疗程",
      "required": true
    }
  ]
}
```

这样就能区分：

* 系统完全没有发现该问题；
* 系统发现疗程问题，但遗漏数值异常；
* 系统发现数值异常，但遗漏长期使用的定性问题；
* 系统发现问题，但没有给出建议；
* 系统完整覆盖。

---

# 四、第二步：建立患者特异的处方关系覆盖图

这里的“图”不是语料知识图谱，而是每个病例运行时生成的**审查任务图**。

## 1. 建议采用分层槽位，而不是机械枚举所有关系

对每种药物建立一个`medication_profile`槽位，统一检查：

* 是否具有明确适应证；
* 是否属于一般老年PIM；
* 当前剂量和频次是否合理；
* 当前疗程是否合理；
* 肾功能和肝功能是否需要调整；
* 是否需要特定监测；
* 是否存在给药途径或剂型问题。

药物之间和药物—疾病之间再单独建立关系槽位：

```python
class ReviewSlot(BaseModel):
    slot_id: str

    slot_type: Literal[
        "medication_profile",
        "drug_disease",
        "drug_drug",
        "organ_function",
        "duplication",
        "cumulative_burden",
        "prescribing_omission"
    ]

    subject_ids: list[str]
    target_ids: list[str]

    patient_constraints: dict
    required_attributes: list[str]

    priority: Literal["high", "medium", "low"]
    generation_reason: str

    status: Literal[
        "unsearched",
        "evidence_found",
        "insufficient",
        "conflicting",
        "completed"
    ] = "unsearched"
```

## 2. 强制生成的槽位

每个病例都必须生成：

1. 每种药物一个`medication_profile`；
2. 每种药物与患者肾功能、肝功能的适配检查；
3. 每种疾病一个`prescribing_omission`检查；
4. 一个重复用药检查；
5. 一个病例级累积风险检查。

这样可以保证至少不会漏掉某种药物。

## 3. 候选生成的槽位

药物—疾病和药物—药物关系数量可能迅速增加。建议使用以下规则。

当药物数不超过8种时：

[
N_{\mathrm{DDI}}=\frac{n(n-1)}{2}
]

可以全部建立候选药物对，但不要求每个药物对都执行独立检索。

当药物数超过8种时，先由一个高召回候选筛选器筛出可能需要重点核验的药物对。筛选器一次读取全部药物，只返回病例中实际存在的药物ID，禁止生成新药物。

药物—疾病关系也采用类似策略：

* 当药物数×疾病数不超过30时，全部建立候选槽位；
* 超过30时，先筛选可能存在治疗关系或冲突的组合；
* 所有被排除的组合仍应记录为`screened_out`，便于计算规划层查全率。

## 4. 将审查槽位和实际检索查询分开

不要让一个槽位等于一次工具调用。建议增加`QueryBundle`：

```python
class QueryBundle(BaseModel):
    bundle_id: str
    slot_ids: list[str]

    query_role: Literal[
        "support",
        "challenge",
        "condition",
        "recommendation",
        "gap"
    ]

    retrieval_view: Literal[
        "lexical",
        "semantic"
    ]

    query_text: str
    expected_evidence_types: list[str]
```

例如，同一种药物的适应证、PIM、疾病冲突和监测要求可以由两至四个查询束覆盖，而不必为每个属性发起一次Agent工具调用。

---

# 五、第三步：实现双向、分类型查询束

## 1. 为什么要检索正反两方面证据

对于完整的用药合理性评估：

* 发现负向规则可以支持“不适当”；
* 找不到负向规则不能自动证明“合理”；
* 判断“合理”需要适应证、推荐方案、剂量或疗程等正向支持；
* 正向和负向证据同时存在时，需要判断患者条件和例外。

因此每个关键槽位至少应包含：

* 支持合理性的查询；
* 挑战合理性的查询。

## 2. 查询模板

### 药物整体审查

词法查询：

```text
{药物通用名} {药物类别} 老年人
适应证 推荐 禁忌 慎用 避免
剂量 频次 疗程 肾功能 监测
```

语义查询：

```text
对于{年龄}岁、患有{主要疾病}的患者，
目前以{剂量、频次、疗程}使用{药物}是否符合指南？
需要检索其适应证、老年人使用限制、剂量疗程和监测要求。
```

### 药物—疾病

支持向：

```text
{疾病} {药物} 老年患者 推荐治疗 适应证 用药方案
```

风险向：

```text
{药物} {疾病} 老年人 禁忌 慎用 加重 避免 风险
```

### 药物—药物

```text
{药物A} {药物B} 老年人
药物相互作用 合并使用 避免 剂量调整 监测 风险
```

### 肾功能或剂量

```text
{药物} eGFR {患者数值} CrCl
{当前剂量} {当前频次}
肾功能不全 剂量调整 禁用 监测
```

### 疗程

```text
{药物} {适应证}
当前疗程 {疗程天数}
推荐疗程 长期使用 避免 停药 减量
```

### 处方遗漏

```text
{疾病} 老年患者 推荐药物 一线治疗
必要治疗 预防治疗 治疗不足
```

## 3. 查询生成约束

每个查询必须满足：

* 至少包含目标药物通用名；
* 药物—疾病槽位必须包含目标疾病；
* 药物—药物槽位必须包含两种药物；
* 数值问题必须保留数值和单位；
* 肾功能查询必须保留指标名称，不能将eGFR和CrCl混用；
* 不允许删除否定、例外、疗程和给药途径；
* 单条查询最多包含6个核心实体，防止查询过长。

可以由程序生成词法查询，由LLM生成语义查询。这样能够避免全部查询都受到LLM改写随机性的影响。

---

# 六、第四步：绕过通用`query_kb`，实现一个批量检索服务

现有`query_kb`适合普通对话，却不适合实验级多查询控制。建议在新增Agent中直接调用知识库管理层或具体Retriever，而不是让模型逐次调用工具。

## 1. 批量检索接口

```python
class RetrievalConfig(BaseModel):
    retrieval_mode: Literal["dense", "hybrid"] = "dense"

    candidate_top_k: int = 20
    rerank_top_k: int = 8
    final_per_slot: int = 3

    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    max_retrieval_rounds: int = 2
    max_gap_queries: int = 8
    max_context_tokens: int = 12000


class RetrievalRequest(BaseModel):
    case_id: str
    patient_case: PatientCase
    slots: list[ReviewSlot]
    query_bundles: list[QueryBundle]
    config: RetrievalConfig
```

返回结果应保留：

```python
class RetrievedEvidence(BaseModel):
    evidence_id: str

    slot_ids: list[str]
    bundle_id: str

    file_id: str | None
    chunk_id: str | None
    chunk_index: int | None
    source_document: str
    source_version: str | None
    section: str | None

    raw_text: str

    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float | None

    rerank_score: float | None
    applicability_score: float | None

    retrieval_status: Literal[
        "success",
        "success_empty",
        "backend_error",
        "embedding_error",
        "reranker_error",
        "timeout"
    ]
```

这能解决Yuxi目前将部分后端异常转换为空列表或空字符串的问题。工具调用“成功”与检索后端成功必须分开记录。

## 2. 候选池大小

开发阶段建议：

```text
每个查询束初始召回：top 20
合并去重后每个槽位：最多30条
通用重排后：保留8条
患者条件判断后：保留2—3条
```

不要直接把30条全部送给生成模型。

## 3. 排名融合

如果使用向量和BM25，建议先用RRF而不是继续微调固定权重：

[
\operatorname{RRF}(d)
=====================

\sum_{q}
\frac{1}{60+\operatorname{rank}_{q}(d)}
]

这里的(q)不仅可以表示向量和BM25，也可以表示同一槽位的支持向、风险向和语义查询。

你当前结果显示BM25对最终答案影响有限，因此BM25应该作为可消融模块，而不应成为新方法的核心。BGE-M3本身支持稠密、稀疏和多向量检索；如果后续仍需增强细粒度匹配，多向量检索比继续调整单一BM25权重更值得研究，但它应放在第二阶段。([ACL Anthology][2])

---

# 七、第五步：患者适用条件感知的两阶段重排

普通reranker只能判断“文本是否与问题相关”，无法充分判断“这条规则是否适用于当前患者”。

适用条件应至少包括：

* 年龄；
* 疾病和适应证；
* 肾功能和肝功能；
* 剂量、频次和疗程；
* 给药途径；
* 合并用药；
* 例外条件；
* 照护阶段。

2026年的研究已将“药物—疾病关系的适用条件抽取”作为独立任务，原因正是单纯抽取药物—疾病关系无法表达关系成立的患者条件。([ACL Anthology][3])

## 1. 第一阶段：程序化硬匹配

对每条候选证据计算：

```text
药物名称是否匹配
药物类别是否匹配
疾病是否匹配
另一药物是否匹配
数值和单位是否出现
给药途径是否匹配
是否存在明显人口学不匹配
是否存在否定或例外表达
```

建议定义：

```python
class HardMatchResult(BaseModel):
    entity_match: bool
    target_match: bool
    numeric_match: bool | None
    route_match: bool | None

    hard_mismatch: bool
    mismatch_reasons: list[str]
```

明显属于另一药物、另一疾病、另一给药途径或完全不同患者群体的证据应进入`rejected_evidence`，而不是直接删除。这样可以在错误分析中看到为什么被排除。

## 2. 第二阶段：LLM适用性判断

仅对每个槽位排名前8的候选进行判断。提示词要求模型输出结构化结果：

```python
class ApplicabilityJudgement(BaseModel):
    relevance: int = Field(ge=0, le=3)
    applicability: int = Field(ge=0, le=3)

    polarity: Literal[
        "supports_appropriate",
        "supports_inappropriate",
        "conditional",
        "neutral"
    ]

    matched_conditions: list[str]
    mismatched_conditions: list[str]
    missing_conditions: list[str]

    contains_threshold: bool
    contains_exception: bool
    contains_recommendation: bool

    evidence_span: str
```

提示词中必须规定：

```text
只能依据给定文本判断。
没有明确说明时不得推断患者适用性。
必须复制能够支持判断的最小证据片段。
若药物、疾病、年龄、剂量、疗程或器官功能条件不一致，
应在mismatched_conditions中明确指出。
```

## 3. 综合得分

开发阶段可以使用：

[
S(e,z)
======

0.30S_{\text{retrieval}}
+
0.20S_{\text{entity}}
+
0.20S_{\text{relation}}
+
0.25S_{\text{applicability}}
+
0.05S_{\text{source}}
---------------------

0.30I_{\text{hard mismatch}}
]

这里的权重不是最终理论参数，只是开发初始值。应在60例开发集上确定，然后冻结。

---

# 八、第六步：覆盖和多样性感知的证据选择

目前常见问题是top-8结果中有多条重复描述同一风险，却没有覆盖剂量、疗程或建议。

DF-RAG的研究表明，单纯最大化相似度容易产生冗余上下文，而同时考虑相关性和多样性可以改善复杂推理任务的信息覆盖。([ACL Anthology][4])

在你的任务中，可以使用更简单、临床可解释的选择规则。

## 1. 每个槽位至少保留不同极性的证据

优先选择：

1. 最强的合理性支持证据；
2. 最强的不合理或风险证据；
3. 能够补充缺失阈值、例外或建议的证据。

每个槽位最多保留3条。

## 2. 定义证据新增价值

[
Gain(e,z)
=========

\text{新增必需属性数}
+
\text{新增证据极性}
+
\text{新增来源}
-----------

\lambda\cdot \text{与已选证据的重复度}
]

贪心选择：

```python
while token_budget_not_exceeded:
    choose evidence with maximum Gain
    update covered_attributes
    update covered_polarities
    update selected_sources
```

## 3. 临床槽位的必需属性

| 槽位    | 必需属性                |
| ----- | ------------------- |
| 药物整体  | 适应证、患者条件、剂量/疗程、建议   |
| 药物—疾病 | 关系、风险、适用条件、处理建议     |
| 药物—药物 | 两种药物、相互作用、后果、处理建议   |
| 肾功能   | 指标、阈值、当前患者值、剂量或禁用建议 |
| 疗程    | 推荐疗程、当前疗程、比较结果、建议   |
| 重复用药  | 药物类别、重复关系、累积风险、建议   |
| 处方遗漏  | 疾病、推荐治疗、患者适用条件、例外   |

---

# 九、第七步：建立证据账本

证据账本是新方法的核心状态。它既控制后续检索，又控制最终答案生成。

```python
class EvidenceLedgerEntry(BaseModel):
    slot_id: str

    positive_evidence_ids: list[str] = []
    negative_evidence_ids: list[str] = []
    conditional_evidence_ids: list[str] = []
    rejected_evidence_ids: list[str] = []

    covered_attributes: list[str] = []
    missing_attributes: list[str] = []

    decision: Literal[
        "appropriate_supported",
        "inappropriate",
        "conditional",
        "conflicting",
        "insufficient"
    ]

    core_finding: str | None = None
    quantitative_issue: str | None = None
    qualitative_issue: str | None = None
    clinical_risk: str | None = None
    recommended_action: str | None = None
    monitoring: str | None = None

    missing_patient_information: list[str] = []

    coverage_score: float
    decision_confidence: float
```

## 1. 建议采用保守判定规则

```text
存在适用于患者的负向证据，且没有明确例外
    → inappropriate

存在明确正向证据，且相关风险维度已完成核验
    → appropriate_supported

同时存在相互冲突的适用证据
    → conflicting

结论依赖于病例中缺失的患者变量
    → insufficient

只有“没有检索到风险”，但没有正向证据
    → insufficient或no_inappropriateness_identified
```

为了与现有二元金标准比较，可以在评价时映射：

```text
appropriate_supported → appropriate
inappropriate → inappropriate
conditional → 按预设规则映射
conflicting/insufficient → abstain
```

同时报告：

* 非弃答样本准确率；
* 系统覆盖率；
* 弃答率；
* 错误自信输出率。

CrossDDI采用“先抽取证据、再进行独立裁决”的验证优先架构，并要求正向药物相互作用判断与显式证据绑定。这个思想适合你的任务：LLM负责解释和抽取，状态机负责限制无证据结论。([ACL Anthology][5])

---

# 十、第八步：缺口驱动的第二轮检索

SEMA-RAG将医学问题解释、证据探索和证据充分性判断拆开，其核心价值是根据证据缺口继续检索，而不是让模型自由重复搜索。([ACL Anthology][6])

你的实现无需使用三个Agent，只需在单个状态图中加入`assess_gaps`节点。

## 1. 覆盖分数

设槽位所需属性集合为(A_z)，已有证据支持的属性为(\hat A_z)：

[
Coverage(z)
===========

\frac{
\sum_{a\in \hat A_z}w_a
}{
\sum_{a\in A_z}w_a
}
]

建议权重：

```text
关系本身：2
患者适用条件：2
数值阈值：2
临床风险：1
处理建议：1
监测：1
```

## 2. 触发第二轮检索的条件

满足任一条件：

* 高优先级槽位的`coverage_score < 0.80`；
* 一般槽位的`coverage_score < 0.67`；
* 已找到风险，但缺少处理建议；
* 已找到数值异常，但缺少定性适用条件；
* 正负证据冲突；
* 证据涉及患者未提供的关键变量；
* 某一金标准关系类型在第一轮完全没有候选。

## 3. 缺口查询不再重复原查询

例如第一轮已找到：

```text
该药物长期使用可能不适当
```

但缺少推荐疗程和调整建议，第二轮查询应当是：

```text
{药物} {适应证} 推荐疗程 停药 减量 替代 监测
```

而不是重复：

```text
{药物} 老年人 不适当
```

## 4. 严格限制轮次

建议：

```text
最大检索轮次：2
复杂病例可设为3，但第三轮只处理高风险冲突
第二轮最大新增查询数：8
单槽位最多新增1个缺口查询
```

停止条件：

[
\Delta Coverage < 0.05
]

或：

[
\text{新增证据中重复比例}>0.80
]

或达到token和查询预算。

这样，图谱组“检索轮次更多但没有新增事实”的问题就能被明确量化。

---

# 十一、第九步：从证据账本生成答案，并做完整性回查

最终答案不应再次读取全部原始检索文本并自由总结。生成模型只读取：

* 结构化患者信息；
* 审查槽位；
* 证据账本；
* 被选中的最小证据片段。

## 1. 先生成机器可评估JSON

```python
class FinalFinding(BaseModel):
    finding_id: str
    slot_id: str

    subject: str
    target: str | None

    judgement: Literal[
        "appropriate",
        "inappropriate",
        "conditional",
        "insufficient"
    ]

    core_finding: str
    quantitative_issue: str | None
    qualitative_issue: str | None
    clinical_risk: str | None
    recommended_action: str | None
    monitoring: str | None

    evidence_ids: list[str]
    missing_information: list[str]


class FinalMedicationReview(BaseModel):
    case_id: str
    findings: list[FinalFinding]
    overall_summary: str
    unresolved_questions: list[str]
```

## 2. 再将JSON转成临床可读文本

例如：

```text
药物或治疗方案：
判断：
适应证或合理性依据：
剂量、频次或疗程问题：
药物—疾病或药物—药物风险：
建议：
证据来源：
缺失信息：
```

## 3. 程序化完整性检查

完成生成后执行：

```python
def verify_completeness(state):
    ledger_slot_ids = set(state.ledger)
    answer_slot_ids = {x.slot_id for x in state.final_output.findings}

    missing_slots = ledger_slot_ids - answer_slot_ids

    for finding in state.final_output.findings:
        required = state.slot_map[finding.slot_id].required_attributes
        check_required_fields(finding, required)

    return missing_slots, missing_fields
```

模型只补写缺失槽位或字段，不重新生成整份答案。

这一步预计主要改善：

* 证据中有5点、答案只写4点；
* 已发现数值异常，但遗漏定性问题；
* 已发现风险，但遗漏处理建议；
* 检索到了正确用药依据，但最终答案只报告不合理部分。

---

# 十二、建议的LangGraph状态图

```python
from typing import TypedDict


class MedicationReviewState(TypedDict):
    case_id: str
    raw_question: str

    patient_case: dict
    review_slots: list[dict]
    query_bundles: list[dict]

    retrieval_round: int
    retrieval_results: list[dict]
    selected_evidence: list[dict]

    evidence_ledger: dict[str, dict]
    unresolved_slot_ids: list[str]

    final_structured_answer: dict
    final_text_answer: str

    query_log: list[dict]
    error_log: list[dict]
    usage_log: dict
```

状态图：

```python
builder.add_node("parse_case", parse_case_node)
builder.add_node("build_slots", build_slots_node)
builder.add_node("build_queries", build_query_bundles_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("select_evidence", select_evidence_node)
builder.add_node("update_ledger", update_ledger_node)
builder.add_node("assess_gaps", assess_gaps_node)
builder.add_node("build_gap_queries", build_gap_queries_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("verify_answer", verify_answer_node)

builder.add_edge("parse_case", "build_slots")
builder.add_edge("build_slots", "build_queries")
builder.add_edge("build_queries", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "select_evidence")
builder.add_edge("select_evidence", "update_ledger")
builder.add_edge("update_ledger", "assess_gaps")

builder.add_conditional_edges(
    "assess_gaps",
    route_after_gap_assessment,
    {
        "retrieve_gap": "build_gap_queries",
        "generate": "generate_answer",
    }
)

builder.add_edge("build_gap_queries", "retrieve")
builder.add_edge("generate_answer", "verify_answer")
```

其中Agentic部分体现在：

* LLM解析病例；
* LLM识别候选关系；
* LLM生成语义查询；
* LLM判断证据适用性；
* LLM识别证据缺口；
* LLM生成最终说明。

检索执行、预算、轮次、状态转移和停止条件是确定性的。这比通用ReAct循环更适合临床实验，也更容易复现。

确定性多阶段临床检索研究已经表明，将检索拆成可观测、可消融的阶段，可以显著提高临床证据召回，并明确每一阶段的错误来源。([ACL Anthology][7])

---

# 十三、最快的实现顺序

## 阶段A：两三天内可验证的最小版本

先不做规则卡片，不修改文档索引。

实现：

1. 病例结构化；
2. 关系槽位生成；
3. 每槽位生成一个风险向语义查询；
4. 直接调用现有向量Retriever；
5. 每槽位保留top-3；
6. 建立简单证据账本；
7. 从账本生成答案；
8. 执行完整性检查。

实验：

```text
A0：当前Yuxi向量Agent
A1：关系槽位规划 + 向量检索
A2：A1 + 证据账本 + 完整性回查
```

如果A1显著提高关系召回，说明主要瓶颈是通用Agent的审查规划。

如果A1没有提高，但A2提高属性完整性，说明主要瓶颈在证据利用和答案生成。

## 阶段B：增加双向查询和适用条件重排

实现：

1. 每槽位支持向与风险向查询；
2. 多查询RRF融合；
3. 程序化条件匹配；
4. LLM适用性判断；
5. 每槽位保留正负两类证据。

实验：

```text
B0：A2
B1：A2 + 双向查询
B2：B1 + 适用条件重排
```

## 阶段C：增加缺口驱动第二轮检索

实现：

1. 槽位覆盖分数；
2. 缺失属性识别；
3. 第二轮定向查询；
4. 新增证据收益和冗余监测。

实验：

```text
C0：B2，单轮
C1：B2，允许缺口驱动第二轮
```

## 阶段D：可选的规则卡片和表格增强

只有前面三阶段已经显示正向增益时，再进行：

* 表格行＋表头＋脚注联合索引；
* 临床规则卡片抽取；
* 数值阈值和例外条件结构化；
* 规则卡片用于候选生成和过滤。

---

# 十四、建议的数据划分

在继续调参前立即冻结划分。

按你原有的250、135和40三类问题，可采用：

| 子集         | 开发集 | 锁定测试集 |
| ---------- | --: | ----: |
| Beers及相关问题 |  24 |   226 |
| 单病种指南问题    |  24 |   111 |
| 跨文档复杂病例    |  12 |    28 |
| 合计         |  60 |   365 |

开发集用于：

* 查询模板；
* top-k；
  -覆盖阈值；
* 重排权重；
* 最大查询数；
  -提示词调整。

锁定测试集用于最终一次性比较。

如果你此前已经查看过全部425例的系统输出，应在论文中说明该数据集属于内部评测集，存在一定的开发接触风险。从现在开始仍应冻结365例，避免继续依据其中的个别结果调整方法。

---

# 十五、完整实验矩阵

## 1. 诊断实验

先在60例开发集和全部40例复杂病例上运行。

| 组别 | 配置                      | 目的          |
| -- | ----------------------- | ----------- |
| D0 | 当前Yuxi向量Agentic RAG     | 真实基线        |
| D1 | 原始问题直接调用向量Retriever     | 去除Agent查询改写 |
| D2 | Agent实际查询词直接调用Retriever | 判断查询改写影响    |
| D3 | 金标准关系槽位＋普通检索            | 估计规划上限      |
| D4 | 金标准证据＋普通答案生成            | 估计检索之后的生成上限 |
| D5 | 金标准结构化账本＋答案生成           | 估计最终生成上限    |

结果解释：

* D3远高于方法：槽位规划是主要瓶颈；
* D4远高于D3：检索或重排是主要瓶颈；
* D4仍低、D5高：证据解析和账本构建是主要瓶颈；
* D5仍低：答案生成或评价器存在问题。

## 2. 方法消融

开发集运行：

| 组别   | 关系规划 | 双向查询 | 条件重排 | 证据账本 | 缺口检索 | 完整性回查 |
| ---- | ---: | ---: | ---: | ---: | ---: | ----: |
| M0   |    否 |    否 |    否 |    否 |    否 |     否 |
| M1   |    是 |    否 |    否 |    否 |    否 |     否 |
| M2   |    是 |    是 |    否 |    否 |    否 |     否 |
| M3   |    是 |    是 |    是 |    否 |    否 |     否 |
| M4   |    是 |    是 |    是 |    是 |    否 |     否 |
| Full |    是 |    是 |    是 |    是 |    是 |     是 |

锁定测试集只需要运行：

1. 当前Yuxi向量Agent；
2. 当前LightRAG图谱Agent；
3. M1关系规划系统；
4. M3条件重排系统；
5. Full完整方法。

如果成本较高，可以只在全部365例上运行基线、M3和Full，其他消融在预先固定的100例消融子集上运行。

---

# 十六、需要记录的分层指标

## 1. 规划层

[
\text{Plan Recall}
==================

\frac{
\text{被系统生成槽位覆盖的金标准关系数}
}{
\text{全部金标准关系数}
}
]

另报告：

* 每病例槽位数；
* 药物—疾病候选数；
* 药物—药物候选数；
* 无效槽位比例；
* 高风险槽位漏生成率。

## 2. 候选召回层

[
\text{Candidate Evidence Recall@K}
==================================

\frac{
\text{候选池中存在支持证据的金标准关系数}
}{
\text{全部金标准关系数}
}
]

## 3. 重排层

[
\text{Context Evidence Recall@K}
================================

\frac{
\text{最终上下文中有支持证据的金标准关系数}
}{
\text{全部金标准关系数}
}
]

[
\text{Applicability Precision@K}
================================

\frac{
\text{前K条中适用于当前患者的证据数}
}{
K
}
]

证据适用性指标建议在全部40个复杂病例加60个分层样本上人工标注。

## 4. 证据覆盖层

[
\text{Slot Coverage}
====================

\frac{
\text{达到覆盖阈值的槽位数}
}{
\text{全部槽位数}
}
]

[
\text{Round-2 Unique Gain}
==========================

\frac{
\text{第二轮新增覆盖的金标准关系数}
}{
\text{全部金标准关系数}
}
]

[
\text{Retrieval Redundancy}
===========================

1-
\frac{
|\bigcup_t E_t|
}{
\sum_t |E_t|
}
]

## 5. 最终关系判断

对合理、不合理和条件性关系计算：

* 微平均Precision、Recall、F1；
* 宏平均Precision、Recall、F1；
* 各关系类型F1；
* 合理用药F1；
* 不合理用药F1；
* 弃答率；
* 非弃答样本准确率。

## 6. 内容完整性

[
\text{Core Finding Recall}
==========================

\frac{
\text{正确识别的核心问题数}
}{
\text{金标准核心问题数}
}
]

[
\text{Attribute Completeness}
=============================

\frac{
\text{正确覆盖的必需属性数}
}{
\text{金标准必需属性数}
}
]

[
\text{Full Finding Rate}
========================

\frac{
\text{所有必需属性均被覆盖的关系项数}
}{
\text{全部金标准关系项数}
}
]

## 7. 证据利用

[
\text{Evidence Utilization}
===========================

\frac{
\text{已进入账本且出现在最终答案中的关系数}
}{
\text{账本中有充分证据的关系数}
}
]

## 8. 病例级严格指标

[
\text{Complete Case Accuracy}
=============================

\frac{
\text{所有金标准关系均判断正确的病例数}
}{
N
}
]

该指标会很严格，但对复杂处方具有较强解释性。

## 9. 资源消耗

必须分开报告：

* Agent外层检索轮次；
* 内部子查询数量；
* 候选证据数量；
* 最终证据数量；
* 输入和输出token；
* 上下文token；
* LLM调用次数；
* 总延迟；
* API成本；
* 后端异常率。

不能再以“工具调用次数”单独评价检索效率，因为一次批量工具调用可以执行多个有目的的子查询。

---

# 十七、建议的LLM-as-judge协议

由于金标准已经完整，Judge不再分解参考答案，只接收固定金标准项目。

输出格式：

```json
{
  "finding_id": "C012_F03",
  "core_status": "fully_covered",
  "label_status": "correct",
  "attribute_results": [
    {
      "attribute_name": "quantitative_issue",
      "status": "covered",
      "answer_span": "……"
    },
    {
      "attribute_name": "qualitative_issue",
      "status": "missing",
      "answer_span": null
    }
  ],
  "contradiction": false,
  "unsupported_addition": false
}
```

Judge提示词应规定：

* 不得创建新的金标准事实；
* 必须返回系统答案中的对应原文；
* 语义等价可以判为覆盖；
* 只提到相关药物但没有完成临床判断，不算完整覆盖；
* 数值问题和定性问题分别判断；
* 风险和建议分别判断；
* 对合理与不合理判断采用相同严格程度。

在至少100例上由人工核验Judge，报告：

* 百分比一致率；
* Cohen’s (\kappa)；
* 完整覆盖判断一致率；
* 部分覆盖判断一致率。

---

# 十八、实验前必须完成的日志改造

每个病例保存一份完整JSON：

```json
{
  "case_id": "C001",
  "split": "test",
  "config_hash": "...",
  "thread_id": "...",

  "raw_question": "...",
  "parsed_case": {},

  "planned_slots": [],
  "query_bundles": [],

  "retrieval_rounds": [
    {
      "round_id": 1,
      "queries": [],
      "raw_results": [],
      "selected_evidence": []
    }
  ],

  "evidence_ledger": {},
  "final_structured_answer": {},
  "final_text_answer": "...",

  "metrics": {
    "agent_rounds": 2,
    "subquery_count": 12,
    "candidate_count": 148,
    "selected_evidence_count": 17,
    "context_tokens": 8650,
    "input_tokens": 13200,
    "output_tokens": 1800,
    "latency_seconds": 31.4
  },

  "errors": []
}
```

同时保存：

* Agent模型精确ID；
* Embedding模型精确ID；
* reranker精确ID；
* 知识库版本；
* 文档版本；
* Prompt文件哈希；
* 查询模板版本；
  -代码commit；
  -运行日期。

---

# 十九、实用的成功判定标准

下面不是论文统计阈值，而是工程上的继续投入标准。

## 关系规划值得保留的条件

在开发集上：

```text
Plan Recall ≥ 0.95
无效槽位比例 ≤ 0.25
```

## 双向查询值得保留的条件

相较单一风险查询：

```text
合理用药Recall提高至少3个百分点
总体Candidate Evidence Recall提高至少3个百分点
Precision下降不超过2个百分点
```

## 条件重排值得保留的条件

```text
Applicability Precision@5提高至少5个百分点
Context Evidence Recall不下降
```

## 第二轮检索值得保留的条件

复杂病例中：

```text
Round-2 Unique Gain ≥ 0.05
```

若第二轮大部分查询没有新增证据，应删除第二轮，而不是为了体现Agentic而保留。

## 完整性回查值得保留的条件

```text
Attribute Completeness提高至少5个百分点
新增无依据内容不增加
```

---

# 二十、可选的规则卡片增强

当上述流程已经稳定后，可以对指南文本抽取轻量规则卡片：

```json
{
  "rule_id": "R001",
  "drug_or_class": ["药物A", "药物类别A"],
  "relation_type": "drug_disease",
  "polarity": "negative",

  "target": "疾病D",

  "age_condition": ">=65",
  "renal_condition": "eGFR < 30",
  "dose_condition": null,
  "duration_condition": "> 7 days",
  "co_medications": [],
  "exceptions": [],

  "risk": "……",
  "recommended_action": "……",

  "source_document": "……",
  "source_version": "……",
  "section": "……",
  "original_span": "……"
}
```

规则卡片的角色是：

```text
规则卡片发现候选关系
→ 原始文本检索验证
→ 患者条件匹配
→ 最终结论引用原始文本
```

不要让规则卡片替代原文证据。

图谱相关研究已经指出，启发式子图扩展可能引入冗余和噪声事实，需要聚焦检索和渐进剪枝才能提高医学问答效果。你的规则卡片可以理解为比通用图谱更适合临床条件表示的轻量结构化层。([ACL Anthology][8])

---

# 二十一、最终推荐的主方法版本

考虑时间、开发成本和论文可解释性，最终论文方法建议包含四个实质组件：

1. **处方关系覆盖规划**：明确枚举药物整体、药物—疾病、药物—药物、器官功能、重复用药、累积风险和处方遗漏；
2. **双向多查询证据召回**：同时检索合理性支持和不合理性风险证据；
3. **患者适用条件重排与覆盖选择**：依据年龄、疾病、剂量、疗程和器官功能筛选互补证据；
4. **证据账本驱动的有界检索与完整性回查**：只针对缺失属性继续检索，并确保最终答案覆盖全部已确认关系。

论文中可以提出三个主要研究假设：

[
H_1:
\text{关系覆盖规划提高复杂病例的金标准关系召回率}
]

[
H_2:
\text{患者条件重排提高证据适用性精度和最终判断精度}
]

[
H_3:
\text{证据账本与完整性回查提高属性完整率，并减少已检索事实在生成阶段的损失}
]

这一方案能够充分利用你已经完善的完整标准答案，也能把当前“向量优于图谱、BM25没有明显帮助、Agent检索轮次不下降”的负结果纳入合理的研究逻辑：

> 老年多重用药审查的主要瓶颈并非更换通用检索器，而是完整规划需要审查的临床关系、识别证据对患者的适用条件、控制证据覆盖以及确保最终答案忠实表达全部已确认问题。

这将使论文从通用Agentic RAG系统比较，转为一项针对老年多重用药审查任务、具有明确方法设计和可检验机制的应用研究。

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12478568/?utm_source=chatgpt.com "American Geriatrics Society 2023 updated AGS Beers Criteria ..."
[2]: https://aclanthology.org/2024.findings-acl.137/ "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation - ACL Anthology"
[3]: https://aclanthology.org/2026.findings-acl.154/?utm_source=chatgpt.com "Applicability Condition Extraction for Therapeutic Drug- ..."
[4]: https://aclanthology.org/2026.findings-eacl.150/ "DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation - ACL Anthology"
[5]: https://aclanthology.org/2026.bionlp-1.73/ "CrossDDI: Cross-Source Evidence-Grounded Drug-Drug Interaction Verification - ACL Anthology"
[6]: https://aclanthology.org/2026.findings-acl.917/ "SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning - ACL Anthology"
[7]: https://aclanthology.org/2026.bionlp-1.53/ "A Deterministic Multi-Stage Retrieval Pipeline for Longitudinal EHR Question Answering - ACL Anthology"
[8]: https://aclanthology.org/2026.findings-acl.408/ "Iterative Knowledge Graph Refinement and Integration for Medical Question Answering - ACL Anthology"
