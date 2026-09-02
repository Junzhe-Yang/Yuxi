你这次把问题推进到了更接近根因的位置。前一版把主要干预放在首次停止之后，已经错过了最关键的决策窗口。等模型准备结束时，它通常已经形成了“关键问题已解决”的内部结论。此时再要求补查，模型很容易把任务理解成答案检查，继续检索的动力仍然很弱。

我会正式替换上一版方案。新的主控制点前移到 Investigation 的生成、调度、补充检索和关闭全过程。首次停止检查只保留为结构性兜底，不再承担主要召回改进职责。

## 一、重新审计以后，证据支持你的判断

我重新统计了 175 个 ACM PRIM 配对病例的 Investigation 生命周期。

1. 175 个病例一共创建 268 个 Investigation，平均每题 1.53 个。
2. 106 个病例只有 1 个 Investigation，占全部病例的 60.57%。47 个病例有 2 个，20 个病例有 3 个，只有 2 个病例有 4 个。
3. 249 个 Investigation 最终标记为 `answered`。其中 148 个只进行过一次查询，比例为 59.44%。
4. 在这 148 个单查询关闭的 Investigation 中，106 个连原文窗口都没有打开。也就是说，模型得到一次候选块以后便直接关闭调查。
5. 594 条搜索记录中有 128 条没有关联任何 Investigation，未分组搜索比例为 21.55%。
6. 564 个 PlanAnchor 中，只有 6 个真正出现在 Investigation 的 `focus_plan_ids` 中，结构关联率只有 1.06%。264 个 Investigation 同时没有 PlanAnchor 和 PatientModifier 关联。

这些结果说明，当前 Investigation 尚未真正承担“组织完整调查”的功能。它主要承担了查询记录容器和关闭标记的功能。

源码中的关闭条件也印证了这一点。`update_investigation` 对 `answered` 的核心要求只有一条，即至少选择一个属于该调查候选集合的 Evidence ID。程序没有检查查询次数，没有要求互补查询，没有检查调查边界，也没有要求该调查在不同提示下重新检索。[当前 Investigation 工具实现](sandbox:/mnt/data/acm_handoff/source/agents/medication_review_prim/tools.py)

搜索工具在没有 `investigation_id` 和 `question` 时仍然正常执行，所以当前未分组查询属于设计允许的行为。已有 Investigation 再次检索时会重新设为 `open`，但 Agent 没有任何程序性义务继续检索。[当前 Investigation 工具实现](sandbox:/mnt/data/acm_handoff/source/agents/medication_review_prim/tools.py)

Yuxi 的底层工具循环本来就把是否检索、查询内容、是否继续和何时停止交给模型。源码分析已经明确指出，Prompt 可以改变倾向，无法形成调用层面的统一约束。真正的努力标准需要进入工具、中间件或 Harness 控制流。

这次新增审计文件如下：

[Investigation 生命周期审计汇总](sandbox:/mnt/data/acm_prim_investigation_lifecycle_audit_summary_20260818.json)

[Investigation 生命周期审计脚本](sandbox:/mnt/data/acm_prim_investigation_lifecycle_audit_20260818.py)

## 二、当前方法可能产生了“探索坍缩”

你提出的解释很有说服力。当前 ACM PRIM 可能同时发生了两件事。

第一，结构化提示帮助小模型形成了更精确的查询。模型更快命中一个具有决定性的块，例如禁忌、剂量或者关键风险。

第二，精确命中强化了模型的完成感。模型把“这个 Investigation 已经有一条关键证据”理解成“该病例的相关调查已经充分”。随后它关闭 Investigation，也很少再提出新的 Investigation。

这可以称为调查层面的探索坍缩：

[
\text{更精确的首轮查询}
\rightarrow
\text{更早取得关键证据}
\rightarrow
\text{更高的主观完成感}
\rightarrow
\text{更少的新 Investigation}
\rightarrow
\text{更窄的总体证据覆盖}
]

这个解释目前属于机制假设，还没有形成因果证明。事后分组结果与它一致：

1. 只有 1 个 Investigation 的 106 个病例，ACM 相对同题 baseline 平均落后 5.18 个百分点。
2. 至少 2 个 Investigation 的 69 个病例，ACM 相对 baseline 平均领先 0.94 个百分点。
3. 搜索次数不超过 3 次的 105 个病例，ACM 相对 baseline 平均落后 5.18 个百分点。
4. 搜索次数达到 5 次及以上的 38 个病例，ACM 相对 baseline 平均领先 3.76 个百分点。

这些分组受到病例难度、金标数量和模型决策的共同影响，不能直接解释成“多查一定更好”。它们至少说明，当前召回损失高度集中在调查数量少和搜索次数少的运行轨迹中。

这里还需要补充一个重要限定。单纯要求每个 Investigation 继续增加查询，也可能让模型在同一个问题上反复改写。已有 BPH 轨迹执行了 8 次搜索，后期查询重复比例达到 0.8 和 1.0，模型持续围绕特拉唑嗪与低血压风险重复检索，最终还因状态契约问题没有生成正常答案。

因此，新的方法需要同时约束调查广度和单个调查深度。

## 三、主方法改为“调查覆盖与努力契约”

我建议下一版将 Investigation 从可选记录对象提升为 Agent 检索循环的强制调度单位。

程序只裁决可确定的过程事实。医学问题内容、查询文本、证据含义和最终判断继续由同一个 Agent 生成。

整个流程变为：

```text
生成调查议程
→ 对所有调查执行第一轮检索
→ 对所有调查执行互补检索
→ 使用剩余预算进行自适应补充
→ 显式关闭每个调查
→ 生成最终回答
```

### 1. 首先建立调查议程

Agent 在看到原始病例、完整 PlanAnchor 清单和完整静态 Atlas 文档地图后，必须先创建至少 (K) 个自然语言 Investigation。

首版建议使用：

[
K=3
]

选择 3 的原因很直接。当前最大搜索预算为 8。每个 Investigation 执行两轮查询时，最低需要 6 次搜索，还保留 2 次自适应搜索空间。这个调用量略高于 baseline 的平均 5 次，足以检验当前差距是否主要来自努力不足。

(K=2) 与 (K=3) 应先在开发集进行比较，测试集运行前冻结。不能查看测试集结果后再选择参数。

议程工具可以使用如下结构：

```python
set_investigation_agenda(
    investigations=[
        {
            "question": "...",
            "focus_plan_ids": ["PE001"],
            "distinctive_scope": "该调查与其它调查的区别"
        },
        {
            "question": "...",
            "focus_plan_ids": ["PE002", "PE003"],
            "distinctive_scope": "该调查与其它调查的区别"
        },
        {
            "question": "...",
            "focus_plan_ids": ["PE001", "PE003"],
            "distinctive_scope": "该调查与其它调查的区别"
        }
    ]
)
```

其中只有三项程序约束：

1. Investigation 数量达到 (K)。
2. 每个有效 PlanAnchor 至少出现在一个 Investigation 的 `focus_plan_ids` 中。
3. Investigation 问题不能完全重复。

程序不判断问题是否属于剂量、禁忌、相互作用或者监测，也不自动生成任何问题。`distinctive_scope` 由 Agent 自己说明，主要用于提示和离线审计。

如果 PlanAnchor 抽取失败，系统仍然要求 (K) 个 Investigation，只跳过 PlanAnchor 结构覆盖检查。原始病例始终保留。

这一改动直接修复目前 564 个 PlanAnchor 只有 6 个进入 Investigation 的问题，也能够防止模型只围绕一个最显眼风险建立调查。

### 2. 所有搜索必须属于某个 Investigation

`search_review_kb` 不再接受无 Investigation 的在线搜索。

模型执行搜索时必须提供以下一种信息：

```text
已有 investigation_id
```

或者：

```text
新 investigation_question
```

在调查议程建立以后，通常只允许使用已有 `investigation_id`。

如果模型遗漏 Investigation ID，工具返回结构错误，本次不调用 Milvus，也不消耗搜索预算。模型需要重新选择一个调查对象。

这项规则可以把当前 21.55% 的未分组搜索降到 0，也让每次查询都能够进入一个清楚的调查生命周期。

### 3. 每个 Investigation 强制执行两种不同阶段的查询

每个 Investigation 至少经历两个有效搜索阶段。

第一阶段叫初始探查。Agent 根据病例、Atlas 地图和 Investigation 问题形成第一条精确查询。

第二阶段叫互补探查。Agent 阅读第一轮 Evidence 后，进入一个不同的提示上下文。此时关闭工具不可见，模型只能识别尚未覆盖的边界并执行第二条查询。

互补探查提示可以写成：

> 当前 Investigation 已经完成一次检索，并获得了候选 Evidence。请暂时不要关闭调查。重新阅读调查问题、第一条查询和返回证据，找出一个仍未得到覆盖的实质边界。该边界可以涉及已有证据的适用条件、相反证据、例外、具体实施要求、患者条件、联合方案影响、监测、疗程、替代处理，或者其它由当前调查自然产生的问题。请说明第二条查询与第一条查询的差异，并立即执行检索。

这里列出的内容只作为语言提示，不形成程序化属性槽。

搜索工具增加一个确定性字段：

```python
investigation_pass: Literal[
    "initial_probe",
    "complementary_probe",
    "adaptive_probe",
]
```

程序检查以下事实：

1. `initial_probe` 已经真实执行。
2. `complementary_probe` 已经真实执行。
3. 两条 `query_text` 不能完全相同。
4. 第二轮必须填写 `uncovered_aspect`。
5. `success` 和 `success_empty` 都算作一次真实调查尝试。
6. `technical_failed` 不满足努力契约，技术重试成功后才计入。

程序不判断两条查询在医学语义上是否足够不同。查询差异、证据新颖度和新增金标收益在离线评价中计算。

### 4. 使用广度优先的轮转调度

这是整个方案中非常关键的一部分。

在所有必需 Investigation 完成第一轮查询以前，任何 Investigation 都不能执行第三次查询。在所有 Investigation 完成互补探查以前，任何 Investigation 都不能关闭。

使用 (K=3) 时，搜索调度固定为：

```text
第1至3次搜索
三个 Investigation 各完成一次 initial_probe

第4至6次搜索
三个 Investigation 各完成一次 complementary_probe

第7至8次搜索
Agent 根据当前剩余缺口自适应分配
```

这样可以防止一个高显著性 Investigation 连续吸收全部预算。

形式上，设第 (j) 个 Investigation 的有效查询数为 (n_j)，最低努力条件为：

[
|\mathcal I|\ge K
]

[
n_j\ge 2,\qquad \forall j\in\mathcal I
]

[
\operatorname{Close}(I_j)
\Rightarrow
\operatorname{InitialProbe}(I_j)
\land
\operatorname{ComplementaryProbe}(I_j)
]

当前方法只要求：

[
\operatorname{CloseAnswered}(I_j)
\Rightarrow
|\operatorname{SelectedEvidence}(I_j)|\ge 1
]

新条件增加了统一的过程充分性，仍然没有让程序判断医学充分性。

### 5. 调查关闭改为显式、分阶段关闭

Investigation 状态建议调整为：

```python
InvestigationPhase = Literal[
    "planned",
    "initial_probed",
    "complementary_probed",
    "closeable",
    "closed_answered",
    "closed_insufficient",
    "dismissed",
]
```

关闭工具接收：

```python
complete_investigation(
    investigation_id="INV...",
    outcome="answered",
    selected_evidence_ids=["EV..."],
    working_conclusion="...",
    residual_uncertainty="...",
)
```

Harness 只检查：

1. 两轮必需搜索已经完成。
2. `answered` 至少选择一个候选 Evidence。
3. `insufficient` 明确记录剩余证据缺口。
4. 提前 `dismissed` 的 Investigation 不计入 (K)，系统必须创建替代 Investigation。

Agent 即使在第一轮后宣称“证据已经充分”，关闭请求也会被程序拒绝。它仍需完成互补探查。这正是独立于模型完成偏好的统一努力标准。

## 四、上下文管理确实需要调整

加强 Investigation 生命周期以后，模型调用次数和 Evidence 数量都会增加。继续把全部原始 ToolMessage、所有 Evidence Card 和所有旧推理平铺到每轮上下文中，容易造成两个问题。

第一，最早发现的高显著性证据会持续占据注意中心，后续 Investigation 仍然围绕它展开。

第二，小模型在长上下文中更容易复述现有结论，也更容易忽略尚未执行的调查阶段。

我建议采用阶段化 Investigation 上下文。

### 调查议程阶段

模型看到：

```text
原始病例全文
全部 PlanAnchor
完整静态 Atlas 文档地图
搜索预算
议程生成要求
```

此时不显示 Evidence，因为尚未检索。

### 初始探查阶段

模型看到：

```text
原始病例全文
完整静态 Atlas 文档地图
当前 Investigation 全文
其它 Investigation 的一行摘要
当前查询历史
剩余预算
```

### 互补探查阶段

模型看到：

```text
当前 Investigation 问题
第一条 query_text 和 search_reason
第一轮去重 Evidence Card
当前 Agent 工作结论
明确要求寻找未覆盖边界
其它 Investigation 的阶段状态
完整静态 Atlas 文档地图
```

### 最终回答阶段

模型看到：

```text
原始病例
全部关闭 Investigation 的工作结论
每个 Investigation 选中的 Evidence
所有 PlanAnchor
完整证据来源信息
```

原始 ToolMessage、完整原文块和全部历史继续保存在 state 与 trace 中。关闭 Investigation 的紧凑摘要由 Agent 自己填写，程序只负责保存和展示。这样不会由程序压缩医学语义。

上下文改造建议作为独立消融。核心方法先验证 Investigation 努力契约。随后再检验阶段化上下文是否为小模型带来额外收益。

## 五、Atlas 在新方法中的位置

Atlas 的完整静态文档地图继续在调查议程、初始探查和互补探查阶段显示。

Agent 可以依据地图提出不同 Investigation，也可以在任一 Investigation 中选择全库检索、打开文档卡或文档内检索。

本轮继续保持以下边界：

1. 不检索 Atlas cue。
2. 不筛选 Atlas cue。
3. 不设置 cue 数量上限。
4. 不注入 Atlas 源块。
5. 不合并或重排 Milvus 返回结果。
6. 不加入 BM25、融合排序或 reranker。
7. 不强制使用 document scope。

Atlas 在新方法中主要帮助 Agent扩展调查问题空间。纯向量检索继续负责证据块召回。

## 六、为什么这次方案真正改变了当前方法

当前 ACM PRIM 的运行逻辑可以概括为：

```text
Agent 自主提出一个调查
→ 执行一次精确查询
→ 得到一个关键候选块
→ 选择该块
→ 标记 answered
→ 进入答案生成
```

新方法的运行逻辑为：

```text
Agent 在完整地图上建立至少三个不同调查
→ 每个调查先执行一次初始探查
→ 每个调查再执行一次互补探查
→ 剩余预算根据调查缺口自适应分配
→ 每个调查显式关闭
→ 完成最终回答
```

它增加了四项当前系统实际缺失的功能：

1. Investigation 数量具有最低要求。
2. 所有显式治疗方案要素必须进入调查议程。
3. 每个 Investigation 都有最低两轮检索努力。
4. Harness 控制广度优先的预算调度。

因此，它能够直接处理你提出的两个问题。

针对 Investigation 太少，系统通过议程数量下限和 PlanAnchor 覆盖要求扩大调查范围。

针对找到一个关键块后立即结束，系统通过互补探查和关闭门控强制继续调查。

Agent 自检仍然承担“下一条查询具体问什么”。Harness 承担“最低调查努力是否完成”。这两个职责边界清楚，也符合通用 Agent 方法的要求。

## 七、需要防范的失败方式

这套方案仍然存在四种可证伪风险。

第一，模型可能建立三个表述不同、内容高度接近的 Investigation。需要离线评价 Investigation 问题的语义差异和金标覆盖，测试集运行期间不进行在线语义裁决。

第二，互补探查可能只产生同义改写。需要报告查询相似度、重复块比例和第二轮新增金标目标数。

第三，最低六次搜索可能只通过扩大候选池提高召回。需要设置相同调用数的无结构控制组。

第四，阶段化上下文可能隐藏有用的早期证据。它必须作为独立消融，底层 state 和 trace 保留完整信息。

已有长轨迹也说明，调用数量增加以后，重复查询、状态契约和最终输出可靠性需要同时检查。强调查约束不能重新引入“证据已经找到，回答却因程序状态失败”的问题。

## 八、最有解释力的实验设计

建议下一轮使用四个 ACM 条件，再与外部 baseline 比较。

### A0，当前 ACM PRIM

保持现有 Investigation 创建、关闭和软反思逻辑。

### A1，无结构最低六次搜索

保持当前 Prompt 和 Investigation 逻辑，只禁止模型在完成 6 次搜索前结束。

这一组检验纯粹增加搜索努力能带来多少收益。

### A2，调查覆盖与努力契约

使用 (K=3)，每个 Investigation 执行一次初始探查和一次互补探查，最低 6 次搜索。上下文仍沿用当前消息管理。

A2 与 A1 使用相同的最低搜索次数。两者差异能够检验 Investigation 级组织和广度优先调度是否具有独立价值。

### A3，A2 加阶段化 Investigation 上下文

A3 检验上下文控制对于小模型是否必要。

开发阶段额外比较 (K=2) 与 (K=3)，测试阶段只运行冻结后的一个值。

四组保持以下条件一致：

```text
相同主模型和参数
相同完整 Atlas 地图
相同 Atlas 文档卡
相同 Milvus 纯向量 Top 10
相同最大搜索预算 8
相同 open 预算
相同 Evidence Card
相同病例顺序
相同最终回答要求
```

正式块检索指标继续使用 All、Core 和 Supporting 的 Macro Recall 与 Micro Recall，同时报告 Complete Coverage、Target MRR、Target nDCG、Autonomous Recall AUC、Productive Call Rate、New Targets per Call、Duplicate Item Rate 和 Document conditioned Chunk Recall。

新增机制指标应包括：

```text
每病例 Investigation 数量
PlanAnchor 调查覆盖率
未分组搜索率
单查询关闭率
互补探查执行率
互补探查新增金标目标数
互补探查零收益率
每个 Investigation 的查询数
跨 Investigation 的新增目标分布
搜索预算耗尽率
总输入 token
运行时间
```

结果解释需要提前固定：

1. A1 高于 A0，说明搜索努力不足具有独立影响。
2. A2 高于 A1，说明调查议程、两轮生命周期和广度优先调度具有独立价值。
3. A1 与 A2 提升接近，说明主要收益来自最低搜索次数，复杂 Investigation 结构的必要性有限。
4. A3 高于 A2，说明小模型需要阶段化上下文才能利用新增轮次。
5. A2 增加调用量，却没有提高 Recall 和 Complete Coverage，说明模型无法生成有效的互补查询，或者当前 Atlas 与查询提示没有提供足够的新方向。
6. A2 提高 Recall，同时 Duplicate Item Rate 明显上升、Productive Call Rate明显下降，说明努力契约过强，需要降低 (K) 或调整互补探查提示。

## 九、最终判断

你的最新判断应当成为下一版设计的主要依据。

首次停止补查降为最终兜底。文档内检索门控从主方法中移除。核心改动前移到 Investigation 生命周期，形成以下统一标准：

```text
至少三个不同 Investigation
全部 PlanAnchor 进入调查议程
每次搜索必须绑定 Investigation
所有 Investigation 先完成第一轮探查
所有 Investigation 再完成互补探查
两轮完成以前禁止关闭
剩余预算允许自适应补充
所有调查关闭以后生成答案
```

这套方法保留当前精确查询的优势，同时为小模型设置一个与主观完成感无关的最低研究努力。它也把科学问题收缩得很清楚：

> 在检索器、知识地图和模型保持一致时，面向 Investigation 的广度与深度努力契约，能否纠正小模型的过早调查收敛并提高证据块召回？

我认为这比此前的首次停止补查更接近当前系统真正需要的 Harness 方法。