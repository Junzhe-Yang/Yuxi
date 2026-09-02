# DA-PRIM：文档地图引导的患者—治疗方案调查记忆 RAG

下面给出一个能够直接进入实现和实验的算法级方案。它是在 V4.2 PRIM-RAG 上的增量设计，不改变 V4.2 的核心 Agent loop、PlanAnchor、PatientModifier、RelationInvestigation、Evidence Card 和一次方案要素覆盖反思。V4.2 当前将检索后端固定为单个 Milvus 纯向量 Top-5，明确不进行自动查询扩展、文档路由或固定关系枚举；其关系调查超图只有在 Agent 已经主动提出关系问题后才会形成。

因此，DA-PRIM 只解决一个新增问题：

> **在 Agent 尚不知道某种药物—疾病、药物—药物或药物—患者因素联系时，如何利用29篇本地指南自身的文档和章节结构，向Agent暴露可能相关的检索方向，同时避免预先枚举临床关系或强制接受程序判断？**

下述 DA-PRIM 是基于当前实验结果和外部研究形成的新设计，不是 V4.2 文档中已经存在的内容。

------

## 一、研究问题与科学假设

### 1. 问题发现

目前检索失败可能来自两个相互关联的原因。

第一，Milvus 实际排序的是全部文档切分后的 chunk，而不是29篇文档本身。一篇篇幅较长、广泛出现“老年、风险、剂量、监测”等词的综合指南，会拥有更多进入Top-5的机会。专题高度匹配、但相关条目数量较少的文档，可能完全不进入Top-5。

第二，V4.2 的关系调查依赖 Agent 先形成某个医学假设：

[
\text{Agent意识到关系}
\rightarrow
\text{提出 relation_question}
\rightarrow
\text{组织查询}
\rightarrow
\text{发现证据}
]

对于模型参数知识中不熟悉的关系，流程会停在第一步。这就是“蛋生鸡”问题。

平坦 chunk 检索容易出现文档路由失败和证据碎片化，分层文档路由与粗到细检索是近期RAG研究中较稳定的改进方向。HiKEY将文档层级作为一等检索信号，报告了相较平坦检索更高的检索召回；医学多来源RAG中的HERO-QA也采用文档、章节、子chunk的层次化检索，并保留全局或全文回退路径。([arXiv](https://arxiv.org/abs/2605.29606))

### 2. 核心假设

建议将新增研究假设定义为：

> **H4：语料地图引导假设。**
> 在主Agent、向量模型、知识库、检索调用次数、最终Evidence Card数量和生成预算均相同的条件下，利用病例要素对本地文档—章节地图进行软路由，并将多个病例对象共同命中的章节作为非结论性的调查机会提供给Agent，可以提高文档召回率和患者特异性关系发现率，而不显著增加无依据临床判断。

这一假设可拆为三个可检验子假设：

[
H_{4a}:
\text{文档地图提高目标文档召回}
]

[
H_{4b}:
\text{软路由提高目标证据召回}
]

[
H_{4c}:
\text{共同路由产生的调查机会提高未知关系发现率}
]

------

# 二、主方法只保留四个必要组件

DA-PRIM 主方法只增加以下四项。

1. **Corpus Atlas**：29篇文档的文档卡和章节卡。
2. **病例多视图软路由**：PlanAnchor、PatientModifier、完整方案和病例全文分别路由文档。
3. **全局＋文档内双路径检索**：保留全局检索作为逃生路径，同时在高相关候选文档内进行局部检索。
4. **语料诱导的调查机会**：当不同病例对象共同命中同一章节时，向Agent显示“可能值得调查”的线索。

不进入主方法的内容包括：

- BM25 文档路由；
- Cross-encoder reranker；
- 额外查询扩展模型；
- 硬文档过滤；
- 固定关系枚举；
- 文档访问完成性门控；
- Evidence适用性分类器；
- 强制来源多样性；
- 全文回退；
- 第二个审计Agent。

这些可以在主方法确认有效后作为后续消融。

------

# 三、总体流程

```text
离线阶段
29篇原始文档
→ 文档结构解析
→ DocumentCard
→ SectionCard
→ 文档/章节向量矩阵
→ Atlas快照

在线阶段
病例原文
→ V4.2 PlanAnchor
→ V4.2 PatientModifier
→ 病例多视图文档路由
→ Top-K文档地图
→ 共同章节命中
→ Top-L调查机会
→ V4.2 PRIM Agent loop
    ├─ Agent自主提出relation_question
    ├─ Agent自主生成查询
    ├─ 全局chunk检索
    ├─ 候选文档内检索
    ├─ 排名融合
    ├─ Evidence Card
    └─ open原文
→ V4.2一次方案要素覆盖反思
→ 最终答案
```

程序只提供语料导航。临床关系是否存在、证据是否适用、药物是否合理、是否需要换药，仍由同一个Agent判断。

------

# 四、离线算法：构建 Corpus Atlas

## 4.1 输入

设知识库包含：

[
D={d_1,d_2,\ldots,d_{29}}
]

每篇文档 (d) 已被Yuxi解析为Markdown和chunk集合：

[
C_d={c_{d1},c_{d2},\ldots,c_{dn_d}}
]

Atlas构建必须在测试集运行前完成，只读取知识库文档，不读取：

- 测试问题；
- 标准答案；
- 金标准文档；
- 测试检索结果；
- 人工错误分析结果。

这样可以防止把测试集知识写入路由器。

------

## 4.2 DocumentCard

每篇文档生成一个DocumentCard：

```python
class DocumentCard(BaseModel):
    doc_id: str
    file_id: str

    title: str
    year: str | None
    source_type: str | None

    heading_paths: list[str]
    table_titles: list[str]

    routing_text: str
```

核心字段 `routing_text` 采用确定性拼接：

```text
文档标题
＋一级标题
＋二级标题
＋表格标题
```

例如：

```text
老年人良性前列腺增生症/下尿路症状药物治疗共识（2015）
α1受体阻滞剂
5α还原酶抑制剂
老年患者应用注意事项
体位性低血压
神经系统不良反应
联合用药
```

### 为什么主方法不用LLM生成文档摘要

LLM摘要可能使文档路由更流畅，但会引入三种额外变量：

- 摘要可能遗漏原文中的次要药物或患者条件；
- 摘要可能加入文档没有明确包含的概括；
- 路由提升无法区分来自文档结构还是摘要模型知识。

29篇文档数量很少，标题和高层章节已经提供很强的专题信号。主方法使用可复现的确定性文本即可。LLM摘要可以作为后续消融。

------

## 4.3 SectionCard

每个具有明确标题路径的章节生成一个SectionCard：

```python
class SectionCard(BaseModel):
    section_id: str
    doc_id: str
    file_id: str

    heading_path: str
    lead_text: str
    table_titles: list[str]

    chunk_ids: list[str]
    routing_text: str
```

其中：

```text
routing_text =
文档标题
＋完整标题路径
＋章节开头若干句
＋本章节中的表格标题
```

例如：

```text
老年人BPH/LUTS药物治疗共识
> α1受体阻滞剂
> 老年患者应用注意事项
> 体位性低血压

体位性低血压是老年人应用α1受体阻滞剂的药物不良反应……
```

`lead_text`只取章节开头约200至400字，用于章节导航，不作为最终回答证据。最终证据仍由原始chunk提供。

### 章节—chunk映射

按照每个chunk在文档中的位置，将其绑定到最近的前置标题路径：

```python
section.chunk_ids.append(chunk.chunk_id)
```

若文档没有可识别标题：

- 整篇文档仍有DocumentCard；
- 按固定窗口划分伪章节；
- `heading_path`使用“文档片段1、文档片段2”等中性标识；
- 不调用LLM补造标题。

------

## 4.4 向量索引

使用与当前知识库相同的Embedding模型，对以下对象生成归一化向量：

[
\mathbf{u}_d = \operatorname{Embed}(\text{DocumentCard.routing_text})
]

[
\mathbf{v}_s = \operatorname{Embed}(\text{SectionCard.routing_text})
]

由于只有29个DocumentCard和数量有限的SectionCard，主方法不需要新建复杂向量数据库。可以保存为：

```text
corpus_atlas/
├── atlas_manifest.json
├── document_cards.json
├── section_cards.json
├── document_embeddings.npy
└── section_embeddings.npy
```

运行时在内存中进行矩阵余弦相似度计算即可。

Atlas快照必须记录：

```python
class AtlasSnapshot(BaseModel):
    kb_id: str
    kb_snapshot_hash: str
    embedding_model_id: str

    document_count: int
    section_count: int

    builder_version: str
    created_at: str
```

知识库文档更新、Embedding模型更新或切块更新后，Atlas必须重建。

------

## 4.5 离线算法伪代码

```python
def build_corpus_atlas(documents, embedding_model):
    document_cards = []
    section_cards = []

    for document in documents:
        parsed = parse_markdown_structure(document.text)

        doc_card = DocumentCard(
            doc_id=stable_doc_id(document),
            file_id=document.file_id,
            title=parsed.title,
            year=extract_year(parsed.title),
            source_type=infer_source_type_from_title(parsed.title),
            heading_paths=parsed.level_1_and_2_paths,
            table_titles=parsed.table_titles,
            routing_text=join_nonempty([
                parsed.title,
                *parsed.level_1_and_2_paths,
                *parsed.table_titles,
            ]),
        )
        document_cards.append(doc_card)

        for section in parsed.sections:
            section_cards.append(
                SectionCard(
                    section_id=stable_section_id(document, section),
                    doc_id=doc_card.doc_id,
                    file_id=document.file_id,
                    heading_path=section.heading_path,
                    lead_text=section.lead_text[:400],
                    table_titles=section.table_titles,
                    chunk_ids=section.chunk_ids,
                    routing_text=join_nonempty([
                        parsed.title,
                        section.heading_path,
                        section.lead_text[:400],
                        *section.table_titles,
                    ]),
                )
            )

    doc_vectors = normalize(
        embedding_model.embed([x.routing_text for x in document_cards])
    )
    section_vectors = normalize(
        embedding_model.embed([x.routing_text for x in section_cards])
    )

    save_atlas(
        document_cards,
        section_cards,
        doc_vectors,
        section_vectors,
    )
```

------

# 五、在线算法第一步：病例多视图路由

## 5.1 不生成药物×疾病笛卡尔积

DA-PRIM不为下列组合逐一构造查询：

[
|P|\times|M|
]

也不枚举：

[
\frac{|P|(|P|-1)}{2}
]

其中 (P) 为方案要素，(M) 为患者事实。

系统只让每个原始病例对象独立接触语料地图，再根据它们在语料中的共同落点产生稀疏候选。

------

## 5.2 路由视图

设V4.2已经产生：

[
P={p_1,\ldots,p_m}
]

和：

[
M={m_1,\ldots,m_n}
]

构造四类路由视图。

### 视图A：完整病例视图

```text
原始病例全文的清理版本
```

保留诊断、症状、药物、剂量和用户问题，但去掉机构编号等无关模板字段。

记为：

[
v_{\mathrm{case}}
]

### 视图B：完整治疗方案视图

```text
全部PlanAnchor按原文顺序拼接
```

记为：

[
v_{\mathrm{regimen}}
]

### 视图C：逐方案要素视图

每个PlanAnchor单独作为一个视图：

[
V_P={v_{p_1},\ldots,v_{p_m}}
]

### 视图D：逐患者事实视图

每个PatientModifier的原始source span单独作为一个视图：

[
V_M={v_{m_1},\ldots,v_{m_n}}
]

这里不需要为患者事实增加“体位性低血压”“高跌倒风险”等模型诊断标签。使用病例原文即可。

------

## 5.3 文档路由得分

对每个视图 (v)，在29个DocumentCard中计算排名：

[
r_v(d)
]

使用类型归一化的RRF聚合：

[
S_{\mathrm{case}}(d)=
\frac{1}{60+r_{\mathrm{case}}(d)}
+
\frac{1}{60+r_{\mathrm{regimen}}(d)}
+
\frac{1}{|P|}
\sum_{p\in P}
\frac{1}{60+r_p(d)}
+
\frac{1}{|M|}
\sum_{m\in M}
\frac{1}{60+r_m(d)}
]

若 (P) 或 (M) 为空，对应项省略。

### 为什么按对象类型归一化

如果直接累加全部PlanAnchor和Modifier，药物数量多的病例会天然提高方案视图权重，患者事实数量多的病例则会提高Modifier权重。分别除以 (|P|) 和 (|M|) 后，四类路由信息具有近似稳定的总贡献：

- 整体病例；
- 整体方案；
- 单项方案；
- 患者修饰因素。

这是一种结构归一，不涉及医学权重。

------

## 5.4 初始候选文档地图

取：

[
D_{\mathrm{case}}=\operatorname{TopK}
\left(
S_{\mathrm{case}},K_{\mathrm{doc}}
\right)
]

建议开发初始值：

```text
K_doc = 6
```

选择6篇的原因是：

- 约占29篇文档的20%；
- 足以覆盖常见的主病指南、老年用药标准和共病指南；
- 文档内局部检索仍然可控；
- 不会像硬选1至2篇那样造成路由错误不可恢复。

最终数值应在开发集冻结。

------

## 5.5 给Agent显示的文档地图

每轮不需要重复完整Atlas。首次Agent调用时显示：

```text
【本地知识库候选来源地图】

[D03] 老年人良性前列腺增生症/下尿路症状药物治疗共识
匹配病例对象：
- [PE001] 特拉唑嗪 2 mg 每晚一次
- [PE002] 非那雄胺 5 mg 每日一次
- [PM004] 卧位150/90，立位110/70并伴头晕

高匹配章节：
- α1受体阻滞剂
- 老年患者应用注意事项
- 体位性低血压
- 5α还原酶抑制剂

[D07] 中国老年高血压管理指南
匹配病例对象：
- [PE001] 特拉唑嗪
- [PE003] 氨氯地平
- [PM004] 卧立位血压变化
- [PM003] 既往晕倒并发生骨折

高匹配章节：
- 降压药物选择
- 直立性低血压
- 跌倒风险

说明：
这些来源只是根据本地语料结构产生的候选导航，
不表示文档必然支持或反对当前方案，也不限制你检索其它文档。
```

这个地图不计为Evidence，不能被Agent引用为临床依据。

FinSAgent将“模型先验与本地语料结构不匹配”定义为检索失败来源，并通过轻量语料视图辅助查询规划；DA-PRIM采用的是规模更小、更可控的文档和章节地图，而不引入其多Agent结构。([arXiv](https://arxiv.org/abs/2607.18102))

------

# 六、在线算法第二步：生成语料诱导的调查机会

这是解决“Agent不知道关系便不会查询”问题的核心。

## 6.1 节点—章节路由

对每个病例节点：

[
N=P\cup M
]

分别在SectionCard中获取Top-(R_s)：

```text
R_s = 3
```

记节点 (n) 对章节 (s) 的排名为：

[
r_n(s)
]

------

## 6.2 机会生成条件

对每个SectionCard (s)，收集命中该章节的病例节点集合：

[
N_s=
{n\in N\mid r_n(s)\le R_s}
]

只有满足以下任一条件时才生成调查机会：

### 类型A：方案—患者事实机会

[
|N_s\cap P|\ge1
\quad\land\quad
|N_s\cap M|\ge1
]

例如：

```text
特拉唑嗪
＋卧立位血压下降
＋既往晕厥骨折
→ 共同命中“体位性低血压”章节
```

### 类型B：多方案机会

[
|N_s\cap P|\ge2
]

例如：

```text
特拉唑嗪
＋氨氯地平
→ 共同命中“联合降压与直立性低血压”章节
```

只有患者事实、没有任何PlanAnchor的共同命中，不生成用药调查机会。

------

## 6.3 调查机会得分

对章节 (s) 的节点集合 (N_s)，定义：

[
O(s)=
\log(1+|N_s|)
\cdot
\frac{|N_s|}
{\sum_{n\in N_s}(60+r_n(s))}
]

该得分同时奖励：

- 多个病例对象共同命中；
- 每个对象都具有较高章节排名；
- 多因素超边，而非单一对象匹配。

为了避免一个章节包含过多泛化对象，每个机会最多保留得分最高的4个节点。

同一文档下语义近似或父子标题重复的机会，只保留得分更高者。

最终显示：

```text
L_opportunity = 3
```

个机会。

------

## 6.4 调查机会数据结构

```python
class RetrievalOpportunity(BaseModel):
    opportunity_id: str

    plan_ids: list[str]
    modifier_ids: list[str]

    doc_id: str
    section_id: str

    document_title: str
    heading_path: str

    score: float
    cue_text: str
```

示例：

```text
[OP001]
共同路由对象：
- [PE001] 特拉唑嗪2 mg每晚一次
- [PM004] 卧位150/90，立位110/70伴头晕
- [PM003] 既往起床晕倒并发生骨折

共同命中：
- 老年人BPH/LUTS药物治疗共识
- α1受体阻滞剂 > 老年患者应用注意事项 > 体位性低血压

该提示仅表示这些病例对象在本地语料中共同指向同一章节，
不表示它们之间已存在禁忌、相互作用或因果关系。
```

------

## 6.5 与RelationInvestigation的衔接

在V4.2的 `search_review_kb` 中增加可选字段：

```python
opportunity_id: str | None = None
```

Agent若采用该机会，可调用：

```python
search_review_kb(
    query_text="已有症状性体位性低血压并有跌倒史的老年患者使用特拉唑嗪的禁忌和处理建议",
    opportunity_id="OP001",
    relation_question=(
        "患者的卧立位血压变化和晕厥骨折史，"
        "是否改变特拉唑嗪的适用性和监测要求？"
    ),
    focus_plan_ids=["PE001"],
    focus_modifier_ids=["PM003", "PM004"],
    reason="调查语料地图提示的患者—方案联系",
)
```

有效 `opportunity_id` 只做三件事：

1. 写入trace；
2. 将机会中的节点作为默认focus对象；
3. 将机会指向的文档加入本次软路由候选集。

它不做：

- 自动创建临床结论；
- 自动判定RelationInvestigation成立；
- 自动认定该文档适用于患者；
- 强制Agent必须调查；
- 阻止Agent检索其它来源。

------

# 七、在线算法第三步：全局＋文档内双路径检索

A-RAG的核心观点是向Agent提供不同粒度的检索接口，而不把Agent限制在固定工作流中。DA-PRIM保持一个简单Agent loop，但让现有search工具内部具备文档级和chunk级两个检索入口。([arXiv](https://arxiv.org/abs/2602.03442))

------

## 7.1 查询特异的文档路由

Agent第 (t) 次提出查询：

[
q_t
]

若提供focus节点，则构造仅用于文档路由的文本：

```text
Agent原始查询
＋focus PlanAnchor原文
＋focus PatientModifier原文
```

记为：

[
q_t^{\mathrm{route}}
]

对DocumentCard计算查询文档排名：

[
r_{q_t}(d)
]

将其与病例级文档先验合并：

[
S_t(d)=
\frac{1}{60+\operatorname{rank}*{\mathrm{case}}(d)}
+
\frac{1}{60+r*{q_t}(d)}
]

若Agent提供有效 `opportunity_id`，其对应文档进入候选文档集合，但不会得到无限大分数，也不会排除其它文档。

取前6篇：

[
D_t=\operatorname{Top6}(S_t)
]

------

## 7.2 全局分支

在全部chunk中执行：

```text
global vector search top 10
```

得到全局排名：

[
L_g=(c_1,c_2,\ldots,c_{10})
]

保留全局分支非常重要。文档路由器可能选错文档，若只在候选文档中检索，错误将无法恢复。

------

## 7.3 文档内分支

对每个候选文档 (d\in D_t)，执行带 `file_id` 过滤的向量检索：

```text
per-document top 2
```

得到：

[
L_d=(c_{d,1},c_{d,2})
]

六篇文档最多产生12个局部候选。

Yuxi当前知识库工具已经支持文件范围概念，但为了实验可复现性，建议在Milvus直接检索接口中加入严格的：

```python
filter_file_ids: list[str]
```

使用稳定 `file_id`，不要依赖模糊文件名。Yuxi原始Agent的检索由模型驱动，查询工具本身并不固定执行单次检索，因此这一后端改造不会改变其Agentic性质。

------

## 7.4 构建文档路由排名

各候选文档的局部得分不能直接横向比较，因为它们来自不同的过滤集合。因此采用轮转式合并：

```text
候选文档第1名的Top-1 chunk
候选文档第2名的Top-1 chunk
……
候选文档第6名的Top-1 chunk
候选文档第1名的Top-2 chunk
……
候选文档第6名的Top-2 chunk
```

若chunk (c_{d,j}) 来自文档排名 (r_t(d))，定义其路由排名：

[
r_r(c_{d,j})=
(j-1)\cdot K_{\mathrm{doc}}
+
r_t(d)
]

其中：

[
K_{\mathrm{doc}}=6
]

这会自然避免长文档在局部分支中占满所有位置，又不需要人为判断哪些来源更权威。

------

## 7.5 全局与局部融合

对去重后的候选chunk (c)，使用：

[
S_{\mathrm{chunk}}(c)=
\mathbb{I}(c\in L_g)
\frac{1}{60+r_g(c)}
+
\mathbb{I}(c\in L_r)
\frac{1}{60+r_r(c)}
]

其中：

- (r_g(c)) 为全局排名；
- (r_r(c)) 为路由分支排名；
- 同时被全局和局部分支召回的chunk会获得双重加分；
- 只被路由分支发现的专题chunk仍能进入最终结果；
- 只被全局分支发现的意外相关文档也不会丢失。

最终仍然返回：

```text
Top-5 Evidence Cards
```

因此主Agent看到的Evidence数量、最终上下文大小和baseline一致。

------

## 7.6 工具返回格式

```json
{
  "query_id": "Q003",
  "relation_id": "RI001",
  "opportunity_id": "OP001",

  "route_documents": [
    {
      "doc_id": "D03",
      "title": "老年人BPH/LUTS药物治疗共识",
      "route_rank": 1,
      "matched_headings": [
        "α1受体阻滞剂",
        "体位性低血压"
      ]
    }
  ],

  "evidence": [
    {
      "evidence_id": "EV012",
      "source_document": "老年人BPH/LUTS药物治疗共识",
      "chunk_index": 12,
      "retrieval_paths": [
        "global",
        "routed_local"
      ],
      "excerpt": "已有体位性低血压或血压过低的老年人应禁用α1受体阻滞剂……"
    }
  ],

  "remaining_search_calls": 5,
  "remaining_open_calls": 2
}
```

Agent可以直接读取Evidence全文窗口，现有V4.2的Evidence Card和open逻辑继续保留。

------

# 八、完整在线算法伪代码

```python
async def run_da_prim(case_text, context):
    # V4.2已有步骤
    plan_anchors = extract_plan_anchors(case_text)
    patient_modifiers = extract_patient_modifiers(case_text)

    # 新增：病例级语料路由
    case_views = build_case_views(
        case_text=case_text,
        plan_anchors=plan_anchors,
        patient_modifiers=patient_modifiers,
    )

    case_doc_ranking = route_case_to_documents(
        views=case_views,
        atlas=context.corpus_atlas,
    )

    section_matches = route_nodes_to_sections(
        nodes=[*plan_anchors, *patient_modifiers],
        atlas=context.corpus_atlas,
        top_r=3,
    )

    opportunities = build_retrieval_opportunities(
        section_matches=section_matches,
        max_opportunities=3,
    )

    state = initialize_v42_prim_state(
        case_text=case_text,
        plan_anchors=plan_anchors,
        patient_modifiers=patient_modifiers,
        case_doc_ranking=case_doc_ranking,
        opportunities=opportunities,
    )

    while True:
        action = await agent_decide(state)

        if action.type == "final_answer":
            draft = action.content
            break

        if action.type == "open_evidence":
            opened = await open_review_evidence(
                evidence_id=action.evidence_id,
                state=state,
            )
            state = update_state(state, opened)
            continue

        if action.type == "search":
            query_route_text = build_query_route_text(
                query_text=action.query_text,
                focus_plan_ids=action.focus_plan_ids,
                focus_modifier_ids=action.focus_modifier_ids,
                state=state,
            )

            query_doc_ranking = route_query_to_documents(
                query_route_text,
                atlas=context.corpus_atlas,
            )

            routed_docs = combine_case_and_query_routes(
                case_doc_ranking=state.case_doc_ranking,
                query_doc_ranking=query_doc_ranking,
                opportunity_id=action.opportunity_id,
                top_k=6,
            )

            global_results = await global_chunk_search(
                query=action.query_text,
                top_k=10,
            )

            local_results = []
            for doc in routed_docs:
                local_results.extend(
                    await document_filtered_search(
                        query=action.query_text,
                        file_id=doc.file_id,
                        top_k=2,
                    )
                )

            fused_results = fuse_global_and_routed_results(
                global_results=global_results,
                local_results=local_results,
                routed_docs=routed_docs,
                final_top_k=5,
            )

            tool_message = build_evidence_cards(
                fused_results,
                include_route_metadata=True,
            )

            state = update_v42_relation_memory(
                state=state,
                action=action,
                routed_docs=routed_docs,
                evidence=fused_results,
            )

            state.messages.append(tool_message)

    # 继续使用V4.2已有的一次PlanAnchor结构覆盖检查
    final_answer = await v42_bounded_coverage_reflection(
        draft=draft,
        state=state,
    )

    return final_answer, build_trace(state)
```

------

# 九、为什么这样设计

## 9.1 为什么先路由文档，再检索chunk

当前错误对象是“目标文档没有进入任何Top-K结果”。继续调chunk向量或让Agent重复生成查询，未必能解决文档层的竞争偏差。

文档卡提供的是：

```text
该问题应该优先进入哪一类指南？
```

chunk检索回答的是：

```text
在这些指南中，哪段原文最相关？
```

将两者分开，才能分别诊断：

- 文档选择错误；
- 文档内证据定位错误。

近期临床检索研究也强调将检索拆解为可单独消融和测量的阶段；一项BioNLP 2026工作通过多阶段临床检索相对强dense基线获得了22%至23%的相对召回增益。([ACL Anthology](https://aclanthology.org/2026.bionlp-1.53/))

## 9.2 为什么是软路由

硬路由意味着：

```text
路由器未选中的文档
→ 后续永远不可见
```

老年多重用药中，某个事实可能同时存在于：

- 专病指南；
- Beers标准；
- 老年高血压指南；
- 老年综合用药共识；
- 器官功能指南。

因此必须保留全局检索。DA-PRIM中的文档地图是候选先验，不能成为访问控制。

## 9.3 为什么不枚举临床关系

完整枚举会带来：

[
O(|P||M|+|P|^2)
]

级别的关系数量，并重新引入此前失败的固定审查矩阵。

DA-PRIM的复杂度主要为：

[
O(|P|+|M|)
]

因为每个病例对象只独立路由语料。只有两个或多个对象在同一章节中共同出现时，才产生少量调查机会。

这是一种语料驱动的稀疏候选生成，而非规则驱动的全组合检查。

## 9.4 为什么调查机会不能直接作为关系

“特拉唑嗪”和“体位性低血压”共同命中同一章节，只能证明：

> 本地语料中存在同时涉及二者的区域。

它不能证明：

- 药物禁忌；
- 药物导致该风险；
- 患者一定不适合使用；
- 应当停药；
- 应当换成某种药物。

这些结论仍需Agent读取Evidence后形成。

## 9.5 为什么不用第二个关系规划Agent

当前V4.2已经由主Agent自主提出relation_question。再增加一个关系规划Agent，会产生：

- 两套关系假设；
- 额外token；
- 关系冲突；
- 重新出现“哪个模型拥有语义决定权”的问题。

语料地图和调查机会由确定性检索产生，只向现有Agent提供线索，避免产生第二个临床推理主体。

## 9.6 为什么不增加在线Evidence审计器

此前失败已经显示，在线相关性、适用性、极性和Evidence合同会否定Agent已经理解的正确证据。DA-PRIM只审计：

- 路由了哪些文档；
- 检索了哪些文档；
- 哪些结果来自全局或局部路径；
- Agent是否采用某个调查机会。

证据是否真正支持结论，继续在离线评价中判断。

------

# 十、代码级接入

建议在V4.2目录旁新增：

```text
backend/package/yuxi/agents/buildin/medication_review_prim/
├── corpus_atlas/
│   ├── models.py
│   ├── builder.py
│   ├── loader.py
│   ├── router.py
│   └── opportunities.py
├── routed_retrieval.py
└── atlas_memory.py
```

V4.2已有的以下模块不需要重写：

- PlanAnchor抽取；
- PatientModifier抽取；
- RelationInvestigation；
- QueryRecord；
- Evidence Store；
- Evidence Card；
- open evidence；
- search/open预算；
- 一次PlanAnchor覆盖反思；
- Trace 5.0主体。

------

## 10.1 新增状态

```python
class AtlasRouteRecord(BaseModel):
    view_id: str
    view_type: Literal[
        "case",
        "regimen",
        "plan",
        "modifier",
        "agent_query",
    ]
    linked_node_ids: list[str]
    ranked_doc_ids: list[str]
    ranked_section_ids: list[str]


class RoutedSearchMetadata(BaseModel):
    case_route_doc_ids: list[str]
    query_route_doc_ids: list[str]
    effective_route_doc_ids: list[str]

    global_candidate_ids: list[str]
    local_candidate_ids: list[str]

    fused_evidence_ids: list[str]
    retrieval_path_by_evidence: dict[str, list[str]]


class DaPrimState(MedicationReviewPrimState, total=False):
    corpus_atlas_snapshot: dict
    atlas_route_records: list[AtlasRouteRecord]
    case_doc_ranking: list[str]

    retrieval_opportunities: list[RetrievalOpportunity]
    adopted_opportunity_ids: list[str]

    routed_search_metadata: list[RoutedSearchMetadata]
```

------

## 10.2 Context参数

主方法只需增加：

```python
@dataclass(kw_only=True)
class DaPrimContext(MedicationReviewPrimContext):
    corpus_atlas_path: str

    atlas_doc_top_k: int = 6
    atlas_section_top_r: int = 3
    max_retrieval_opportunities: int = 3

    global_candidate_top_k: int = 10
    per_document_candidate_top_k: int = 2

    final_evidence_top_k: int = 5
```

不增加：

- 疾病特殊规则；
- 药品类别规则；
- 关系类型开关；
- 数值阈值；
- 必查文档；
- 每种药物最低查询次数。

------

# 十一、主实验组

建议把方法实验压缩为四组。

| 组别            | 文档地图 | 双路径检索 | 调查机会 |
| --------------- | -------- | ---------- | -------- |
| B0：V4.2 Full   | 否       | 否         | 否       |
| M1：Atlas-Map   | 是       | 否         | 否       |
| M2：Atlas-Route | 是       | 是         | 否       |
| Full：DA-PRIM   | 是       | 是         | 是       |

### B0：V4.2 Full

保持当前纯向量Top-5和关系调查记忆。

### M1：Atlas-Map

Agent可以看到Top-6文档和高匹配章节，但实际search仍使用原有平坦向量Top-5。

检验：

> 仅向Agent提供语料地图，能否改善其查询规划？

### M2：Atlas-Route

加入全局＋候选文档内双路径检索，但不显示调查机会。

检验：

> 检索增益是否主要来自文档路由？

### Full：DA-PRIM

加入Top-3语料诱导调查机会。

检验：

> 调查机会能否帮助Agent发现其先验中没有主动想到的关系？

------

## 11.1 必要的候选池对照

双路径检索内部最多产生：

```text
全局10条
＋6篇文档×每篇2条
＝22条候选
```

因此应增加一个检索层对照：

> **Flat-Deep**：在全部chunk中直接召回Top-22，然后按当前相似度选Top-5。

Flat-Deep不需要运行全部端到端病例，可以在文档召回诊断集上完成。

若Atlas-Route优于Flat-Deep，才能说明收益来自结构化文档路由，而非简单扩大候选池。

------

# 十二、主方法之外的消融模块

以下功能不建议进入第一版主方法。

## 12.1 DocumentCard上的Dense＋BM25

DocumentCard主要由标题和章节名称构成，词法信号可能比正文chunk更强。可以比较：

```text
Doc-Dense
Doc-Dense + BM25 RRF
```

chunk层仍保持纯向量。

这是优先级最高的可选消融。

## 12.2 章节导航直接进入证据检索

主方法中的SectionCard只用于地图和调查机会。可选增强是：

```text
查询SectionCard
→ 定位章节
→ 在章节所属chunk中局部检索
```

这可以形成第三条检索路径，但会增加实现和候选融合复杂度，应在文档内检索效果仍不足时再增加。

## 12.3 来源多样性约束

可测试最终Top-5中每篇文档最多2条：

```text
max_chunks_per_document = 2
```

但它可能损害单文档病例，因此不进入主方法。

## 12.4 一次“未访问候选来源”软反思

Agent首次准备回答时，如果存在高排名但从未被检索的文档，可提示一次：

```text
以下高相关来源尚未被访问……
```

不得阻止回答。由于V4.2已经有一次方案要素覆盖反思，再增加一次检索反思可能提高成本，因此只作为消融。

## 12.5 Cross-encoder reranker

可用于候选22条到最终5条的重排，但会引入新的模型和延迟，不宜与文档路由首次实验同时加入。

## 12.6 短文档全文回退

医学多来源RAG中的HERO-QA对较短手册直接提供全文，对长文档使用分层检索，以减少短文档中的检索遗漏。([arXiv](https://arxiv.org/html/2605.29084v1))

你们的指南篇幅较长，全文回退可能产生较大token开销，建议只对非常短的标准或表格型文档进行后续消融。

## 12.7 LLM生成文档摘要

只在标题和章节路由效果仍不足时测试，不能与主实验一起加入。

------

# 十三、评价指标

## 13.1 先澄清Top-1

若一个病例有多篇金标准文档，定义：

# [ \operatorname{Recall@1}

\frac{
|\operatorname{Top1}\cap G|
}{
|G|
}
]

那么当 (|G|=2) 时，上限就是0.5。

因此需要同时报告：

# [ \operatorname{Hit@1}

\mathbb{I}(
\operatorname{Top1}\cap G\neq\varnothing
)
]

建议主文档指标包括：

- Doc Hit@1；
- Doc MRR；
- Doc Recall@3；
- Doc Recall@6；
- All-Doc@6。

------

## 13.2 Atlas路由指标

# [ \operatorname{RouterRecall@6}

\frac{
|D_{\mathrm{case}}@6\cap G|
}{
|G|
}
]

# [ \operatorname{RouterAll@6}

\mathbb{I}(G\subseteq D_{\mathrm{case}}@6)
]

它只评价文档地图，不受Agent查询影响。

------

## 13.3 检索分层指标

### 文档内条件召回

# [ \operatorname{WithinDocRecall}

P(
\text{金标准Evidence进入候选池}
\mid
\text{金标准文档已进入路由集}
)
]

### 路由独有增益

# [ \operatorname{RoutedUniqueGain}

\frac{
|\text{仅由局部分支召回的金标准文档}|
}{
|G|
}
]

### 全局逃生率

# [ \operatorname{GlobalEscapeRate}

\frac{
|\text{仅由全局分支召回的金标准文档}|
}{
|\text{全部成功召回的金标准文档}|
}
]

若该值明显大于零，说明保留全局分支是必要的。

------

## 13.4 调查机会指标

在复杂病例子集上，为金标准Finding标注依赖的方案要素和患者事实。

### 机会召回率

# [ \operatorname{OpportunityRecall}

\frac{
\text{至少有一个机会覆盖其相关节点的金标准关系Finding数}
}{
\text{全部关系Finding数}
}
]

### 机会采用率

# [ \operatorname{OpportunityAdoption}

\frac{
\text{被Agent用于查询的机会数}
}{
\text{展示给Agent的机会数}
}
]

### 机会证据收益

# [ \operatorname{OpportunityEvidenceGain}

\frac{
\text{采用机会后新增的金标准文档或Evidence数}
}{
\text{采用机会产生的查询数}
}
]

### 机会诱发误报率

# [ \operatorname{OpportunityFalseFindingRate}

\frac{
\text{由机会触发但不属于金标准的无依据关系Finding数}
}{
\text{采用机会产生的全部关系Finding数}
}
]

------

## 13.5 最终回答指标

继续使用已有的：

- Finding Micro-F1；
- Finding Macro-F1；
- 合理项F1；
- 不合理/需调整项F1；
- 属性完整率；
- 完整Finding率；
- 完整病例正确率；
- Citation蕴含率；
- Citation患者适用率；
- 无来源具体建议率。

另增加：

# [ \operatorname{PostRetrievalSurvival}

\frac{
\text{最终答案保留的已召回金标准事实数}
}{
\text{检索结果中已经存在的金标准事实数}
}
]

文档召回提高、最终答案没有提高时，该指标可以判断问题是否转移到了证据利用阶段。

------

# 十四、开发阶段的参数建议

下面是用于启动实验的参数，不是不可修改的临床规则。

| 参数                  | 初始值   |
| --------------------- | -------- |
| 文档候选数            | 6        |
| 每节点章节候选数      | 3        |
| 调查机会数            | 3        |
| 全局chunk候选数       | 10       |
| 每候选文档局部chunk数 | 2        |
| 最终Evidence Card数   | 5        |
| RRF常数               | 60       |
| Agent最大search次数   | 沿用V4.2 |
| Agent最大open次数     | 沿用V4.2 |

这些参数只能在开发集调整，锁定测试前冻结。

------

# 十五、实施顺序与退出条件

## P0：Atlas检索单测

先不接Agent。

输入现有病例的：

- 原始病例；
- PlanAnchor；
- PatientModifier。

直接评价29个DocumentCard的：

- Hit@1；
- MRR；
- Recall@6；
- All-Doc@6。

退出条件建议为：

```text
开发集Router Recall@6接近或超过95%
```

如果DocCard路由本身仍只有85%，不应继续实现Agent集成，应先检查：

- 文档标题和章节是否被正确提取；
- 金标准文档是否唯一；
- 文档卡是否缺少重要章节标题；
- Embedding是否适合文档级短文本。

## P1：Atlas-Map

只向V4.2 Agent显示文档地图，检索后端不变。

若查询和文档召回没有变化，说明Agent没有利用地图，需改进地图呈现方式，而不是立即增加更多模块。

## P2：Atlas-Route

实现全局＋文档内双路径检索。

先比较：

```text
Flat Top-5
Flat-Deep Top-22→5
Atlas-Route Top-22→5
```

只有Atlas-Route优于Flat-Deep，才继续。

## P3：调查机会

生成Top-3机会并接入RelationInvestigation。

若文档召回提高，但关系Finding没有提高，调查机会模块应删除，论文只保留文档路由方法。

## P4：全量测试

锁定：

- Atlas快照；
- Prompt；
- 参数；
- 模型；
  -知识库；
- search/open预算。

运行：

```text
B0
M1
M2
Full
```

------

# 十六、论文中的方法贡献表述

完成验证后，方法贡献可以写为：

> 针对老年多重用药审查中Agent查询规划依赖模型先验、平坦chunk检索难以稳定定位专题指南的问题，本文提出文档地图引导的患者—治疗方案调查记忆RAG方法。该方法从本地指南的标题和章节结构构建语料地图，利用患者事实与治疗方案要素执行病例级软文档路由，并通过全局检索与候选文档内检索的融合提高来源覆盖。进一步地，当多个病例对象共同指向同一章节时，系统生成非结论性的调查机会，供Agent自主建立关系问题和开展后续检索。程序不预设临床关系，不判断Evidence极性或患者适用性，最终临床解释仍由Agent结合原始证据完成。

这套设计相较V4.2增加的是**语料侧导航和未知关系发现入口**；相较此前失败的固定审查流程，它没有恢复关系笛卡尔积、Evidence合同或在线医学裁决。它的主干足够聚焦，文档路由、关系机会和最终答案三个层次也能够分别进行消融和归因。