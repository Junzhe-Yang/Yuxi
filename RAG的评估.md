RAG 的检索评估可以拆成三个问题。

第一，相关文档或文本块有没有进入候选集合。第二，找回的相关证据是否完整。第三，相关证据在排序中是否足够靠前。传统信息检索通常使用 Recall、Precision、MAP、MRR、nDCG 等指标，LlamaIndex 当前的检索评估模块也直接提供 Hit Rate、MRR、Precision、Recall、AP 和 nDCG。BEIR 等检索基准则常用 nDCG 与 Recall 评价排序质量和深层候选召回。([Developer Documentation](https://developers.llamaindex.ai/python/examples/evaluation/retrieval/retriever_eval/))

下面先统一符号。对查询 (q)：

[
G_q={\text{该查询的全部金标准相关文档或块}}
]

检索器返回一个有序列表：

[
L_q=(d_1,d_2,\ldots,d_K)
]

前 (k) 个结果组成集合：

[
T_q(k)={d_1,\ldots,d_k}
]

二值相关性定义为：

[
\operatorname{rel}_q(i)=
\begin{cases}
1,&d_i\in G_q\
0,&d_i\notin G_q
\end{cases}
]

同一套公式既能用于文档级，也能用于块级。区别只在于 (d_i) 表示文档 ID，还是 chunk ID。

## 一、Recall@k

Recall@k 是最直接的召回覆盖指标：

# [ \operatorname{Recall@k}(q)

\frac{|T_q(k)\cap G_q|}{|G_q|}
]

它回答的问题是：

> 金标准相关文档或块中，有多少比例出现在前 (k) 个结果里？

例如，一个问题需要三个相关块 (A,B,C)，检索前五名找到了 (A,B)，那么：

[
\operatorname{Recall@5}=\frac{2}{3}\approx0.667
]

Recall@k 不考虑相关块在前 (k) 中的具体位置。相关块排第 1 和第 5，对 Recall@5 的贡献相同。Precision 和 Recall 最初属于集合指标，在有序检索中通常固定一个 top (k) 截断位置进行计算。([Stanford NLP Group](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html))

### 文档级 Recall@k

假设每个块都有父文档映射：

[
\operatorname{parent}(d_i)=\text{块 }d_i\text{ 所属的文档}
]

将检索块映射成父文档集合：

[
T_q^{doc}(k)={\operatorname{parent}(d_i)\mid i\leq k}
]

那么：

# [ \operatorname{DocRecall@k}(q)

\frac{|T_q^{doc}(k)\cap G_q^{doc}|}{|G_q^{doc}|}
]

文档级指标适合判断“来源文档有没有找对”。它的粒度较粗。检索到正确文档中的无关章节，也会得到文档级命中。

### 块级 Recall@k

# [ \operatorname{ChunkRecall@k}(q)

\frac{|T_q^{chunk}(k)\cap G_q^{chunk}|}{|G_q^{chunk}|}
]

它能更准确地评价证据位置，但会受到切块方法影响。更换 chunk size、overlap 或分句规则后，原来的 chunk ID 往往失效。

因此，实际实验中最好同时报告：

[
\text{DocRecall@k}
\quad\text{和}\quad
\text{ChunkRecall@k}
]

前者反映来源覆盖，后者反映具体证据覆盖。

## 二、Hit Rate@k 或 Success@k

单个查询的 Hit@k 定义为：

# [ \operatorname{Hit@k}(q)

\mathbb{I}\bigl(T_q(k)\cap G_q\neq\varnothing\bigr)
]

其中 (\mathbb{I}) 是指示函数。只要前 (k) 个结果中至少出现一个相关项，得分就是 1，否则为 0。

整个测试集的 Hit Rate@k 为：

# [ \operatorname{HitRate@k}

\frac{1}{|Q|}
\sum_{q\in Q}\operatorname{Hit@k}(q)
]

例如 100 个问题中有 82 个问题在 top 5 内至少出现一个相关块：

[
\operatorname{HitRate@5}=0.82
]

Hit Rate 衡量“能否碰到至少一条有用证据”。IBM 对 RAG Hit Rate 的定义同样是检查检索上下文中是否至少包含一个相关上下文。([IBM](https://www.ibm.com/docs/en/watsonx/saas?topic=metrics-hit-rate&utm_source=chatgpt.com))

它适合单跳问答，例如每个问题只需要一个块。多跳问题中，它可能明显高估系统效果。一个问题需要三个证据，系统只找到其中一个，Hit@k 仍然等于 1。

## 三、Complete Evidence Success@k

对于需要多条证据的问题，可以增加一个更严格的派生指标：

# [ \operatorname{CompleteHit@k}(q)

\mathbb{I}\bigl(G_q\subseteq T_q(k)\bigr)
]

测试集平均值为：

# [ \operatorname{CompleteEvidenceSuccess@k}

\frac{1}{|Q|}
\sum_{q\in Q}\operatorname{CompleteHit@k}(q)
]

它回答：

> 前 (k) 个结果是否覆盖了该问题所需的全部金标准证据？

假设一个问题需要 (A,B,C)，系统只找到了 (A,B)：

[
\operatorname{Hit@k}=1
]

[
\operatorname{Recall@k}=\frac{2}{3}
]

[
\operatorname{CompleteHit@k}=0
]

这个指标没有完全统一的名称，也常被称为 All Evidence Recall、All Relevant Hit 或 Complete Retrieval Rate。对于多跳 QA、法规问答、跨文档推理，它很有解释力。

## 四、Precision@k

这里的 Precision 指检索精度，与最终答案准确率无关。

# [ \operatorname{Precision@k}(q)

# \frac{|T_q(k)\cap G_q|}{k}

\frac{1}{k}\sum_{i=1}^{k}\operatorname{rel}_q(i)
]

如果前五个块中有两个相关块：

[
\operatorname{Precision@5}=\frac{2}{5}=0.4
]

Recall@k 可以通过不断增大 (k) 来提高，Precision@k 会揭示随之进入上下文的噪声量。

两个系统都达到 Recall@10 (=1) 时：

- 系统 A 的十个块中有八个相关块，Precision@10 (=0.8)
- 系统 B 的十个块中有两个相关块，Precision@10 (=0.2)

系统 B 会向生成模型输入更多无关内容，增加上下文长度和证据冲突。

### F1@k

可以把 Recall@k 和 Precision@k 合并：

# [ F1@k

\frac{2\cdot\operatorname{Precision@k}\cdot\operatorname{Recall@k}}
{\operatorname{Precision@k}+\operatorname{Recall@k}}
]

它适合需要在覆盖率与上下文噪声之间取得平衡的场景。不过 F1@k 会隐藏两项原始指标的差异，建议同时保留 Precision@k 和 Recall@k。

## 五、MRR@k

MRR 全称 Mean Reciprocal Rank，关注首个相关结果出现得多早。

对查询 (q)，令第一个相关结果的位置为：

[
r_q=\min{i\mid \operatorname{rel}_q(i)=1}
]

则 Reciprocal Rank 为：

[
\operatorname{RR}(q)=
\begin{cases}
\frac{1}{r_q},&\text{存在相关结果}\
0,&\text{没有相关结果}
\end{cases}
]

MRR 是所有查询 RR 的平均：

# [ \operatorname{MRR}

\frac{1}{|Q|}
\sum_{q\in Q}\operatorname{RR}(q)
]

例如：

- 首个相关块排第 1，RR (=1)
- 首个相关块排第 2，RR (=0.5)
- 首个相关块排第 5，RR (=0.2)
- 没有找到相关块，RR (=0)

MRR 强调首个命中的位置，后续相关结果不会继续提高分数。它适合每个查询只需要一条主要证据，或生成器倾向优先使用靠前上下文的场景。TREC 检索评估中也将 MRR 定义为每个查询首个相关结果排名倒数的平均值。([TREC](https://trec.nist.gov/pubs/trec34/papers/DUTH.tot.pdf?utm_source=chatgpt.com))

MRR 与用于提高检索多样性的 MMR，也就是 Maximal Marginal Relevance，是两个不同概念。

## 六、Average Precision 和 MAP

MRR 只看第一个相关项。Average Precision 会考虑所有相关项出现的位置。

首先计算每个位置的 Precision@i：

[
P@i=\frac{\sum_{j=1}^{i}\operatorname{rel}_q(j)}{i}
]

单个查询的 AP@k 可以定义为：

# [ \operatorname{AP@k}(q)

\frac{1}{|G_q|}
\sum_{i=1}^{k}
P@i\cdot\operatorname{rel}_q(i)
]

只有当第 (i) 个结果相关时，该位置的 Precision@i 才会被计入。

MAP 是所有查询 AP 的平均值：

# [ \operatorname{MAP@k}

\frac{1}{|Q|}
\sum_{q\in Q}\operatorname{AP@k}(q)
]

假设金标准相关块为 (A,B,C)，检索结果为：

[
[X,A,Y,B,Z]
]

相关性序列为：

[
[0,1,0,1,0]
]

相关结果出现在第 2 和第 4 位：

[
P@2=\frac{1}{2}
]

[
P@4=\frac{2}{4}=\frac{1}{2}
]

所以：

# [ AP@5

# \frac{P@2+P@4+0}{3}

\frac{1}{3}
]

未被找回的 (C) 对 AP 的贡献为 0。Stanford 的信息检索教材也将 AP 描述为每次检索到相关文档时 Precision 值的平均，并将未检索到的相关文档贡献记为 0，MAP 则在查询集合上继续求平均。([Stanford NLP Group](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf))

AP 和 MAP 同时奖励：

1. 找回更多相关项
2. 将相关项排在更前面

需要留意不同库的 AP@k 实现。有些使用 (|G_q|) 作为分母，有些使用 (\min(|G_q|,k))，还有些只除以 top (k) 中实际找到的相关项数量。实验报告中应明确公式。

## 七、nDCG@k

nDCG 适合相关性存在等级的情况。例如：

- 0，完全无关
- 1，部分相关
- 2，相关
- 3，直接包含完整答案证据

令第 (i) 个结果的相关性等级为 (g_i)，常见 DCG 定义为：

# [ \operatorname{DCG@k}

\sum_{i=1}^{k}
\frac{2^{g_i}-1}{\log_2(i+1)}
]

相关性越高，增益越大。位置越靠后，增益经过对数折扣。

再把所有候选项按照真实相关性从高到低排列，得到理想排序的：

[
\operatorname{IDCG@k}
]

最后：

# [ \operatorname{nDCG@k}

\frac{\operatorname{DCG@k}}
{\operatorname{IDCG@k}}
]

取值一般位于 0 到 1。完美排序为 1。

nDCG 同样可以使用二值相关性。BEIR 选择 nDCG@10 作为主要指标，原因之一是它能同时处理二值与分级相关性，并考虑排序位置。

nDCG 的优势包括：

- 允许“部分相关”和“高度相关”获得不同分数
- 对前排结果赋予更高权重
- 可以比较不同查询，因为使用 IDCG 进行了归一化

它仍然只评价截断位置 (k) 以内的排序。需要深层候选召回时，应配合 Recall@50、Recall@100 等指标。

## 八、R-Precision

对于查询 (q)，令相关项总数为：

[
R_q=|G_q|
]

R-Precision 定义为前 (R_q) 个结果的 Precision：

# [ \operatorname{RPrec}(q)

\operatorname{Precision@}R_q
]

例如一个查询共有三个相关块，检查前 3 个结果。如果其中一个相关：

[
\operatorname{RPrec}=\frac{1}{3}
]

它会根据每个查询的相关项数量自动调整截断位置。某些查询只有一个相关块，另一些查询有十个相关块时，R-Precision 比固定 Precision@10 更容易进行查询间比较。Stanford 的定义同样是在存在 (|Rel|) 个已知相关文档时，检查前 (|Rel|) 个结果。([Stanford NLP Group](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf))

它依赖较完整的相关性标注。金标准只记录一个来源块时，R-Precision 的含义会退化为首位命中情况。

## 九、RAG 专用的 Context Recall

传统 Recall@k 需要知道正确文档或 chunk ID。实际 RAG 数据中，经常只有标准回答，没有完整的 gold chunk 列表。RAGAS 提供了一种基于事实声明的 Context Recall。

首先将参考回答拆成若干独立 claim：

[
C_q={c_1,c_2,\ldots,c_m}
]

然后判断每个 claim 能否由检索上下文支持：

[
s_j=
\begin{cases}
1,&c_j\text{ 能由检索上下文支持}\
0,&\text{无法支持}
\end{cases}
]

最终：

# [ \operatorname{ContextRecall}

\frac{\sum_{j=1}^{m}s_j}{m}
]

RAGAS 当前文档给出的定义正是“参考回答中被检索上下文支持的 claim 数量，除以参考回答中的 claim 总数”。reference 在这里充当证据清单的代理，评分对象仍是检索上下文。([Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/))

这种方法适合以下情况：

- 切块方案频繁变化
- 多个不同块都能支持同一事实
- 检索结果经过摘要或压缩
- gold chunk ID 无法直接对应

它会受到参考回答完整度和裁判模型判断差异的影响。建议在一部分人工标注数据上验证 LLM judge 与人工判断的一致程度。

## 十、ID-Based Context Recall

当数据集中已经保存 reference chunk ID 时，可以直接计算：

# [ \operatorname{IDContextRecall}

\frac{
|\text{retrieved context IDs}\cap
\text{reference context IDs}|
}{
|\text{reference context IDs}|
}
]

这与标准 Chunk Recall 本质相同，优点是计算快、结果可重复、无需 LLM 判断。RAGAS 当前也提供 IDBasedContextRecall，并允许字符串或整数 ID。([Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/))

它的主要限制来自切块版本绑定。相同文本使用新的 chunk size 重建索引后，旧 ID 很可能无法继续使用。

较好的 gold 标注方式是同时保存：

[
(\text{document ID},\text{字符起点},\text{字符终点},\text{claim ID})
]

这样可以把新旧切块映射到同一证据区间。

## 十一、Context Precision

RAGAS 的 Context Precision 是一种与 AP 相似的排序指标。令第 (i) 个检索块的相关性判断为 (v_i\in{0,1})：

# [ \operatorname{ContextPrecision@K}

\frac{
\sum_{i=1}^{K}
P@i\cdot v_i
}{
\sum_{i=1}^{K}v_i
}
]

它奖励相关块出现在更靠前的位置。RAGAS 可以通过参考回答、系统回答或 reference contexts 判断每个块是否相关。([Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/))

它与标准 AP 的一个重要差异在分母：

标准 AP 常用金标准相关项总数 (|G_q|)。

RAGAS Context Precision 使用检索结果中被判定为相关的项数 (\sum v_i)。

因此，它主要反映已找回相关块的排序情况，对遗漏的 gold 块惩罚较弱。使用时最好与 Context Recall 同时报告。

## 十二、Context Entity Recall

实体密集型任务还可以计算实体召回率。令：

[
E_{ref}=\text{参考证据中的实体集合}
]

[
E_{ret}=\text{检索上下文中的实体集合}
]

则：

# [ \operatorname{EntityRecall}

\frac{|E_{ref}\cap E_{ret}|}{|E_{ref}|}
]

例如标准证据涉及“药物 A、药物 B、CYP3A4、肝功能不全”，检索上下文只覆盖前三个实体：

[
\operatorname{EntityRecall}=\frac{3}{4}=0.75
]

RAGAS 将该指标称为 Context Entity Recall，适合历史事实、旅游、医学知识、人物关系等实体覆盖很重要的场景。([Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall/?utm_source=chatgpt.com))

它无法判断实体之间的关系是否正确。例如上下文同时出现药物 A 和药物 B，却没有包含两者相互作用的事实，实体召回仍然可能很高。

## 十三、用一个例子比较所有主要指标

设金标准相关块为：

[
G={A,B,C}
]

系统返回：

| 排名 | 检索块 | 是否相关 |
| ---- | ------ | -------- |
| 1    | X      | 0        |
| 2    | A      | 1        |
| 3    | Y      | 0        |
| 4    | B      | 1        |
| 5    | Z      | 0        |

那么：

[
\operatorname{Recall@5}=\frac{2}{3}=0.667
]

[
\operatorname{Hit@1}=0
]

[
\operatorname{Hit@3}=1
]

[
\operatorname{CompleteHit@5}=0
]

[
\operatorname{Precision@5}=\frac{2}{5}=0.4
]

# [ F1@5

\frac{2\times0.4\times0.667}{0.4+0.667}
\approx0.5
]

[
\operatorname{RR}=\frac{1}{2}=0.5
]

# [ \operatorname{AP@5}

# \frac{P@2+P@4}{3}

\frac{0.5+0.5}{3}
=0.333
]

# [ \operatorname{RPrec}

# P@3

\frac{1}{3}
]

使用二值相关性的 nDCG：

# [ DCG@5

\frac{1}{\log_2 3}
+
\frac{1}{\log_2 5}
\approx1.062
]

理想排序会把 (A,B,C) 放在前三名：

# [ IDCG@5

1+
\frac{1}{\log_2 3}
+
\frac{1}{\log_2 4}
\approx2.131
]

[
nDCG@5\approx\frac{1.062}{2.131}\approx0.498
]

如果采用 RAGAS 风格 Context Precision：

# [ \operatorname{ContextPrecision@5}

# \frac{P@2+P@4}{2}

0.5
]

这个例子能体现各指标观察的角度：

- Hit Rate 认为系统已经找到有用内容
- Recall 表明仍有三分之一证据缺失
- MRR 表明第一条有用证据排在第 2
- AP 和 nDCG 会评价整个排序
- CompleteHit 表明该问题尚未获得完整证据

## 十四、测试集上的聚合方式

### Macro Average

先计算每个查询的指标，再对查询求平均：

# [ \operatorname{MacroRecall@k}

\frac{1}{|Q|}
\sum_{q\in Q}\operatorname{Recall@k}(q)
]

每个查询权重相同。一个单跳问题和一个需要五个证据的多跳问题，各贡献一次。

### Micro Average

先累计所有查询的命中数量：

# [ \operatorname{MicroRecall@k}

\frac{
\sum_q|T_q(k)\cap G_q|
}{
\sum_q|G_q|
}
]

需要更多证据的查询权重更高。

常规报告可以将 Macro Recall 作为主指标，再补充 Micro Recall。测试集中单跳和多跳问题比例差异很大时，两者可能产生明显不同的结论。

对于没有任何 gold relevant item 的查询，需要提前制定规则。通常将这类查询从 Recall 计算中排除，并单独统计为无答案查询。

## 十五、常见评估陷阱

### 1. Gold 集合不完整

很多 QA 数据只记录了生成问题时使用的来源块。语料库中可能还有其他同样能回答问题的块。此时所谓 Chunk Recall 更准确地说是“标注来源块召回率”。

解决方法包括人工补充相关性标注、合并多个检索器结果后进行 pooled judging、使用 claim 支持判断。

### 2. Chunk ID 对切块方法过度敏感

原 gold chunk 为 500 tokens，实验改成 300 tokens 后，完全相同的事实可能落入两个新块。Exact ID Recall 会将这些结果记成遗漏。

可以同时报告：

[
\text{Exact ID Recall}
]

[
\text{Evidence Span Recall}
]

[
\text{Claim Context Recall}
]

### 3. 重叠块导致重复计数

同一证据由于 chunk overlap 出现在三个相邻块中，检索器可能把三者全部返回。Recall 计算应对 gold ID 或 claim ID 去重，每条金标准证据最多计一次。

还可以增加：

# [ \operatorname{Redundancy@k}

1-\frac{\text{top }k\text{ 中的唯一证据或唯一父文档数}}{k}
]

### 4. 固定 (k) 与上下文 token 数不一致

五个 100 token 的块和五个 1000 token 的块对生成器的负担差异很大。比较不同 chunk size 时，可以增加 token budget 版本：

# [ \operatorname{Recall@Btokens}

\frac{
|\text{在前 }B\text{ 个上下文 token 内找回的 gold evidence}|
}{
|G_q|
}
]

它更接近真实 RAG 推理时的上下文预算。

### 5. 文档召回掩盖块级失败

正确文档可能有数百个块。系统检索到了该文档中的目录或无关章节，DocRecall 仍然会记为命中。文档级指标适合评估粗召回，最终送入 LLM 的结果还需要 Chunk Recall 或 Claim Recall。

### 6. 单个 (k) 容易产生片面结论

推荐至少报告一组截断位置：

[
k\in{1,3,5,10,20}
]

两阶段检索系统可以分别评价：

- 初始候选检索器，Recall@20、Recall@50、Recall@100
- reranker，MRR@5、MAP@10、nDCG@10
- 最终输入上下文，Chunk Recall、Context Precision、Claim Context Recall

## 十六、推荐的实际指标组合

对于一般的 RAG 文档问答系统，我建议保留以下主指标：

1. **Chunk Recall@5、@10、@20**，衡量具体证据覆盖。
2. **Document Recall@5、@10、@20**，衡量来源覆盖。
3. **Hit Rate@k**，衡量多少查询至少得到一条相关证据。
4. **Complete Evidence Success@k**，衡量多证据问题是否完整召回。
5. **MRR@k**，衡量首个相关块的位置。
6. **nDCG@k 或 MAP@k**，衡量整体排序。
7. **Precision@k 或 Context Precision**，衡量上下文噪声。
8. **Claim Context Recall**，降低 chunk 边界变化带来的误判。

如果只能选择四个指标，可以采用：

[
\boxed{
\text{Chunk Recall@k}
+
\text{Hit Rate@k}
+
\text{MRR@k}
+
\text{nDCG@k}
}
]

对于多跳 RAG，再加入：

[
\boxed{\text{Complete Evidence Success@k}}
]

对于同时返回实体、关系和原始 chunk 的图 RAG 系统，可以分别计算 Entity Recall、Relation Recall、Chunk Recall 和 Document Recall，最后使用 Claim Context Recall 检查融合后的证据集合是否覆盖问题所需事实。这样能够定位失败发生在实体检索、关系扩展、原文回溯或最终排序中的哪一个环节。