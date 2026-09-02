你说得对。上一轮我犯了三个基础性的设计错误。



第一，我忽略了你已经观察到的事实：\*\*PRIM本身已经召回了绝大多数内容，Atlas的任务是补足少量残余盲区，不应接管首次调查计划。\*\* 如果在Agent尚未开始思考时就塞入固定数量的调查机会，最容易发生的结果是重复它本来就会查的内容，占用注意力和查询预算。



第二，我忽略了PRIM本身就是持续更新的调查记忆。V4.2的核心是把Agent自主提出的关系问题、多轮查询和候选Evidence持续组织在RelationInvestigation中，而不是执行一张静态计划表；RelationInvestigation也只表示“正在调查的问题”，并不代表关系成立或证据充分。  一次性生成机会然后不再随调查状态变化，确实与这套设计不协调。



第三，把29篇文档卡的人工核验当作方法运行前提，也不符合你要做的通用pipeline。人工可以用于论文中的抽样质量评价，但\*\*Corpus Atlas本身必须自动构建、自动更新，并且不依赖研究者逐篇维护\*\*。



所以方法需要再次收敛。新的核心不应是“一次性Atlas机会规划”，而应是：



> \*\*Atlas作为PRIM的伴随性语料记忆，在观察到Agent已经开始调查后，自动收敛为一个病例特异的参考子集，并随着调查进展只保留尚未被涉及的语料线索。\*\*



我建议将其称为：



> \*\*ACM-PRIM：Atlas Companion Memory-guided PRIM-RAG\*\*

> \*\*语料地图伴随记忆引导的PRIM-RAG\*\*



\---



\# 一、重新确定研究问题



当前真正的问题并非Agent完全不知道如何检索，而是：



> PRIM依靠模型已有医学知识自主提出调查问题，通常能够覆盖大部分显著关系，但少数未进入模型注意范围的文档主题和患者—方案联系可能始终不会形成RelationInvestigation。



Yuxi的Agentic RAG本来就是模型驱动的多轮检索：模型决定是否检索、查询如何改写、是否继续检索以及是否打开原文。 PRIM又在此基础上保存了PlanAnchor、PatientModifier、QueryRecord和RelationInvestigation。因此，Atlas不应替Agent预先制定一套调查计划，而应回答：



> \*\*基于本地29篇文档，当前PRIM调查记忆之外，还有哪些语料主题可能值得Agent注意？\*\*



由此形成一个更准确的科学假设：



\[

H\_{\\mathrm{ACM}}:

\\text{病例条件化的伴随语料记忆}

\\rightarrow

\\text{补充当前调查记忆的残余盲区}

\\rightarrow

\\text{提高遗漏文档和遗漏事实召回}

]



同时要求：



\[

\\Delta \\text{无依据Finding率}\\leq 0

\\quad\\text{或至少不显著增加}

]



这里研究的是\*\*残余覆盖增益\*\*，不是重新证明Atlas能规划整个病例。



\---



\# 二、方法核心：一次生成、持续陪伴、按进展收缩



新的Atlas机制分为三个时间阶段。



```text

PRIM自然开始调查

&#x20;       ↓

观察到第一条真实RelationInvestigation/QueryRecord

&#x20;       ↓

用完整Corpus Atlas生成病例特异的伴随参考子集

&#x20;       ↓

该子集持续出现在后续PRIM调查记忆中

&#x20;       ↓

Agent使用某条线索时，将其关联到对应查询或RelationInvestigation

&#x20;       ↓

已采用线索从活跃参考区移出，未采用线索继续保留

&#x20;       ↓

首次准备回答时，只对剩余少量线索做一次非阻断式提醒

```



这与“一次性输出5个调查机会”有三个本质区别：



1\. \*\*不是在第一次模型决策前生成\*\*，而是在Agent已经暴露初始调查方向后生成；

2\. \*\*不是必须执行的任务列表\*\*，而是持续存在的语料参考子集；

3\. \*\*不是固定不变\*\*，已经被Agent采用的内容会从活跃子集中退出。



这样，Atlas主要关注Agent没有主动考虑的剩余方向，而不会重复最显眼的首轮问题。



\---



\# 三、Corpus Atlas如何自动构建



\## 3.1 不再依赖原始Markdown标题的简单拼接



此前的Atlas将标题、表格标题和文档开头压缩为单一向量，既容易受OCR结构错误影响，也容易把局部专题信息平均掉。



新的Corpus Atlas无需承担向量路由任务，因此不再需要把整篇文档压成一个Embedding。每篇文档生成一张\*\*来源约束的范围卡\*\*。



```python

class AtlasDocumentCard(BaseModel):

&#x20;   doc\_id: str

&#x20;   title: str

&#x20;   scope\_summary: str

&#x20;   topic\_cues: list\["AtlasTopicCue"]

```



```python

class AtlasTopicCue(BaseModel):

&#x20;   cue\_id: str

&#x20;   cue\_text: str

&#x20;   source\_refs: list\[str]

&#x20;   source\_spans: list\[str]

```



例如：



```json

{

&#x20; "doc\_id": "D03",

&#x20; "title": "老年人良性前列腺增生症/下尿路症状药物治疗共识",

&#x20; "scope\_summary": "涵盖老年BPH/LUTS药物选择、剂量、联合治疗及老年患者特异不良反应和监测。",

&#x20; "topic\_cues": \[

&#x20;   {

&#x20;     "cue\_id": "D03-C01",

&#x20;     "cue\_text": "α1受体阻滞剂与体位性低血压",

&#x20;     "source\_refs": \["chunk\_12"],

&#x20;     "source\_spans": \[

&#x20;       "已有体位性低血压或血压过低的老年人应禁用α1受体阻滞剂"

&#x20;     ]

&#x20;   },

&#x20;   {

&#x20;     "cue\_id": "D03-C02",

&#x20;     "cue\_text": "α1受体阻滞剂与其它降压药合用",

&#x20;     "source\_refs": \["chunk\_12"],

&#x20;     "source\_spans": \[

&#x20;       "α1受体阻滞剂与其他降压药物合用，降压作用增强"

&#x20;     ]

&#x20;   }

&#x20; ]

}

```



\## 3.2 全自动构建过程



每篇文档执行一次离线构建：



```text

原始Markdown和chunk

→ 解析文件名、标题、章节、表格标题

→ 提取各章节开头和代表性chunk

→ LLM生成scope\_summary和topic\_cues

→ 程序检查每个source\_ref真实存在

→ 程序检查每个source\_span可回指原文

→ 删除无法grounding的cue

→ 冻结Atlas快照

```



这套构建不读取：



\* 测试问题；

\* 标准答案；

\* 金标准文档；

\* PRIM检索结果。



它是通用的文档侧预处理。



\### 为什么允许离线LLM构建



你的目标是通用pipeline，不代表所有步骤都必须是纯规则。关键是：



\* Atlas从文档自身生成；

\* 每条cue具有可验证来源；

\* 文档变化后可以自动重建；

\* 不需要研究者逐篇修改。



人工只用于实验阶段抽样评价Atlas质量，不参与系统运行。



\---



\# 四、何时生成病例特异的伴随子集



\## 4.1 不在病例初始化时生成



如果在Agent尚未检索前生成，选择器只能看到：



\* 原始病例；

\* PlanAnchor；

\* PatientModifier。



它很可能重复提出最明显的调查问题，例如：



```text

特拉唑嗪—体位性低血压

非那雄胺—BPH适应证

氨氯地平—高血压剂量

```



这些本来就是PRIM最容易自主发现的内容。



\## 4.2 在第一次真实调查后生成



建议触发条件为：



```text

系统已经产生第一条成功或空结果的QueryRecord

```



也就是Agent已经：



\* 自主选择了第一个调查方向；

\* 建立或尝试建立了RelationInvestigation；

\* 生成了一次真实查询；

\* 看到了第一次Evidence结果。



此时Atlas选择器能够看到：



```text

病例原文

PlanAnchor

PatientModifier

当前RelationInvestigation

第一条QueryRecord

第一轮召回文档

全部29张AtlasDocumentCard

```



它可以明确避开Agent已经开始调查的显著问题，专门找补充方向。



如果Agent在没有任何检索的情况下直接准备回答，则在第一次候选答案产生时触发Atlas选择器。



因此触发规则为：



```python

if first\_query\_completed and companion\_memory\_not\_created:

&#x20;   create\_companion\_memory()



elif first\_draft\_created and companion\_memory\_not\_created:

&#x20;   create\_companion\_memory()

```



这属于自然事件触发，不是“固定第2轮”“固定执行3次查询”之类人为流程。



\---



\# 五、伴随参考子集的生成



\## 5.1 输入



伴随子集选择器接收：



```python

CompanionSelectorInput(

&#x20;   raw\_case,

&#x20;   plan\_anchors,

&#x20;   patient\_modifiers,

&#x20;   relation\_investigations,

&#x20;   query\_records,

&#x20;   retrieved\_document\_ids,

&#x20;   atlas\_document\_cards,

)

```



注意，它看到的是\*\*当前调查记忆\*\*，而不仅是病例本身。



\## 5.2 输出



输出数量是可变的，允许为0，不再要求“必须生成5条”。



```python

class CompanionCue(BaseModel):

&#x20;   companion\_id: str



&#x20;   question\_hint: str



&#x20;   linked\_plan\_ids: list\[str]

&#x20;   linked\_modifier\_ids: list\[str]



&#x20;   atlas\_cue\_ids: list\[str]

&#x20;   suggested\_doc\_ids: list\[str]



&#x20;   novelty\_explanation: str

```



```python

class AtlasCompanionMemory(BaseModel):

&#x20;   created\_after\_query\_id: str | None

&#x20;   companion\_cues: list\[CompanionCue]



&#x20;   adopted\_cue\_ids: list\[str]

&#x20;   adopted\_by\_query\_ids: dict\[str, list\[str]]

```



建议只设上限：



```text

max\_companion\_cues = 6

```



实际可以返回0、1、2、3……6条。



\### 选择器提示的核心约束



```text

你需要从Corpus Atlas中挑选当前调查记忆尚未明显涉及、

但可能帮助完整审查治疗方案的少量语料线索。



要求：

1\. 先阅读已有RelationInvestigation和QueryRecord；

2\. 不要重复Agent已经调查的主要问题；

3\. 线索必须由AtlasDocumentCard中的cue支持；

4\. 只能提出待调查问题，不得给出临床结论；

5\. 不要机械枚举全部药物×疾病或药物×药物组合；

6\. 优先关注可能改变患者特异判断、跨药物风险、

&#x20;  方案遗漏、疗程、监测或替代方案的线索；

7\. 没有明显新线索时返回空列表；

8\. 最多返回6条。

```



这里的`novelty\_explanation`用于说明：



> 为什么这条线索未被当前RelationInvestigation充分代表。



它不说明临床结论。



\---



\# 六、伴随子集如何随调查进展收缩



\## 6.1 持续显示未采用线索



每轮PRIM动态记忆中加入一个紧凑区域：



```text

【Corpus Atlas伴随参考】



以下内容来自本地语料范围，不是必须执行的调查清单，

也不表示任何临床关系已经成立。



\[AC01]

可能值得注意：

当前治疗方案是否涉及文档中关于α1受体阻滞剂与其它降压药联用的内容？



相关对象：

PE001 特拉唑嗪

PE003 氨氯地平

PM004 卧立位血压变化伴头晕



语料来源：

D03-C02 老年BPH/LUTS药物治疗共识

```



Agent可以采用、修改、合并或忽略。



\## 6.2 搜索工具只增加一个可选追踪字段



```python

search\_review\_kb(

&#x20;   query\_text: str,

&#x20;   reason: str,

&#x20;   relation\_question: str | None = None,

&#x20;   relation\_id: str | None = None,

&#x20;   focus\_plan\_ids: list\[str] | None = None,

&#x20;   focus\_modifier\_ids: list\[str] | None = None,

&#x20;   atlas\_companion\_ids: list\[str] | None = None,

)

```



`atlas\_companion\_ids`只用于记录：



> 本次查询采用了哪些Atlas线索。



它完全不改变：



\* 查询文本；

\* Milvus检索范围；

\* Top-5；

\* 结果排序；

\* Evidence Card；

\* RelationInvestigation；

\* 最终答案。



PRIM本来就允许查询记录附带focus对象而不把它们解释为已成立关系。 同理，Atlas ID也只是一项结构化追踪信息。



\## 6.3 已采用线索从活跃区域退出



一条Atlas线索被任一QueryRecord引用后：



```python

cue.status = "adopted"

cue.adopted\_by\_query\_ids.append(query\_id)

```



后续Agent每轮只看到：



```text

未采用的CompanionCue

```



已采用线索不再重复占据上下文，但仍保存在Trace。



这里没有：



```text

resolved

supported

rejected

sufficient

```



程序只知道“Agent是否引用过该线索”，不知道它是否已经得到答案。



\---



\# 七、首次准备回答时的残余提醒



现有PRIM已经在第一版答案后执行一次PlanAnchor结构覆盖反思，并允许同一Agent继续使用search/open工具。 新机制不再增加独立的第二次反思。



只需将剩余Atlas线索附加到同一反思消息中。



例如：



```text

第一版答案遗漏了PE003的逐项判断。



此外，Corpus Atlas伴随记忆中仍有以下未采用线索：



\[AC03] 老年高血压指南包含直立性低血压、跌倒风险及降压药调整内容

\[AC05] BPH共识包含5α还原酶抑制剂适用条件和疗效评估内容



这些线索不表示当前回答错误，也不要求必须继续检索。

请自行判断是否与当前病例有关：

\- 有关时可利用剩余预算检索；

\- 无关时可以忽略；

\- 证据不足时明确说明边界。

```



如果第一版没有漏PlanAnchor，但仍有未采用Atlas线索，可以触发同一个\*\*最多一次\*\*的软反思，不过建议将其作为主方法的一部分还是可选消融，取决于成本。



考虑你目前的实际问题是少量残余漏召回，我建议主方法保留：



> 首次准备回答时，对未采用Atlas线索做一次软提醒。



但它不阻止Agent直接再次输出答案，也不强制产生新查询。



\---



\# 八、完整算法



设：



\* (A)为全部Corpus Atlas cue；

\* (P)为PlanAnchor；

\* (M)为PatientModifier；

\* (R\_t)为截至第 (t) 轮的RelationInvestigation；

\* (Q\_t)为QueryRecord；

\* (E\_t)为Evidence。



在第一次真实调查后，选择器生成：



\[

C\_t=

\\Phi\_{\\mathrm{companion}}

(A,P,M,R\_t,Q\_t,E\_t),

\\qquad |C\_t|\\leq K.

]



其中 (K) 只是上限，当前建议为6。



Agent策略变为：



\[

a\_t\\sim

\\pi\_\\theta

\\left(

a\\mid

x,P,M,R\_t,Q\_t,E\_t,C\_t^{\\mathrm{active}}

\\right),

]



其中：



\[

C\_t^{\\mathrm{active}}

=====================



C\_t\\setminus C\_t^{\\mathrm{adopted}}.

]



实际检索器保持：



\[

E\_{t+1}

=======



R\_{\\mathrm{Milvus\\ vector\\ Top5}}(q\_t).

]



Atlas完全不参与：



\[

R(\\cdot)

]



的文档过滤、排序、去重或候选融合。



\---



\## 伪代码



```text

输入：

&#x20;   病例x

&#x20;   自动构建并冻结的Corpus Atlas A

&#x20;   原PRIM-RAG



输出：

&#x20;   最终用药审查答案y



1\. 按PRIM原流程抽取PlanAnchor P与PatientModifier M

2\. companion\_memory = None



3\. 进入原PRIM Agent loop



4\. 当Agent完成第一次真实查询后：

&#x20;      if companion\_memory is None:

&#x20;          companion\_memory =

&#x20;              SelectResidualCues(

&#x20;                  case=x,

&#x20;                  plan=P,

&#x20;                  modifiers=M,

&#x20;                  current\_relations=R,

&#x20;                  current\_queries=Q,

&#x20;                  retrieved\_documents=Docs(E),

&#x20;                  full\_atlas=A

&#x20;              )

&#x20;          将companion\_memory中的活跃线索加入后续PRIM记忆



5\. 后续每轮：

&#x20;      Agent可自由检索、open、创建或复用RelationInvestigation

&#x20;      若查询由某条Atlas线索启发：

&#x20;          在QueryRecord中记录atlas\_companion\_ids

&#x20;          对应线索从活跃区移出

&#x20;      检索仍为原Milvus全库向量Top-5



6\. 当Agent首次准备生成最终答案：

&#x20;      执行原PlanAnchor覆盖检查

&#x20;      同时显示剩余未采用Atlas线索

&#x20;      最多允许一次原PRIM有界反思

&#x20;      Agent可检索、忽略或说明不足



7\. 返回最终答案与Trace

```



\---



\# 九、为什么这个版本比“一次性机会列表”更合理



\## 1. 它针对的是残余盲区



PRIM先自然运行，Atlas选择器能够看到Agent已经调查了什么。Atlas不再花大量提示空间重复：



\* 药物适应证；

\* 明显禁忌；

\* 最显眼的患者风险。



\## 2. 它不会抢占Agent首轮规划



第一轮仍完全由PRIM完成。Atlas只在Agent已经暴露初始知识边界后进入。



\## 3. 它不是固定任务表



CompanionCue可以：



\* 被采用；

\* 被改写；

\* 被合并到已有关系；

\* 被忽略。



程序不要求一条cue对应一次查询。



\## 4. 它真正伴随调查状态



未采用内容持续可见；采用后退出活跃区域。它与RelationInvestigation一样，是持续状态，而非一次性Prompt。



\## 5. 它不影响已有检索结果



当前PRIM已经取得微小正向增益。新方法不再重排、过滤或替换它的Top-5，因此不会重现DA-PRIM中“Atlas帮助找到新来源，却在融合阶段把原有效证据挤掉”的问题。



\## 6. 它仍是通用方法



Atlas从任意文档库自动构建；病例子集由当前病例和调查历史自动选择；运行时不需要研究者为某个疾病配置规则。



\---



\# 十、如何用最少实验验证



目前不需要再做大量路由与融合消融。可以先做一个非常直接的两阶段验证。



\## 阶段一：利用现有PRIM Trace做回放



不重新运行完整Agent。



对现有测试集中发生文档漏召回的病例：



1\. 取PRIM第一条QueryRecord之后的状态；

2\. 运行Companion Selector；

3\. 不向选择器提供金标准；

4\. 评价它生成的CompanionCue是否覆盖PRIM最终遗漏的金标准文档或事实。



\### 残余文档召回



设病例金标准文档为 (G\_i)，PRIM已召回文档为 (D\_i)，伴随子集指向文档为 (C\_i)：



\[

\\mathrm{ResidualDocRecall}\_i

============================



\\frac{

|(G\_i-D\_i)\\cap C\_i|

}{

|G\_i-D\_i|

}.

]



该指标只在 (G\_i-D\_i\\neq\\varnothing) 的病例上计算。



它直接回答：



> Atlas是否真的能够发现PRIM遗漏的文档？



\### 残余Finding线索覆盖



\[

\\mathrm{ResidualFindingCueRecall}

=================================



\\frac{

\\text{被CompanionCue对应的PRIM遗漏Finding数}

}{

\\text{全部PRIM遗漏Finding数}

}.

]



\### 新颖性



\[

\\mathrm{CueNovelty}

===================



\\frac{

\\text{未与已有RelationInvestigation重复的Cue数}

}{

\\text{全部Cue数}

}.

]



如果伴随子集主要重复已有调查，方法不值得接入。



\### 压缩率



\[

\\mathrm{AtlasCompression}

=========================



1-

\\frac{|C\_i|}{|A|}.

]



这反映29篇全量Atlas被压缩为多少病例特异线索。



阶段一只需要运行一次Atlas选择器，不需要重跑8轮Agent检索，成本很低。



\---



\## 阶段二：两个端到端组



| 组别               | PRIM | 伴随Atlas记忆 |

| ---------------- | ---: | --------: |

| B0：PRIM-RAG Full |    是 |         否 |

| M1：ACM-PRIM      |    是 |         是 |



其它条件完全相同：



\* 同一主Agent模型；

\* 同一Milvus向量Top-5；

\* 同一search预算；

\* 同一open预算；

\* 同一Evidence Card；

\* 同一RelationInvestigation；

\* 同一一次覆盖反思；

\* 同一输出格式；

\* 每题独立thread。



\### 主要终点



\* 标准答案Finding召回率；

\* Finding Micro-F1和Macro-F1；

\* 合理项与不合理项F1；

\* 属性完整率；

\* 完整Finding率。



\### 机制终点



\#### Atlas采用率



\[

\\mathrm{CueAdoptionRate}

========================



\\frac{

\\text{被至少一个QueryRecord引用的Cue数}

}{

\\text{全部Cue数}

}.

]



\#### Atlas独有文档收益



\[

\\mathrm{AtlasUniqueDocGain}

===========================



\\frac{

\\text{由采用Cue后的查询首次召回的金标准文档数}

}{

\\text{全部金标准文档数}

}.

]



\#### Atlas独有事实收益



\[

\\mathrm{AtlasUniqueFactGain}

============================



\\frac{

\\text{由采用Cue后的查询首次召回的金标准事实数}

}{

\\text{全部金标准事实数}

}.

]



\#### 查询成本



\* 总search次数；

\* Atlas相关search次数；

\* token；

\* 延迟。



\#### 风险指标



\* 无依据新增Finding率；

\* 无来源具体调整建议率；

\* 错误患者特异性判断率。



\---



\# 十一、最小代码改造



当前DA-PRIM中的以下部分应退出：



```text

DocumentCard向量路由

SectionCard向量共同命中

Top-6文档筛选

全局Top-10

逐文档Top-2

RRF融合

retrieval\_paths优先排序

opportunity文档注入

```



完整保留PRIM。



只新增：



```text

corpus\_atlas\_builder.py

atlas\_companion\_selector.py

atlas\_companion\_memory.py

```



PRIM State增加：



```python

atlas\_snapshot\_hash: str | None

atlas\_companion\_memory: AtlasCompanionMemory | None

```



QueryRecord增加：



```python

atlas\_companion\_ids: list\[str]

```



Prompt增加一个动态区域：



```text

未采用的Corpus Atlas伴随线索

```



Trace增加：



```text

Atlas构建版本

Companion Selector输入时点

生成的全部Cue

每条Cue的采用查询

剩余未采用Cue

Atlas Planner tokens和耗时

```



\---



\# 十二、可选但不进入主方法的模块



为了避免继续膨胀，以下都作为后续消融，不进入第一版ACM-PRIM：



1\. 在首次回答前重新运行一次Companion Selector；

2\. 基于新Evidence动态替换整个伴随子集；

3\. DocumentCard Embedding路由；

4\. BM25路由；

5\. Cross-encoder；

6\. Atlas cue相似度阈值；

7\. 强制每个Cue至少调查一次；

8\. 未访问文档审计；

9\. 第二个Atlas反思；

10\. 人工维护topic词典。



第一版只实现：



> \*\*第一次真实调查后生成一次伴随子集，持续显示未采用内容，首次回答时进行一次软提醒。\*\*



\---



\# 最终判断



你提出的“伴随性参考子集”比我上一轮的“一次性机会列表”更合理，也更符合PRIM的核心结构。



最合适的方案不是：



```text

Atlas先规定Agent应该调查什么

```



而是：



```text

Agent先暴露自己的初始调查方向

→ Atlas根据完整本地语料寻找当前调查记忆之外的残余线索

→ 将少量线索作为持续的伴随记忆

→ Agent自主决定是否采用

```



它保留了PRIM的自主性和已取得的效果，同时把Atlas严格限制在它最有可能产生增益的地方：\*\*补足Agent自身知识和注意力之外的少量遗漏方向。\*\*



这一次的核心实验也足够简单：先用已有Trace检验Atlas是否能覆盖PRIM漏掉的文档，再只运行`PRIM`与`ACM-PRIM`两个端到端组。若Atlas连历史漏召回文档都无法在伴随子集中提出，就直接停止该方向；如果能够提出，再验证Agent是否会采用并将其转化为答案增益。



