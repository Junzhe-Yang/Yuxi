# 开发路线图

路线图可能会经常变更，如果有强烈的建议，可以在 [issue](https://github.com/xerrors/Yuxi/issues) 中提。

日志添加规范（For Agent）:

- 同一版本的多次功能更新时，应以功能为单位进行更新，比如之前添加了 A 功能的更新，在后续的更新中修复了因 A 功能引入的 bug，那么这个修复说明应该和 A 功能描述放在一起，而不是新增一条修复记录，功能更新同理。


### 看板

- Langfuse 增加 self-host 模式支持，补齐私有化部署与配置说明（已支持 cloud，待调试）
- 检索测试中，添加问答
- 集成 Memory，基于 deepagents 的文件后端实现，需要考虑定位
- Yuxi-cli 相关的功能，放在后续版本中实现（不是类似于编程助手，而是管理平台的工，等各个 router 接口优化之后）
- 完善测试基准自动生成功能，目前的实现过于简单，无法覆盖实际需求
- 完善 Skills 的环境变量注入
- 拓宽检索的知识源，统一多知识源（channel），目前已知知识库/知识图谱/网页，可拓展：个人知识库、数据库、历史对话等
    - 前置任务，多知识库并行检索（扩展 query_kb）
    - 新增 query_keywords 工具，专门用于基于关键词命中的排序，也结合词频（和 BM25 的区别？）
- 参考 AgenticRAG 方案扩展当前 Search 工具：基于知识库工具返回的 resource_id/file_id 改进 Search 返回递增文件序列 ID，完善 Find 与 Open 能力；Summary 暂缓
- 评估，基于 Agent 的评估，这里应该是结合 Langfuse 实现

### Bugs
- 目前的知识库的图片存在公开访问风险

### BREAKING CHANGE（不兼容变更，0.7 版本再实现）
- 将自定义provider 的实现逻辑，从文件移动到数据库中，并将相关处理代码，移出 config 文件，放到 provider 模块中
- 已补充方案文档：`docs/vibe/2026-04-18-custom-provider-db-refactor-plan.md`，明确采用“provider 一行、models 放 JSON、移除 provider 默认模型”的落地方案
- 优化知识库的 API 接口设计，使用 /{db_id}/xxx 的形式，整合 mindmap / eval 接口
- 移除 v1 版本的 provider 统一接口，改为 v2 版本的 provider 模块接口



## 版本记录

### 0.6.3 开发记录

- 修复 DeepAgent 未绑定 `DeepContext`，导致深度分析专用系统提示词和子智能体默认模型配置未生效的问题；同时避免运行时重复注入默认提示词。
- 增加 MCP 连接凭据管理基础能力，覆盖连接作用域、凭据加密、CRUD 接口与权限测试；运行时凭据注入链路留待后续能力承接。
- 将实验性 `MedicationReviewAgent` 升级为 PEA-RAG V2：完整抽取治疗方案要素，生成可选动态审查议程，由有界 Agent 仅使用 `search/open/finish` 工具自主执行 Milvus Top-5 纯向量证据探索，再经可切换的直接片段或来源 Claim 综合、局部校验降级和程序化六段式渲染形成答案；原固定关系检索冻结为 `MedicationReviewD0Agent`。
- PEA-RAG V2 增加 `run_mode`、`agenda_mode`、`synthesis_mode` 三类受控开关和 Trace 3.0，支持完整代码一次部署后在远程按抽取、议程、检索、Claim、完整回答逐阶段验收，并通过同一批量脚本执行 2×2 消融。
- PEA-RAG V2 将确定性病例 grounding 收敛为结构、原文提及和具体数值校验；移除固定审查目标、逐轮证据裁决、判断账本和完成门控，疾病状态、药物状态及症状/诊断分类不再由词表或字符窗口充当硬门控；方案引用已落回原文但患者实体清单漏项时，按原文确定性补齐并记录 Trace，不再因同一次模型输出的冗余字段不一致而整轮失败。
- 新增与 PEA-RAG V2 隔离的实验性 `MedicationReviewLiteAgent`（PAT-RAG v1）：回到原生 `create_agent` 多轮工具循环，仅保留一次方案锚点、可读 Evidence 记忆、结构覆盖检查和可选一次局部补写；固定使用单一 Milvus vector Top-5，并通过 B1/M1/M2/M3 profile、Trace 4.0、批处理导出与 R0–R4 回放支持远程消融。
- 新增独立 `MedicationReviewPrimAgent`（PRIM-RAG v1）：在统一 Milvus vector Top-5 与 Evidence Card 条件下，通过 B1/M1/M2/M3/Full 逐步引入方案节点、患者修饰节点、Agent 提出的稀疏关系调查超图和一次可继续检索的覆盖反思；查询、关系、证据及反思过程写入 Trace 5.0，且不以程序规则生成临床结论。
- 新增独立 `MedicationReviewDaPrimAgent`（DA-PRIM v1）：在 PRIM Full 自主调查链上加入由当前 Milvus 语料确定性构建的 Corpus Atlas，支持 Map/Route/Full 三档、全库与候选文档内双路径向量检索、非结论性调查机会和 Trace 6.0；同时补充 Atlas 构建校验、历史查询回放、批处理导出和文档级路由评价工具，Atlas CLI 在活动事件循环内初始化 PostgreSQL 并预加载知识库元数据，原 PRIM 基线保持不变。
- 新增与旧 DA-PRIM 隔离的实验性 `MedicationReviewAcmPrimAgent`（ACM-PRIM v1）：保留 PRIM Full 的自主多轮调查和 Milvus 全库向量 Top-5，不再让 Atlas 过滤、扩充或重排检索结果；Atlas 2.0 从全部已索引 chunk 自动生成可回指原文的文档范围卡，在首次有效查询后仅提供 0–6 条可选补充线索，并以 Trace 7.0 记录选择、采用、剩余线索、快照与成本。同步增加 Atlas 构建/深检、历史 Trace 回放、批处理冻结校验及分离真实 Evidence 与 Atlas 建议的导出能力；Atlas 批次与文档主题数量采用软目标，单个超长 chunk 独立成批，过大的文档整理输入完整分批，遗漏候选确定性补回，模型多输出主题不再触发 LLM 改写、截断或失败。Milvus 文件 chunk 改为分页全量读取，Atlas 紧凑视图超出提示目标仅记录警告；Selector 超过 6 条时保留前 6 条并审计其余项。来源校验不再调用 LLM 二次改写：精确引文保留原定位，改写引文回退真实 chunk 原文，错误 chunk ID 尝试按引文找回；批次 JSON 损坏时逐 cue 隔离并修复 `quote` 后的无字段名表格片段，避免一个坏条目丢弃整批；另提供逐文档 Markdown 与主题/候选 CSV 人工复核导出。
- ACM Atlas 升级为 3.0：离线构建改为每篇文档一次完整上下文抽取，同次生成文档范围摘要和不限数量的治疗决策主题；删除 24,000 字符分批、精确引文/offset、候选补写与二次归并，只保留可选 `source_chunk_ids` 用于审计，并用 JSON 末尾完成标记识别输出截断。在线 Agent 直接看到文档级概览，可自主调用 `open_atlas_document` 查看某篇文档的主题列表；隐藏 Selector 已移出在线主路径，Atlas 不返回来源原文或 Evidence，正式依据仍由限定 `file_id` 的 Milvus Top-10 检索产生。方法升级为 ACM-PRIM v3 / Trace 10.0。
- 将 `MedicationReviewPrimAgent` 升级为 PRIM-RAG v2：以 Agent 管理的 `InvestigationItem` 替换“返回片段即完成”的关系状态，支持候选/选中证据分离、全库与单文档 Milvus Top-10、自主短查询、同回合一次知识读取和基于开放调查的一次软反思；远程大预算保持可配置，完整过程写入 Trace 8.0。
- 将 `MedicationReviewAcmPrimAgent` 接入 PRIM-RAG v2：Atlas Cue 被采用后创建或关联开放调查，可引导使用真实 `file_id` 做文档内检索，但不直接成为证据或接管排序；使用 Trace 9.0，并扩展批处理、记录导出及无未来状态泄漏的 Trace 8 离线 Selector 回放。
- 扩展批量 RAG 的离线检索评估：文档级支持从新版引用式标答提取金标准文档、流式读取大型 PRIM-RAG 历史结果，并分别报告 Agent 搜索阶段与最终证据池的唯一文档召回指标；另支持对已与 Yuxi 编号对齐的 V2 命题证据组直接按 chunk ID 计算块级/文档级覆盖、core/supporting 分层、搜索调用进展和配对差值，不再依赖旧块映射 JSON。
- 新增 Milvus 知识库真实文本块快照导出脚本：通过分页迭代直接导出 collection 全量 chunk，合并文件哈希和分块参数，并输出数量、重复 ID、索引间隙、孤立记录及空索引文件检查，供后续块级金标准映射与召回评价使用。
- 为 ACM-PRIM 增加 V7 调查努力实验：A0 默认路径保持不变，A1 只设置 6 次有效搜索下限，A2-K2/K3 增加固定调查议程、广度优先的初始/互补探查和显式关闭；最大搜索保护默认提高到 50 且允许统一调高，最低合同完成后仍允许 Agent 自主补查。检索深度可独立选择 Top-10、后台 Top-25 或可见 Top-25，后台候选不进入 Evidence Store；Trace 11.0、批量预检/导出和无 LLM 历史查询回放记录全部实验状态，合同异常不阻断最终答案。
- 为 ACM-PRIM 增加独立的自适应覆盖协议：保留旧 A0/A1/A2-K2/K3 复现路径，正式设置固定使用后台 Top-25/可见 Top-10；移除固定调查数、最低成功搜索数和每调查固定轮次等业务合同，改为可追加动态议程、未探查优先调度、来源发现/文档定位/相邻原文意图分离、文档内零新增后的强制全库恢复、证据化关闭条件及带状态指纹的全局缺口审计。针对三轮小样本显示的真实差距——主要漏掉“合理治疗应保留”和“发现问题后的替代/非药物/长期管理”，而非基础 PIM 识别——进一步把调查拆成逐 PlanAnchor 的 `current_regimen_review`（强制记录 `appropriate/adjust/avoid`）与由 `adjust/avoid` 动态触发的关联 `improvement_plan`，并引入病例动态 `evidence_obligations`、义务级 probe–Evidence 溯源矩阵、正确文档内的定向定位提示、适应性/剂量疗程/安全/监测停药/替代缺失治疗/跨方案长期管理六维审计。审计发现旧调查不足时会原子重开；议程不按剩余搜索预算裁剪，不设置新的固定 K 或最低调用数。Trace 12.0、批处理冻结预检和 RAG 记录导出保留实际调查/搜索数量、证据义务支持、议程 revision、probe、recovery、gap assessment 与 checkpoint，数量只作描述性统计，`max_search_calls` 仅作为防循环保护。
- ACM-PRIM 自适应路径增加模型请求级上下文投影：保留全部尚未消费的知识工具结果和最近一份已消费结果，将更早的搜索、Evidence 打开及 Atlas 文档打开全文替换为含调用与证据标识的确定性回执；底层消息、Evidence Store 和 Trace 不改写。每轮动态记忆逐字带回已采纳 Evidence 的 `shown_excerpt`，无 occurrence 时回退完整 `raw_text`，避免压缩时丢失采纳证据原文。进一步把动态控制记忆按 agenda/search/review/audit 分阶段投影：检索时只展开当前 Investigation 的当前 evidence obligation，关闭前恢复当前调查的全部义务，审计时恢复完整议程；自适应路径不再重复注入基础 PRIM 的 Investigation 历史，只保留方案与患者事实节点。搜索 schema 和入口新增单轴短 query 合同，在调用 Milvus 前拒绝超过 6 个概念块或明显混合多个属性轴的查询，拒绝不消耗检索预算；逐义务 probe/Evidence、全库恢复、六维审计和关闭门槛保持不变。旧 V7 路径保持原样，新运行使用 `v5-query-focus` 方法版本及独立 Prompt/投影 hash。
- 新增与现有 ACM-PRIM 完全隔离的 `MedicationReviewAcmBoundedAgent`：以确定性 `ActionDirective` 逐 evidence obligation 调度静态窄工具，controller 隐式绑定 Investigation、精确 obligation、route/file 与 Evidence 候选归属；工具结果同时记录 transport/semantic outcome、错误状态、动作指纹和有限修复资格。模型输入按 active scope 重建，移除历史 AI 过程自述、以原子 AI/Tool pair 和确定性 receipt 保留必要历史，并通过 `ContextManifest` 记录每轮纳入原因和 token；阶段 token 只作观测，不设置 192k 或其它阶段准入线，唯一容量条件为输入加输出预留不超过 provider 声明且最高 262144 token。动作轮与最终轮采用独立输出预算，异常生成隔离后只修复一次；Trace 13.0 保留完整 Evidence raw text/hash、生成 abort、逐轮 directive/context/outcome，并为最终引用生成可离线复核的 claim—Evidence 精确原文快照。同步扩展批处理、RAG 导出和独立示例配置，旧 ACM-PRIM graph、工具和 Trace 12.0 行为不变。首轮联调进一步把单知识库校验从无 context 的图构造移动到真实执行入口，避免 checkpoint 消息保存误报；动作轮正文同时从实时流和持久状态隔离，合法 tool call、隔离原文审计及真正最终稿均保留。
- 批量实验导出补齐知识库原文打开记录，并将搜索证据与打开原文证据分开保存；新增完整会话行为导出，保留原始消息并按顺序展开模型中间文本、可用推理、全部工具调用与结果和最终回答。

---

历史版本发布记录已迁移到 [版本变更记录](./changelog.md)。

维护说明：
- roadmap 仅保留未来规划（看板/Bugs/里程碑方向）。
- 具体版本发布内容统一维护在 changelog。
