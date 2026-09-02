# Yuxi 0.6.3 批量 RAG、PEA-RAG v2、PRIM-RAG 与调查地图远程验收

本目录中的脚本只通过 HTTP 调用远程 Yuxi，不要求本机启动 Yuxi、Milvus 或模型。每条数据、每个实验条件都会创建独立 thread，并只提交一次用户问题，避免问题之间互相干扰。

## 文件

- `batch_yuxi_rag.py`：批量创建 thread/run，跟踪 SSE，保存回答、工具记录和 Trace。
- `discover_yuxi.py`：只读发现 Agent、AgentConfig、知识库和模型 ID。
- `replay_medication_review_trace.py`：不重新检索，使用 Trace 2.0/3.0 的证据回放后检索流程。
- `replay_pat_rag.py`：不重新检索，用 V2 Evidence 对 PAT-RAG 的 R0–R4 组做回放。
- `backend/scripts/replay_da_prim_retrieval.py`：用历史 PRIM 查询比较 Flat Top-22 与 DA-PRIM 候选池。
- `export_answers.py`：导出 `question`、`response`。
- `export_rag_records.py`：导出问题、回答、检索片段和打开的原文；兼容 Trace 2.0–12.0。
- `export_session_records.py`：无损保留会话消息，并生成便于分析的模型文本、推理、工具调用/结果和最终回答时间线。
- `evaluate_document_retrieval.py`：不调用 LLM 的文档层检索评价。
- `evaluate_evidence_group_retrieval.py`：按 V2 命题证据组，以直接块 ID 同时评价块级和文档级检索。
- `backend/scripts/export_milvus_chunks.py`：在 API 容器内分页导出一个 Milvus 知识库实际入库的全部文本块。

## 导出 Milvus 知识库的全部文本块

该命令直接读取 Milvus collection 中现存的 chunk，不重新分块、不调用 Embedding/LLM，
也不修改知识库。建议在没有文档入库或重新索引任务运行时执行，以冻结稳定快照。

按网页中的知识库名称导出：

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.export_milvus_chunks \
  --knowledge-name "用药助手-md" \
  --output-dir /app/saves/exports/medication-kb-chunks
```

也可以使用 `--db-id <DB_ID>`。输出目录位于宿主机
`saves/exports/medication-kb-chunks/`，包含：

- `chunks.jsonl`：每行一个真实 chunk，含 `file_id`、`chunk_id`、`chunk_index`、正文、
  文件哈希、分块参数和正文 SHA-256；
- `manifest.json`：知识库、文件及 chunk 数量、完整性检查、孤立 chunk 与空索引文件告警。

仅当明确要覆盖同一目录里的旧结果时增加 `--overwrite`。脚本失败时不会把部分结果冒充
正式导出，而是保留 `chunks.jsonl.part` 并在 `manifest.json`（覆盖旧快照时为
`manifest.failed.json`）中写入失败原因。

## 文档级检索评估

`evaluate_document_retrieval.py` 支持旧版 `documents` /
`must_retrieve_documents` 标注，也支持从新版标答 `reference` 中的
`【依据：文档名 · 章节 · #块号】` 提取金标准文档。结果文件可使用 JSON 数组或
JSONL；大 JSON 数组会流式读取，不会整体载入内存。

```bat
python scripts\yuxi_batch_rag\evaluate_document_retrieval.py ^
  --gold scripts\yuxi_batch_rag\reference_file\dataset_425.json ^
  --results scripts\yuxi_batch_rag\reference_file\prim_rag_full_rag_records.json ^
  --output-dir output\document-retrieval-evaluation
```

主结果 `search_union` 只统计成功的检索调用，并按各调用的执行顺序和局部排名形成
唯一文档首次发现序列；`final_evidence_pool` 统计检索及证据窗口打开后最终提供给
模型的文档集合。所有 @K 均以唯一文档为单位，不计算 chunk 召回或 chunk 匹配。
未标注的召回文档按 `unjudged` 处理，不能将
`annotated_required_document_fraction` 解释为完整相关性标注下的严格精确率。

## V2 证据组直接 ID 检索评估

`evaluate_evidence_group_retrieval.py` 直接比较 V2 金标与 RAG 记录中的
`chunk_id`，不读取旧 `manual_to_auto.json`，也不执行 quote 模糊映射。同一命题的
证据组之间按 OR、组内块按 AND 计分，并分别输出 all/core/supporting、块级/文档级、
搜索阶段/最终证据池结果。搜索阶段保留每个成功非 open 调用的 Top-K；MRR 使用跨调用
最佳局部排名，调用间检索进展另由 coverage AUC 和 productive call rate 表达。

```bat
python scripts\yuxi_batch_rag\evaluate_evidence_group_retrieval.py ^
  --gold "0824金标数据集交付\output\P3_最终打磨版_标准数据集_v2.json" ^
  --snapshot "0824金标数据集交付\standards\milvus_snapshot\chunks.jsonl" ^
  --result "ACM-PRIM=outputs\acm-prim_ragrecords_test.json" ^
  --result "baseline=outputs\yuxi_vector_ragrecords_175.json" ^
  --gids 72 359 244 102 414 164 161 ^
  --top-k 10 ^
  --pair ACM-PRIM baseline ^
  --output-dir "outputs\eval\test7_v2_direct_id_top10"
```

输出包含逐系统 JSONL、`summary.json`、`summary.csv` 和 `REPORT.md`，并记录输入
SHA-256、金标 ID 的快照缺失审计及检索返回的未知 ID。顶层 `result_status` 与检索调用
状态分开审计：来源运行即使在生成阶段失败，只要存在成功搜索调用，检索轨迹仍可计分。
未标注召回块按 `unjudged` 处理，因此不计算 precision/F1。

## PAT-RAG v1 快速开始

PAT-RAG 使用独立的 `MedicationReviewLiteAgent`，不会替换或迁移已有
`MedicationReviewAgent` 会话。它保留原生多轮 Agent 工具调用，只提供
`search_review_kb` 和 `open_review_evidence` 两个工具；检索固定为所选单一
Milvus 知识库的 vector Top-5，不使用 BM25、LightRAG 或 reranker。

在网页中复制四份新 AgentConfig，除 `experiment_profile` 外保持模型、知识库、
Prompt 和预算一致：

| profile | 锚点 | 给模型的检索正文 | 覆盖补写 |
| --- | ---: | --- | ---: |
| `b1` | 否 | 完整 chunk | 否 |
| `m1` | 是 | 完整 chunk | 否 |
| `m2` | 是 | query-centered Evidence card | 否 |
| `m3` | 是 | query-centered Evidence card | 最多一次 |

复制 `config.pat-rag.example.json`，替换四个 AgentConfig ID 和知识库名称后先做
预检：

```bat
set YUXI_LOGIN_ID=你的登录ID
set YUXI_PASSWORD=你的密码
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.pat-rag.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.pat-rag.local.json --variants m2_anchor_evidence_card --log-level DEBUG --log-file output\pat-rag-m2.log
```

Trace 4.0 的 `completed` 和 `partial` 都会作为有答案的成功记录保存；只有远程
Run 失败、Trace 为 `failed` 或最终答案为空才触发 job 重试。检索记录应从
`medication_review_trace.search_records/open_records/evidence_store` 读取，后者
保留完整 raw chunk，不能用页面中展示的 Evidence card 代替。

导出：

```bat
python scripts\yuxi_batch_rag\export_answers.py --input output\pat-rag-v1-ablation\results\m3_anchor_card_patch.jsonl --output output\pat-rag-m3-answers.json
python scripts\yuxi_batch_rag\export_rag_records.py --input output\pat-rag-v1-ablation\results\m3_anchor_card_patch.jsonl --output output\pat-rag-m3-rag-records.json
```

基于旧 Trace 2.0/3.0 Evidence 做不重新检索的回放：

```bat
python scripts\yuxi_batch_rag\replay_pat_rag.py --input output\pea-rag-v2-ablation\results\full_dynamic_claims.jsonl --output-dir output\pat-rag-replay --groups r0,r1,r2,r3,r4
```

`R2/R3/R4` 对同一病例复用一次锚点抽取；`R0` 只复制旧答案，单独运行时不要求
模型；`R1` 不执行锚点抽取。

## PRIM-RAG v2 快速开始

PRIM-RAG v2 使用独立的 `MedicationReviewPrimAgent`。所有实验组共用一个 Milvus
知识库、纯向量 Top-10、相同的相似度阈值、Evidence Card、检索/打开预算和最终答案协议；
不在主检索中加入 BM25、reranker 或 RRF。五个 profile 只逐步增加下表中的信息：

| profile | 方案要素 | 患者事实 | 证据调查记忆 | 最终软反思 |
| --- | ---: | ---: | ---: | ---: |
| `b1` | 否 | 否 | 否 | 否 |
| `m1` | 是 | 否 | 否 | 否 |
| `m2` | 是 | 是 | 否 | 否 |
| `m3` | 是 | 是 | 是 | 否 |
| `full` | 是 | 是 | 是 | 是 |

检索仍由 Agent 自主决定。查询应写成贴近中文语料的短证据表达；全库命中可能正确的
文档后，Agent 可以使用 Evidence Card 中的稳定 `file_id` 做该文档内的 Top-10 向量
补查。每个模型回合最多执行一次 `search` 或 `open`，使模型先看到本次结果再规划下一步；
这不会缩小配置的总预算，也不要求用完预算。`m3/full` 中，任意 Top-10 都只进入候选池，
调查只有在 Agent 调用 `update_investigation` 后才会变为 `answered`、`insufficient` 或
`dismissed`。

复制 `config.prim-rag.example.json` 为 `config.prim-rag.local.json`，在网页中建立五份
只改变 `experiment_profile` 的 AgentConfig，并替换样例中的数字 ID 和知识库名称。
远程实验已经放大的 `max_search_calls` / `max_open_calls` 应在各对照组保持一致；样例脚本
不会覆盖网页保存的预算。Trace 8.0 的主要字段如下：

Top-10 与旧 Top-5 属于不同实验条件。样例已使用新的输出目录并校验精确
`method_version`，不要把新结果续跑到旧 Top-5 的 `output_dir`。

- `query_records`：每次全库/文档内查询、`file_id`、状态和 Evidence ID；
- `investigations`：问题、候选证据、Agent 选择的证据、状态和当前缺口；
- `deferred_knowledge_calls`：因同一模型回合只允许一次读取或总预算耗尽而未执行的调用；
- `evidence_store`：可用于块召回评估的完整真实 Milvus 片段；
- `reflection_report`：唯一一次软反思前后的开放调查和未覆盖方案要素。

先准备只含一条病例的输入文件，以 `concurrency=1` 验证 Full；确认后再换回全量数据集：

```bat
set "YUXI_LOGIN_ID=你的登录ID"
set "YUXI_PASSWORD=你的密码"
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.prim-rag.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.prim-rag.local.json --variants full_investigation_reflection --log-level DEBUG --log-file outputs\prim-rag-v2-smoke.log
python scripts\yuxi_batch_rag\export_answers.py --input outputs\prim-rag-v2-top10-ablation\results\full_investigation_reflection.jsonl --output outputs\prim-rag-v2-top10-answers.json
python scripts\yuxi_batch_rag\export_rag_records.py --input outputs\prim-rag-v2-top10-ablation\results\full_investigation_reflection.jsonl --output outputs\prim-rag-v2-top10-rag-records.json
```

批处理对每道题、每个实验组建立独立 thread，因此病例之间不会共享问题、回答或调查记忆。

## DA-PRIM v1（历史方法）快速开始

本节仅用于复现实验历史。当前正式方法不再继续扩展 DA-PRIM 的多文档路由和融合，而是
使用后文的 ACM-PRIM v3，把两层调查地图作为 PRIM v2 上的可选提示层。

DA-PRIM 使用独立的 `MedicationReviewDaPrimAgent`。它固定继承 PRIM `full` 的
Agent 自主检索、关系调查、打开原文、覆盖反思和最终答案协议，只增加确定性的
Corpus Atlas 导航；不使用 BM25、LightRAG、reranker 或疾病/药物特例。

| atlas_profile | 病例文档地图 | 全局+文档内双路径检索 | 调查机会 |
| --- | ---: | ---: | ---: |
| `map` | 是 | 否，仍为原 Flat Top-5 | 否 |
| `route` | 是 | 是 | 否 |
| `full` | 是 | 是 | 是 |

首次运行前必须在远程 API 容器内为目标 Milvus 知识库构建 Atlas。`--check` 只做
当前快照校验，不会重建：

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.build_corpus_atlas \
  --knowledge-name "<知识库名称>"
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.build_corpus_atlas \
  --db-id <DB_ID> --check
```

在网页中为新 Agent 建立三份配置，只改变 `atlas_profile`。复制
`config.da-prim.example.json` 为本地配置并替换 AgentConfig ID 与知识库名称。
如果只批量运行当前完整方法，可以删除 `atlas_map`、`atlas_route`，仅保留
`da_prim_full`。`config.da-prim.local.json`、`discovery.json` 和结果目录都是
用户环境生成的文件，不随源码提供。

配置中的相对路径以配置文件所在目录为基准；样例的 `../../outputs/...` 指向仓库
根目录的 `outputs`。跨机器运行时也可以填写使用 `/` 的绝对路径。

DA-PRIM 的 PRIM 基础档位由 Agent 代码隐藏并固定为 `full`，不会写入网页可配置的
AgentConfig，因此样例不对 `expected_experiment_profile` 做预检；正式 Trace 中仍会
记录 `requested_profile=full`，可用于事后核验。

```bat
set "YUXI_LOGIN_ID=你的登录ID"
set "YUXI_PASSWORD=你的密码"
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.da-prim.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.da-prim.local.json --variants da_prim_full --log-level INFO --log-file outputs\da-prim-full.log
python scripts\yuxi_batch_rag\export_answers.py --input outputs\da-prim-rag-v1-ablation\results\da_prim_full.jsonl --output outputs\da-prim-full-answers.json
python scripts\yuxi_batch_rag\export_rag_records.py --input outputs\da-prim-rag-v1-ablation\results\da_prim_full.jsonl --output outputs\da-prim-full-rag-records.json
```

Trace 6.0 在 Trace 5.0 上增加 `atlas_snapshot`、`case_route_record`、
`retrieval_opportunities`、`adopted_opportunity_ids` 和
`routed_retrieval_records`。每个 routed search 通常执行一次批量 embedding、一次
全库 Top-10 和六次严格 `file_id` 文档内 Top-2，再融合为最多五条 Evidence。
单个分支失败会标记 degraded，但不会取消其他分支或阻止 Agent 后续补查。

先用历史 PRIM records 做不调用 LLM 的候选池回放；`--gold` 可选，提供后会同时
计算会话级文档召回、RoutedUniqueGain 和 GlobalEscapeRate：

```bat
docker cp output/prim-rag-full-records.json api-dev:/tmp/prim-rag-full-records.json
docker cp scripts/yuxi_batch_rag/reference_file/dataset_425.json api-dev:/tmp/dataset_425.json
docker compose exec api uv run python scripts/replay_da_prim_retrieval.py \
  --db-id <DB_ID> \
  --records /tmp/prim-rag-full-records.json \
  --gold /tmp/dataset_425.json \
  --output-dir /tmp/da-prim-replay \
  --limit 20
docker cp api-dev:/tmp/da-prim-replay output/da-prim-replay
```

上述容器命令使用 Linux shell 的续行符 `\`；不要把 Windows CMD 的 `^` 带入
远程 Linux shell。回放摘要分别报告：局部分支独有的 `RoutedUniqueGain`、全局分支
独有的 `GlobalEscapeRate`，以及 DA union 相对 Flat Top-22 新增命中的文档数，三者
含义不同。

Atlas 路由本身在远程容器内评价：

```bash
docker compose exec api uv run python scripts/evaluate_corpus_atlas.py \
  --db-id <DB_ID> \
  --gold /tmp/dataset_425.json \
  --records /tmp/prim-rag-full-records.json \
  --output-dir /tmp/atlas-router-evaluation
docker cp api-dev:/tmp/atlas-router-evaluation output/atlas-router-evaluation
```

P0 阶段可重复增加例如 `--opportunity-min-similarity 0.35`，比较机会数量、病例覆盖率
和明细中的节点 cosine；该参数只影响离线校准脚本，不会偷偷改变在线 Full 方法。
确定开发集阈值后，应修改并冻结代码中的 `MIN_SECTION_SIMILARITY`，重启 API/worker
并重新运行预实验，以 Trace `atlas_snapshot.method_parameters` 为准记录最终值。

正式消融使用两个批量配置：现有 `config.prim-rag.example.json` 的 PRIM `full`
作为 B0；DA 配置运行 `map/route/full`。四组必须固定同一 LLM、知识库、Prompt、
search/open 预算和并发。先依次跑 1 条、10–20 条开发集，冻结参数后再全量运行；
CPU embedding 环境保持 `concurrency=1`。
首轮一条病例应使用单独的一行输入文件和新的 `output_dir`，不要把一次全量任务
当作 smoke test 启动。

远程部署后的最小验收：

```bash
docker compose up -d
docker compose exec api uv run --group test pytest \
  test/unit/plugins/test_milvus_kb.py \
  test/unit/agents/medication_review_prim \
  test/unit/agents/medication_review_da_prim \
  test/unit/tools/test_yuxi_batch_rag_script.py \
  test/unit/tools/test_yuxi_batch_rag_export.py \
  test/unit/tools/test_da_prim_replay.py \
  -q
docker logs api-dev --tail 200
docker logs worker-dev --tail 200
```

再用 Windows Anaconda Prompt 依次对同一条病例运行 `atlas_map`、`atlas_route`、
`da_prim_full`。三组最终回答的 `medication_review_trace.schema_version` 均应为
`6.0`，`atlas_profile` 与配置一致；Route/Full 的每条正常 routed search 应记录
`embedding_batch_count=1`、通常 `backend_search_count=7`，Map 不应产生 routed
record。首轮必须保持 `concurrency=1`。模型自动发现、真实 tool call、CPU embedding
延迟、Milvus 严格 file_id 过滤、多轮 search/open、最终答案与流式 history 均只能
由该远程冒烟确认。

## ACM-PRIM v3（PRIM + 两层调查地图）快速开始

ACM-PRIM v3 使用 `MedicationReviewAcmPrimAgent`，完整继承 PRIM v2 Full 的自主多轮
调查。Atlas 不过滤、扩充或重排 Milvus 结果，也不直接充当证据。Agent 每轮可以看到
知识库全部文档的标题和范围摘要，并自主决定是否调用 `open_atlas_document` 打开某篇
文档的详细主题列表。这里不再运行一个隐藏 Selector 替 Agent 挑选主题。

Agent 既可做普通全库 Top-10，也可用 Atlas 文档的真实 `file_id` 做文档内 Top-10；
后者只是调查地图指出“可去哪里深挖”，真实 Evidence 仍必须来自 Milvus。

### 1. 构建和检查 Atlas 3.0

以下命令在远程 Linux 主机的仓库根目录执行。`--model` 使用 Yuxi 已配置的完整模型名，
例如网页模型列表中显示的 `provider:model`。构建器会读取目标知识库的全部已索引 chunk，
每篇文档完整拼接后只调用一次构建模型，并逐篇建立缓存；只有所有文档成功后才切换
current 快照。构建命令不再接受 `--batch-max-chars` 或主题数量限制。

Atlas 构建提示或 Schema 更新后必须重新执行构建；旧 2.0 快照会被 Agent 明确拒绝，
不会静默复用。输出截断时构建会明确失败，需提高模型或网关的最大输出 token 后重建。

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.build_acm_corpus_atlas \
  --knowledge-name "用药助手-md" \
  --model "provider:model"

docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.build_acm_corpus_atlas \
  --knowledge-name "用药助手-md" \
  --check --deep-check
```

第二条命令会复用快照中记录的模型和构建参数。`--check` 只看知识库元数据，
`--deep-check` 还会重新读取全部 chunk 并核对内容哈希。知识库内容或构建提示、Schema、
参数发生变化后，在线 Agent 会明确提示 Atlas 过期，不能静默退回普通 PRIM。

构建完成后可导出一份只读人工复核包。默认输出到宿主机
`saves/exports/acm_corpus_atlas/`，其中 Markdown 适合按“文档概览→详细主题”通读，
`atlas_cues.csv` 适合筛选主题及轻量 `source_chunk_ids`。导出不调用 LLM 或 Milvus，
也不会修改快照：

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.export_acm_corpus_atlas \
  --knowledge-name "用药助手-md"
```

先查看命令输出中的 `output_dir`，再打开其中的 `atlas_review.md`。`source_chunk_ids`
只用于人工回溯，不会通过在线 Atlas 工具暴露给 Agent，也不会直接计为 Evidence。

### 2. 可选：离线回放历史 Selector

回放只重新运行一次伴随线索选择器，不调用 Milvus，也不重跑主 Agent。输入应优先使用
PRIM v2 Trace 8 的原始 batch JSONL，也可以使用 `export_rag_records.py` 导出的 JSON；
旧 PRIM v1 Trace 5 仍兼容。回放只重建首个有效检索时点可见的 open investigation，
不会把后续 Agent 的 answered 状态、工作结论或 selected evidence 泄漏给 Selector。
把输入文件放到宿主机 `saves/` 下，以便容器从 `/app/saves/` 读取；标答文件也复制到
同一目录：

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.replay_atlas_companion \
  --input /app/saves/prim-rag-v2-rag-records.json \
  --gold /app/saves/dataset_425.json \
  --output /app/saves/acm-companion-replay.jsonl \
  --db-id '<DB_ID>' \
  --model "provider:model" \
  --limit 20
```

`--gold` 只在选择器完成之后用于残余文档统计，不进入选择器输入。逐例结果会记录
`reconstruction_mode`：原始 batch history 可按 AI 消息精确识别触发批次；新版在线流程
每回合最多一次知识读取，正常情况下该批次只有一个查询。精简导出缺少消息边界时会标为
时间重叠近似。`trigger_investigations` 是 Selector 实际看到的调查状态；Atlas 建议文档
始终与真实 Evidence 分开，不计入正式召回。`residual_document` 以 PRIM 最终仍未召回
的金标准文档为分母；`trigger_point_document` 只描述触发时尚未召回的文档，二者不能混用。

历史 Selector 回放只用于分析旧方法，不属于 Atlas 3.0 在线主流程。

### 3. 建立 AgentConfig 并批量运行

先在网页中为 `MedicationReviewAcmPrimAgent` 新建一份配置。模型、系统 Prompt、
单一 Milvus 知识库、相似度阈值和 search/open 预算应与 PRIM Full 基线一致。这个 Agent 没有
`atlas_profile`，也不需要把前端配置名称 `da-prim-full` 写进脚本；批处理需要的是
AgentConfig 的数字 ID。

在 Windows Anaconda Prompt 中发现 ID：

```bat
set "YUXI_LOGIN_ID=你的登录ID"
set "YUXI_PASSWORD=你的密码"
python scripts\yuxi_batch_rag\discover_yuxi.py --base-url https://your-yuxi.example.com --output scripts\yuxi_batch_rag\discovery.json
```

复制 `config.acm-prim.example.json` 为 `config.acm-prim.local.json`，替换
`agent_config_id`、知识库名称、地址和输入文件。使用新的 `output_dir`，不要续跑旧
PRIM/DA-PRIM 目录。ACM 的请求档位由代码固定为 `full`，隐藏字段不会保存在网页
AgentConfig 中，所以样例不对它做请求档位预检；正式 Trace 则用
`expected_effective_profile=full` 核验实际运行档位。方案要素或患者事实抽取失败不会关闭
Full 的核心 Agent 循环，但会把 Trace 标为 `partial`，供正式分析时单独报告。先预检和
单病例，再全量运行：

```bat
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-prim.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-prim.local.json --variants acm_prim --log-level DEBUG --log-file outputs\acm-prim-smoke.log
```

正式记录必须是 Trace 10.0，`method_family=acm-prim-rag-v3`，且
`effective_profile` 仍为 `full`。首次成功结果会把 Atlas snapshot hash 锁入 manifest，
续跑时不允许混入另一版本。每条问题仍建立独立 thread，因此不同病例不会共享问答
历史。CPU 向量模型环境建议保持 `concurrency=1`。

### 4. 导出答案、RAG 证据和完整会话行为

```bat
python scripts\yuxi_batch_rag\export_answers.py --input outputs\acm-prim-rag-v3-top10\results\acm_prim.jsonl --output outputs\acm-prim-v3-top10-answers.json
python scripts\yuxi_batch_rag\export_rag_records.py --input outputs\acm-prim-rag-v3-top10\results\acm_prim.jsonl --output outputs\acm-prim-v3-top10-rag-records.json
python scripts\yuxi_batch_rag\export_session_records.py --input outputs\acm-prim-rag-v3-top10\results\acm_prim.jsonl --output outputs\acm-prim-v3-top10-sessions.json
```

第二份文件面向 RAG 评价。`retrieval_calls` 按执行顺序同时保留检索和打开原文，
`search_calls` 只含 Milvus/LightRAG 检索，`document_open_calls` 只含
`open_kb_document`、`open_evidence_source` 或 `open_review_evidence`；相应证据分别位于
`searched_evidence` 和 `opened_evidence`。`retrieved_evidence` 是 Agent 最终可用证据池。
`atlas_document_open_records` 只是查看调查地图文档的轨迹，不属于打开知识库原文。

第三份文件面向模型行为分析。`messages` 无损保留 Yuxi 历史消息，`timeline` 将其中的
模型中间文本、可获得的推理文本、全部工具调用与结果、最终回答按顺序展开，
`tool_calls` 则提供全部工具的统一列表。若已经从网页导出了单个会话 JSON，也可直接运行：

```bat
python scripts\yuxi_batch_rag\export_session_records.py --input outputs\conversations_dump\会话ID.json --output outputs\会话ID-session.json
```

推理内容只可能导出服务端实际保存的部分，包括 `reasoning_content`、`<think>` 内容和
调用工具前的模型中间文本。模型网关未返回或 Yuxi 未保存的隐藏推理无法事后恢复。

### 5. V7 调查努力实验

V7 不改动 A0 的检索和停止逻辑。另在网页中复制四份 ACM AgentConfig，分别设置：

| 配置 | `v7_experiment_arm` | `v7_retrieval_depth` | `max_search_calls` |
| --- | --- | --- | ---: |
| A1 后台 Top-25 | `a1` | `shadow_top25` | 50 |
| A2-K2 后台 Top-25 | `a2_k2` | `shadow_top25` | 50 |
| A2-K3 后台 Top-25 | `a2_k3` | `shadow_top25` | 50 |
| T25 可见消融 | `a1` | `visible_top25` | 50 |

6 次是 A1/A2 的最低有效搜索数，不是停止目标。完成最低合同后，Agent 仍可根据证据
缺口继续搜索，直到自主结束或达到 50 次运行保护上限。后端不设 50 的固定上界，确有
需要时可在所有对照组中统一配置得更高。`technical_failed` 不计入 6 次；
合同无法继续时仍保留最终答案，并在 Trace 11.0 的 `contract_report` 中标记原因。

历史 A0 和 Agentic RAG baseline 是否需要重跑只看一个确定事实：检查其 Trace 中
`budgets.executed_search_calls` 的最大值。若没有病例达到旧的 20 次上限，可以复用；若有
任何病例达到 20，则把需要比较的 A0、baseline 和 V7 组统一设置为 50（或统一的更高值）
重新运行。不同实验组不能使用会实际截断轨迹的不同搜索上限。

当前工作区三份历史导出已经检查过：ACM 425 例最多 9 次、向量 Agentic baseline
175 例最多 10 次、PRIM 425 例最多 14 次，均未触及 20，因此这三份记录可以复用。

`shadow_top25` 每次让 Milvus 返回 25 个块，但只把前 10 个登记到 Evidence Store 并
展示给 Agent；全部 25 个排名只写入 `retrieval_records`。`visible_top25` 才会把 25 个
块都展示给 Agent，因此应作为单独消融，不要与 A1 主组混用。

复制 `config.acm-v7.example.json` 为本地配置，替换四个数字 ID 后先预检，再按需运行
40 题开发集：

```bat
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-v7.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-v7.local.json --variants a1_shadow25,a2_k2_shadow25,a2_k3_shadow25 --log-level INFO --log-file outputs\acm-v7-dev.log
python scripts\yuxi_batch_rag\export_rag_records.py --input outputs\acm-prim-v7-dev\results\a2_k3_shadow25.jsonl --output outputs\acm-v7-a2-k3-rag-records.json
```

若要先回放现有 A0 或 baseline 的历史查询，不调用 LLM，可在远程 API 容器运行：

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m scripts.replay_acm_top25 \
  --knowledge-name "用药助手-md" \
  --records /app/outputs/acm-prim_ragrecords_425.json \
  --output /app/outputs/acm-top25-replay.jsonl \
  --timeout-seconds 600
```

回放严格保留历史查询文本、`global/document` 范围和 `file_id`，输出可直接交给现有
文档或块检索评估脚本；金标准不会进入回放查询过程。

### 5.1 自适应覆盖正式实验

自适应协议用于完善实验设置，不再把“恰好 K 个调查”“至少 6 次成功搜索”或
“每项两轮探查”作为业务完成条件。网页中的 AgentConfig 固定设置：

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `acm_protocol` | `adaptive_coverage` | 启用动态议程和覆盖缺口合同 |
| `v7_retrieval_depth` | `shadow_top25` | 后台取 25、Agent 只看前 10 |
| `max_search_calls` | `50` 或统一更高值 | 仅作为防失控保护，不是搜索目标 |

Agent 首次议程应为每个 PlanAnchor 分别建立 `current_regimen_review`，并记录
`appropriate / adjust / avoid`，从而同时覆盖“应保留的合理治疗”和“应调整的不合理治疗”。
其中 `adjust` 也包括“保留当前基础治疗，但需加用/强化缺失治疗”，不能因单药合理而忽略方案不完整。
一旦结论为 `adjust/avoid`，必须追加关联同一 PlanAnchor 的 `improvement_plan`，分别调查具体纠正、
药物/非药物替代、监测随访和长期管理；跨药方案可使用 `cross_regimen_review`。每项调查再拆成
可由直接证据回答的 `evidence_obligations`；后续只在现用药结论或全局缺口审计发现真实新问题时追加。
议程及义务数量不按剩余搜索预算裁剪，`max_search_calls` 只限制实际运行。
每次检索只能定向一个尚未覆盖义务；调查只有在所有义务都有成功 probe，且各自绑定由该 probe
返回或由其打开的相邻 Evidence 时才能设为 `answered`。全库检索命中候选文档却缺具体原文时，优先
按 `file_id` 做 `within_document_localization`；全库空结果允许缩窄/改写原子查询，文档内检索零新增后才强制转为全库来源发现。
单次 `query_text` 还必须保持单轴：使用不超过 6 个空格分隔的概念块，只表达一个主体、一个待查属性和至多一个患者限定，
不得把适应性、剂量、安全、监测、替代或长期管理枚举在同一查询里；不合规调用会在 Milvus 前被拒绝且不计入实际搜索。
动态控制记忆在检索阶段只展开当前 obligation，在调查关闭前恢复该调查全部 obligations，在最终审计阶段恢复完整议程；
该投影只减少重复提示，不放松逐义务证据合同或关闭门槛，已采纳 Evidence 原文仍逐字保留。
所有调查关闭后还要按适应性/获益、剂量/疗程、患者安全、监测/停药、替代/缺失治疗、跨方案/长期管理六维审计；
发现旧调查证据不足时会原子重开，不得只记录缺口后结束。Trace 12.0 记录实际调查数、证据义务支持矩阵、
实际搜索数、动态议程 revision、检索意图、恢复任务、六维审计与 checkpoint，
但不将任何数量写成质量目标。

复制 `config.acm-adaptive.example.json` 并替换 AgentConfig ID 后运行：

```bat
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-adaptive.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.acm-adaptive.local.json --variants adaptive_shadow25 --log-level INFO --log-file outputs\acm-adaptive-dev.log
python scripts\yuxi_batch_rag\export_rag_records.py --input outputs\acm-prim-adaptive-dev\results\adaptive_shadow25.jsonl --output outputs\acm-adaptive-rag-records.json
```

### 6. ACM-PRIM 远程代码验收

Atlas 构建和正式 batch 之前，先在 API 容器内执行下面这组测试。它覆盖 PRIM 默认
扩展位置不改变基线、整篇 Atlas 构建、两层在线导航、Trace 10.0、历史 Selector
离线回放，以及批处理和导出。这里使用 Compose 服务名 `api`；若部署文件使用其它
服务名，请按实际名称替换。

```bash
docker compose exec -w /app api uv run --no-sync --no-dev python -m pytest \
  test/unit/agents/medication_review_prim/test_harness.py \
  test/unit/agents/medication_review_acm_prim \
  test/unit/scripts/test_build_acm_corpus_atlas_script.py \
  test/unit/scripts/test_export_acm_corpus_atlas_script.py \
  test/unit/scripts/test_replay_acm_top25_script.py \
  test/unit/scripts/test_replay_atlas_companion_script.py \
  test/unit/plugins/test_milvus_kb.py \
  test/unit/tools/test_yuxi_batch_rag_script.py \
  test/unit/tools/test_yuxi_batch_rag_export.py \
  -q
docker compose exec -w /app api uv run --no-sync --no-dev ruff check \
  package/yuxi/agents/buildin/medication_review_prim \
  package/yuxi/agents/buildin/medication_review_acm_prim \
  scripts/build_acm_corpus_atlas.py \
  scripts/export_acm_corpus_atlas.py \
  scripts/replay_acm_top25.py \
  scripts/replay_atlas_companion.py
```

这两条命令只使用镜像中已经安装的包，不会根据 `uv.lock` 重新安装 Torch。旧系统的
镜像若没有 `pytest` 或 `ruff`，不要在实验机器上强行同步开发依赖；改在可用的开发/CI
环境完成代码级测试，实验机器继续执行下面的 Atlas 深检和在线 smoke test。

随后先运行 Atlas `--check --deep-check`，再分别用一条“Agent 从文档概览打开主题后按
file_id 深挖”、一条“首轮空结果”和一条“模型不检索直接作答”的病例验收。确认 Trace
10.0 中 `atlas_document_open_records` 与真实工具调用一致、地图消息未暴露来源 chunk、
实际证据来自全库或单文档 vector Top-10、最终答案存在后，再做 10～20 条 smoke test。
只有 PRIM v2 先通过预设检索门槛，才正式比较 P+A。

V7 另用同一条复杂病例依次验收 A1、A2-K2 和 A2-K3：Trace 应为 11.0；A1 的
`successful_search_calls` 不少于 6；A2 的议程数量分别为 2/3，所有必需调查都有
`initial_probe` 和 `complementary_probe`。在 `shadow_top25` 下，`fetch_k=25`、
`visible_k=10`，第 11—25 名只出现在 `retrieval_records`，不能进入 Evidence Store。
若 smoke 轨迹出现预算耗尽、工具参数连续错误或未关闭调查，还应确认 Markdown 最终答案
仍被保留，且 `contract_report.status=incomplete`；不要把这类轨迹当作完成合同的样本。

自适应协议另验收一条同时包含合理治疗保留、方案调整、交互作用、安全边界和长期随访的复杂病例：
Trace 应为 12.0，`protocol=adaptive_coverage`，`retrieval_depth=shadow_top25`；议程数量由
病例决定而非固定值，每个 PlanAnchor 有独立 `current_regimen_review`；`adjust/avoid` 必须触发关联
`improvement_plan`，每个 evidence obligation 都有独立定向 probe 和可追溯的 Evidence 支持；命中正确文档但块不相关时
出现 `within_document_localization`，文档内零新增后存在并完成全库恢复；最终六个 `coverage_audit` 维度完整，
`adaptive_coverage_report.status=completed` 且当前 gap assessment 无实质缺口。实际调查数和搜索数只做描述性记录，
不能作为通过阈值。

## 1. PEA-RAG v2 的三个开关

开关配置在网页中的 `MedicationReviewAgent` AgentConfig 上，而不是批量脚本临时修改远程配置。

### `run_mode`

| 值 | 停止位置 | 结果状态 |
| --- | --- | --- |
| `stop_after_plan` | 病例及方案要素抽取后 | `debug_stopped` |
| `stop_after_agenda` | 动态审查问题生成后 | `debug_stopped` |
| `stop_after_retrieval` | Agent 检索和最终证据选择后 | `debug_stopped` |
| `stop_after_claims` | Claim 抽取后 | `debug_stopped` |
| `full` | 患者级综合及六段式答案后 | `completed`/`partial` |

### `agenda_mode`

- `none`：不做预先议程，检索 Agent 根据病例、方案要素和当前证据自主检索。
- `dynamic`：先由模型生成自由文本审查问题，再由同一个检索 Agent 自主执行、补查和停止。

### `synthesis_mode`

- `direct_chunks`：最终 Synthesizer 直接读取选中的原始片段。
- `claims`：先从片段抽取带 source span 的 Claim，再进行患者级综合。

引用校验、短 Evidence ID、技术错误隔离、局部降级、Trace 3.0 和六段式渲染始终启用，不作为消融变量。

## 2. 推荐的实现与远程验证方式

代码是一条完整纵向链路，P0—P4 只作为远程验收闸门，不是五套 Agent。为每个实验条件复制一份 AgentConfig，并在网页中设置对应开关。批量脚本的 `variants` 只引用这些固定的 AgentConfig ID。

推荐配置矩阵：

| 配置名 | run_mode | agenda_mode | synthesis_mode | 用途 |
| --- | --- | --- | --- | --- |
| `p0_plan` | `stop_after_plan` | `none` | `direct_chunks` | 验证自由文本解析 |
| `p1_retrieval_no_agenda` | `stop_after_retrieval` | `none` | `direct_chunks` | 验证自主检索基本链 |
| `p2_retrieval_dynamic_agenda` | `stop_after_retrieval` | `dynamic` | `direct_chunks` | 验证动态议程 |
| `full_direct_chunks` | `full` | `dynamic` | `direct_chunks` | 完整直接片段条件 |
| `full_dynamic_claims` | `full` | `dynamic` | `claims` | 默认完整候选方法 |

所有配置都应：

- 只选择同一个 Milvus 知识库；
- `retrieval_top_k=5`；
- `max_search_calls=8`；
- `max_open_calls=2`；
- `max_agent_steps=12`；
- `max_subqueries_per_action=3`；
- 首轮 `concurrency=1`；
- 正式批量关闭 `diagnostic_trace`。

检索实现固定为 Milvus vector-only，不使用 BM25、LightRAG 或 reranker。一次 `search_evidence` 可提交 1–3 条自然语言临床命题，后端按顺序编码和检索，避免 CPU 向量模型被并发压满。

## 3. 配置与 Windows Anaconda Prompt

复制 `config.example.json` 为 `config.local.json`，把 AgentConfig ID 和知识库名称替换为远程实例中的真实值。每个 variant 的 `expected_run_mode`、`expected_agenda_mode` 和 `expected_synthesis_mode` 会在 preflight 中核对远程 AgentConfig，避免实验 ID 配错后产生不可归因的结果。阶段停止记录只有在本地配置设置以下选项时才视为成功：

```json
"allow_debug_stopped": true
```

正式全量实验建议改回 `false`，防止把诊断记录误当正式答案。

Anaconda Prompt 使用 CMD 语法：

```bat
set YUXI_LOGIN_ID=你的登录ID
set YUXI_PASSWORD=你的密码
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --preflight-only
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants p0_plan --log-level DEBUG --log-file output\p0.log
```

不要在 Anaconda Prompt 中使用 PowerShell 的 `$env:NAME=...` 语法。

不知道 ID 时：

```bat
python scripts\yuxi_batch_rag\discover_yuxi.py --base-url https://your-yuxi.example.com --output scripts\yuxi_batch_rag\discovery.json
```

输入文件必须是 JSON 列表，每项至少包含非空 `question`：

```json
[
  {
    "case_id": "case-001",
    "question": "病例自由文本……",
    "answer": "金标准答案……"
  }
]
```

`answer` 等额外字段只原样保存在结果中，不会发送给 Yuxi。

## 4. 建议的远程验收顺序

每个闸门先跑 1 条复杂病例，确认 Trace 后再跑 12 条固定开发集。

```bat
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants p0_plan
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants p1_retrieval_no_agenda
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants p2_retrieval_dynamic_agenda
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants full_direct_chunks
python scripts\yuxi_batch_rag\batch_yuxi_rag.py --config scripts\yuxi_batch_rag\config.local.json --variants full_dynamic_claims
```

每个闸门检查：

1. `schema_version` 为 `3.0`，`effective_profile` 与 AgentConfig 开关一致。
2. `debug_stopped` 的 `last_completed_stage` 正确，且不会被答案导出器当正式答案。
3. 每个显式方案要素都有且只有一个 `ElementReview`。
4. `search_records[].subquery.query_text` 是单一自然语言临床命题，最多执行 8 条。
5. `search_evidence` 内多个子查询按顺序完成，没有 CPU 编码并发。
6. Evidence 使用 `EV001...`，同内容跨查询合并但保留全部 occurrence。
7. 引用的 `source_span` 可在对应 Evidence 原文中找到。
8. 单个无效 Claim 或引用只产生 `local_validation_events` 和局部降级。
9. `full` 输出含六个固定章节，批量记录状态为 `completed` 或 `partial`。

通过阶段闸门后，再对同一 12 条开发集运行以下 2×2 消融：

- `agenda_mode=none` / `dynamic`
- `synthesis_mode=direct_chunks` / `claims`

不要同时改变 Top-K、向量模型、知识库或并发，否则无法归因。

## 5. Trace 3.0 与导出

主要字段：

- `plan_elements`、`patient_facts`
- `review_agenda`
- `agent_steps`
- `search_records`、`open_records`
- `evidence`、`evidence_selection`
- `evidence_claims`
- `review_synthesis`
- `local_validation_events`
- `final_review`
- `usage`

正式回答：

```bat
python scripts\yuxi_batch_rag\export_answers.py --input output\batch\results\full_dynamic_claims.jsonl --output output\answers.json
```

检索与 Claim 记录：

```bat
python scripts\yuxi_batch_rag\export_rag_records.py --input output\batch\results\full_dynamic_claims.jsonl --output output\rag_records.json
```

完整会话行为：

```bat
python scripts\yuxi_batch_rag\export_session_records.py --input output\batch\results\full_dynamic_claims.jsonl --output output\session_records.json
```

`export_answers.py` 会拒绝 Trace 3.0 的 `debug_stopped` 记录，避免将阶段诊断文本用于答案评价。

## 6. 不重新检索的后检索回放

回放 Trace 2.0/3.0 中已保存的证据，可只比较两种综合方式：

```bat
python scripts\yuxi_batch_rag\replay_medication_review_trace.py --input output\batch\results\full_dynamic_claims.jsonl --output-dir output\replay-direct --synthesis-mode direct_chunks
python scripts\yuxi_batch_rag\replay_medication_review_trace.py --input output\batch\results\full_dynamic_claims.jsonl --output-dir output\replay-claims --synthesis-mode claims
```

可用 `--model` 覆盖 Trace 中记录的模型。回放会重新调用 Claim/Synthesis 模型，但不访问 Milvus。

## 7. 输出与恢复

```text
output/pea-rag-v2-ablation/
  manifest.json
  state.jsonl
  results/<variant>.jsonl
  events/<variant>/<row>.attempt<n>.jsonl
```

- 相同 `variant + row_index` 使用独立 thread。
- 已成功落盘的任务自动跳过。
- 未完成 Run 会优先续接。
- 恢复批次时会重新读取远程 AgentConfig；同一 ID 的上下文若已被修改，脚本会拒绝继续，必须使用新的 `output_dir`。
- 每个结果保留完整 history、工具调用、Trace 和原始 SSE 文件路径。
- CPU BGE-M3 环境先保持 `concurrency=1`；确认单条延迟和资源稳定后最多小幅提高。

## 8. 本地与远程测试边界

本地不连接模型和知识库时，可执行语法检查、纯函数/fixture 单测和批处理 HTTP 假对象测试。真正的 LangGraph 编译、模型 structured-output fallback、Milvus 检索与最终效果必须在远程 Docker 环境执行：

```bash
docker compose up -d
docker exec api-dev pytest \
  test/unit/agents/medication_review \
  test/unit/tools/test_yuxi_batch_rag_script.py \
  -q
docker exec api-dev pytest \
  test/integration/api/test_medication_review_agent.py \
  -q
```

正式批量前还应按项目规范执行 format、lint，并用一条真实病例完成 `full_dynamic_claims` 端到端冒烟。
