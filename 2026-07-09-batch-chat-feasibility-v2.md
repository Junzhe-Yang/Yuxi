# Yuxi 0.6.3 批量会话可行性分析报告（修订版）

- **日期**: 2026-07-09
- **目标版本**: Yuxi 0.6.3（Docker 部署，本地大模型 + bge-m3 向量模型）
- **分析依据**: 仓库源码（`backend/` 全量阅读）+ 在线文档 `https://xerrors.github.io/Yuxi/`
- **本次需求重述**:
  1. **不使用** Yuxi 内置 RAG 评估系统（指标固定、不支持 LightRAG）。
  2. **批量完成会话**：每条问题单独开一个会话询问一次。
  3. **保留 Yuxi 会话设置能力**：选择某个知识库、修改系统提示词等。
  4. **导出内容**：回答 + 具体的工具调用 + 调用知识库检索时的具体检索内容。
  5. **同时覆盖向量库（Milvus）与图谱库（LightRAG）** 各做一次。

---

## 一、结论先行

**完全可行。** 核心通路是「通过 API 创建/复用 AgentConfig（设置知识库与系统提示词）→ 用同步对话接口 `POST /api/chat/agent/sync` 逐条发问 → 拉取 `GET /api/chat/thread/{thread_id}/history` 拿到回答与全部工具调用及检索内容 → 落盘导出」。

关键源码事实（已逐一核对，非文档转述）：

1. **会话设置完全可编程**：`AgentConfig.config_json.context` 里 `system_prompt` / `knowledges` / `model` / `tools` / `mcps` / `skills` 全是普通字段，可通过 `PUT /api/chat/agent/{agent_id}/configs/{config_id}` 用 API 直接修改。前端 UI 改这些字段时，最终也是写到这里。
2. **知识库选择**：`context.knowledges` 是**知识库名称列表**（不是 db_id），用名称做白名单过滤。设为 `null` 表示"当前用户可访问的全部知识库"。
3. **检索内容可取**：Agent 调用 `query_kb` 工具时，工具结果会落库到 `tool_calls.tool_output`，并通过 `GET /api/chat/thread/{thread_id}/history` 返回（`tool_calls[].tool_call_result.content`，是 JSON 字符串，需 `json.loads`）。
4. **向量库 vs 图谱库无差异**：Agent 都是通过 `query_kb` 工具检索；底层 `KnowledgeBaseManager.get_retrievers` 统一闭包，固定传 `agent_call=True`，无论 Milvus 还是 LightRAG 都返回结构化的 chunks / entities / relationships。
5. **不依赖内置评估**：本方案绕开 `/api/evaluation/*`，自行用会话 API 拉数据后自算指标。

---

## 二、会话设置能力总览（已核对源码）

`AgentConfig` 表存于 Postgres，关键字段为 `config_json`（JSONB）。其 `context` 子对象就是 LangGraph 智能体的运行时上下文，字段定义见 [context.py](file:///workspace/backend/package/yuxi/agents/context.py)：

| 字段 | 类型 | 作用 | 默认 |
| --- | --- | --- | --- |
| `system_prompt` | str | 系统提示词 | "You are a helpful assistant." |
| `model` | str | 主模型标识 `provider_id:model_id` | 系统默认 |
| `tools` | list[str] | 启用的内置工具（如 `tavily_search`） | `["ask_user_question","tavily_search"]` |
| **`knowledges`** | list[str] \| null | **知识库名称列表**（白名单）；`null`=全部可访问 KB | `null` |
| `mcps` | list[str] | 启用的 MCP 服务器 | `[]` |
| `skills` | list[str] | 关联的 Skills（含内置 `knowledge-base`） | `[]` |
| `subagents` | list[str] | 子智能体 | `[]` |
| `subagents_model` | str | 子智能体默认模型 | 系统默认 |
| `summary_threshold` | int | 上下文摘要阈值（KB） | 100 |

> ⚠️ 注意：`knowledge-base` Skill 是**默认激活**的内置技能，它派生出 `list_kbs` / `query_kb` / `find_kb_document` / `open_kb_document` / `get_mindmap` 五个工具。即使 `skills` 留空，这些 KB 工具也会被挂载。所以**无需在 `skills` 里手动加 `knowledge-base`**。

**知识库绑定机制**（源码：[knowledge_base_backend.py](file:///workspace/backend/package/yuxi/agents/backends/knowledge_base_backend.py)）：运行时按 `context.knowledges`（名称列表）从用户可访问的全部 KB 中过滤出 `_visible_knowledge_bases`，`query_kb` 工具执行时只在这批 KB 里检索。**与 mention 机制无关**——mention 是前端编辑器里"@"提及文件的功能，最终也只影响附件，不是绑定知识库的方式。

---

## 三、向量库 vs 图谱库的会话层差异

| 维度 | Milvus（向量库） | LightRAG（图谱库） |
| --- | --- | --- |
| Agent 调用工具 | `query_kb`（统一） | `query_kb`（统一） |
| 工具入参 | `kb_name` + `query_text` + 可选 `file_name` | `kb_name` + `query_text` |
| `agent_call=True` 返回结构 | `[{"content","metadata":{"source","chunk_id","file_id","chunk_index"},"score"}]` 列表 | 取决于知识库配置 `retrieval_content_scope`：`chunks` / `graph` / `all` |
| `chunks` scope | 与 Milvus 同结构 | `{content, metadata:{file_id, source_id...}}` 列表 |
| `graph` scope | 不适用 | `{entities:[{entity_name,description,...}], relationships:[{src/tgt,description,...}], references:[]}` |
| `all` scope | 不适用 | chunks + entities + relationships 合并 |
| 多轮检索能力 | 单次向量+可选 rerank | LightRAG 内部已融合向量+图谱 |
| 是否支持内置评估 | ✅（但你不用） | ❌（这正是你换方案的原因） |
| 会话 API 调用方式 | **完全相同** | **完全相同** |

源码确认（[base.py:1055-1077](file:///workspace/backend/package/yuxi/knowledge/base.py) + [lightrag.py:561-631](file:///workspace/backend/package/yuxi/knowledge/implementations/lightrag.py) + [milvus.py:559](file:///workspace/backend/package/yuxi/knowledge/implementations/milvus.py)）。

---

## 四、批量会话完整链路

### 4.1 鉴权（与上版相同，简述）

- API Key 形如 `yxkey_<48位十六进制>`，绑定 admin 身份可访问所有接口。
- 创建：登录 Web → 用户菜单 → API Key 管理；或 `POST /api/apikey/`（先用账号密码换 JWT）。
- 所有请求头：`Authorization: Bearer yxkey_xxx`
- 后续示例默认 `BASE=http://localhost:5050`、`KEY=...`、`AUTH="Authorization: Bearer $KEY"`。

### 4.2 准备阶段（一次性）

#### 4.2.1 列出知识库，拿到名称

```bash
curl -s "$BASE/api/knowledge/databases" -H "$AUTH" | jq '.[] | {name, db_id, kb_type, description}'
```

记下两个知识库的**名称**（不是 db_id，因为 `knowledges` 字段用名称）与 `kb_type`：
- 向量库：`MILVUS_KB_NAME=<名称>`、`kb_type=milvus`
- 图谱库：`LIGHTRAG_KB_NAME=<名称>`、`kb_type=lightrag`

#### 4.2.2 列出智能体与已有配置

```bash
# 1) 智能体 slug（形如 chatbot / deep_agent）
curl -s "$BASE/api/chat/agent" -H "$AUTH" | jq '.agents[] | {id, name}'

# 2) 该智能体下已有配置
AGENT_ID=chatbot
curl -s "$BASE/api/chat/agent/$AGENT_ID/configs" -H "$AUTH" | jq
```

#### 4.2.3 创建两套专用配置（向量库一套、图谱库一套）

为避免污染默认配置，建议**新建两个独立配置**，分别只绑定一个知识库。

```bash
# 向量库配置
MILVUS_CFG=$(curl -s -X POST "$BASE/api/chat/agent/$AGENT_ID/configs" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "name": "批量会话-向量库",
    "description": "仅绑定向量库，用于批量评测",
    "set_default": false,
    "config_json": {
      "context": {
        "system_prompt": "你是一个严谨的问答助手，请优先检索知识库后再回答；若知识库无相关内容，请如实回答不知道。",
        "model": "<provider_id>:<chat_model_id>",
        "knowledges": ["'"$MILVUS_KB_NAME"'"],
        "tools": [],
        "skills": []
      }
    }
  }' | jq -r '.config.id')
echo "向量库配置 ID: $MILVUS_CFG"

# 图谱库配置（同样模式，仅改 knowledges 与可选提示词）
LIGHTRAG_CFG=$(curl -s -X POST "$BASE/api/chat/agent/$AGENT_ID/configs" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "name": "批量会话-图谱库",
    "description": "仅绑定图谱库，用于批量评测",
    "set_default": false,
    "config_json": {
      "context": {
        "system_prompt": "你是一个严谨的问答助手，请优先检索知识库后再回答；若知识库无相关内容，请如实回答不知道。",
        "model": "<provider_id>:<chat_model_id>",
        "knowledges": ["'"$LIGHTRAG_KB_NAME"'"],
        "tools": [],
        "skills": []
      }
    }
  }' | jq -r '.config.id')
echo "图谱库配置 ID: $LIGHTRAG_CFG"
```

> 提示：源码 `AgentConfigCreate` 的 `config_json` 是可选字段，但**完整填 context**最稳妥；`tools` 留空表示该 agent 不挂额外工具（KB 工具由 `knowledge-base` Skill 自动派生，不受此影响）。

#### 4.2.4 若需修改系统提示词（批量调试用）

不必每次新建配置，直接 PUT 更新：

```bash
curl -s -X PUT "$BASE/api/chat/agent/$AGENT_ID/configs/$MILVUS_CFG" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "config_json": {
      "context": {
        "system_prompt": "你的新提示词...",
        "model": "<provider_id>:<chat_model_id>",
        "knowledges": ["'"$MILVUS_KB_NAME"'"],
        "tools": [],
        "skills": []
      }
    }
  }' | jq
```

> ⚠️ PUT 是**全量覆盖 `config_json`**（源码 `AgentConfigRepository.update`），所以每次更新都要把完整的 context 传一遍，否则会丢字段。

### 4.3 批量会话执行阶段（每条问题重复）

对每条问题 `q`：

#### 步骤 1：发起同步对话

```bash
THREAD_ID=$(uuidgen)
curl -s -X POST "$BASE/api/chat/agent/sync" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "query": "<问题文本>",
    "agent_config_id": '"$MILVUS_CFG"',
    "thread_id": "'"$THREAD_ID"'",
    "meta": {"request_id": "'"$(uuidgen)"'"}
  }' | jq
```

返回结构（源码 `agent_chat`）：

```jsonc
{
  "status": "finished",       // finished / error / interrupted
  "response": "AI 完整答案",    // ← LLM 回答
  "thread_id": "...",
  "agent_state": {"todos":[],"files":{},"artifacts":[]},
  "time_cost": 3.21,
  "request_id": "..."
}
```

**关键**：每条问题用**全新 `thread_id`**（UUID），即可保证上下文隔离，无需显式 `POST /api/chat/thread`（sync 接口不传 `thread_id` 也会自动生成，但建议显式传以便后续拉历史）。

#### 步骤 2：拉取会话历史，提取工具调用与检索内容

```bash
curl -s "$BASE/api/chat/thread/$THREAD_ID/history" -H "$AUTH" | jq > "history_$THREAD_ID.json"
```

返回结构（源码 `get_thread_history_view`）：

```jsonc
{
  "history": [
    {"id":1, "type":"human", "content":"问题", ...},
    {"id":2, "type":"ai", "content":"AI 答案片段（可能为空）", "tool_calls":[
      {
        "id":"1", "name":"query_kb",
        "args":{"kb_name":"<向量库名称>","query_text":"关键词"},
        "tool_call_result": {
          "content": "[{\"content\":\"...\",\"metadata\":{\"source\":\"...\",\"chunk_id\":\"...\",\"file_id\":\"...\",\"chunk_index\":0},\"score\":0.83}]"  // ← JSON 字符串
        },
        "status":"success"
      }
    ]},
    {"id":3, "type":"ai", "content":"最终 LLM 答案", "tool_calls":[]}
  ]
}
```

**提取要点**：

1. **回答**：取最后一条 `type=="ai"` 且 `tool_calls` 为空（或非空但 content 非空）的消息 `content`；若用同步接口，直接取 `sync.response` 更省事。
2. **工具调用**：遍历所有 `type=="ai"` 消息的 `tool_calls[]`，记录 `name` / `args` / `status`。
3. **检索内容**：对 `name=="query_kb"` 的工具调用，`json.loads(tool_call_result.content)` 得到 chunk 列表。
   - 向量库：`[{content, metadata:{source,chunk_id,file_id,chunk_index}, score}]`
   - 图谱库（`scope=chunks`）：同结构
   - 图谱库（`scope=graph`）：`{entities:[...], relationships:[...], references:[...]}`
   - 图谱库（`scope=all`）：两者合并
4. 若 Agent 还调用了 `find_kb_document` / `open_kb_document`，这些工具的输出也都在对应 `tool_call_result.content` 里。

### 4.4 重要补充：异步 run（SSE）是否需要？

**不需要。** 经源码核对（[run_worker.py](file:///workspace/backend/package/yuxi/services/run_worker.py) + [agent_run_service.py](file:///workspace/backend/package/yuxi/services/agent_run_service.py)）：

- 异步 run 的 SSE 事件流（`/api/chat/runs/{run_id}/events`）只把 `stream_agent_chat` 的 chunk 转发，**loading 状态的 chunk 只包含 LLM 输出的增量 token**（`AIMessageChunk.content`）。
- 工具调用的具体结果**不会出现在 SSE 流中**，它们只通过 `save_messages_from_langgraph_state` 落库到 conversation history。
- 所以无论用同步（sync）还是异步（run），**最终都必须从 `/api/chat/thread/{thread_id}/history` 拿工具调用与检索内容**。

同步 sync 接口的优势：阻塞到完成、直接返回 `response`、不依赖 SSE 消费、单条 timeout 即可控制。**本方案推荐用 sync**。

异步 run 的唯一优势是可中断、可观察中间 token 流；对纯导出场景无用，且增加 SSE 消费复杂度。

---

## 五、完整可运行脚本（向量库 + 图谱库各跑一遍）

```python
# batch_chat_export.py
# 用法: python batch_chat_export.py
import json, time, uuid, requests

BASE = "http://localhost:5050"
KEY  = "yxkey_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
AUTH = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

# 一次性准备好的两个配置 ID（见 §4.2.3）
AGENT_CONFIGS = {
    "milvus":   {"id": 123, "kb_name": "<向量库名称>"},
    "lightrag": {"id": 124, "kb_name": "<图谱库名称>"},
}

# 数据集：每行一个 JSON，至少含 query；可选 gold_answer 用于自算指标
DATASET_PATH = "questions.jsonl"
# 输出目录
OUT_DIR = "results"

import os
os.makedirs(OUT_DIR, exist_ok=True)

def run_one(query: str, agent_config_id: int, kb_name: str) -> dict:
    thread_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    payload = {
        "query": query,
        "agent_config_id": agent_config_id,
        "thread_id": thread_id,
        "meta": {"request_id": request_id},
    }
    # 1) 同步对话（最多等 5 分钟，按你的 LLM 调整）
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/api/chat/agent/sync", headers=AUTH,
                               json=payload, timeout=300)
            r.raise_for_status()
            sync = r.json()
            break
        except Exception as e:
            if attempt == 2:
                return {"status": "error", "error": str(e), "thread_id": thread_id}
            time.sleep(2 ** attempt)

    if sync.get("status") != "finished":
        return {"status": sync.get("status"), "error": sync.get("error_message"),
                "thread_id": thread_id, "raw": sync}

    answer = sync.get("response", "")

    # 2) 拉历史，提取工具调用与检索内容
    hr = requests.get(f"{BASE}/api/chat/thread/{thread_id}/history",
                      headers=AUTH, timeout=60)
    hr.raise_for_status()
    history = hr.json().get("history", [])

    tool_calls = []
    retrieved_chunks = []
    for msg in history:
        if msg.get("type") != "ai":
            continue
        for tc in msg.get("tool_calls", []):
            entry = {
                "name": tc.get("name"),
                "args": tc.get("args"),
                "status": tc.get("status"),
            }
            raw = (tc.get("tool_call_result") or {}).get("content", "")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) and raw else raw
            except json.JSONDecodeError:
                parsed = raw
            entry["result"] = parsed
            tool_calls.append(entry)
            if tc.get("name") == "query_kb" and isinstance(parsed, list):
                retrieved_chunks.extend(parsed)

    return {
        "status": "finished",
        "thread_id": thread_id,
        "answer": answer,
        "tool_calls": tool_calls,                 # ← 所有工具调用（含 args + result）
        "retrieved_chunks": retrieved_chunks,      # ← query_kb 检索到的具体内容
        "kb_name": kb_name,
        "time_cost": sync.get("time_cost"),
    }

# 主循环：两套配置各跑一遍
questions = [json.loads(l) for l in open(DATASET_PATH, encoding="utf-8") if l.strip()]
for kb_kind, cfg in AGENT_CONFIGS.items():
    out_path = f"{OUT_DIR}/results_{kb_kind}.jsonl"
    done_queries = set()
    if os.path.exists(out_path):  # 断点续跑
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_queries.add(json.loads(line)["query"])
                except Exception:
                    pass
    with open(out_path, "a", encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            if q["query"] in done_queries:
                continue
            result = run_one(q["query"], cfg["id"], cfg["kb_name"])
            record = {
                "query": q["query"],
                "gold_answer": q.get("gold_answer", ""),
                **{k: v for k, v in q.items() if k not in ("query","gold_answer")},
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{kb_kind} {i}/{len(questions)}] status={result['status']} "
                  f"chunks={len(result.get('retrieved_chunks',[]))} "
                  f"tools={len(result.get('tool_calls',[]))} "
                  f"time={result.get('time_cost')}s")
    print(f"[{kb_kind}] done -> {out_path}")
```

输出每行 JSON 字段：

```jsonc
{
  "query": "...",
  "gold_answer": "...",          // 你数据集里的标准答案（若有），便于自算指标
  "status": "finished",
  "thread_id": "...",
  "answer": "AI 回答",
  "tool_calls": [
    {"name":"query_kb","args":{"kb_name":"...","query_text":"..."},"status":"success",
     "result":[{"content":"...","metadata":{"source":"...","chunk_id":"...","file_id":"...","chunk_index":0},"score":0.83}]}
  ],
  "retrieved_chunks": [
    {"content":"...","metadata":{...},"score":0.83}
  ],
  "kb_name":"...",
  "time_cost": 3.21
}
```

---

## 六、并发与稳定性建议

1. **并发度**：本地大模型推理是瓶颈，建议 `concurrent.futures.ThreadPoolExecutor(max_workers=2~4)`，按你的 GPU/内存调。两套 KB 可以并行跑（互不冲突），但 LLM 还是同一个，会争资源。
2. **timeout**：sync 接口阻塞调用，`timeout=300` 起步；若问题复杂、Agent 多轮检索，可放到 600。
3. **断点续跑**：脚本已实现按 `query` 去重，中断后重跑会跳过已处理项。
4. **错误隔离**：单条失败不影响整体，落盘时记 `status=error` 与 `error_message`，事后可单独重试。
5. **请求追踪**：每条带唯一 `meta.request_id`，便于 `docker logs api-dev --tail 200 | grep <request_id>` 定位单条问题。
6. **图谱库特殊参数**：LightRAG 的检索行为受知识库配置 `retrieval_content_scope`（chunks/graph/all）影响。若你想让导出包含实体/关系，确保该知识库配置为 `all` 或 `graph`（可在 Web 知识库详情页改，或 `PUT /api/knowledge/databases/{db_id}/query-params`）。
7. **`tools` 字段注意**：若想让 Agent 只检索不联网，把 `tools` 设为 `[]`（已默认）。但 `tavily_search` 等若未在 `tools` 里则不会被启用——本场景正合适。

---

## 七、自定义指标计算（在脚本侧自行实现）

由于不使用内置评估，导出 JSONL 后用 pandas / 自写脚本计算你的定制指标。常见模式：

```python
# compute_metrics.py
import json

results = [json.loads(l) for l in open("results/results_milvus.jsonl", encoding="utf-8")]

for r in results:
    retrieved = r.get("retrieved_chunks", [])
    # 你的定制检索指标（如 MRR、NDCG、命中率等）
    # 你已知的"期望文档"如何与 retrieved_chunks 里的 source/chunk_id 对齐，由你定义

    answer = r.get("answer", "")
    gold = r.get("gold_answer", "")
    # 你的定制答案指标（如 BLEU、ROUGE、语义相似度、自定义 LLM Judge）
    # 可在此调用本地大模型做 judge，提示词完全由你控制
    pass
```

这种模式的优势：指标定义、judge 提示词、对照基准全部由你掌控，比内置评估灵活。

---

## 八、关键源码索引（修订版重点）

| 关注点 | 文件 |
| --- | --- |
| AgentConfig CRUD 路由 | [chat_router.py](file:///workspace/backend/server/routers/chat_router.py) (create/update/delete agent config) |
| AgentConfig 仓储 | [agent_config_repository.py](file:///workspace/backend/package/yuxi/repositories/agent_config_repository.py) |
| 智能体上下文字段 | [context.py](file:///workspace/backend/package/yuxi/agents/context.py) (BaseContext) |
| 同步对话实现 | [chat_service.py](file:///workspace/backend/package/yuxi/services/chat_service.py) (agent_chat) |
| 会话历史装配（含 tool_calls） | [conversation_service.py](file:///workspace/backend/package/yuxi/services/conversation_service.py) (get_thread_history_view) |
| 知识库绑定逻辑 | [knowledge_base_backend.py](file:///workspace/backend/package/yuxi/agents/backends/knowledge_base_backend.py) |
| KB 工具（query_kb 等） | [kbs/tools.py](file:///workspace/backend/package/yuxi/agents/toolkits/kbs/tools.py) |
| Retriever 闭包（统一 agent_call=True） | [base.py](file:///workspace/backend/package/yuxi/knowledge/base.py) (get_retrievers) |
| Milvus 查询返回结构 | [milvus.py](file:///workspace/backend/package/yuxi/knowledge/implementations/milvus.py) (aquery/_build_chunk_from_hit) |
| LightRAG 查询返回结构 | [lightrag.py](file:///workspace/backend/package/yuxi/knowledge/implementations/lightrag.py) (aquery, scope 分支) |
| 异步 run SSE（不用于导出） | [run_worker.py](file:///workspace/backend/package/yuxi/services/run_worker.py) + [agent_run_service.py](file:///workspace/backend/package/yuxi/services/agent_run_service.py) |
| API Key 鉴权 | [auth_middleware.py](file:///workspace/backend/server/utils/auth_middleware.py) |
| 同步对话测试参考 | [test_chat_agent_sync.py](file:///workspace/backend/test/integration/api/test_chat_agent_sync.py) |

---

## 九、与上一版报告的差异

| 维度 | 上版（被弃用部分） | 本版（聚焦） |
| --- | --- | --- |
| 主方案 | 内置 RAG 评估系统（方案 A） | 批量会话 API 循环（sync + history） |
| 是否依赖内置评估 | 是 | 否 |
| 是否支持 LightRAG | ❌ 内置评估不支持 | ✅ 与 Milvus 完全同链路 |
| 指标 | 内置 Recall/F1/答案准确性 | 不计算，由你导出后自算 |
| 工具调用导出 | 不导出（直查不走 Agent） | ✅ 完整导出 args + result |
| 系统提示词 | 固定 | ✅ 可通过 API 任意修改 |
| 知识库选择 | 评估直查 KB | ✅ 通过 `context.knowledges` 绑定，复现真实对话 |

---

## 十、最终建议

1. **按 §4.2.3 创建两套配置**（向量库一套、图谱库一套），分别只绑定一个知识库，并设好你想要的系统提示词。
2. **用 §5 脚本批量跑**：每条问题独立 `thread_id`，sync 接口拿答案，history 接口拿工具调用与检索内容。
3. **导出 JSONL 后自算指标**：检索指标对齐方式（用 chunk_id / source / file_id）由你定义，答案指标用你自己的 judge 提示词，灵活度最高。
4. **向量库与图谱库的导出格式一致**（都是 `tool_calls` + `retrieved_chunks`），便于横向对比；图谱库若想要实体/关系，把 KB 的 `retrieval_content_scope` 改为 `all`。
5. **并发建议 2~4**，断点续跑脚本已内置，失败条目单独重试即可。

如需把脚本里的占位符（`<provider_id>:<chat_model_id>`、配置 ID、知识库名称）填好实际值跑通，把你的实际参数告诉我即可。
