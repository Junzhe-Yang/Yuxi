# Yuxi 0.6.3 批量会话脚本实现计划

日期：2026-07-12

## 目标

在不启动本地 Yuxi、Docker、Milvus、LightRAG 或模型服务的前提下，提供一个可以从本地 JSON 列表读取问题、远程调用 Yuxi 0.6.3、导出检索结果和 LLM 回答的脚本。

每个“输入行 × 知识库实验类型”必须是独立会话，不能把其它问题或标准答案发送给当前 Agent，也不能复用产生过其它答案的 thread。

## 已确定的实现方案

1. 使用 Python 3.12 和 `httpx`，脚本只依赖远程 Yuxi HTTP API；认证优先使用 `/api/auth/token` 登录换取内存中的 JWT，也兼容 API Key。
2. 正式批量使用 `POST /api/chat/thread`、`POST /api/chat/runs`、Run SSE 和 `GET /api/chat/thread/{thread_id}/history`。
3. 向量库与 LightRAG 使用两个固定 AgentConfig；每个 AgentConfig 必须只启用一个知识库名称。
4. 每个任务生成唯一 `thread_id`，只提交一次 `question`。
5. `gold_answer`、`documents` 等输入字段只写入本地输出，不放进请求 Prompt、`meta` 或检索过滤条件。
6. 从 history 提取所有工具调用；`query_kb` 保留实际 `query_text`、原始返回值和解析结果；LightRAG 保留 entities、relationships、references 和 chunks。
7. Run SSE 断开时使用 `after_seq` 重连；`state.jsonl` 保存已提交 Run，脚本重启后优先接管未完成 Run。
8. 已成功结果写入 JSONL 后跳过；失败或不完整任务在 `max_attempts` 内使用新的 thread 和新的 request attempt 重试。

## 待办清单

- [x] 创建独立工具目录。
- [x] 增加只读 discovery，自动列出 Agent、AgentConfig、知识库和配置内模型信息。
- [x] 支持使用登录账号自动换取 JWT，不再要求手工复制 token。
- [x] 实现 JSON 配置和输入列表校验。
- [x] 实现 Bearer token 认证的 HTTP 客户端，并兼容 API Key。
- [x] 实现线程创建、Run 创建、Run 状态查询和 history 查询。
- [x] 实现 SSE 解析、Redis stream sequence 去重和重连。
- [x] 实现向量检索结果、LightRAG 图谱结果和 `open_kb_document` 提取。
- [x] 实现 JSONL 结果、原始事件、manifest 和 state 持久化。
- [x] 实现并发上限、断点恢复和失败任务处理。
- [x] 在 `backend/test/unit/tools/test_yuxi_batch_rag_script.py` 编写不连接远程服务的伪响应测试。
- [x] 使用 Anaconda Python 3.12.4 完成语法编译和离线单元测试。
- [ ] 在远程 Yuxi 上用一条向量问题和一条 LightRAG 问题做最小真实验收。
- [ ] 最小验收通过后，将 `concurrency` 从 1 调整到适合局域网模型服务的值。

## 验收标准

- 输入有 N 条数据并启用两个 variant 时，输出最多产生 N×2 条成功或失败记录。
- 每条记录的 `thread_id` 唯一，history 只包含当前问题。
- 输出包含最终答案、全部工具调用、`query_kb` 的实际参数和实际检索内容。
- LightRAG 结果在 `retrieval_content_scope=all` 时保留图谱字段与 chunks。
- 标准答案和标准文档不会出现在发送给 Yuxi 的 `query` 或 `meta` 中。
- 进程中断后能够根据 `state.jsonl` 重新连接已有 Run，而不是重复提交同一个问题。
- 已成功写入的任务再次运行时被跳过。

## 当前验证边界

当前验证没有访问远程地址，因此只能证明本地输入校验、HTTP/SSE 数据处理、history 解析、线程隔离和 JSONL 状态逻辑。真实模型回答、知识库检索结果、API Key 权限和远程部署的网络连通性必须在用户的 Anaconda 环境中执行最小样本验收。
