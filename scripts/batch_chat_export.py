#!/usr/bin/env python3
"""
Yuxi 批量会话导出脚本

基于 Yuxi 的同步对话接口（POST /api/chat/agent/sync）与会话历史接口
（GET /api/chat/thread/{thread_id}/history），对一份问题数据集进行批量调用，
导出每条问题的：
  1. LLM 最终回答
  2. 全部工具调用记录（含入参与结果）
  3. RAG 检索内容（query_kb 等工具返回的 chunks / graph 结构）

关键设计：
  - 每条问题使用**全新 thread_id**（uuid4），保证会话隔离、避免记忆干扰。
  - 同步接口阻塞至 Agent 执行完成（含内部工具调用），返回后立即拉取历史，
    即可拿到完整工具调用与检索内容。
  - 支持断点续跑：输出文件中已存在的 query 会被跳过。
  - 支持并发：本地大模型推理为瓶颈，建议 --concurrency 2~4。
  - 不依赖 Yuxi 内置评估系统，导出 JSONL 后可自行计算指标。

依赖：仅使用 Python 标准库（urllib），无需 pip install。

用法示例：

  # 1) 发现可用 agent_config_id
  python scripts/batch_chat_export.py list \\
      --base http://localhost:5050 --key yxkey_xxx

  # 2) 批量执行（单套配置）
  python scripts/batch_chat_export.py run \\
      --base http://localhost:5050 --key yxkey_xxx \\
      --config-id 123 --dataset questions.jsonl \\
      --output results_milvus.jsonl --tag milvus

  # 3) 横向对比：图谱库再跑一遍
  python scripts/batch_chat_export.py run \\
      --base http://localhost:5050 --key yxkey_xxx \\
      --config-id 124 --dataset questions.jsonl \\
      --output results_lightrag.jsonl --tag lightrag

数据集格式（JSONL，每行一个 JSON 对象，至少含 query 字段）：
  {"query": "你的问题", "gold_answer": "可选标准答案", "id": "可选编号"}
  其他字段会原样保留在输出中。

输出格式（JSONL，每行一条）：
  {
    "query": "...",
    "gold_answer": "...",          # 数据集原样保留
    "id": "...",                    # 数据集原样保留
    "tag": "milvus",
    "agent_config_id": 123,
    "status": "finished",
    "thread_id": "...",
    "request_id": "...",
    "answer": "AI 最终回答",
    "tool_calls": [
      {"name":"query_kb","args":{...},"status":"success","result":[...]}
    ],
    "retrieved_chunks": [...],       # 从 query_kb 结果中抽取的 chunk 列表
    "retrieval_raw": [...],         # 每次 query_kb 的完整解析结果（含 graph 结构）
    "time_cost": 3.21,
    "error": null
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# HTTP 基础封装（基于 stdlib urllib，零外部依赖）
# ---------------------------------------------------------------------------


class RequestError(Exception):
    """HTTP 请求异常，携带状态码与响应体便于上层判断。"""

    def __init__(self, message: str, *, status_code: int = 0, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class YuxiClient:
    """Yuxi REST API 极简封装，仅覆盖批量导出所需接口。"""

    def __init__(self, base_url: str, api_key: str, timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # -- 通用请求 ---------------------------------------------------------
    def _request(self, method: str, path: str, *, body: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        data = None
        headers = dict(self._headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RequestError(
                f"HTTP {exc.code} {exc.reason} for {method} {path}",
                status_code=exc.code,
                body=text,
            ) from exc
        except urllib.error.URLError as exc:
            raise RequestError(f"URL error for {method} {path}: {exc.reason}") from exc

        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RequestError(f"响应非 JSON: {exc}; body[:200]={raw[:200]!r}") from exc

    # -- 发现接口 ---------------------------------------------------------
    def list_agents(self) -> list[dict]:
        data = self._request("GET", "/api/chat/agent")
        return data.get("agents", [])

    def list_agent_configs(self, agent_id: str) -> list[dict]:
        data = self._request("GET", f"/api/chat/agent/{agent_id}/configs")
        return data.get("configs", [])

    def list_knowledge_bases(self) -> list[dict]:
        data = self._request("GET", "/api/knowledge/databases")
        return data.get("databases", [])

    # -- 会话接口 ---------------------------------------------------------
    def sync_chat(
        self,
        *,
        query: str,
        agent_config_id: int,
        thread_id: str,
        request_id: str,
    ) -> dict:
        payload = {
            "query": query,
            "agent_config_id": agent_config_id,
            "thread_id": thread_id,
            "meta": {"request_id": request_id},
        }
        return self._request("POST", "/api/chat/agent/sync", body=payload)

    def get_thread_history(self, thread_id: str) -> list[dict]:
        data = self._request("GET", f"/api/chat/thread/{thread_id}/history")
        return data.get("history", [])


# ---------------------------------------------------------------------------
# 结果抽取
# ---------------------------------------------------------------------------


def _parse_tool_content(raw: str | None):
    """工具结果 content 是 JSON 字符串时解析为对象，否则原样返回。"""
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _flatten_chunks(parsed) -> list[dict]:
    """从 query_kb 的解析结果中抽取 chunk 列表。

    兼容三种结构：
      - 列表 [{content, metadata, score}, ...]            （Milvus / LightRAG chunks）
      - 字典 {chunks: [...]}                              （部分 LightRAG 实现）
      - 字典 {entities, relationships, references}        （LightRAG graph scope，无 chunks）
    """
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        if isinstance(parsed.get("chunks"), list):
            return [item for item in parsed["chunks"] if isinstance(item, dict)]
        # graph scope 没有 chunk，返回空列表；实体/关系保留在 retrieval_raw 中
        if "entities" in parsed or "relationships" in parsed:
            return []
    return []


def extract_from_history(history: list[dict]) -> dict:
    """从会话历史中抽取工具调用、检索内容。

    返回:
        {
            "tool_calls": [...],          # 全部工具调用（含 args + result）
            "retrieved_chunks": [...],    # query_kb 抽取的 chunk 列表
            "retrieval_raw": [...],       # 每次 query_kb 的完整解析结果
        }
    """
    tool_calls: list[dict] = []
    retrieved_chunks: list[dict] = []
    retrieval_raw: list[dict] = []

    for msg in history:
        if msg.get("type") != "ai":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("name")
            raw_content = (tc.get("tool_call_result") or {}).get("content")
            parsed = _parse_tool_content(raw_content)
            entry = {
                "name": name,
                "args": tc.get("args"),
                "status": tc.get("status"),
                "error_message": tc.get("error_message"),
                "result": parsed,
            }
            tool_calls.append(entry)

            if name == "query_kb" and parsed is not None:
                retrieval_raw.append(
                    {
                        "args": tc.get("args"),
                        "result": parsed,
                    }
                )
                retrieved_chunks.extend(_flatten_chunks(parsed))

    return {
        "tool_calls": tool_calls,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_raw": retrieval_raw,
    }


# ---------------------------------------------------------------------------
# 单条执行
# ---------------------------------------------------------------------------


def run_one(
    client: YuxiClient,
    *,
    record: dict,
    agent_config_id: int,
    tag: str,
    retries: int,
    backoff_base: float = 2.0,
) -> dict:
    """对单条数据集记录执行一次独立会话并抽取结果。"""
    query = record["query"]
    thread_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    output: dict = {
        **record,
        "tag": tag,
        "agent_config_id": agent_config_id,
        "status": "error",
        "thread_id": thread_id,
        "request_id": request_id,
        "answer": "",
        "tool_calls": [],
        "retrieved_chunks": [],
        "retrieval_raw": [],
        "time_cost": None,
        "error": None,
    }

    # 1) 同步对话（含重试）
    sync = None
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            sync = client.sync_chat(
                query=query,
                agent_config_id=agent_config_id,
                thread_id=thread_id,
                request_id=request_id,
            )
            break
        except RequestError as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(backoff_base**attempt)
                continue
            output["error"] = f"sync_chat failed: {last_err}"
            return output

    if sync is None:
        output["error"] = f"sync_chat failed: {last_err}"
        return output

    status = sync.get("status")
    output["status"] = status
    output["time_cost"] = sync.get("time_cost")

    if status != "finished":
        output["error"] = sync.get("error_message") or sync.get("message") or f"sync status={status}"
        output["answer"] = sync.get("response", "") or ""
        # 即便失败也尝试拉取历史（部分工具调用可能已落库）
    else:
        output["answer"] = sync.get("response", "") or ""

    # 2) 拉取会话历史，抽取工具调用与检索内容
    try:
        history = client.get_thread_history(thread_id)
    except RequestError as exc:
        output["error"] = f"history fetch failed: {exc}"
        return output

    extracted = extract_from_history(history)
    output["tool_calls"] = extracted["tool_calls"]
    output["retrieved_chunks"] = extracted["retrieved_chunks"]
    output["retrieval_raw"] = extracted["retrieval_raw"]

    # 若 finished 但 answer 为空，兜底从历史最后一条 AI 消息取
    if status == "finished" and not output["answer"]:
        for msg in reversed(history):
            if msg.get("type") == "ai" and msg.get("content"):
                output["answer"] = msg["content"]
                break

    return output


# ---------------------------------------------------------------------------
# 数据集与断点续跑
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"数据集第 {lineno} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(obj, dict) or "query" not in obj:
                raise SystemExit(f"数据集第 {lineno} 行缺少 query 字段")
            records.append(obj)
    if not records:
        raise SystemExit("数据集为空")
    return records


def load_done_keys(output_path: str) -> set[str]:
    """读取已输出文件，返回已处理记录的去重键集合（与 record_key 一致）。"""
    if not os.path.exists(output_path):
        return set()
    done: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                done.add(record_key(obj))
    return done


def record_key(record: dict) -> str:
    """返回单条记录的去重键：优先 id，其次 query 文本。"""
    if "id" in record and record["id"] is not None:
        return f"id::{record['id']}"
    return f"query::{record.get('query', '')}"


# ---------------------------------------------------------------------------
# 子命令：list（发现 agent / config / 知识库）
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    client = YuxiClient(args.base, args.key, timeout=30)

    print("=" * 60)
    print("知识库列表（name 字段即 AgentConfig.context.knowledges 使用的名称）")
    print("=" * 60)
    try:
        kbs = client.list_knowledge_bases()
    except RequestError as exc:
        print(f"  获取失败: {exc}")
        kbs = []
    for kb in kbs:
        print(
            f"  - name={kb.get('name')!r}  kb_type={kb.get('kb_type')!r}  "
            f"db_id={kb.get('db_id')!r}  desc={(kb.get('description') or '')[:40]}"
        )

    print()
    print("=" * 60)
    print("智能体列表")
    print("=" * 60)
    try:
        agents = client.list_agents()
    except RequestError as exc:
        print(f"  获取失败: {exc}")
        agents = []
    if not agents:
        print("  （无智能体）")
    for ag in agents:
        print(f"  - id={ag.get('id')!r}  name={ag.get('name')!r}")

    print()
    print("=" * 60)
    print("智能体配置列表（需要 agent_config_id 用于 --config-id）")
    print("=" * 60)
    target = args.agent
    if not target and agents:
        target = agents[0].get("id")
        print(f"  （未指定 --agent，默认使用第一个: {target}）")
    if not target:
        print("  （无可用智能体，无法列出配置）")
        return 0
    try:
        configs = client.list_agent_configs(target)
    except RequestError as exc:
        print(f"  获取失败: {exc}")
        configs = []
    if not configs:
        print("  （无配置）")
    for cfg in configs:
        print(f"  - config_id={cfg.get('id')}  name={cfg.get('name')!r}  is_default={cfg.get('is_default')}")

    return 0


# ---------------------------------------------------------------------------
# 子命令：run（批量执行）
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)

    output_path = args.output
    done_keys = load_done_keys(output_path) if args.resume else set()

    pending = [r for r in dataset if record_key(r) not in done_keys]
    skipped = len(dataset) - len(pending)
    print(
        f"数据集共 {len(dataset)} 条；"
        f"{'跳过已处理 ' + str(skipped) + ' 条；' if args.resume and skipped else ''}"
        f"待执行 {len(pending)} 条；并发={args.concurrency}；"
        f"输出={output_path}"
    )

    if not pending:
        print("无待执行记录，退出。")
        return 0

    client = YuxiClient(args.base, args.key, timeout=args.timeout)

    # 以追加模式写入，每条完成后 flush，保证断点安全
    out_f = open(output_path, "a", encoding="utf-8")
    total = len(pending)
    success = 0
    failed = 0

    def _execute(idx: int, record: dict) -> tuple[int, dict]:
        result = run_one(
            client,
            record=record,
            agent_config_id=args.config_id,
            tag=args.tag,
            retries=args.retries,
        )
        return idx, result

    try:
        if args.concurrency <= 1:
            for i, record in enumerate(pending, 1):
                _, result = _execute(i, record)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                _log_progress(i, total, result, args.tag)
                if result["status"] == "finished":
                    success += 1
                else:
                    failed += 1
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(_execute, i, rec): i for i, rec in enumerate(pending, 1)}
                for fut in as_completed(futures):
                    _, result = fut.result()
                    # 并发场景下按完成顺序写入
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    idx = futures[fut]
                    _log_progress(idx, total, result, args.tag)
                    if result["status"] == "finished":
                        success += 1
                    else:
                        failed += 1
    finally:
        out_f.close()

    print()
    print(f"完成：成功 {success} 条，失败 {failed} 条，共 {total} 条。输出已写入 {output_path}")
    return 0 if failed == 0 else 1


def _log_progress(idx: int, total: int, result: dict, tag: str) -> None:
    status = result.get("status")
    chunks = len(result.get("retrieved_chunks", []))
    tools = len(result.get("tool_calls", []))
    t = result.get("time_cost")
    t_str = f"{t:.2f}s" if isinstance(t, (int, float)) else "-"
    err = result.get("error") or ""
    if err:
        err = f" err={err[:80]}"
    print(f"[{tag} {idx}/{total}] status={status} tools={tools} chunks={chunks} time={t_str}{err}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_chat_export",
        description="Yuxi 批量会话导出脚本：对数据集逐条独立会话调用 Agent，"
        "导出 LLM 回答、工具调用记录与 RAG 检索内容。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("YUXI_BASE", "http://localhost:5050"),
        help="Yuxi API 基址（默认 http://localhost:5050，可用环境变量 YUXI_BASE）",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("YUXI_API_KEY", ""),
        help="API Key（yxkey_xxx，可用环境变量 YUXI_API_KEY）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="列出知识库、智能体与配置，用于发现 config_id")
    p_list.add_argument("--agent", default=None, help="指定智能体 id（默认第一个）")

    # run
    p_run = sub.add_parser("run", help="批量执行数据集")
    p_run.add_argument("--config-id", type=int, required=True, help="AgentConfig ID")
    p_run.add_argument("--dataset", required=True, help="数据集 JSONL 路径")
    p_run.add_argument(
        "--output",
        default=None,
        help="输出 JSONL 路径（默认 results_<config_id>.jsonl）",
    )
    p_run.add_argument("--tag", default="", help="本次运行的标签，写入输出（如 milvus/lightrag）")
    p_run.add_argument("--timeout", type=int, default=600, help="单次同步对话超时秒数（默认 600）")
    p_run.add_argument("--retries", type=int, default=2, help="单条失败重试次数（默认 2）")
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发线程数（本地大模型建议 2~4，默认 1）",
    )
    p_run.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="禁用断点续跑（默认启用，按 id 优先、query 兜底跳过已处理项）",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.key:
        parser.error("缺少 API Key，请通过 --key 或环境变量 YUXI_API_KEY 提供")

    if args.command == "list":
        return cmd_list(args)
    if args.command == "run":
        if not args.output:
            args.output = f"results_{args.config_id}.jsonl"
        return cmd_run(args)

    parser.error("未知子命令")
    return 2


if __name__ == "__main__":
    sys.exit(main())
