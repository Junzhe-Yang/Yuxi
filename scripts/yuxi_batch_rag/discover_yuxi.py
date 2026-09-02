"""Discover the IDs needed by the Yuxi 0.6.3 batch configuration.

This command is read-only from the Yuxi application perspective.  It logs in
or uses an API Key, then lists the selected Agent, its AgentConfig profiles,
the model/knowledge settings stored inside each profile, and the current
user's accessible knowledge bases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from batch_yuxi_rag import (
    BatchConfigError,
    YuxiClient,
    YuxiError,
    extract_config_context,
    login_for_access_token,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="读取 Yuxi 0.6.3 的 Agent、AgentConfig、知识库和模型信息")
    parser.add_argument("--base-url", required=True, help="远程 Yuxi 地址，例如 https://yuxi.example.com")
    parser.add_argument("--agent-id", default=None, help="Agent ID；不填写时使用 Yuxi 默认 Agent")
    parser.add_argument("--auth-mode", choices=("login", "api_key"), default="login")
    parser.add_argument("--login-id-env", default="YUXI_LOGIN_ID", help="登录用户 ID/手机号所在环境变量")
    parser.add_argument("--password-env", default="YUXI_PASSWORD", help="登录密码所在环境变量")
    parser.add_argument("--api-key-env", default="YUXI_API_KEY", help="API Key 所在环境变量")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--insecure", action="store_true", help="关闭 TLS 证书校验，仅用于自签名测试环境")
    parser.add_argument("--output", type=Path, default=None, help="可选，将 discovery JSON 写入本地文件")
    return parser


def get_token(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.auth_mode == "login":
        login_id = os.environ.get(args.login_id_env, "")
        password = os.environ.get(args.password_env, "")
        response = login_for_access_token(
            args.base_url,
            login_id,
            password,
            args.timeout_seconds,
            not args.insecure,
        )
        return str(response["access_token"]), {
            "mode": "login",
            "user_id": response.get("user_id"),
            "username": response.get("username"),
            "user_id_login": response.get("user_id_login"),
            "role": response.get("role"),
            "department_id": response.get("department_id"),
            "department_name": response.get("department_name"),
        }

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise BatchConfigError(f"环境变量 {args.api_key_env} 未设置")
    return api_key, {"mode": "api_key"}


def normalize_accessible_databases(response: dict[str, Any]) -> list[dict[str, Any]]:
    databases = response.get("databases", []) if isinstance(response, dict) else []
    if not isinstance(databases, list):
        return []
    return [database for database in databases if isinstance(database, dict)]


def discover(args: argparse.Namespace) -> dict[str, Any]:
    token, auth_info = get_token(args)
    with YuxiClient(args.base_url, token, args.timeout_seconds, not args.insecure) as client:
        agents_response = client.get_agents()
        agents = agents_response.get("agents", []) if isinstance(agents_response, dict) else []
        agents = [agent for agent in agents if isinstance(agent, dict)]
        agent_ids = {str(agent.get("id")) for agent in agents if agent.get("id")}

        default_agent_id = None
        try:
            default_agent_id = client.get_default_agent().get("default_agent_id")
        except YuxiError:
            pass

        agent_id = args.agent_id or default_agent_id or ("ChatbotAgent" if "ChatbotAgent" in agent_ids else None)
        if not agent_id:
            raise BatchConfigError("无法确定 agent_id，请使用 --agent-id；/api/chat/agent 未提供可用 Agent")
        if agent_ids and agent_id not in agent_ids:
            raise BatchConfigError(f"agent_id {agent_id!r} 不在远程 Agent 列表中：{sorted(agent_ids)}")

        config_list_response = client.get_agent_configs(agent_id)
        config_list = config_list_response.get("configs", []) if isinstance(config_list_response, dict) else []
        accessible_databases = normalize_accessible_databases(client.get_accessible_databases())
        databases_by_name: dict[str, list[dict[str, Any]]] = {}
        for database in accessible_databases:
            name = str(database.get("name") or "").strip()
            if name:
                databases_by_name.setdefault(name, []).append(database)

        profiles: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        for summary in config_list:
            if not isinstance(summary, dict) or summary.get("id") is None:
                continue
            config_id = int(summary["id"])
            detail = client.get_agent_config(agent_id, config_id)
            context = extract_config_context(detail)
            knowledge_names = context.get("knowledges")
            if not isinstance(knowledge_names, list):
                knowledge_names = []
            knowledge_names = [str(name) for name in knowledge_names]

            matched_databases: list[dict[str, Any]] = []
            for name in knowledge_names:
                matches = databases_by_name.get(name, [])
                matched_databases.extend(
                    {
                        "name": database.get("name"),
                        "db_id": database.get("db_id"),
                        "description": database.get("description", ""),
                    }
                    for database in matches
                )

            profile = {
                "id": config_id,
                "name": summary.get("name"),
                "description": summary.get("description"),
                "is_default": bool(summary.get("is_default")),
                "agent_id": agent_id,
                "model": context.get("model"),
                "subagents_model": context.get("subagents_model"),
                "knowledges": knowledge_names,
                "tools": context.get("tools", []),
                "mcps": context.get("mcps", []),
                "skills": context.get("skills", []),
                "knowledge_bases": matched_databases,
            }
            profiles.append(profile)

            if len(knowledge_names) == 1 and len(matched_databases) == 1:
                recommendations.append(
                    {
                        "agent_id": agent_id,
                        "agent_config_id": config_id,
                        "agent_config_name": summary.get("name"),
                        "knowledge_base_name": knowledge_names[0],
                        "knowledge_db_id": matched_databases[0].get("db_id"),
                        "model": context.get("model"),
                    }
                )

    return {
        "base_url": args.base_url.rstrip("/"),
        "auth": auth_info,
        "default_agent_id": default_agent_id,
        "selected_agent_id": agent_id,
        "agents": agents,
        "accessible_knowledge_bases": accessible_databases,
        "agent_configs": profiles,
        "single_knowledge_base_recommendations": recommendations,
        "notes": [
            "agent_config_id 是批处理 Run 请求需要的 ID。",
            "model 是 AgentConfig.context.model，批处理不需要单独传模型 ID。",
            "knowledge_db_id 只在需要调用管理员知识库接口（例如修改 query params）时才需要；当前 chunks 实验可以不填写。",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = discover(args)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        print(rendered, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (BatchConfigError, YuxiError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
