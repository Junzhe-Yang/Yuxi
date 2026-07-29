#!/usr/bin/env python3
"""语析会话批量删除脚本

重要说明:
1. 后端删除接口是软删除(status="deleted"), 不会真正从数据库删除数据行,
   不会释放存储空间。如需真正释放空间, 需直接操作数据库(见脚本末尾"硬清理"说明)。
2. 只能删除当前登录用户自己的会话(后端有所有权校验, 删他人会话返回404)。
3. 脚本默认 dry-run 模式, 仅预览不删除, 需 --confirm 才真正执行。
4. 删除前可用 --backup-dir 备份会话内容到本地 JSON。

用法:
    # 1. 登录(复用爬取脚本的 token 文件 yuxi_token.json)
    python delete_conversations.py login --base-url http://localhost:8000 --username admin --password xxx

    # 2. 预览将被删除的会话(dry-run, 默认安全模式)
    python delete_conversations.py delete --all
    python delete_conversations.py delete --agent-id ChatbotAgent
    python delete_conversations.py delete --older-than-days 30
    python delete_conversations.py delete --thread-ids uuid1,uuid2,uuid3

    # 3. 删除前备份 + 确认执行
    python delete_conversations.py delete --all --backup-dir ./backup --confirm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# 默认配置
DEFAULT_TOKEN_FILE = Path("./yuxi_token.json")
DEFAULT_PAGE_SIZE = 100
REQUEST_TIMEOUT = 30
RETRY_TIMES = 3
RETRY_BACKOFF = 2.0


class YuxiDeleter:
    """语析会话批量删除器

    复用爬取脚本的 token 文件, 登录一次即可同时用于爬取和删除。
    """

    def __init__(self, base_url: str, token_file: Path = DEFAULT_TOKEN_FILE):
        self.base_url = base_url.rstrip("/")
        self.token_file = token_file
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.token: str | None = None
        self.user_info: dict[str, Any] = {}
        self._load_token()

    # ------------------------------------------------------------------
    # Token 管理 (与爬取脚本共用 yuxi_token.json)
    # ------------------------------------------------------------------

    def _load_token(self) -> None:
        if self.token_file.exists():
            data = json.loads(self.token_file.read_text(encoding="utf-8"))
            self.token = data.get("access_token")
            self.user_info = {
                k: v for k, v in data.items() if k != "access_token"
            }
            if self.token:
                self.session.headers.update(
                    {"Authorization": f"Bearer {self.token}"}
                )

    def _save_token(self, token_data: dict[str, Any]) -> None:
        self.token = token_data["access_token"]
        self.user_info = {
            k: v for k, v in token_data.items() if k != "access_token"
        }
        self.session.headers.update(
            {"Authorization": f"Bearer {self.token}"}
        )
        self.token_file.write_text(
            json.dumps(token_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def login(self, username: str, password: str) -> None:
        """登录并保存 token

        登录接口使用 application/x-www-form-urlencoded (OAuth2PasswordRequestForm),
        不是 JSON, 这是后端 auth_router.py:147 的硬性要求。
        """
        url = f"{self.base_url}/api/auth/token"
        resp = requests.post(
            url,
            data={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 423:
            remaining = resp.headers.get("X-Lock-Remaining", "?")
            print(f"[login] 账户被锁定, 剩余 {remaining} 秒", file=sys.stderr)
            sys.exit(1)
        if resp.status_code != 200:
            print(f"[login] 登录失败 {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)
        token_data = resp.json()
        self._save_token(token_data)
        print(f"[login] 登录成功, 用户: {token_data.get('username')}")

    def _check_token(self) -> None:
        if not self.token:
            print("[error] 未找到 token, 请先执行 login 子命令", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # HTTP 请求封装
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(1, RETRY_TIMES + 1):
            try:
                resp = self.session.request(
                    method, url, timeout=REQUEST_TIMEOUT, **kwargs
                )
                if resp.status_code == 401:
                    print("[error] token 已失效或过期(7天), 请重新 login", file=sys.stderr)
                    sys.exit(2)
                if resp.status_code == 429:
                    wait = RETRY_BACKOFF ** attempt
                    print(f"[rate] 被限流, 等待 {wait:.1f}s 后重试")
                    time.sleep(wait)
                    continue
                # DELETE 成功返回 200, 404 表示不存在或无权访问
                if resp.status_code == 404:
                    return {"_not_found": True}
                resp.raise_for_status()
                # DELETE 可能返回空 body
                if resp.text:
                    return resp.json()
                return {}
            except requests.RequestException as e:
                last_exc = e
                if attempt < RETRY_TIMES:
                    wait = RETRY_BACKOFF ** attempt
                    print(f"[retry] 请求失败({e}), 第 {attempt}/{RETRY_TIMES} 次重试, 等待 {wait:.1f}s")
                    time.sleep(wait)
        raise RuntimeError(f"请求 {url} 失败, 已重试 {RETRY_TIMES} 次: {last_exc}")

    # ------------------------------------------------------------------
    # 会话列表查询
    # ------------------------------------------------------------------

    def list_threads(
        self, agent_id: str | None = None, page_size: int = DEFAULT_PAGE_SIZE
    ) -> list[dict[str, Any]]:
        """分页拉取所有活跃会话

        返回结构: [{id(thread_id), user_id, agent_id, title, is_pinned,
                   created_at, updated_at, metadata}]
        只返回 status="active" 的会话(软删除的不在列表中)。
        """
        self._check_token()
        all_threads: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, Any] = {"limit": page_size, "offset": offset}
            if agent_id:
                params["agent_id"] = agent_id
            data = self._request("GET", "/api/chat/threads", params=params)
            threads = data if isinstance(data, list) else (data or {}).get("threads", [])
            if not threads:
                break
            all_threads.extend(threads)
            if len(threads) < page_size:
                break
            offset += page_size
        return all_threads

    # ------------------------------------------------------------------
    # 备份 (删除前可选)
    # ------------------------------------------------------------------

    def _backup_thread(self, thread: dict[str, Any], backup_dir: Path) -> bool:
        """备份单个会话的完整历史到本地 JSON 文件"""
        thread_id = thread["id"]
        try:
            data = self._request("GET", f"/api/chat/thread/{thread_id}/history")
            history = (data or {}).get("history", [])
            backup_path = backup_dir / f"{thread_id}.json"
            backup_path.write_text(
                json.dumps(
                    {
                        "thread_id": thread_id,
                        "title": thread.get("title"),
                        "agent_id": thread.get("agent_id"),
                        "created_at": thread.get("created_at"),
                        "updated_at": thread.get("updated_at"),
                        "message_count": len(history),
                        "history": history,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return True
        except Exception as e:
            print(f"[backup] {thread_id} 备份失败: {e}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> dict[str, Any]:
        """删除单个会话 (软删除)

        后端 delete_thread_view 调用 conv_repo.delete_conversation(soft_delete=True),
        只把 status 改为 "deleted", 不删除 messages/tool_calls/stats。
        返回 {"message": "删除成功"} 或 {"_not_found": True}。
        """
        return self._request("DELETE", f"/api/chat/thread/{thread_id}")

    # ------------------------------------------------------------------
    # 过滤逻辑
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_iso_time(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            # 兼容带/不带时区的 ISO 格式
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _filter_by_age(
        threads: list[dict[str, Any]], older_than_days: int
    ) -> list[dict[str, Any]]:
        """筛选 N 天未更新的会话(基于 updated_at)"""
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        result = []
        for t in threads:
            updated = YuxiDeleter._parse_iso_time(t.get("updated_at"))
            if updated is None or updated.timestamp() < cutoff:
                result.append(t)
        return result

    @staticmethod
    def _filter_by_before_date(
        threads: list[dict[str, Any]], before_date: str
    ) -> list[dict[str, Any]]:
        """筛选创建时间早于指定日期的会话(基于 created_at)"""
        try:
            cutoff_dt = datetime.fromisoformat(before_date).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[error] --before 日期格式无效: {before_date} (应为 YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)
        result = []
        for t in threads:
            created = YuxiDeleter._parse_iso_time(t.get("created_at"))
            if created is None or created < cutoff_dt:
                result.append(t)
        return result

    @staticmethod
    def _filter_by_keep_pinned(
        threads: list[dict[str, Any]], keep_pinned: bool
    ) -> list[dict[str, Any]]:
        """如果 keep_pinned=True, 从待删列表中排除置顶会话"""
        if not keep_pinned:
            return threads
        return [t for t in threads if not t.get("is_pinned")]

    # ------------------------------------------------------------------
    # 主删除流程
    # ------------------------------------------------------------------

    def run_delete(
        self,
        *,
        delete_all: bool = False,
        agent_id: str | None = None,
        thread_ids: list[str] | None = None,
        older_than_days: int | None = None,
        before_date: str | None = None,
        keep_pinned: bool = False,
        confirm: bool = False,
        backup_dir: Path | None = None,
        delay: float = 0.2,
    ) -> None:
        """执行批量删除流程

        策略:
        1. 确定"待删候选集": 通过 list_threads 拉取活跃会话, 再按过滤条件筛选
        2. 如果指定了 thread_ids, 直接用这些 ID(不查列表)
        3. dry-run 时只打印候选, 不调用 DELETE
        4. confirm 时逐个调用 DELETE, 记录成功/失败
        """
        self._check_token()

        # --- 确定待删候选集 ---
        if thread_ids:
            # 直接用指定 ID, 不查列表(这些 ID 可能已被列表过滤掉)
            candidates = [{"id": tid, "title": "(指定)", "agent_id": None,
                           "created_at": None, "updated_at": None, "is_pinned": False}
                          for tid in thread_ids]
        else:
            # 拉取列表后筛选
            if not delete_all and not agent_id and not older_than_days and not before_date:
                print("[error] 请指定至少一个过滤条件: --all / --agent-id / "
                      "--older-than-days / --before / --thread-ids", file=sys.stderr)
                sys.exit(1)

            print("[scan] 正在拉取会话列表...")
            threads = self.list_threads(agent_id=agent_id)
            print(f"[scan] 共 {len(threads)} 个活跃会话")

            candidates = list(threads)

            if older_than_days is not None:
                before_count = len(candidates)
                candidates = self._filter_by_age(candidates, older_than_days)
                print(f"[filter] --older-than-days {older_than_days}: "
                      f"{before_count} -> {len(candidates)}")

            if before_date is not None:
                before_count = len(candidates)
                candidates = self._filter_by_before_date(candidates, before_date)
                print(f"[filter] --before {before_date}: "
                      f"{before_count} -> {len(candidates)}")

            if keep_pinned:
                before_count = len(candidates)
                candidates = self._filter_by_keep_pinned(candidates, keep_pinned)
                print(f"[filter] --keep-pinned: "
                      f"{before_count} -> {len(candidates)} (排除置顶)")

        if not candidates:
            print("\n[done] 没有符合条件的会话, 无需删除。")
            return

        # --- 打印待删清单 ---
        print(f"\n{'='*60}")
        print(f"待删除会话: {len(candidates)} 个")
        print(f"{'='*60}")
        for i, t in enumerate(candidates, 1):
            tid = t["id"]
            title = t.get("title") or "(无标题)"
            agent = t.get("agent_id") or "-"
            updated = t.get("updated_at") or "-"
            pinned = "📌" if t.get("is_pinned") else "  "
            print(f"  {pinned} {i:>4}. {tid}  |  {title[:30]:<30}  |  {agent:<16}  |  {updated}")

        # --- dry-run 模式: 到此为止 ---
        if not confirm:
            print(f"\n[dry-run] 以上 {len(candidates)} 个会话将被删除(软删除)。")
            print("[dry-run] 这是预览模式, 未执行任何删除操作。")
            print("[dry-run] 确认无误后, 添加 --confirm 参数执行删除。")
            if backup_dir:
                print("[dry-run] 如需删除前备份, 备份也会在 --confirm 时执行。")
            return

        # --- 备份 ---
        if backup_dir:
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[backup] 开始备份到 {backup_dir} ...")
            backup_ok = 0
            backup_fail = 0
            for i, t in enumerate(candidates, 1):
                ok = self._backup_thread(t, backup_dir)
                if ok:
                    backup_ok += 1
                else:
                    backup_fail += 1
                if i % 50 == 0:
                    print(f"[backup] 进度: {i}/{len(candidates)}")
                time.sleep(delay)
            print(f"[backup] 完成: 成功 {backup_ok}, 失败 {backup_fail}")

        # --- 执行删除 ---
        print(f"\n[delete] 开始删除 {len(candidates)} 个会话(软删除)...")
        success = 0
        not_found = 0
        failed = 0
        failed_list: list[tuple[str, str]] = []

        for i, t in enumerate(candidates, 1):
            tid = t["id"]
            try:
                result = self.delete_thread(tid)
                if result and result.get("_not_found"):
                    not_found += 1
                    print(f"[delete] ({i}/{len(candidates)}) {tid} -> 404 不存在或无权访问")
                else:
                    success += 1
                    if i % 50 == 0 or i == len(candidates):
                        print(f"[delete] ({i}/{len(candidates)}) 进度... 成功 {success}")
            except Exception as e:
                failed += 1
                failed_list.append((tid, str(e)))
                print(f"[delete] ({i}/{len(candidates)}) {tid} 失败: {e}", file=sys.stderr)
            time.sleep(delay)

        # --- 汇总 ---
        print(f"\n{'='*60}")
        print(f"删除完成汇总")
        print(f"{'='*60}")
        print(f"  总数:     {len(candidates)}")
        print(f"  成功:     {success}")
        print(f"  不存在:   {not_found}")
        print(f"  失败:     {failed}")
        if backup_dir:
            print(f"  备份目录: {backup_dir}")
        if failed_list:
            print(f"\n  失败列表:")
            for tid, err in failed_list:
                print(f"    {tid}: {err}")
        print(f"\n[info] 注意: 这是软删除, 数据仍在数据库中, 未释放存储空间。")
        print(f'[info] 如需真正释放空间, 请参考脚本末尾的"硬清理"说明。')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="语析会话批量删除脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 登录
  %(prog)s login -u admin -p xxx

  # 预览删除全部会话(dry-run)
  %(prog)s delete --all

  # 删除30天未更新的会话, 保留置顶
  %(prog)s delete --older-than-days 30 --keep-pinned

  # 删除前备份 + 确认执行
  %(prog)s delete --all --backup-dir ./backup --confirm

  # 删除指定会话
  %(prog)s delete --thread-ids uuid1,uuid2,uuid3 --confirm
        """,
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="后端地址")
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_FILE), help="token 文件路径")
    sub = parser.add_subparsers(dest="command", required=True)

    # login
    p_login = sub.add_parser("login", help="登录并保存 token")
    p_login.add_argument("-u", "--username", required=True)
    p_login.add_argument("-p", "--password", required=True)

    # delete
    p_del = sub.add_parser("delete", help="批量删除会话")
    g = p_del.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="删除全部活跃会话")
    g.add_argument("--thread-ids", help="逗号分隔的 thread_id 列表")
    p_del.add_argument("--agent-id", default=None, help="只删除指定 agent 的会话")
    p_del.add_argument("--older-than-days", type=int, default=None,
                       help="删除 N 天未更新的会话(基于 updated_at)")
    p_del.add_argument("--before", default=None,
                       help="删除创建时间早于此日期的会话(YYYY-MM-DD)")
    p_del.add_argument("--keep-pinned", action="store_true",
                       help="排除置顶会话(不删除置顶的)")
    p_del.add_argument("--confirm", action="store_true",
                       help="确认执行删除(默认 dry-run 预览模式)")
    p_del.add_argument("--backup-dir", default=None,
                       help="删除前备份会话内容到此目录")
    p_del.add_argument("--delay", type=float, default=0.2,
                       help="请求间隔秒数(避免压垮服务器)")

    args = parser.parse_args()
    token_file = Path(args.token_file)
    deleter = YuxiDeleter(base_url=args.base_url, token_file=token_file)

    if args.command == "login":
        deleter.login(args.username, args.password)

    elif args.command == "delete":
        thread_ids = None
        if args.thread_ids:
            thread_ids = [tid.strip() for tid in args.thread_ids.split(",") if tid.strip()]

        backup_dir = Path(args.backup_dir) if args.backup_dir else None

        deleter.run_delete(
            delete_all=args.all,
            agent_id=args.agent_id,
            thread_ids=thread_ids,
            older_than_days=args.older_than_days,
            before_date=args.before,
            keep_pinned=args.keep_pinned,
            confirm=args.confirm,
            backup_dir=backup_dir,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
