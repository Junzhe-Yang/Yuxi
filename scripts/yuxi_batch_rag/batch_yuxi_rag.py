"""Batch-run isolated Yuxi 0.6.3 Agent conversations.

The script intentionally talks to Yuxi over HTTP only.  It does not import
Yuxi internals and therefore can run on a workstation that has the input data
and network access to the remote deployment, but does not run Yuxi itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
SUCCESS_RUN_STATUS = "completed"
SCHEMA_VERSION = 1
SSE_PROGRESS_LOG_INTERVAL_SECONDS = 15.0
TERMINAL_SSE_EVENT_STATUSES = {
    "finished": "completed",
    "error": "failed",
    "interrupted": "interrupted",
    "cancelled": "cancelled",
    "ask_user_question_required": "interrupted",
}
DIRECT_REVIEW_TRACE_VERSIONS = {
    "4.0",
    "5.0",
    "6.0",
    "7.0",
    "8.0",
    "9.0",
    "10.0",
    "11.0",
    "12.0",
    "13.0",
}
KNOWN_REVIEW_TRACE_VERSIONS = {"2.0", "3.0", *DIRECT_REVIEW_TRACE_VERSIONS}
VALID_COMPANION_SELECTOR_STATUSES = {"success", "repaired", "empty"}
LOGGER = logging.getLogger("yuxi_batch_rag")
LOGGER.addHandler(logging.NullHandler())


class BatchConfigError(ValueError):
    """Raised when the local batch configuration or input is invalid."""


class YuxiError(RuntimeError):
    """Base class for errors raised while talking to Yuxi."""


class YuxiApiError(YuxiError):
    """Raised for a non-successful Yuxi HTTP response."""

    def __init__(self, method: str, url: str, status_code: int, body: str):
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body
        super().__init__(f"{method} {url} returned HTTP {status_code}: {body[:500]}")


class YuxiTransportError(YuxiError):
    """Raised when the HTTP connection cannot be completed."""


class RunStreamError(YuxiError):
    """Raised when a Run cannot be observed until a terminal state."""


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Configure console logging and, optionally, a UTF-8 log file."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise BatchConfigError(f"无效的日志级别：{level!r}")

    LOGGER.handlers.clear()
    LOGGER.setLevel(numeric_level)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


@dataclass(frozen=True)
class VariantConfig:
    """The fixed AgentConfig used for one experiment variant."""

    name: str
    agent_config_id: int
    expected_knowledge_base_name: str | None = None
    expected_run_mode: str | None = None
    expected_agenda_mode: str | None = None
    expected_synthesis_mode: str | None = None
    expected_method_family: str | None = None
    expected_method_version: str | None = None
    expected_experiment_profile: str | None = None
    expected_effective_profile: str | None = None
    expected_atlas_profile: str | None = None
    expected_acm_protocol: str | None = None
    expected_v7_experiment_arm: str | None = None
    expected_v7_retrieval_depth: str | None = None
    expected_max_search_calls: int | None = None
    expected_trace_schema_version: str | None = None
    expected_controller_version: str | None = None
    expected_context_view_version: str | None = None
    expected_model_context_window_tokens: int | None = None
    knowledge_db_id: str | None = None
    ensure_retrieval_content_scope_all: bool = False
    require_companion_selector: bool = False


@dataclass(frozen=True)
class BatchSettings:
    """Validated settings loaded from the local JSON configuration file."""

    config_path: Path
    base_url: str
    auth_mode: str
    api_key_env: str | None
    login_id_env: str | None
    password_env: str | None
    agent_id: str
    input_file: Path
    output_dir: Path
    variants: dict[str, VariantConfig]
    concurrency: int = 1
    request_timeout_seconds: float = 60.0
    run_timeout_seconds: float = 960.0
    max_attempts: int = 2
    max_sse_reconnects: int = 8
    sse_reconnect_delay_seconds: float = 1.0
    verify_tls: bool = True
    write_raw_events: bool = True
    allow_debug_stopped: bool = False
    batch_id: str | None = None


@dataclass(frozen=True)
class Job:
    """One dataset row executed against one fixed AgentConfig."""

    batch_id: str
    row_index: int
    input_record: dict[str, Any]
    variant: VariantConfig
    attempt: int

    @property
    def job_key(self) -> str:
        return f"{self.variant.name}:{self.row_index:06d}"

    @property
    def question(self) -> str:
        return str(self.input_record["question"])

    @property
    def request_id(self) -> str:
        raw = f"{self.batch_id}-{self.variant.name}-{self.row_index:06d}-a{self.attempt}"
        if len(raw) <= 64:
            return raw
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"{self.batch_id[:40]}-{self.variant.name[:10]}-{self.row_index:06d}-{digest}"[:64]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except OSError as exc:
        raise BatchConfigError(f"无法读取 JSON 文件 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BatchConfigError(f"JSON 文件格式错误 {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise BatchConfigError(f"配置文件顶层必须是 JSON 对象：{path}")
    return value


def resolve_config_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_settings(config_path: Path) -> BatchSettings:
    raw = load_json_object(config_path)

    def required_string(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BatchConfigError(f"配置项 {key} 必须是非空字符串")
        return value.strip()

    base_url = required_string("base_url").rstrip("/")
    agent_id = required_string("agent_id")
    input_file = resolve_config_path(required_string("input_file"), config_path)
    output_dir = resolve_config_path(required_string("output_dir"), config_path)

    auth_raw = raw.get("auth", {})
    if not isinstance(auth_raw, dict):
        raise BatchConfigError("配置项 auth 必须是 JSON 对象")
    auth_mode = str(auth_raw.get("mode", raw.get("auth_mode", "api_key"))).strip().lower()
    if auth_mode not in {"api_key", "login"}:
        raise BatchConfigError("auth.mode 只能是 api_key 或 login")

    api_key_env: str | None = None
    login_id_env: str | None = None
    password_env: str | None = None
    if auth_mode == "api_key":
        api_key_env = str(auth_raw.get("api_key_env", raw.get("api_key_env", "YUXI_API_KEY"))).strip()
        if not api_key_env:
            raise BatchConfigError("API Key 认证必须配置 api_key_env")
    else:
        login_id_env = str(auth_raw.get("login_id_env", auth_raw.get("username_env", "YUXI_LOGIN_ID"))).strip()
        password_env = str(auth_raw.get("password_env", "YUXI_PASSWORD")).strip()
        if not login_id_env or not password_env:
            raise BatchConfigError("login 认证必须配置 login_id_env 和 password_env")

    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, dict) or not variants_raw:
        raise BatchConfigError("配置项 variants 必须是非空 JSON 对象")

    variants: dict[str, VariantConfig] = {}
    for name, variant_raw in variants_raw.items():
        if not isinstance(name, str) or not name.strip():
            raise BatchConfigError("variants 的名称必须是非空字符串")
        if not isinstance(variant_raw, dict):
            raise BatchConfigError(f"variants.{name} 必须是 JSON 对象")
        config_id = variant_raw.get("agent_config_id")
        if isinstance(config_id, bool) or not isinstance(config_id, int):
            raise BatchConfigError(f"variants.{name}.agent_config_id 必须是整数")
        expected_name = variant_raw.get("expected_knowledge_base_name")
        if not isinstance(expected_name, str) or not expected_name.strip():
            raise BatchConfigError(
                f"variants.{name}.expected_knowledge_base_name 必须填写，脚本需要用它确认 AgentConfig 只启用一个知识库"
            )
        kb_id = variant_raw.get("knowledge_db_id")
        if kb_id is not None and not isinstance(kb_id, (str, int)):
            raise BatchConfigError(f"variants.{name}.knowledge_db_id 必须是字符串或整数")
        expected_modes: dict[str, str | None] = {}
        allowed_modes = {
            "expected_run_mode": {
                "full",
                "stop_after_plan",
                "stop_after_agenda",
                "stop_after_retrieval",
                "stop_after_claims",
            },
            "expected_agenda_mode": {"none", "dynamic"},
            "expected_synthesis_mode": {"direct_chunks", "claims"},
        }
        for key, allowed in allowed_modes.items():
            value = variant_raw.get(key)
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise BatchConfigError(f"variants.{name}.{key} 必须是 {sorted(allowed)} 之一")
            expected_modes[key] = value
        expected_method_family = variant_raw.get("expected_method_family")
        if expected_method_family is not None and (
            not isinstance(expected_method_family, str) or not expected_method_family.strip()
        ):
            raise BatchConfigError(f"variants.{name}.expected_method_family 必须是非空字符串")
        expected_method_version = variant_raw.get("expected_method_version")
        if expected_method_version is not None and (
            not isinstance(expected_method_version, str)
            or not expected_method_version.strip()
        ):
            raise BatchConfigError(
                f"variants.{name}.expected_method_version 必须是非空字符串"
            )
        expected_profile = variant_raw.get("expected_experiment_profile")
        if expected_profile is not None and expected_profile not in {
            "b1",
            "m1",
            "m2",
            "m3",
            "full",
        }:
            raise BatchConfigError(f"variants.{name}.expected_experiment_profile 必须是 " "b1、m1、m2、m3、full 之一")
        expected_effective_profile = variant_raw.get(
            "expected_effective_profile"
        )
        if (
            expected_effective_profile is not None
            and expected_effective_profile
            not in {"b1", "m1", "m2", "m3", "full"}
        ):
            raise BatchConfigError(
                f"variants.{name}.expected_effective_profile 必须是 "
                "b1、m1、m2、m3、full 之一"
            )
        expected_atlas_profile = variant_raw.get("expected_atlas_profile")
        if expected_atlas_profile is not None and expected_atlas_profile not in {
            "map",
            "route",
            "full",
        }:
            raise BatchConfigError(f"variants.{name}.expected_atlas_profile 必须是 " "map、route、full 之一")
        expected_acm_protocol = variant_raw.get("expected_acm_protocol")
        if expected_acm_protocol is not None and expected_acm_protocol not in {
            "legacy_v7",
            "adaptive_coverage",
        }:
            raise BatchConfigError(
                f"variants.{name}.expected_acm_protocol 必须是 "
                "legacy_v7、adaptive_coverage 之一"
            )
        expected_v7_arm = variant_raw.get("expected_v7_experiment_arm")
        if expected_v7_arm is not None and expected_v7_arm not in {
            "a0",
            "a1",
            "a2_k2",
            "a2_k3",
        }:
            raise BatchConfigError(
                f"variants.{name}.expected_v7_experiment_arm 必须是 "
                "a0、a1、a2_k2、a2_k3 之一"
            )
        expected_v7_depth = variant_raw.get("expected_v7_retrieval_depth")
        if expected_v7_depth is not None and expected_v7_depth not in {
            "top10",
            "shadow_top25",
            "visible_top25",
        }:
            raise BatchConfigError(
                f"variants.{name}.expected_v7_retrieval_depth 必须是 "
                "top10、shadow_top25、visible_top25 之一"
            )
        expected_max_search_calls = variant_raw.get(
            "expected_max_search_calls"
        )
        if expected_max_search_calls is not None and (
            isinstance(expected_max_search_calls, bool)
            or not isinstance(expected_max_search_calls, int)
            or expected_max_search_calls < 1
        ):
            raise BatchConfigError(
                f"variants.{name}.expected_max_search_calls 必须是正整数"
            )
        expected_trace_schema = variant_raw.get("expected_trace_schema_version")
        if expected_trace_schema is not None and (
            not isinstance(expected_trace_schema, str) or not expected_trace_schema.strip()
        ):
            raise BatchConfigError(f"variants.{name}.expected_trace_schema_version 必须是非空字符串")
        bounded_versions: dict[str, str | None] = {}
        for key in ("expected_controller_version", "expected_context_view_version"):
            value = variant_raw.get(key)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise BatchConfigError(f"variants.{name}.{key} 必须是非空字符串")
            bounded_versions[key] = value.strip() if isinstance(value, str) else None
        expected_context_window = variant_raw.get(
            "expected_model_context_window_tokens"
        )
        if expected_context_window is not None and (
            isinstance(expected_context_window, bool)
            or not isinstance(expected_context_window, int)
            or expected_context_window < 1
        ):
            raise BatchConfigError(
                f"variants.{name}.expected_model_context_window_tokens 必须是正整数"
            )
        require_companion_selector = variant_raw.get(
            "require_companion_selector",
            False,
        )
        if not isinstance(require_companion_selector, bool):
            raise BatchConfigError(
                f"variants.{name}.require_companion_selector 必须是布尔值"
            )
        variants[name] = VariantConfig(
            name=name,
            agent_config_id=config_id,
            expected_knowledge_base_name=expected_name.strip(),
            expected_run_mode=expected_modes["expected_run_mode"],
            expected_agenda_mode=expected_modes["expected_agenda_mode"],
            expected_synthesis_mode=expected_modes["expected_synthesis_mode"],
            expected_method_family=(
                expected_method_family.strip() if isinstance(expected_method_family, str) else None
            ),
            expected_method_version=(
                expected_method_version.strip()
                if isinstance(expected_method_version, str)
                else None
            ),
            expected_experiment_profile=expected_profile,
            expected_effective_profile=expected_effective_profile,
            expected_atlas_profile=expected_atlas_profile,
            expected_acm_protocol=expected_acm_protocol,
            expected_v7_experiment_arm=expected_v7_arm,
            expected_v7_retrieval_depth=expected_v7_depth,
            expected_max_search_calls=expected_max_search_calls,
            expected_trace_schema_version=(
                expected_trace_schema.strip() if isinstance(expected_trace_schema, str) else None
            ),
            expected_controller_version=bounded_versions[
                "expected_controller_version"
            ],
            expected_context_view_version=bounded_versions[
                "expected_context_view_version"
            ],
            expected_model_context_window_tokens=expected_context_window,
            knowledge_db_id=str(kb_id) if kb_id is not None else None,
            ensure_retrieval_content_scope_all=bool(variant_raw.get("ensure_retrieval_content_scope_all", False)),
            require_companion_selector=require_companion_selector,
        )

    settings = BatchSettings(
        config_path=config_path.resolve(),
        base_url=base_url,
        auth_mode=auth_mode,
        api_key_env=api_key_env,
        login_id_env=login_id_env,
        password_env=password_env,
        agent_id=agent_id,
        input_file=input_file,
        output_dir=output_dir,
        variants=variants,
        concurrency=int(raw.get("concurrency", 1)),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 60)),
        run_timeout_seconds=float(raw.get("run_timeout_seconds", 960)),
        max_attempts=int(raw.get("max_attempts", 2)),
        max_sse_reconnects=int(raw.get("max_sse_reconnects", 8)),
        sse_reconnect_delay_seconds=float(raw.get("sse_reconnect_delay_seconds", 1)),
        verify_tls=bool(raw.get("verify_tls", True)),
        write_raw_events=bool(raw.get("write_raw_events", True)),
        allow_debug_stopped=bool(raw.get("allow_debug_stopped", False)),
        batch_id=raw.get("batch_id"),
    )

    if settings.concurrency < 1:
        raise BatchConfigError("concurrency 必须大于等于 1")
    if settings.request_timeout_seconds <= 0 or settings.run_timeout_seconds <= 0:
        raise BatchConfigError("HTTP 和 Run 超时必须大于 0")
    if settings.max_attempts < 1:
        raise BatchConfigError("max_attempts 必须大于等于 1")
    if settings.max_sse_reconnects < 0:
        raise BatchConfigError("max_sse_reconnects 不能小于 0")
    if not isinstance(settings.batch_id, (str, type(None))):
        raise BatchConfigError("batch_id 必须是字符串")
    return settings


def load_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except OSError as exc:
        raise BatchConfigError(f"无法读取数据集 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BatchConfigError(f"数据集 JSON 格式错误 {path}: {exc}") from exc

    if not isinstance(value, list):
        raise BatchConfigError("输入数据集顶层必须是 JSON 列表")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BatchConfigError(f"数据集第 {index} 项不是 JSON 对象")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise BatchConfigError(f"数据集第 {index} 项的 question 必须是非空字符串")
        rows.append(dict(item))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonlStore:
    """Append-only JSONL writer used for results and resumable state."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        line = json_dumps(record) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line)
                file.flush()
                os.fsync(file.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if line_number == len(lines):
                print(f"警告：忽略 {path} 末尾不完整记录，第 {line_number} 行：{exc}", file=sys.stderr)
                continue
            raise BatchConfigError(f"JSONL 文件损坏 {path}:{line_number}: {exc}") from exc
        if isinstance(record, dict):
            records.append(record)
    return records


class StateStore:
    """Persists submitted Run IDs so a later process can resume the same Run."""

    def __init__(self, path: Path):
        self.store = JsonlStore(path)
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        self.store.append(record)

    def latest_by_job(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in read_jsonl(self.path):
            job_key = record.get("job_key")
            if isinstance(job_key, str):
                latest[job_key] = record
        return latest


def login_for_access_token(
    base_url: str,
    login_id: str,
    password: str,
    timeout_seconds: float = 60.0,
    verify_tls: bool = True,
) -> dict[str, Any]:
    """Use the 0.6.3 OAuth2 password endpoint and return its JSON response."""

    if not login_id.strip() or not password:
        raise BatchConfigError("登录认证需要非空的 login_id 和 password")
    url = f"{base_url.rstrip('/')}/api/auth/token"
    LOGGER.info(
        "AUTH login start endpoint=%s login_id_present=%s password_present=%s timeout=%.1fs",
        url,
        bool(login_id.strip()),
        bool(password),
        timeout_seconds,
    )
    started = time.monotonic()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_seconds), verify=verify_tls) as client:
            response = client.post(
                url,
                data={"username": login_id, "password": password},
                headers={"Accept": "application/json"},
            )
    except httpx.RequestError as exc:
        LOGGER.error("AUTH login transport error elapsed=%.3fs error=%s", time.monotonic() - started, exc)
        raise YuxiTransportError(f"POST {url} 登录连接失败：{exc}") from exc

    LOGGER.info(
        "AUTH login response status=%s elapsed=%.3fs",
        response.status_code,
        time.monotonic() - started,
    )

    if response.is_error:
        detail = response.text
        try:
            error_data = response.json()
            detail = str(error_data.get("detail") or detail)
        except ValueError:
            pass
        raise YuxiApiError("POST", url, response.status_code, detail)

    try:
        result = response.json()
    except ValueError as exc:
        raise YuxiError(f"登录接口返回内容不是 JSON：{response.text[:500]}") from exc
    if not isinstance(result, dict) or not result.get("access_token"):
        raise YuxiError(f"登录响应缺少 access_token：{result!r}")
    return result


def load_auth_token(settings: BatchSettings) -> str:
    """Load an API Key or exchange configured login credentials for a JWT."""

    if settings.auth_mode == "login":
        login_id = os.environ.get(settings.login_id_env or "", "")
        password = os.environ.get(settings.password_env or "", "")
        LOGGER.info(
            "AUTH mode=login login_id_env=%s password_env=%s login_id_present=%s password_present=%s",
            settings.login_id_env,
            settings.password_env,
            bool(login_id.strip()),
            bool(password),
        )
        result = login_for_access_token(
            settings.base_url,
            login_id,
            password,
            settings.request_timeout_seconds,
            settings.verify_tls,
        )
        LOGGER.info(
            "AUTH login succeeded token_type=%s user_id=%s",
            result.get("token_type", "unknown"),
            result.get("user_id", "unknown"),
        )
        return str(result["access_token"])

    api_key = os.environ.get(settings.api_key_env or "", "")
    if not api_key:
        raise BatchConfigError(f"环境变量 {settings.api_key_env} 未设置")
    LOGGER.info("AUTH mode=api_key api_key_env=%s api_key_present=true", settings.api_key_env)
    return api_key


def terminal_sse_status(event: dict[str, Any]) -> str | None:
    """Map a terminal Yuxi SSE event to the corresponding Run status."""

    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    chunk = payload.get("chunk") if isinstance(payload.get("chunk"), dict) else {}
    candidates = (event.get("event"), data.get("event_type"), chunk.get("status"))
    for candidate in candidates:
        if candidate is not None:
            status = TERMINAL_SSE_EVENT_STATUSES.get(str(candidate))
            if status:
                return status
    return None


def extract_sse_tool_activities(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract privacy-safe tool start/result markers from a loading event."""

    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    items = payload.get("items")
    if not isinstance(items, list):
        return []

    activities: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        message = item.get("msg")
        if not isinstance(message, dict):
            continue

        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = str(call.get("name") or function.get("name") or "").strip()
                call_id = str(call.get("id") or "").strip()
                if name and call_id:
                    activities.append(
                        {
                            "kind": "call",
                            "tool_name": name,
                            "tool_call_id": call_id,
                        }
                    )

        if message.get("type") == "tool":
            name = str(message.get("name") or "").strip()
            call_id = str(message.get("tool_call_id") or "").strip()
            if name and call_id:
                content = message.get("content")
                content_chars = len(content) if isinstance(content, str) else len(json_dumps(content))
                activities.append(
                    {
                        "kind": "result",
                        "tool_name": name,
                        "tool_call_id": call_id,
                        "content_chars": content_chars,
                    }
                )
    return activities


class YuxiClient:
    """Small HTTP client for the 0.6.3 chat, Run, thread and history APIs."""

    def __init__(self, base_url: str, auth_token: str, timeout_seconds: float, verify_tls: bool = True):
        if not auth_token.strip():
            raise BatchConfigError("认证 token 为空")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            verify=verify_tls,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Accept": "application/json",
                "User-Agent": "yuxi-0.6.3-batch-rag/1.0",
            },
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> YuxiClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.url(path)
        headers = {"Content-Type": "application/json"} if payload is not None else None
        is_run_poll = method.upper() == "GET" and path.startswith("/api/chat/runs/")
        request_level = logging.DEBUG if is_run_poll else logging.INFO
        payload_summary = ""
        if payload is not None:
            payload_summary = f" payload_keys={sorted(payload)}"
            query = payload.get("query")
            if isinstance(query, str):
                payload_summary += f" query_chars={len(query)}"
        LOGGER.log(request_level, "HTTP request %s %s%s", method.upper(), path, payload_summary)
        started = time.monotonic()
        try:
            response = self.client.request(method, url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            LOGGER.error(
                "HTTP transport error %s %s elapsed=%.3fs error=%s",
                method.upper(),
                path,
                time.monotonic() - started,
                exc,
            )
            raise YuxiTransportError(f"{method} {url} 连接失败：{exc}") from exc

        elapsed = time.monotonic() - started
        if response.is_error:
            LOGGER.warning(
                "HTTP response %s %s status=%s elapsed=%.3fs body_chars=%s",
                method.upper(),
                path,
                response.status_code,
                elapsed,
                len(response.content),
            )
            raise YuxiApiError(method, url, response.status_code, response.text)
        LOGGER.log(
            request_level,
            "HTTP response %s %s status=%s elapsed=%.3fs body_chars=%s",
            method.upper(),
            path,
            response.status_code,
            elapsed,
            len(response.content),
        )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise YuxiError(f"{method} {url} 返回内容不是 JSON：{response.text[:500]}") from exc

    def get_agents(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/chat/agent")

    def get_default_agent(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/chat/default_agent")

    def get_agent_configs(self, agent_id: str) -> dict[str, Any]:
        return self.request_json("GET", f"/api/chat/agent/{agent_id}/configs")

    def get_agent_config(self, agent_id: str, config_id: int) -> dict[str, Any]:
        return self.request_json("GET", f"/api/chat/agent/{agent_id}/configs/{config_id}")

    def get_accessible_databases(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/knowledge/databases/accessible")

    def get_knowledge_query_params(self, db_id: str) -> dict[str, Any]:
        return self.request_json("GET", f"/api/knowledge/databases/{db_id}/query-params")

    def set_knowledge_query_params(self, db_id: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.request_json("PUT", f"/api/knowledge/databases/{db_id}/query-params", params)

    def create_thread(self, agent_id: str, title: str, metadata: dict[str, Any]) -> dict[str, Any]:
        result = self.request_json(
            "POST",
            "/api/chat/thread",
            {"agent_id": agent_id, "title": title, "metadata": metadata},
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise YuxiError(f"创建线程响应缺少 id：{result!r}")
        return result

    def create_run(
        self,
        query: str,
        agent_config_id: int,
        thread_id: str,
        request_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.request_json(
            "POST",
            "/api/chat/runs",
            {
                "query": query,
                "agent_config_id": agent_config_id,
                "thread_id": thread_id,
                "meta": {"request_id": request_id, **metadata},
            },
        )
        if not isinstance(result, dict) or not result.get("run_id"):
            raise YuxiError(f"创建 Run 响应缺少 run_id：{result!r}")
        return result

    def get_run(self, run_id: str) -> dict[str, Any]:
        result = self.request_json("GET", f"/api/chat/runs/{run_id}")
        if isinstance(result, dict) and isinstance(result.get("run"), dict):
            return result["run"]
        if isinstance(result, dict):
            return result
        raise YuxiError(f"Run 状态响应格式错误：{result!r}")

    def get_history(self, thread_id: str) -> dict[str, Any]:
        result = self.request_json("GET", f"/api/chat/thread/{thread_id}/history")
        if not isinstance(result, dict):
            raise YuxiError(f"history 响应格式错误：{result!r}")
        return result

    def read_sse_once(
        self,
        run_id: str,
        after_seq: str,
        deadline_monotonic: float | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        path = f"/api/chat/runs/{run_id}/events?after_seq={after_seq}"
        url = self.url(path)
        events: list[dict[str, Any]] = []
        close_seen = False
        stream_end_reason = "server_eof"
        LOGGER.info("SSE connect run_id=%s after_seq=%s", run_id, after_seq)
        started = time.monotonic()
        last_progress_log = started
        received_count = 0
        seen_sequences = {after_seq} if after_seq not in {"", "0", "0-0"} else set()
        seen_tool_activities: set[tuple[str, str]] = set()
        tool_call_count = 0
        tool_result_count = 0
        if deadline_monotonic is not None and started >= deadline_monotonic:
            raise RunStreamError(f"Run {run_id} 已达到等待超时，停止建立 SSE 连接")
        try:
            with self.client.stream(
                "GET",
                url,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.is_error:
                    body = response.read().decode("utf-8", errors="replace")
                    LOGGER.warning(
                        "SSE response run_id=%s status=%s elapsed=%.3fs body_chars=%s",
                        run_id,
                        response.status_code,
                        time.monotonic() - started,
                        len(body),
                    )
                    raise YuxiApiError("GET", url, response.status_code, body)
                LOGGER.info(
                    "SSE connected run_id=%s status=%s content_type=%s",
                    run_id,
                    response.status_code,
                    response.headers.get("content-type", "unknown"),
                )
                for event_name, data in iter_sse_events(response.iter_lines()):
                    received_count += 1
                    record: dict[str, Any] = {
                        "event": event_name,
                        "data": data,
                        "received_at": utc_now(),
                    }
                    sequence = event_sequence(record)
                    duplicate = bool(sequence and sequence in seen_sequences)
                    if sequence and not duplicate:
                        seen_sequences.add(sequence)

                    if not duplicate:
                        events.append(record)
                        for activity in extract_sse_tool_activities(record):
                            activity_key = (str(activity["kind"]), str(activity["tool_call_id"]))
                            if activity_key in seen_tool_activities:
                                continue
                            seen_tool_activities.add(activity_key)
                            if activity["kind"] == "call":
                                tool_call_count += 1
                                LOGGER.info(
                                    "SSE tool call run_id=%s tool=%s tool_call_id=%s elapsed=%.1fs",
                                    run_id,
                                    activity["tool_name"],
                                    activity["tool_call_id"],
                                    time.monotonic() - started,
                                )
                            else:
                                tool_result_count += 1
                                LOGGER.info(
                                    "SSE tool result run_id=%s tool=%s tool_call_id=%s content_chars=%s "
                                    "elapsed=%.1fs",
                                    run_id,
                                    activity["tool_name"],
                                    activity["tool_call_id"],
                                    activity["content_chars"],
                                    time.monotonic() - started,
                                )

                    terminal_status = terminal_sse_status(record)
                    if terminal_status:
                        LOGGER.info(
                            "SSE terminal event run_id=%s event=%s inferred_run_status=%s seq=%s "
                            "unique_events=%s elapsed=%.1fs",
                            run_id,
                            event_name,
                            terminal_status,
                            sequence or "-",
                            len(events),
                            time.monotonic() - started,
                        )
                        stream_end_reason = "terminal_event"
                        break

                    if event_name == "close":
                        close_seen = True
                        stream_end_reason = "close_event"
                        LOGGER.info(
                            "SSE close event run_id=%s seq=%s unique_events=%s elapsed=%.1fs",
                            run_id,
                            sequence or "-",
                            len(events),
                            time.monotonic() - started,
                        )
                        break

                    now = time.monotonic()
                    if len(events) == 1 and not duplicate:
                        LOGGER.info(
                            "SSE first event run_id=%s event=%s seq=%s",
                            run_id,
                            event_name,
                            sequence or "-",
                        )
                    elif event_name not in {"loading", "heartbeat"} and not duplicate:
                        LOGGER.debug(
                            "SSE event run_id=%s event=%s seq=%s unique_events=%s",
                            run_id,
                            event_name,
                            sequence or "-",
                            len(events),
                        )

                    if now - last_progress_log >= SSE_PROGRESS_LOG_INTERVAL_SECONDS:
                        LOGGER.info(
                            "SSE progress run_id=%s elapsed=%.1fs received=%s unique_events=%s "
                            "last_event=%s tool_calls=%s tool_results=%s",
                            run_id,
                            now - started,
                            received_count,
                            len(events),
                            event_name,
                            tool_call_count,
                            tool_result_count,
                        )
                        last_progress_log = now

                    if deadline_monotonic is not None and now >= deadline_monotonic:
                        LOGGER.error(
                            "SSE deadline reached run_id=%s elapsed=%.1fs unique_events=%s",
                            run_id,
                            now - started,
                            len(events),
                        )
                        raise RunStreamError(f"Run {run_id} 已达到等待超时，停止 SSE 读取")
        except YuxiApiError:
            raise
        except httpx.RequestError as exc:
            LOGGER.warning(
                "SSE transport/read error run_id=%s elapsed=%.3fs error=%s",
                run_id,
                time.monotonic() - started,
                exc,
            )
            raise YuxiTransportError(f"GET {url} SSE 连接失败：{exc}") from exc
        LOGGER.info(
            "SSE stream ended run_id=%s reason=%s unique_events=%s received=%s close_seen=%s elapsed=%.3fs",
            run_id,
            stream_end_reason,
            len(events),
            received_count,
            close_seen,
            time.monotonic() - started,
        )
        return events, close_seen


def iter_sse_events(lines: Any):
    """Parse the subset of SSE used by Yuxi: event + one or more data lines."""

    event_name = "message"
    data_lines: list[str] = []
    for line in lines:
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield event_name, parse_sse_data("\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield event_name, parse_sse_data("\n".join(data_lines))


def parse_sse_data(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}


def run_payload_status(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    return str(status) if status is not None else None


def event_sequence(event: dict[str, Any]) -> str | None:
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    seq = data.get("seq")
    if seq:
        return str(seq)
    if data.get("last_seq"):
        return str(data["last_seq"])
    return None


def consume_run_events(
    client: YuxiClient,
    run_id: str,
    run_timeout_seconds: float,
    max_reconnects: int,
    reconnect_delay_seconds: float,
) -> list[dict[str, Any]]:
    """Read Run SSE, reconnecting with the last Redis stream sequence."""

    deadline = time.monotonic() + run_timeout_seconds
    after_seq = "0-0"
    reconnects = 0
    events: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    last_status: str | None = None
    LOGGER.info(
        "RUN wait start run_id=%s timeout=%.1fs max_reconnects=%s",
        run_id,
        run_timeout_seconds,
        max_reconnects,
    )

    while time.monotonic() < deadline:
        try:
            new_events, close_seen = client.read_sse_once(run_id, after_seq, deadline)
            for event in new_events:
                sequence = event_sequence(event)
                if sequence:
                    after_seq = sequence
                    if sequence in seen_sequences:
                        continue
                    seen_sequences.add(sequence)
                events.append(event)
            LOGGER.debug(
                "RUN SSE batch run_id=%s received=%s total_unique=%s after_seq=%s reconnects=%s",
                run_id,
                len(new_events),
                len(events),
                after_seq,
                reconnects,
            )

            run = client.get_run(run_id)
            status = run_payload_status(run)
            if status != last_status:
                LOGGER.info("RUN status run_id=%s status=%s", run_id, status or "unknown")
                last_status = status
            if status in TERMINAL_RUN_STATUSES:
                LOGGER.info("RUN wait complete run_id=%s status=%s events=%s", run_id, status, len(events))
                return events
            reconnects = 0
            if close_seen:
                LOGGER.info(
                    "RUN SSE closed before terminal status run_id=%s status=%s after_seq=%s; reconnecting",
                    run_id,
                    status or "unknown",
                    after_seq,
                )
        except (YuxiTransportError, httpx.TimeoutException) as exc:
            reconnects += 1
            LOGGER.warning(
                "RUN SSE reconnect run_id=%s reason=%s reconnect=%s/%s after_seq=%s",
                run_id,
                type(exc).__name__,
                reconnects,
                max_reconnects,
                after_seq,
            )
            if reconnects > max_reconnects:
                raise RunStreamError(f"Run {run_id} SSE 重连次数超过上限：{exc}") from exc
            try:
                run = client.get_run(run_id)
            except YuxiError:
                run = {}
            status = run_payload_status(run)
            if status != last_status:
                LOGGER.info("RUN status run_id=%s status=%s", run_id, status or "unknown")
                last_status = status
            if status in TERMINAL_RUN_STATUSES:
                LOGGER.info("RUN wait complete run_id=%s status=%s events=%s", run_id, status, len(events))
                return events
        except YuxiApiError as exc:
            if exc.status_code < 500:
                raise
            reconnects += 1
            LOGGER.warning(
                "RUN SSE server error run_id=%s status=%s reconnect=%s/%s after_seq=%s",
                run_id,
                exc.status_code,
                reconnects,
                max_reconnects,
                after_seq,
            )
            if reconnects > max_reconnects:
                raise RunStreamError(f"Run {run_id} SSE 服务端错误次数超过上限：{exc}") from exc

        if reconnects > max_reconnects:
            raise RunStreamError(f"Run {run_id} SSE 重连次数超过上限")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(reconnect_delay_seconds, remaining, 1.0))

    LOGGER.error("RUN wait timeout run_id=%s timeout=%.1fs events=%s", run_id, run_timeout_seconds, len(events))
    raise RunStreamError(f"Run {run_id} 在 {run_timeout_seconds:.0f} 秒内未进入终态")


def try_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    if value is None:
        return ""
    return json_dumps(value)


def normalize_tool_call(message: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = call.get("name") or function.get("name") or "unknown"
    result = call.get("tool_call_result")
    result_content = result.get("content") if isinstance(result, dict) else None
    return {
        "message_id": message.get("id"),
        "tool_call_id": call.get("id"),
        "tool_name": str(name),
        "args": call.get("args") or function.get("arguments") or {},
        "status": call.get("status"),
        "error_message": call.get("error_message"),
        "result_raw": result_content,
        "result_parsed": try_parse_json(result_content),
    }


def extract_medication_review_trace(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Read the deterministic MedicationReviewAgent trace from the last AI message."""

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("type") != "ai":
            continue
        extra_metadata = message.get("extra_metadata")
        if not isinstance(extra_metadata, dict):
            continue
        additional_kwargs = extra_metadata.get("additional_kwargs")
        if isinstance(additional_kwargs, dict):
            trace = additional_kwargs.get("medication_review_trace")
            if isinstance(trace, dict):
                return trace
        trace = extra_metadata.get("medication_review_trace")
        if isinstance(trace, dict):
            return trace
    return None


def is_complete_six_section_answer(answer: str) -> bool:
    return bool(answer.strip()) and all(
        marker in answer
        for marker in (
            "①【原方案要素清单】",
            "②【逐项判断】",
            "③【正面判断汇总】",
            "④【负面",
            "⑤【综合建议】",
            "⑥【依据清单】",
        )
    )


def extract_history_data(history: dict[str, Any]) -> dict[str, Any]:
    messages = history.get("history")
    if not isinstance(messages, list):
        messages = []

    all_tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict):
                all_tool_calls.append(normalize_tool_call(message, call))

    answer = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("type") == "ai":
            candidate = content_to_text(message.get("content"))
            if candidate.strip():
                answer = candidate
                break

    retrieval_calls = [
        call
        for call in all_tool_calls
        if call["tool_name"]
        in {
            "query_kb",
            "search_evidence",
            "search_review_kb",
            "search_active_obligation",
        }
    ]
    document_open_calls = [
        call
        for call in all_tool_calls
        if call["tool_name"]
        in {
            "open_kb_document",
            "open_evidence_source",
            "open_review_evidence",
            "open_active_evidence",
        }
    ]
    medication_review_trace = extract_medication_review_trace(messages)
    if medication_review_trace:
        retrieval_records = medication_review_trace.get("search_records")
        if not isinstance(retrieval_records, list):
            retrieval_records = medication_review_trace.get("query_records")
        if not isinstance(retrieval_records, list):
            retrieval_records = medication_review_trace.get("retrieval_records")
        if not isinstance(retrieval_records, list):
            retrieval_records = []
        successful = any(
            isinstance(record, dict)
            and record.get("status")
            not in {
                "technical_failed",
                "assessment_failed",
                "invalid_query",
                "skipped_budget",
                "skipped_early_stop",
            }
            for record in retrieval_records
        )
        retrieval_status = "called" if successful else "failed" if retrieval_records else "not_called"
    else:
        retrieval_status = "called" if retrieval_calls else "not_called"

    return {
        "answer": answer,
        "retrieval_status": retrieval_status,
        "retrieval_calls": retrieval_calls,
        "document_open_calls": document_open_calls,
        "all_tool_calls": all_tool_calls,
        "medication_review_trace": medication_review_trace,
        "history": messages,
    }


def extract_config_context(config_response: dict[str, Any]) -> dict[str, Any]:
    config = config_response.get("config") if isinstance(config_response.get("config"), dict) else config_response
    config_json = config.get("config_json") if isinstance(config, dict) else {}
    if not isinstance(config_json, dict):
        return {}
    context = config_json.get("context")
    return context if isinstance(context, dict) else {}


def build_error(error: Exception) -> dict[str, Any]:
    result = {"type": type(error).__name__, "message": str(error)}
    if isinstance(error, YuxiApiError):
        result["status_code"] = error.status_code
    return result


def terminal_run_status(run: dict[str, Any]) -> str | None:
    status = run.get("status")
    return str(status) if status is not None else None


def make_batch_id(configured: str | None) -> str:
    if configured and configured.strip():
        return configured.strip()
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


class BatchRunner:
    """Coordinates preflight, isolated jobs, persistence and resume."""

    def __init__(self, settings: BatchSettings, auth_token: str):
        self.settings = settings
        self.auth_token = auth_token
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.settings.output_dir / "results"
        self.events_dir = self.settings.output_dir / "events"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.state = StateStore(self.settings.output_dir / "state.jsonl")
        self._manifest_path = self.settings.output_dir / "manifest.json"
        self._manifest_lock = threading.Lock()

    def _variant_expectations(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "method_family": variant.expected_method_family,
                "method_version": variant.expected_method_version,
                "experiment_profile": variant.expected_experiment_profile,
                "effective_profile": variant.expected_effective_profile,
                "atlas_profile": variant.expected_atlas_profile,
                "acm_protocol": variant.expected_acm_protocol,
                "v7_experiment_arm": variant.expected_v7_experiment_arm,
                "v7_retrieval_depth": variant.expected_v7_retrieval_depth,
                "max_search_calls": variant.expected_max_search_calls,
                "trace_schema_version": (
                    variant.expected_trace_schema_version
                ),
                "controller_version": variant.expected_controller_version,
                "context_view_version": variant.expected_context_view_version,
                "model_context_window_tokens": (
                    variant.expected_model_context_window_tokens
                ),
                "require_companion_selector": (
                    variant.require_companion_selector
                ),
            }
            for name, variant in self.settings.variants.items()
        }

    def client(self) -> YuxiClient:
        return YuxiClient(
            self.settings.base_url,
            self.auth_token,
            self.settings.request_timeout_seconds,
            self.settings.verify_tls,
        )

    def preflight(self, batch_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        LOGGER.info(
            "PREFLIGHT start batch_id=%s rows=%s agent_id=%s variants=%s",
            batch_id,
            len(rows),
            self.settings.agent_id,
            ",".join(self.settings.variants),
        )
        with self.client() as client:
            agents_response = client.get_agents()
            agents = agents_response.get("agents", []) if isinstance(agents_response, dict) else []
            LOGGER.info("PREFLIGHT agents received=%s", len(agents) if isinstance(agents, list) else 0)
            agent_ids = {str(agent.get("id")) for agent in agents if isinstance(agent, dict) and agent.get("id")}
            if agent_ids and self.settings.agent_id not in agent_ids:
                raise BatchConfigError(
                    f"agent_id {self.settings.agent_id!r} 不在 /api/chat/agent 返回列表中：{sorted(agent_ids)}"
                )

            config_snapshots: dict[str, Any] = {}
            query_param_snapshots: dict[str, Any] = {}
            for name, variant in self.settings.variants.items():
                LOGGER.info(
                    "PREFLIGHT config start variant=%s agent_config_id=%s expected_kb=%s",
                    name,
                    variant.agent_config_id,
                    variant.expected_knowledge_base_name,
                )
                config_response = client.get_agent_config(self.settings.agent_id, variant.agent_config_id)
                context = extract_config_context(config_response)
                expected_name = variant.expected_knowledge_base_name
                knowledges = context.get("knowledges")
                LOGGER.info(
                    "PREFLIGHT config loaded variant=%s model=%s subagents_model=%s knowledges=%s",
                    name,
                    context.get("model", "unknown"),
                    context.get("subagents_model", "unknown"),
                    knowledges,
                )
                if expected_name:
                    if not isinstance(knowledges, list) or knowledges != [expected_name]:
                        raise BatchConfigError(
                            f"{name} AgentConfig 必须只启用知识库 {expected_name!r}，实际 knowledges={knowledges!r}"
                        )
                for context_key, expected_value in (
                    ("run_mode", variant.expected_run_mode),
                    ("agenda_mode", variant.expected_agenda_mode),
                    ("synthesis_mode", variant.expected_synthesis_mode),
                    (
                        "experiment_profile",
                        variant.expected_experiment_profile,
                    ),
                    ("atlas_profile", variant.expected_atlas_profile),
                    ("acm_protocol", variant.expected_acm_protocol),
                    (
                        "v7_experiment_arm",
                        variant.expected_v7_experiment_arm,
                    ),
                    (
                        "v7_retrieval_depth",
                        variant.expected_v7_retrieval_depth,
                    ),
                    (
                        "max_search_calls",
                        variant.expected_max_search_calls,
                    ),
                ):
                    if expected_value is not None and context.get(context_key) != expected_value:
                        raise BatchConfigError(
                            f"{name} AgentConfig 的 {context_key} 应为 "
                            f"{expected_value!r}，实际为 {context.get(context_key)!r}"
                        )
                config_snapshots[name] = config_response

                if variant.ensure_retrieval_content_scope_all:
                    if not variant.knowledge_db_id:
                        raise BatchConfigError(f"{name} 需要设置 retrieval_content_scope，但缺少 knowledge_db_id")
                    LOGGER.warning(
                        "PREFLIGHT changing knowledge query params variant=%s db_id=%s scope=all",
                        name,
                        variant.knowledge_db_id,
                    )
                    before = client.get_knowledge_query_params(variant.knowledge_db_id)
                    after = client.set_knowledge_query_params(
                        variant.knowledge_db_id,
                        {"retrieval_content_scope": "all"},
                    )
                    query_param_snapshots[name] = {"before": before, "after": after}
                elif variant.knowledge_db_id:
                    LOGGER.info(
                        "PREFLIGHT reading knowledge query params variant=%s db_id=%s",
                        name,
                        variant.knowledge_db_id,
                    )
                    query_param_snapshots[name] = client.get_knowledge_query_params(variant.knowledge_db_id)

        LOGGER.info("PREFLIGHT complete batch_id=%s manifest_path=%s", batch_id, self._manifest_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "created_at": utc_now(),
            "base_url": self.settings.base_url,
            "auth_mode": self.settings.auth_mode,
            "agent_id": self.settings.agent_id,
            "config_file": str(self.settings.config_path),
            "input_file": str(self.settings.input_file),
            "input_sha256": sha256_file(self.settings.input_file),
            "input_count": len(rows),
            "variants": list(self.settings.variants),
            "variant_config_ids": {name: variant.agent_config_id for name, variant in self.settings.variants.items()},
            "variant_expectations": self._variant_expectations(),
            "runtime_fingerprints": {},
            "config_snapshots": config_snapshots,
            "knowledge_query_params": query_param_snapshots,
            "execution": {
                "concurrency": self.settings.concurrency,
                "run_timeout_seconds": self.settings.run_timeout_seconds,
                "max_attempts": self.settings.max_attempts,
                "verify_tls": self.settings.verify_tls,
            },
        }

    def load_or_create_manifest(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        input_hash = sha256_file(self.settings.input_file)
        if self._manifest_path.exists():
            LOGGER.info("MANIFEST existing path=%s; validating input and AgentConfig IDs", self._manifest_path)
            manifest = load_json_object(self._manifest_path)
            if manifest.get("input_sha256") != input_hash:
                raise BatchConfigError(
                    f"输出目录已有不同输入文件的 manifest：{self._manifest_path}；请换一个 output_dir"
                )
            expected_config_ids = {name: variant.agent_config_id for name, variant in self.settings.variants.items()}
            if manifest.get("agent_id") != self.settings.agent_id:
                raise BatchConfigError("输出目录已有不同 agent_id 的 manifest；请换一个 output_dir")
            if manifest.get("variant_config_ids") != expected_config_ids:
                raise BatchConfigError("输出目录已有不同 AgentConfig ID 的 manifest；请换一个 output_dir")
            stored_expectations = manifest.get("variant_expectations")
            if isinstance(stored_expectations, dict):
                stored_expectations = {
                    name: {
                        **(value if isinstance(value, dict) else {}),
                        "effective_profile": (
                            value.get("effective_profile")
                            if isinstance(value, dict)
                            else None
                        ),
                        "method_version": (
                            value.get("method_version")
                            if isinstance(value, dict)
                            else None
                        ),
                        "require_companion_selector": (
                            value.get("require_companion_selector", False)
                            if isinstance(value, dict)
                            else False
                        ),
                    }
                    for name, value in stored_expectations.items()
                }
            if stored_expectations != self._variant_expectations():
                raise BatchConfigError(
                    "输出目录已有不同的实验校验条件；请换一个 output_dir"
                )
            current = self.preflight(str(manifest.get("batch_id") or ""), rows)
            stored_snapshots = manifest.get("config_snapshots")
            current_snapshots = current.get("config_snapshots")
            if isinstance(stored_snapshots, dict) and isinstance(
                current_snapshots,
                dict,
            ):
                for name in self.settings.variants:
                    stored_context = extract_config_context(stored_snapshots.get(name) or {})
                    current_context = extract_config_context(current_snapshots.get(name) or {})
                    if stored_context != current_context:
                        raise BatchConfigError(
                            f"{name} 的远程 AgentConfig 已在本批次开始后变化；" "请换一个 output_dir，避免混合实验条件"
                        )
            LOGGER.info(
                "MANIFEST reused batch_id=%s input_count=%s",
                manifest.get("batch_id", "unknown"),
                manifest.get("input_count", "unknown"),
            )
            return manifest

        batch_id = make_batch_id(self.settings.batch_id)
        LOGGER.info("MANIFEST not found; running remote preflight path=%s", self._manifest_path)
        manifest = self.preflight(batch_id, rows)
        atomic_write_json(self._manifest_path, manifest)
        LOGGER.info("MANIFEST written path=%s", self._manifest_path)
        return manifest

    def validate_runtime_fingerprint(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Freeze method assets first seen in a remote Trace for this batch."""
        trace = record.get("medication_review_trace")
        if not isinstance(trace, dict) or trace.get("schema_version") not in {
            "7.0",
            "9.0",
            "10.0",
            "13.0",
        }:
            return None
        atlas_snapshot = trace.get("atlas_snapshot")
        prompt_hashes = trace.get("prompt_hashes")
        trace_schema = str(trace.get("schema_version") or "")
        if trace_schema == "13.0":
            fingerprint = {
                "atlas_snapshot_hash": (
                    atlas_snapshot.get("snapshot_hash")
                    if isinstance(atlas_snapshot, dict)
                    else None
                ),
                "controller_version": trace.get("controller_version"),
                "context_view_version": trace.get("context_view_version"),
                "prompt_version": trace.get("prompt_version"),
                "model_context_window_tokens": trace.get(
                    "model_context_window_tokens"
                ),
                "provider_context_window_tokens": trace.get(
                    "provider_context_window_tokens"
                ),
                "context_window_verified": trace.get("context_window_verified"),
            }
        else:
            prompt_key = (
                "atlas_navigation_prompt_hash"
                if trace_schema == "10.0"
                else "companion_selector_prompt_hash"
            )
            fingerprint = {
                "atlas_snapshot_hash": (
                    atlas_snapshot.get("snapshot_hash")
                    if isinstance(atlas_snapshot, dict)
                    else None
                ),
                prompt_key: (
                    prompt_hashes.get("atlas_navigation")
                    if trace_schema == "10.0" and isinstance(prompt_hashes, dict)
                    else prompt_hashes.get("companion_selector")
                    if isinstance(prompt_hashes, dict)
                    else None
                ),
            }
        missing = [
            key
            for key, value in fingerprint.items()
            if value is None or value == ""
        ]
        if missing:
            return {
                "type": "missing_runtime_fingerprint",
                "message": (
                    f"Trace {trace.get('schema_version')} 缺少实验指纹字段："
                    + "、".join(missing)
                ),
            }
        variant_name = str(record.get("variant") or "")
        with self._manifest_lock:
            manifest = load_json_object(self._manifest_path)
            fingerprints = manifest.get("runtime_fingerprints")
            if not isinstance(fingerprints, dict):
                fingerprints = {}
            stored = fingerprints.get(variant_name)
            if stored is None:
                fingerprints[variant_name] = fingerprint
                manifest["runtime_fingerprints"] = fingerprints
                atomic_write_json(self._manifest_path, manifest)
                LOGGER.info(
                    "MANIFEST runtime fingerprint frozen variant=%s fingerprint=%s",
                    variant_name,
                    fingerprint,
                )
                return None
            if stored != fingerprint:
                return {
                    "type": "runtime_fingerprint_changed",
                    "message": (
                        f"{variant_name} 的 Atlas/提示/控制器/上下文运行指纹已变化；"
                        "不能续跑到同一 output_dir"
                    ),
                    "expected": stored,
                    "actual": fingerprint,
                }
        return None

    def result_store(self, variant_name: str) -> JsonlStore:
        return JsonlStore(self.results_dir / f"{variant_name}.jsonl")

    def existing_successes(self) -> set[str]:
        completed: set[str] = set()
        for name in self.settings.variants:
            path = self.results_dir / f"{name}.jsonl"
            for record in read_jsonl(path):
                if record.get("result_status") == "success" and isinstance(record.get("job_key"), str):
                    completed.add(record["job_key"])
        return completed

    def latest_attempts(self) -> dict[str, dict[str, Any]]:
        return self.state.latest_by_job()

    def event_path(self, job: Job) -> Path:
        return self.events_dir / job.variant.name / f"{job.row_index:06d}.attempt{job.attempt}.jsonl"

    def write_events(self, path: Path, events: list[dict[str, Any]]) -> None:
        if not self.settings.write_raw_events:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not events:
            path.touch(exist_ok=True)
            return
        store = JsonlStore(path)
        for event in events:
            store.append(event)

    def append_submitted_state(self, job: Job, thread_id: str, run_id: str) -> None:
        self.state.append(
            {
                "event": "submitted",
                "job_key": job.job_key,
                "attempt": job.attempt,
                "batch_id": job.batch_id,
                "variant": job.variant.name,
                "row_index": job.row_index,
                "thread_id": thread_id,
                "run_id": run_id,
                "request_id": job.request_id,
                "submitted_at": utc_now(),
            }
        )

    def append_attempt_failed_state(self, job: Job, error: Exception) -> None:
        self.state.append(
            {
                "event": "attempt_failed",
                "job_key": job.job_key,
                "attempt": job.attempt,
                "error": build_error(error),
                "at": utc_now(),
            }
        )

    def append_completed_state(self, record: dict[str, Any]) -> None:
        self.state.append(
            {
                "event": "completed",
                "job_key": record["job_key"],
                "attempt": record["attempt"],
                "result_status": record["result_status"],
                "run_status": record.get("run_status"),
                "run_id": record.get("run_id"),
                "completed_at": utc_now(),
            }
        )

    def starting_submission(self, job_key: str) -> dict[str, Any] | None:
        latest = self.latest_attempts().get(job_key)
        return latest if latest and latest.get("event") == "submitted" else None

    def next_attempt(self, job_key: str) -> int:
        latest = self.latest_attempts().get(job_key)
        if not latest:
            return 1
        try:
            return int(latest.get("attempt", 0)) + 1
        except (TypeError, ValueError):
            return 1

    def build_job(
        self, batch_id: str, row_index: int, row: dict[str, Any], variant: VariantConfig, attempt: int
    ) -> Job:
        return Job(batch_id=batch_id, row_index=row_index, input_record=row, variant=variant, attempt=attempt)

    def run_one(self, job: Job, submission: dict[str, Any] | None = None) -> dict[str, Any]:
        started_at = utc_now()
        started_clock = time.monotonic()
        current_job = job
        current_submission = submission

        LOGGER.info(
            "JOB start job_key=%s attempt=%s agent_config_id=%s question_chars=%s resume=%s",
            job.job_key,
            job.attempt,
            job.variant.agent_config_id,
            len(job.question),
            bool(submission),
        )

        while current_job.attempt <= self.settings.max_attempts:
            try:
                with self.client() as client:
                    if current_submission:
                        thread_id = str(current_submission["thread_id"])
                        run_id = str(current_submission["run_id"])
                        LOGGER.info(
                            "JOB resume job_key=%s attempt=%s thread_id=%s run_id=%s",
                            current_job.job_key,
                            current_job.attempt,
                            thread_id,
                            run_id,
                        )
                    else:
                        LOGGER.info(
                            "JOB create thread job_key=%s attempt=%s",
                            current_job.job_key,
                            current_job.attempt,
                        )
                        thread = client.create_thread(
                            self.settings.agent_id,
                            f"{current_job.batch_id}-{current_job.variant.name}-{current_job.row_index:06d}",
                            {
                                "batch_id": current_job.batch_id,
                                "row_index": current_job.row_index,
                                "variant": current_job.variant.name,
                            },
                        )
                        thread_id = str(thread["id"])
                        LOGGER.info("JOB thread created job_key=%s thread_id=%s", current_job.job_key, thread_id)
                        run = client.create_run(
                            current_job.question,
                            current_job.variant.agent_config_id,
                            thread_id,
                            current_job.request_id,
                            {
                                "batch_id": current_job.batch_id,
                                "row_index": current_job.row_index,
                                "variant": current_job.variant.name,
                            },
                        )
                        run_id = str(run["run_id"])
                        LOGGER.info(
                            "JOB run created job_key=%s thread_id=%s run_id=%s request_id=%s",
                            current_job.job_key,
                            thread_id,
                            run_id,
                            current_job.request_id,
                        )
                        self.append_submitted_state(current_job, thread_id, run_id)
                        current_submission = {
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "request_id": current_job.request_id,
                            "attempt": current_job.attempt,
                        }

                    initial_status = client.get_run(run_id)
                    initial_status_name = terminal_run_status(initial_status)
                    LOGGER.info(
                        "JOB initial status job_key=%s run_id=%s status=%s",
                        current_job.job_key,
                        run_id,
                        initial_status_name or "unknown",
                    )
                    if initial_status_name not in TERMINAL_RUN_STATUSES:
                        events = consume_run_events(
                            client,
                            run_id,
                            self.settings.run_timeout_seconds,
                            self.settings.max_sse_reconnects,
                            self.settings.sse_reconnect_delay_seconds,
                        )
                    else:
                        events = []
                    final_run = client.get_run(run_id)
                    LOGGER.info(
                        "JOB final status job_key=%s run_id=%s status=%s events=%s",
                        current_job.job_key,
                        run_id,
                        terminal_run_status(final_run) or "unknown",
                        len(events),
                    )
                    history_response = client.get_history(thread_id)
                    LOGGER.info(
                        "JOB history loaded job_key=%s thread_id=%s history_messages=%s",
                        current_job.job_key,
                        thread_id,
                        (
                            len(history_response.get("history", []))
                            if isinstance(history_response.get("history"), list)
                            else 0
                        ),
                    )

                history_data = extract_history_data(history_response)
                run_status = terminal_run_status(final_run)
                medication_review_trace = history_data.get("medication_review_trace")
                medication_review_status = (
                    str(medication_review_trace.get("run_status") or "")
                    if isinstance(medication_review_trace, dict)
                    else ""
                )
                trace_schema_version = (
                    str(medication_review_trace.get("schema_version") or "")
                    if isinstance(medication_review_trace, dict)
                    else ""
                )
                answer_validation = (
                    medication_review_trace.get("answer_validation")
                    if isinstance(medication_review_trace, dict)
                    else None
                )
                has_valid_review_answer = bool(history_data["answer"].strip()) and (
                    trace_schema_version in DIRECT_REVIEW_TRACE_VERSIONS
                    or (
                        trace_schema_version == "3.0"
                        and isinstance(
                            medication_review_trace.get("final_review"),
                            dict,
                        )
                        and is_complete_six_section_answer(history_data["answer"])
                    )
                    or (
                        trace_schema_version == "2.0"
                        and isinstance(answer_validation, dict)
                        and answer_validation.get("valid") is True
                    )
                    or trace_schema_version not in KNOWN_REVIEW_TRACE_VERSIONS
                )
                trace_expectation_errors: list[str] = []
                if isinstance(medication_review_trace, dict):
                    for trace_key, expected_value in (
                        (
                            "method_family",
                            current_job.variant.expected_method_family,
                        ),
                        (
                            "method_version",
                            current_job.variant.expected_method_version,
                        ),
                        (
                            "experiment_profile",
                            current_job.variant.expected_experiment_profile,
                        ),
                        (
                            "effective_profile",
                            current_job.variant.expected_effective_profile,
                        ),
                        (
                            "atlas_profile",
                            current_job.variant.expected_atlas_profile,
                        ),
                        (
                            "protocol",
                            current_job.variant.expected_acm_protocol,
                        ),
                        (
                            "experiment_arm",
                            current_job.variant.expected_v7_experiment_arm,
                        ),
                        (
                            "retrieval_depth",
                            current_job.variant.expected_v7_retrieval_depth,
                        ),
                        (
                            "schema_version",
                            current_job.variant.expected_trace_schema_version,
                        ),
                        (
                            "controller_version",
                            current_job.variant.expected_controller_version,
                        ),
                        (
                            "context_view_version",
                            current_job.variant.expected_context_view_version,
                        ),
                        (
                            "model_context_window_tokens",
                            current_job.variant.expected_model_context_window_tokens,
                        ),
                    ):
                        if (
                            expected_value is not None
                            and (
                                medication_review_trace.get(trace_key)
                                if trace_key != "experiment_profile"
                                else medication_review_trace.get(
                                    "experiment_profile",
                                    medication_review_trace.get("requested_profile"),
                                )
                            )
                            != expected_value
                        ):
                            actual_value = (
                                medication_review_trace.get(trace_key)
                                if trace_key != "experiment_profile"
                                else medication_review_trace.get(
                                    "experiment_profile",
                                    medication_review_trace.get("requested_profile"),
                                )
                            )
                            trace_expectation_errors.append(
                                f"{trace_key} 应为 {expected_value!r}，实际为 " f"{actual_value!r}"
                            )
                if current_job.variant.require_companion_selector:
                    selection = (
                        medication_review_trace.get("companion_selection")
                        if isinstance(medication_review_trace, dict)
                        else None
                    )
                    selector_audit = (
                        selection.get("selector_audit")
                        if isinstance(selection, dict)
                        else None
                    )
                    selector_status = (
                        str(selector_audit.get("status") or "")
                        if isinstance(selector_audit, dict)
                        else ""
                    )
                    if trace_schema_version not in {"7.0", "9.0"}:
                        trace_expectation_errors.append(
                            "require_companion_selector=true 时必须返回 ACM Trace 7.0 或 9.0"
                        )
                    elif medication_review_trace.get(
                        "companion_selection_attempted"
                    ) is not True:
                        trace_expectation_errors.append(
                            "伴随线索选择器没有运行"
                        )
                    elif selector_status not in VALID_COMPANION_SELECTOR_STATUSES:
                        trace_expectation_errors.append(
                            "伴随线索选择器状态必须是 success、repaired 或 empty，"
                            f"实际为 {selector_status or '<缺失>'}"
                        )
                if run_status in {"failed", "cancelled", "interrupted"}:
                    result_status = "failed"
                    result_error = {
                        "type": final_run.get("error_type") or f"run_{run_status}",
                        "message": final_run.get("error_message") or f"Run 进入终态：{run_status}",
                    }
                elif run_status == SUCCESS_RUN_STATUS and medication_review_status:
                    if trace_expectation_errors:
                        result_status = "failed"
                        result_error = {
                            "type": "medication_review_trace_mismatch",
                            "message": "；".join(trace_expectation_errors),
                        }
                    elif medication_review_status in {"completed", "partial"} and has_valid_review_answer:
                        result_status = "success"
                        result_error = None
                    elif (
                        medication_review_status == "debug_stopped"
                        and self.settings.allow_debug_stopped
                        and trace_schema_version == "3.0"
                        and bool(history_data["answer"].strip())
                    ):
                        result_status = "success"
                        result_error = None
                    else:
                        result_status = "failed"
                        trace_errors = medication_review_trace.get("errors")
                        first_error = trace_errors[0] if isinstance(trace_errors, list) and trace_errors else {}
                        result_error = {
                            "type": f"medication_review_{medication_review_status}",
                            "message": (
                                first_error.get("message")
                                if isinstance(first_error, dict) and first_error.get("message")
                                else f"处方关系覆盖运行状态为 {medication_review_status}"
                            ),
                        }
                elif run_status == SUCCESS_RUN_STATUS and history_data["answer"].strip():
                    result_status = "success"
                    result_error = None
                else:
                    result_status = "incomplete"
                    result_error = {"type": "missing_answer", "message": f"Run 状态为 {run_status!r}，未提取到最终答案"}
                record = self.build_record(
                    current_job,
                    started_at,
                    started_clock,
                    thread_id,
                    run_id,
                    run_status,
                    history_data,
                    events,
                    result_status=result_status,
                    error=result_error,
                )
                runtime_fingerprint_error = self.validate_runtime_fingerprint(
                    record
                )
                if runtime_fingerprint_error is not None:
                    result_status = "failed"
                    result_error = runtime_fingerprint_error
                    record["result_status"] = result_status
                    record["error"] = result_error
                self.write_events(self.event_path(current_job), events)

                LOGGER.info(
                    "JOB extracted job_key=%s run_status=%s result_status=%s answer_chars=%s "
                    "retrieval_calls=%s medication_review_status=%s search_records=%s",
                    current_job.job_key,
                    run_status or "unknown",
                    result_status,
                    len(history_data["answer"]),
                    len(history_data["retrieval_calls"]),
                    medication_review_status or "not_applicable",
                    (
                        len(
                            medication_review_trace.get("search_records")
                            or medication_review_trace.get("query_records")
                            or medication_review_trace.get("retrieval_records")
                            or []
                        )
                        if isinstance(medication_review_trace, dict)
                        else 0
                    ),
                )

                if result_status == "success":
                    LOGGER.info("JOB complete job_key=%s attempt=%s", current_job.job_key, current_job.attempt)
                    return record
                if current_job.attempt >= self.settings.max_attempts:
                    LOGGER.warning(
                        "JOB exhausted attempts job_key=%s attempt=%s result_status=%s",
                        current_job.job_key,
                        current_job.attempt,
                        result_status,
                    )
                    return record

                LOGGER.warning(
                    "JOB retrying job_key=%s from_attempt=%s next_attempt=%s result_status=%s",
                    current_job.job_key,
                    current_job.attempt,
                    current_job.attempt + 1,
                    result_status,
                )
                self.append_attempt_failed_state(current_job, RunStreamError(f"Run 未成功完成：{run_status}"))
                current_job = Job(
                    batch_id=current_job.batch_id,
                    row_index=current_job.row_index,
                    input_record=current_job.input_record,
                    variant=current_job.variant,
                    attempt=current_job.attempt + 1,
                )
                current_submission = None
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the batch
                duration = time.monotonic() - started_clock
                LOGGER.error(
                    "JOB failed job_key=%s attempt=%s elapsed=%.3fs error_type=%s error=%s",
                    current_job.job_key,
                    current_job.attempt,
                    duration,
                    type(exc).__name__,
                    exc,
                    exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                )
                record = self.build_record(
                    current_job,
                    started_at,
                    started_clock,
                    str(current_submission.get("thread_id")) if current_submission else None,
                    str(current_submission.get("run_id")) if current_submission else None,
                    None,
                    {
                        "answer": "",
                        "retrieval_status": "unknown",
                        "retrieval_calls": [],
                        "document_open_calls": [],
                        "all_tool_calls": [],
                        "medication_review_trace": None,
                        "history": [],
                    },
                    [],
                    result_status="failed",
                    error=build_error(exc),
                )
                record["duration_seconds"] = round(duration, 3)
                return record

        raise AssertionError("unreachable")

    def build_record(
        self,
        job: Job,
        started_at: str,
        started_clock: float,
        thread_id: str | None,
        run_id: str | None,
        run_status: str | None,
        history_data: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        result_status: str,
        error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace = history_data.get("medication_review_trace")
        review_status = str(trace.get("run_status") or "") if isinstance(trace, dict) else None
        method_version = str(trace.get("method_version") or "") if isinstance(trace, dict) else None
        method_family = str(trace.get("method_family") or "") if isinstance(trace, dict) else None
        experiment_profile = (
            str(trace.get("experiment_profile") or trace.get("requested_profile") or "")
            if isinstance(trace, dict)
            else None
        )
        atlas_profile = str(trace.get("atlas_profile") or "") if isinstance(trace, dict) else None
        acm_protocol = str(trace.get("protocol") or "") if isinstance(trace, dict) else None
        v7_experiment_arm = (
            str(trace.get("experiment_arm") or "")
            if isinstance(trace, dict)
            else None
        )
        v7_retrieval_depth = (
            str(trace.get("retrieval_depth") or "")
            if isinstance(trace, dict)
            else None
        )
        v7_contract = (
            trace.get("contract_report")
            if isinstance(trace, dict)
            and isinstance(trace.get("contract_report"), dict)
            else {}
        )
        adaptive_coverage = (
            trace.get("adaptive_coverage_report")
            if isinstance(trace, dict)
            and isinstance(trace.get("adaptive_coverage_report"), dict)
            else {}
        )
        context_manifests = (
            trace.get("context_manifests")
            if isinstance(trace, dict)
            and isinstance(trace.get("context_manifests"), list)
            else []
        )
        generation_aborts = (
            trace.get("generation_abort_records")
            if isinstance(trace, dict)
            and isinstance(trace.get("generation_abort_records"), list)
            else []
        )
        citation_verification = (
            trace.get("citation_verification")
            if isinstance(trace, dict)
            and isinstance(trace.get("citation_verification"), dict)
            else {}
        )
        atlas_snapshot = (
            trace.get("atlas_snapshot")
            if isinstance(trace, dict) and isinstance(trace.get("atlas_snapshot"), dict)
            else {}
        )
        companion_selection = (
            trace.get("companion_selection")
            if isinstance(trace, dict)
            and isinstance(trace.get("companion_selection"), dict)
            else {}
        )
        selector_audit = (
            companion_selection.get("selector_audit")
            if isinstance(companion_selection.get("selector_audit"), dict)
            else {}
        )
        trace_usage = trace.get("usage") if isinstance(trace, dict) and isinstance(trace.get("usage"), dict) else {}
        return {
            "schema_version": SCHEMA_VERSION,
            "job_key": job.job_key,
            "batch_id": job.batch_id,
            "row_index": job.row_index,
            "variant": job.variant.name,
            "attempt": job.attempt,
            "question": job.question,
            "input_record": job.input_record,
            "agent_id": self.settings.agent_id,
            "agent_config_id": job.variant.agent_config_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "request_id": job.request_id,
            "run_status": run_status,
            "result_status": result_status,
            "review_status": review_status,
            "method_version": method_version,
            "method_family": method_family,
            "experiment_profile": experiment_profile,
            "atlas_profile": atlas_profile,
            "acm_protocol": acm_protocol,
            "adaptive_coverage_status": adaptive_coverage.get("status"),
            "adaptive_investigation_count": adaptive_coverage.get(
                "actual_investigation_count"
            ),
            "adaptive_pending_recovery_ids": adaptive_coverage.get(
                "pending_recovery_ids"
            ) or [],
            "adaptive_gap_assessment_status": adaptive_coverage.get(
                "gap_assessment_status"
            ),
            "bounded_model_call_count": len(context_manifests),
            "bounded_generation_abort_count": len(generation_aborts),
            "bounded_citation_verification_status": citation_verification.get(
                "status"
            ),
            "bounded_max_projected_total_tokens": max(
                (
                    int(value.get("projected_total_tokens") or 0)
                    for value in context_manifests
                    if isinstance(value, dict)
                ),
                default=0,
            ),
            "v7_experiment_arm": v7_experiment_arm,
            "v7_retrieval_depth": v7_retrieval_depth,
            "v7_contract_status": v7_contract.get("status"),
            "atlas_snapshot_hash": atlas_snapshot.get("snapshot_hash"),
            "companion_selector_status": selector_audit.get("status"),
            "companion_selector_prompt_hash": (
                (trace.get("prompt_hashes") or {}).get(
                    "companion_selector"
                )
                if isinstance(trace, dict)
                and isinstance(trace.get("prompt_hashes"), dict)
                else None
            ),
            "atlas_navigation_prompt_hash": (
                (trace.get("prompt_hashes") or {}).get("atlas_navigation")
                if isinstance(trace, dict)
                and isinstance(trace.get("prompt_hashes"), dict)
                else None
            ),
            "trace_schema_version": (trace.get("schema_version") if isinstance(trace, dict) else None),
            "run_mode": trace.get("run_mode") if isinstance(trace, dict) else None,
            "agenda_mode": trace.get("agenda_mode") if isinstance(trace, dict) else None,
            "synthesis_mode": (trace.get("synthesis_mode") if isinstance(trace, dict) else None),
            "effective_profile": (trace.get("effective_profile") if isinstance(trace, dict) else None),
            "budget_usage": {
                "logical_search_count": trace_usage.get("logical_search_count"),
                "technical_attempt_count": trace_usage.get("technical_attempt_count"),
                "open_count": trace_usage.get("open_count"),
                "active_evidence_count": trace_usage.get("active_evidence_count"),
                "logical_agent_steps": trace_usage.get(
                    "logical_agent_steps",
                    trace_usage.get("logical_search_count"),
                ),
                "technical_attempts": trace_usage.get(
                    "technical_attempts",
                    trace_usage.get("technical_attempt_count"),
                ),
                "executed_subqueries": trace_usage.get("executed_subqueries"),
                "open_calls": trace_usage.get(
                    "open_calls",
                    trace_usage.get("open_count"),
                ),
                "unique_evidence_count": trace_usage.get("unique_evidence_count"),
                "selected_evidence_count": trace_usage.get(
                    "selected_evidence_count",
                    trace_usage.get("active_evidence_count"),
                ),
                "pat_rag": (
                    trace.get("budgets") if isinstance(trace, dict) and isinstance(trace.get("budgets"), dict) else None
                ),
                "prim_rag": (
                    trace.get("budgets")
                    if isinstance(trace, dict)
                    and trace.get("schema_version") in {"5.0", "8.0"}
                    and isinstance(trace.get("budgets"), dict)
                    else None
                ),
                "da_prim": (
                    trace.get("budgets")
                    if isinstance(trace, dict)
                    and trace.get("schema_version") == "6.0"
                    and isinstance(trace.get("budgets"), dict)
                    else None
                ),
                "acm_prim": (
                    (
                        {
                            "agent": trace.get("budgets") or {},
                            "atlas_navigation": {
                                "opened_document_count": len(
                                    trace.get("atlas_document_open_records")
                                    or []
                                )
                            },
                        }
                        if trace.get("schema_version") in {"10.0", "11.0", "12.0", "13.0"}
                        else {
                            "agent": trace.get("budgets") or {},
                            "selector": {
                                "status": selector_audit.get("status"),
                                "elapsed_ms": selector_audit.get("elapsed_ms"),
                                "usage": selector_audit.get("usage") or {},
                                "cue_count": len(
                                    companion_selection.get("companion_cues")
                                    or []
                                ),
                            },
                        }
                    )
                    if isinstance(trace, dict)
                    and trace.get("schema_version") in {"7.0", "9.0", "10.0", "11.0", "12.0", "13.0"}
                    else None
                ),
            },
            "answer": history_data.get("answer", ""),
            "retrieval_status": history_data.get("retrieval_status", "unknown"),
            "retrieval_calls": history_data.get("retrieval_calls", []),
            "document_open_calls": history_data.get("document_open_calls", []),
            "all_tool_calls": history_data.get("all_tool_calls", []),
            "medication_review_trace": trace,
            "history": history_data.get("history", []),
            "event_count": len(events),
            "event_file": (
                str(self.event_path(job).relative_to(self.settings.output_dir))
                if self.settings.write_raw_events
                else None
            ),
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started_clock, 3),
            "error": error,
        }

    def write_result(self, record: dict[str, Any]) -> None:
        self.result_store(str(record["variant"])).append(record)
        self.append_completed_state(record)

    def run(self, selected_variants: list[str] | None = None) -> dict[str, int]:
        rows = load_dataset(self.settings.input_file)
        LOGGER.info("BATCH input loaded path=%s rows=%s", self.settings.input_file, len(rows))
        manifest = self.load_or_create_manifest(rows)
        batch_id = str(manifest["batch_id"])
        variants = selected_variants or list(self.settings.variants)
        unknown = set(variants) - set(self.settings.variants)
        if unknown:
            raise BatchConfigError(f"未配置实验类型：{sorted(unknown)}")

        completed = self.existing_successes()
        selected_completed = {job_key for job_key in completed if job_key.partition(":")[0] in variants}
        latest = self.latest_attempts()
        jobs: list[tuple[Job, dict[str, Any] | None]] = []
        for variant_name in variants:
            variant = self.settings.variants[variant_name]
            for row_index, row in enumerate(rows):
                job_key = f"{variant_name}:{row_index:06d}"
                if job_key in completed:
                    continue
                state_record = latest.get(job_key)
                if state_record and state_record.get("event") == "submitted":
                    attempt = int(state_record.get("attempt", 1))
                    job = self.build_job(batch_id, row_index, row, variant, attempt)
                    jobs.append((job, state_record))
                    continue
                attempt = int(state_record.get("attempt", 0)) + 1 if state_record else 1
                if attempt > self.settings.max_attempts:
                    continue
                jobs.append((self.build_job(batch_id, row_index, row, variant, attempt), None))

        counts = {
            "scheduled": len(jobs),
            "success": 0,
            "incomplete": 0,
            "failed": 0,
            "skipped": len(selected_completed),
        }
        LOGGER.info(
            "BATCH schedule batch_id=%s variants=%s scheduled=%s skipped=%s concurrency=%s",
            batch_id,
            ",".join(variants),
            counts["scheduled"],
            counts["skipped"],
            self.settings.concurrency,
        )
        if not jobs:
            LOGGER.info("BATCH no pending jobs; nothing to submit")
            return counts

        batch_started_clock = time.monotonic()
        with ThreadPoolExecutor(max_workers=self.settings.concurrency) as executor:
            futures = {executor.submit(self.run_one, job, submission): job for job, submission in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - defensive boundary
                    record = self.build_record(
                        job,
                        utc_now(),
                        time.monotonic(),
                        None,
                        None,
                        None,
                        {
                            "answer": "",
                            "retrieval_status": "unknown",
                            "retrieval_calls": [],
                            "document_open_calls": [],
                            "all_tool_calls": [],
                            "medication_review_trace": None,
                            "history": [],
                        },
                        [],
                        result_status="failed",
                        error=build_error(exc),
                    )
                self.write_result(record)
                counts[str(record["result_status"])] += 1
                completed_count = counts["success"] + counts["incomplete"] + counts["failed"]
                batch_elapsed = time.monotonic() - batch_started_clock
                average_seconds = batch_elapsed / completed_count
                eta_seconds = average_seconds * (counts["scheduled"] - completed_count)
                LOGGER.info(
                    "BATCH result persisted job_key=%s result_status=%s progress=%s/%s "
                    "average_seconds=%.1f eta_seconds=%.1f",
                    record["job_key"],
                    record["result_status"],
                    completed_count,
                    counts["scheduled"],
                    average_seconds,
                    eta_seconds,
                )
                medication_trace = record.get("medication_review_trace") or {}
                relation_query_count = len(
                    medication_trace.get("search_records")
                    or medication_trace.get("query_records")
                    or medication_trace.get("retrieval_records")
                    or []
                )
                print(
                    f"[{record['result_status']}] {record['job_key']} "
                    f"progress={completed_count}/{counts['scheduled']} "
                    f"answer_chars={len(record.get('answer') or '')} "
                    f"retrieval_calls={len(record.get('retrieval_calls') or [])} "
                    f"relation_queries={relation_query_count} "
                    f"duration={record.get('duration_seconds')}s "
                    f"thread={record.get('thread_id')} run={record.get('run_id')}",
                    flush=True,
                )
        return counts


def parse_variants(value: str | None) -> list[str] | None:
    if not value:
        return None
    variants = [item.strip() for item in value.split(",") if item.strip()]
    if not variants:
        raise BatchConfigError("--variants 不能为空")
    return variants


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量调用 Yuxi 0.6.3 Agent，并导出检索结果与回答")
    parser.add_argument("--config", required=True, type=Path, help="本地 JSON 配置文件")
    parser.add_argument(
        "--variants",
        default=None,
        help="只运行指定实验类型，逗号分隔，例如 vector,lightrag；默认运行配置中的全部类型",
    )
    parser.add_argument("--preflight-only", action="store_true", help="只检查远程配置并写入 manifest，不提交问题")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别，默认 INFO；排查 SSE/HTTP 细节时使用 DEBUG",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="可选的 UTF-8 日志文件路径；日志仍会输出到控制台",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        log_file = args.log_file.expanduser().resolve() if args.log_file else None
        configure_logging(args.log_level, log_file)
        LOGGER.info("START batch script config=%s", config_path)
        settings = load_settings(config_path)
        LOGGER.info(
            "CONFIG loaded base_url=%s auth_mode=%s agent_id=%s input=%s output=%s variants=%s concurrency=%s",
            settings.base_url,
            settings.auth_mode,
            settings.agent_id,
            settings.input_file,
            settings.output_dir,
            ",".join(settings.variants),
            settings.concurrency,
        )
        if settings.concurrency == 1:
            LOGGER.info("EXECUTION serial mode enabled; suitable for a CPU-hosted embedding model")
        else:
            LOGGER.warning(
                "EXECUTION concurrency=%s; concurrent query_kb/search_evidence "
                "calls may saturate a CPU-hosted embedding model. "
                "Start with concurrency=1 and increase only after observing stable tool-result latency.",
                settings.concurrency,
            )
        auth_token = load_auth_token(settings)
        runner = BatchRunner(settings, auth_token)
        rows = load_dataset(settings.input_file)
        LOGGER.info("INPUT validated path=%s rows=%s", settings.input_file, len(rows))
        manifest = runner.load_or_create_manifest(rows)
        if args.preflight_only:
            LOGGER.info("PREFLIGHT_ONLY complete; no question will be submitted")
            print(f"预检查完成：{runner._manifest_path}")
            print(json.dumps({"batch_id": manifest["batch_id"], "input_count": len(rows)}, ensure_ascii=False))
            return 0

        counts = runner.run(parse_variants(args.variants))
        LOGGER.info("END batch script counts=%s", counts)
        print(json.dumps(counts, ensure_ascii=False))
        return 0 if counts["failed"] == 0 and counts["incomplete"] == 0 else 2
    except (BatchConfigError, YuxiError) as exc:
        LOGGER.error("STOP batch script error_type=%s error=%s", type(exc).__name__, exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
