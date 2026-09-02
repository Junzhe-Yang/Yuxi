from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from yuxi.agents.buildin.medication_review_lite.anchor_extraction import (
    _invoke_model,
    _parse_json_object,
    merge_usage,
    message_text,
    response_usage,
)
from yuxi.agents.buildin.medication_review_prim.memory import (
    anchors_from_state,
    investigations_from_state,
    modifiers_from_state,
    queries_from_state,
)
from yuxi.utils.datetime_utils import utc_isoformat

from .corpus_atlas.models import CorpusAtlas
from .models import (
    CompanionCue,
    CompanionSelection,
    CompanionSelectorAudit,
    SelectorTrigger,
)

SELECTOR_PROMPT_VERSION = "acm-companion-selector-v5"
SELECTOR_INPUT_CONTRACT_VERSION = "first-successful-query-investigation-v3"
MAX_COMPANION_CUES = 6

SELECTOR_SYSTEM_PROMPT = """你负责从本地 Corpus Atlas 中挑选当前调查尚未明显涉及、
但可能帮助完整审查当前治疗方案的少量语料线索。

规则：
1. 先阅读已有 InvestigationItem、QueryRecord 和已召回文档；
2. 不要重复 Agent 已经调查的主要问题；
3. 每条线索只能引用输入中真实存在的 atlas cue ID；
4. 只提出待调查问题，不得给出临床结论；
5. 不要机械枚举全部药物×疾病、药物×药物或方案×患者事实组合；
6. 优先关注可能改变患者特异判断、跨药物风险、方案遗漏、疗程、监测或替代方案的线索；
7. 没有明显新线索时返回空列表；
8. 最多返回 6 条。"""

SELECTOR_USER_PROMPT_TEMPLATE = """请根据当前调查状态，从 Atlas 中选择当前尚未明显涉及的少量线索。只能引用输入中存在的 atlas cue ID。

选择器输入：
{input_json}"""

SELECTOR_REPAIR_PROMPT_TEMPLATE = """上一次输出未通过 JSON、Schema 或 ID 校验。
只修复格式和字段，不得增加输入中不存在的病例 ID 或 Atlas cue ID。
最多返回 6 条；没有合法新线索时返回空列表。

校验错误：
{error}

上一次输出：
{raw_output}

原任务：
{user_prompt}"""


class SelectorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompanionCueDraft(SelectorModel):
    question_hint: str = Field(min_length=1)
    linked_plan_ids: list[str] = Field(default_factory=list)
    linked_modifier_ids: list[str] = Field(default_factory=list)
    atlas_cue_ids: list[str] = Field(min_length=1)
    novelty_explanation: str = Field(min_length=1)


class CompanionSelectorEnvelope(SelectorModel):
    cues: list[CompanionCueDraft] = Field(default_factory=list)


def selector_prompt_hash() -> str:
    contract = _stable_json(
        {
            "prompt_version": SELECTOR_PROMPT_VERSION,
            "input_contract_version": SELECTOR_INPUT_CONTRACT_VERSION,
            "max_companion_cues": MAX_COMPANION_CUES,
            "system_prompt": SELECTOR_SYSTEM_PROMPT,
            "user_prompt_template": SELECTOR_USER_PROMPT_TEMPLATE,
            "repair_prompt_template": SELECTOR_REPAIR_PROMPT_TEMPLATE,
            "schema_instructions": _schema_instructions(),
        }
    )
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _schema_instructions() -> str:
    schema = json.dumps(
        CompanionSelectorEnvelope.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "\n\n只返回一个符合下方 JSON Schema 的 JSON 对象。"
        "不要使用 Markdown 代码围栏，不要附加解释或其它文本。"
        "字段名和必填字段必须严格遵守 Schema。\n\n"
        f"目标 JSON Schema：\n{schema}"
    )


def _retrieved_document_ids(
    state: dict[str, Any],
    query_ids: set[str],
) -> list[str]:
    evidence = state.get("evidence_store")
    if not isinstance(evidence, dict):
        return []
    values: list[str] = []
    for raw in evidence.values():
        item = raw if hasattr(raw, "occurrences") else None
        file_id = getattr(item, "file_id", None) if item is not None else None
        occurrences = getattr(item, "occurrences", []) if item is not None else []
        if isinstance(raw, dict):
            file_id = raw.get("file_id")
            occurrences = raw.get("occurrences") or []
        occurrence_query_ids = {
            str(
                value.get("record_id")
                if isinstance(value, dict)
                else getattr(value, "record_id", "")
            )
            for value in occurrences
        }
        if file_id and occurrence_query_ids & query_ids:
            values.append(str(file_id))
    return list(dict.fromkeys(values))


def _first_query_batch_ids(state: dict[str, Any]) -> list[str]:
    queries = queries_from_state(state)
    eligible = [
        value
        for value in queries
        if value.status in {"success", "success_empty"}
    ]
    if not eligible:
        return []
    for message in state.get("messages") or []:
        tool_calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        if not isinstance(tool_calls, list):
            continue
        call_ids = {
            str(call.get("id") or call.get("tool_call_id") or "")
            for call in tool_calls
            if isinstance(call, dict)
        }
        matched = [
            value.query_id
            for value in queries
            if value.tool_call_id in call_ids
        ]
        matched_ids = set(matched)
        if any(value.query_id in matched_ids for value in eligible):
            return matched
    return [eligible[0].query_id]


def selector_input(
    *,
    state: dict[str, Any],
    atlas: CorpusAtlas,
    trigger_type: SelectorTrigger,
    created_after_query_id: str | None,
) -> dict[str, Any]:
    anchors = anchors_from_state(state)
    modifiers = modifiers_from_state(state)
    queries = queries_from_state(state)
    investigations = investigations_from_state(state)
    observed_query_ids = set(_first_query_batch_ids(state))
    if trigger_type == "after_query" and created_after_query_id:
        observed_query_ids.add(created_after_query_id)
    observed_queries = [
        value for value in queries if value.query_id in observed_query_ids
    ]
    observed_investigations = [
        value
        for value in investigations
        if any(query_id in observed_query_ids for query_id in value.query_ids)
    ]
    observed_evidence_ids = {
        evidence_id
        for value in observed_queries
        for evidence_id in value.evidence_ids
    }
    return {
        "trigger_type": trigger_type,
        "created_after_query_id": created_after_query_id,
        "raw_case": str(state.get("raw_case_text") or ""),
        "plan_anchors": [value.model_dump(mode="json") for value in anchors],
        "patient_modifiers": [
            value.model_dump(mode="json") for value in modifiers
        ],
        "investigations": [
            {
                **value.model_dump(mode="json"),
                "query_ids": [
                    query_id
                    for query_id in value.query_ids
                    if query_id in observed_query_ids
                ],
                "candidate_evidence_ids": [
                    evidence_id
                    for evidence_id in value.candidate_evidence_ids
                    if evidence_id in observed_evidence_ids
                ],
            }
            for value in observed_investigations
        ],
        "query_records": [
            value.model_dump(mode="json") for value in observed_queries
        ],
        "retrieved_document_ids": _retrieved_document_ids(
            state,
            observed_query_ids,
        ),
        "atlas_document_cards": atlas.selector_view(),
    }


def _validate_envelope(
    *,
    envelope: CompanionSelectorEnvelope,
    state: dict[str, Any],
    atlas: CorpusAtlas,
) -> tuple[list[CompanionCue], list[dict[str, Any]]]:
    known_plan_ids = {value.element_id for value in anchors_from_state(state)}
    known_modifier_ids = {
        value.modifier_id for value in modifiers_from_state(state)
    }
    cue_index = atlas.cue_index()
    retained: list[CompanionCue] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, draft in enumerate(envelope.cues, start=1):
        atlas_cue_ids = list(dict.fromkeys(draft.atlas_cue_ids))
        unknown_atlas_ids = [
            value for value in atlas_cue_ids if value not in cue_index
        ]
        if unknown_atlas_ids:
            raise ValueError(
                "CompanionCue 引用了不存在的 Atlas cue ID："
                + "、".join(unknown_atlas_ids)
            )
        key = tuple(sorted(atlas_cue_ids))
        if key in seen:
            dropped.append(
                {
                    "item_index": index,
                    "reason": "duplicate_atlas_cue_set",
                    "ids": atlas_cue_ids,
                }
            )
            continue
        seen.add(key)
        if len(retained) >= MAX_COMPANION_CUES:
            dropped.append(
                {
                    "item_index": index,
                    "reason": "soft_budget_exceeded",
                    "ids": atlas_cue_ids,
                }
            )
            continue
        invalid_plan_ids = [
            value
            for value in draft.linked_plan_ids
            if value not in known_plan_ids
        ]
        invalid_modifier_ids = [
            value
            for value in draft.linked_modifier_ids
            if value not in known_modifier_ids
        ]
        if invalid_plan_ids or invalid_modifier_ids:
            dropped.append(
                {
                    "item_index": index,
                    "reason": "unknown_case_ids_removed",
                    "plan_ids": invalid_plan_ids,
                    "modifier_ids": invalid_modifier_ids,
                }
            )
        suggested_docs = list(
            dict.fromkeys(cue_index[value][0].doc_id for value in atlas_cue_ids)
        )
        retained.append(
            CompanionCue(
                companion_id=f"AC{len(retained) + 1:02d}",
                question_hint=draft.question_hint.strip(),
                linked_plan_ids=[
                    value
                    for value in dict.fromkeys(draft.linked_plan_ids)
                    if value in known_plan_ids
                ],
                linked_modifier_ids=[
                    value
                    for value in dict.fromkeys(draft.linked_modifier_ids)
                    if value in known_modifier_ids
                ],
                atlas_cue_ids=atlas_cue_ids,
                suggested_doc_ids=suggested_docs,
                novelty_explanation=draft.novelty_explanation.strip(),
            )
        )
    return retained, dropped


async def select_companion_cues(
    *,
    model: Any,
    state: dict[str, Any],
    atlas: CorpusAtlas,
    trigger_type: SelectorTrigger,
    created_after_query_id: str | None,
    technical_retry_limit: int,
) -> CompanionSelection:
    started_at = utc_isoformat()
    started = time.monotonic()
    raw_output: str | None = None
    repair_raw_output: str | None = None
    errors: list[str] = []
    usage: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    payload = selector_input(
        state=state,
        atlas=atlas,
        trigger_type=trigger_type,
        created_after_query_id=created_after_query_id,
    )
    input_json = _stable_json(payload)
    observed_query_ids = [
        str(value.get("query_id") or "")
        for value in payload["query_records"]
        if str(value.get("query_id") or "")
    ]
    observed_investigation_ids = [
        str(value.get("investigation_id") or "")
        for value in payload["investigations"]
        if str(value.get("investigation_id") or "")
    ]
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    user_prompt = SELECTOR_USER_PROMPT_TEMPLATE.format(
        input_json=input_json,
    )

    try:
        response = await _invoke_model(
            model=model,
            messages=[
                SystemMessage(content=SELECTOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt + _schema_instructions()),
            ],
            technical_retry_limit=technical_retry_limit,
        )
        raw_output = message_text(response)
        usage = merge_usage(usage, response_usage(response))
        envelope = CompanionSelectorEnvelope.model_validate(
            _parse_json_object(raw_output)
        )
        cues, dropped = _validate_envelope(
            envelope=envelope,
            state=state,
            atlas=atlas,
        )
        status = "success" if cues else "empty"
    except Exception as exc:  # noqa: BLE001 - provider/schema boundary
        errors.append(f"{type(exc).__name__}: {exc}")
        repair_prompt = SELECTOR_REPAIR_PROMPT_TEMPLATE.format(
            error=errors[-1],
            raw_output=raw_output or "<无输出>",
            user_prompt=user_prompt,
        )
        try:
            response = await _invoke_model(
                model=model,
                messages=[
                    SystemMessage(content=SELECTOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=repair_prompt + _schema_instructions()
                    ),
                ],
                technical_retry_limit=technical_retry_limit,
            )
            repair_raw_output = message_text(response)
            usage = merge_usage(usage, response_usage(response))
            envelope = CompanionSelectorEnvelope.model_validate(
                _parse_json_object(repair_raw_output)
            )
            cues, dropped = _validate_envelope(
                envelope=envelope,
                state=state,
                atlas=atlas,
            )
            status = "repaired"
        except Exception as repair_exc:  # noqa: BLE001
            errors.append(f"{type(repair_exc).__name__}: {repair_exc}")
            audit = CompanionSelectorAudit(
                status="failed",
                trigger_type=trigger_type,
                created_after_query_id=created_after_query_id,
                observed_query_ids=[
                    *observed_query_ids
                ],
                observed_investigation_ids=[
                    *observed_investigation_ids
                ],
                retrieved_document_ids=payload["retrieved_document_ids"],
                selector_input_hash=input_hash,
                atlas_snapshot_hash=atlas.snapshot_hash,
                started_at=started_at,
                elapsed_ms=round((time.monotonic() - started) * 1000),
                raw_output=raw_output,
                repair_raw_output=repair_raw_output,
                validation_errors=errors,
                usage=usage,
                error_type=type(repair_exc).__name__,
                error_message=str(repair_exc),
            )
            return CompanionSelection(
                trigger_type=trigger_type,
                created_after_query_id=created_after_query_id,
                observed_query_ids=audit.observed_query_ids,
                companion_cues=[],
                selector_audit=audit,
            )

    audit = CompanionSelectorAudit(
        status=status,
        trigger_type=trigger_type,
        created_after_query_id=created_after_query_id,
        observed_query_ids=[
            *observed_query_ids
        ],
        observed_investigation_ids=[
            *observed_investigation_ids
        ],
        retrieved_document_ids=payload["retrieved_document_ids"],
        selector_input_hash=input_hash,
        atlas_snapshot_hash=atlas.snapshot_hash,
        started_at=started_at,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        raw_output=raw_output,
        repair_raw_output=repair_raw_output,
        validation_errors=errors,
        dropped_items=dropped,
        usage=usage,
    )
    return CompanionSelection(
        trigger_type=trigger_type,
        created_after_query_id=created_after_query_id,
        observed_query_ids=audit.observed_query_ids,
        companion_cues=cues,
        selector_audit=audit,
    )
