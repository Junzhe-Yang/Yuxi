from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from yuxi.agents.buildin.medication_review_lite.models import (
    EvidenceItem,
    OpenRecord,
)
from yuxi.agents.buildin.medication_review_prim.harness import (
    PreFinalInterruption,
    ReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_prim.memory import (
    anchors_from_state,
    build_case_node_memory,
    investigations_from_state,
    modifiers_from_state,
    queries_from_state,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    ExperimentProfile,
    MedicationReviewPrimTrace,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    resolve_milvus_retriever,
)

from .adaptive import (
    ADAPTIVE_AUDIT_DIMENSIONS,
    ADAPTIVE_AUDIT_INSTRUCTIONS,
    ADAPTIVE_COVERAGE_INSTRUCTIONS,
    ADAPTIVE_QUERY_INSTRUCTIONS,
    ADAPTIVE_REVIEW_INSTRUCTIONS,
    adaptive_agenda_from_state,
    adaptive_checkpoint_records_from_state,
    adaptive_gap_assessments_from_state,
    adaptive_meta_from_state,
    adaptive_probe_records_from_state,
    adaptive_recoveries_from_state,
    build_adaptive_coverage_report,
    build_adaptive_investigation_memory,
)
from .context import (
    V7_MINIMUM_SEARCH_CALLS,
    MedicationReviewAcmPrimContext,
    adaptive_coverage_enabled,
    v7_required_investigation_count,
    validate_acm_context,
)
from .corpus_atlas import AtlasStore, CorpusAtlasBuilder
from .corpus_atlas.models import CorpusAtlas
from .experiment import (
    agenda_from_state,
    build_v7_contract_memory,
    build_v7_contract_report,
    checkpoint_records_from_state,
    configure_v7_retrieval,
    v7_contract_enabled,
    v7_contract_fingerprint,
    v7_trace_enabled,
)
from .memory import (
    ADAPTIVE_ATLAS_NAVIGATION_PROMPT_HASH,
    ADAPTIVE_ATLAS_NAVIGATION_PROMPT_VERSION,
    ATLAS_NAVIGATION_PROMPT_HASH,
    ATLAS_NAVIGATION_PROMPT_VERSION,
    build_adaptive_atlas_document_memory,
    build_atlas_document_memory,
)
from .models import (
    AdaptiveCheckpointRecord,
    AtlasDocumentOpenRecord,
    MedicationReviewAcmAdaptiveTrace,
    MedicationReviewAcmPrimState,
    MedicationReviewAcmPrimTrace,
    MedicationReviewAcmV7Trace,
    V7CheckpointRecord,
)
from .tools import search_review_kb_adaptive, search_review_kb_v7

V7_CONTRACT_PROMPT_VERSION = "acm-v7-effort-contract-v1"
V7_CONTRACT_PROMPT_HASH = hashlib.sha256(
    b"minimum-six-successful-searches; fixed-k-agenda; "
    b"breadth-first-initial-then-complementary; explicit-close; "
    b"continue-adaptively-up-to-runtime-limit; final-answer-never-blocked"
).hexdigest()
ADAPTIVE_CONTRACT_PROMPT_VERSION = "acm-adaptive-two-track-coverage-v4-query-focus"
ADAPTIVE_CONTRACT_PROMPT_HASH = hashlib.sha256(
    "\n".join(
        (
            ADAPTIVE_COVERAGE_INSTRUCTIONS,
            ADAPTIVE_QUERY_INSTRUCTIONS,
            ADAPTIVE_REVIEW_INSTRUCTIONS,
            ADAPTIVE_AUDIT_INSTRUCTIONS,
            *ADAPTIVE_AUDIT_DIMENSIONS,
        )
    ).encode("utf-8")
).hexdigest()
ADAPTIVE_CONTEXT_PROJECTION_VERSION = "acm-adaptive-context-projection-v2-focused-memory"
ADAPTIVE_CONTEXT_PROJECTION_HASH = hashlib.sha256(
    b"keep-unread-knowledge-results-and-one-consumed-overlap; "
    b"compact-older-results-to-auditable-receipts; "
    b"preserve-selected-evidence-excerpts-verbatim; "
    b"retain-full-evidence-in-state-and-trace; "
    b"replace-base-investigation-memory-with-case-nodes; "
    b"phase-specific-active-obligation-view; "
    b"restore-full-obligations-before-close-and-audit"
).hexdigest()

_KNOWLEDGE_TOOL_NAMES = frozenset(
    {
        "search_review_kb",
        "open_review_evidence",
        "open_atlas_document",
    }
)
_COMPACTED_KNOWLEDGE_PREFIX = "[ACM 历史知识结果已冷却]"


def _build_selected_evidence_memory(state: dict[str, Any]) -> str:
    selected_ids = [
        evidence_id
        for investigation in investigations_from_state(state)
        for evidence_id in investigation.selected_evidence_ids
    ]
    selected_ids.extend(
        evidence_id
        for meta in adaptive_meta_from_state(state)
        for support in meta.obligation_supports
        for evidence_id in support.evidence_ids
    )
    selected_ids = list(dict.fromkeys(selected_ids))
    if not selected_ids:
        return ""

    evidence_store: dict[str, EvidenceItem] = {}
    for evidence_id, raw_item in (state.get("evidence_store") or {}).items():
        try:
            evidence_store[str(evidence_id)] = (
                raw_item if isinstance(raw_item, EvidenceItem) else EvidenceItem.model_validate(raw_item)
            )
        except Exception:  # noqa: BLE001
            continue

    blocks: list[str] = []
    for evidence_id in selected_ids:
        evidence = evidence_store.get(evidence_id)
        if evidence is None:
            continue
        original_text = evidence.occurrences[-1].shown_excerpt if evidence.occurrences else evidence.raw_text
        source = evidence.source_document or "未知来源"
        location = ", ".join(
            value
            for value in (
                f"file_id={evidence.file_id}" if evidence.file_id else "",
                f"chunk_id={evidence.chunk_id}" if evidence.chunk_id is not None else "",
                f"chunk_index={evidence.chunk_index}" if evidence.chunk_index is not None else "",
            )
            if value
        )
        metadata = f"来源={source}" + (f"；{location}" if location else "")
        blocks.append(f"[{evidence_id}]\n{metadata}\n采纳时原文：\n{original_text}")
    if not blocks:
        return ""
    return (
        "【已采纳 Evidence 原文】\n"
        "以下片段逐字来自已采纳 Evidence；不得用历史摘要替代。"
        "完整 raw_text 仍保存在 Evidence Store 与 Trace。\n\n" + "\n\n".join(blocks)
    )


def _knowledge_result_receipt(
    message: ToolMessage,
    state: dict[str, Any],
) -> str:
    lines = [
        _COMPACTED_KNOWLEDGE_PREFIX,
        f"tool={message.name or 'unknown'}; tool_call_id={message.tool_call_id}",
    ]
    if message.name == "search_review_kb":
        record = next(
            (value for value in queries_from_state(state) if value.tool_call_id == message.tool_call_id),
            None,
        )
        if record is not None:
            lines.extend(
                [
                    f"query_id={record.query_id}; "
                    f"investigation_id={record.investigation_id or 'none'}; "
                    f"status={record.status}",
                    f"query={record.query_text}",
                    "evidence_ids=" + (",".join(record.evidence_ids) or "none"),
                ]
            )
    elif message.name == "open_review_evidence":
        for raw_record in state.get("open_records") or []:
            try:
                record = raw_record if isinstance(raw_record, OpenRecord) else OpenRecord.model_validate(raw_record)
            except Exception:  # noqa: BLE001
                continue
            if record.tool_call_id != message.tool_call_id:
                continue
            lines.extend(
                [
                    f"record_id={record.record_id}; "
                    f"investigation_id={record.investigation_id or 'none'}; "
                    f"status={record.status}",
                    f"parent_evidence_id={record.parent_evidence_id}",
                    "evidence_ids=" + (",".join(record.evidence_ids) or "none"),
                ]
            )
            break
    elif message.name == "open_atlas_document":
        for raw_record in state.get("atlas_document_open_records") or []:
            try:
                record = (
                    raw_record
                    if isinstance(raw_record, AtlasDocumentOpenRecord)
                    else AtlasDocumentOpenRecord.model_validate(raw_record)
                )
            except Exception:  # noqa: BLE001
                continue
            if record.tool_call_id != message.tool_call_id:
                continue
            lines.extend(
                [
                    f"record_id={record.record_id}; doc_id={record.doc_id}",
                    f"title={record.title}; cue_ids={','.join(record.cue_ids) or 'none'}",
                ]
            )
            break
    lines.append(
        "原始工具结果未从运行状态删除；完整 Evidence raw_text 仍保存在 "
        "Evidence Store 与 Trace；已采纳 Evidence 原文由系统记忆逐字保留。"
    )
    return "\n".join(lines)


class AcmReviewHarnessMiddleware(ReviewHarnessMiddleware):
    state_schema = MedicationReviewAcmPrimState

    async def abefore_agent(
        self,
        state: MedicationReviewAcmPrimState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        context: MedicationReviewAcmPrimContext = runtime.context
        validate_acm_context(context)
        configure_v7_retrieval(context)
        if adaptive_coverage_enabled(context):
            setattr(
                context,
                "_prim_retrieval_diagnostic_state_key",
                "adaptive_retrieval_records",
            )
        selection = await resolve_milvus_retriever(context)
        store = AtlasStore()
        atlas = store.load_current(selection.db_id)
        builder = CorpusAtlasBuilder(
            model_name=atlas.builder_model,
            store=store,
        )
        await builder.validate_runtime(atlas)
        setattr(context, "_acm_prim_atlas", atlas)
        atlas_file_ids = {value.doc_id for value in atlas.document_cards} | {
            value.file_id for value in atlas.source_records
        }
        cached_file_ids = getattr(context, "_prim_allowed_file_ids", None)
        setattr(
            context,
            "_prim_allowed_file_ids",
            atlas_file_ids | ({str(value) for value in cached_file_ids} if isinstance(cached_file_ids, set) else set()),
        )
        return await super().abefore_agent(state, runtime)

    async def augment_initial_state(
        self,
        *,
        state: dict[str, Any],
        update: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any]:
        del update
        context: MedicationReviewAcmPrimContext = runtime.context
        atlas = self._atlas(context)
        previous = dict(state.get("atlas_snapshot") or {})
        if previous and previous.get("snapshot_hash") != atlas.snapshot_hash:
            raise ValueError("同一 ACM-PRIM thread 的 Atlas 快照已变化，请新建会话")
        protocol_snapshot = {
            "protocol": context.acm_protocol,
            "experiment_arm": context.v7_experiment_arm,
            "retrieval_depth": context.v7_retrieval_depth,
            "maximum_search_calls": context.max_search_calls,
        }
        previous_protocol = dict(state.get("adaptive_protocol_snapshot") or {})
        if previous_protocol and previous_protocol != protocol_snapshot:
            raise ValueError("同一 ACM-PRIM thread 不能切换调查协议、检索深度或预算，请新建会话")
        experiment_snapshot = {
            "experiment_arm": context.v7_experiment_arm,
            "retrieval_depth": context.v7_retrieval_depth,
            "minimum_search_calls": (V7_MINIMUM_SEARCH_CALLS if v7_contract_enabled(context) else 0),
            "maximum_search_calls": context.max_search_calls,
        }
        previous_experiment = dict(state.get("v7_experiment_snapshot") or {})
        if previous_experiment and previous_experiment != experiment_snapshot:
            raise ValueError("同一 ACM-PRIM thread 不能切换 V7 实验臂、检索深度或预算，请新建会话")
        result: dict[str, Any] = {
            "atlas_snapshot": previous or self._atlas_snapshot(atlas),
            "adaptive_protocol_snapshot": (previous_protocol or protocol_snapshot),
        }
        if not previous:
            result["atlas_document_open_records"] = []
        if adaptive_coverage_enabled(context):
            if not previous_protocol:
                result.update(
                    {
                        "adaptive_investigation_meta": [],
                        "adaptive_probe_records": [],
                        "adaptive_recovery_requirements": [],
                        "adaptive_gap_assessments": [],
                        "adaptive_checkpoint_records": [],
                        "adaptive_retrieval_records": [],
                    }
                )
        elif v7_trace_enabled(context) and not previous_experiment:
            result.update(
                {
                    "v7_experiment_snapshot": experiment_snapshot,
                    "v7_probe_records": [],
                    "v7_retrieval_records": [],
                    "v7_checkpoint_records": [],
                }
            )
        return result

    async def abefore_model(
        self,
        state: MedicationReviewAcmPrimState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del state, runtime
        return None

    def project_model_messages(
        self,
        *,
        messages: list[Any],
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
    ) -> list[Any]:
        if not adaptive_coverage_enabled(context):
            return messages

        knowledge_result_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, ToolMessage) and message.name in _KNOWLEDGE_TOOL_NAMES
        ]
        if not knowledge_result_indexes:
            return messages

        last_ai_index = max(
            (index for index, message in enumerate(messages) if isinstance(message, AIMessage)),
            default=-1,
        )
        consumed_indexes = [index for index in knowledge_result_indexes if index < last_ai_index]
        full_result_indexes = {index for index in knowledge_result_indexes if index > last_ai_index}
        if consumed_indexes:
            # Keep one already-read result as overlap for the next reasoning turn.
            full_result_indexes.add(consumed_indexes[-1])
        compacted_indexes = [index for index in knowledge_result_indexes if index not in full_result_indexes]
        if not compacted_indexes:
            return messages

        projected = list(messages)
        for index in compacted_indexes:
            message = projected[index]
            projected[index] = message.model_copy(update={"content": _knowledge_result_receipt(message, state)})
        return projected

    async def prepare_candidate_state(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
        candidate_body: str,
    ) -> dict[str, Any]:
        del candidate_body
        if adaptive_coverage_enabled(context):
            report = build_adaptive_coverage_report(state, context)
            if report.status != "incomplete":
                return {}
            return {
                "warnings": list(
                    dict.fromkeys(
                        [
                            *(state.get("warnings") or []),
                            "Adaptive coverage contract incomplete at final "
                            "answer: " + ",".join(report.incomplete_reasons),
                        ]
                    )
                )
            }
        if not v7_contract_enabled(context):
            return {}
        report = build_v7_contract_report(state, context)
        if report.status != "incomplete":
            return {}
        return {
            "warnings": list(
                dict.fromkeys(
                    [
                        *(state.get("warnings") or []),
                        "V7 effort contract incomplete at final answer: " + ",".join(report.incomplete_reasons),
                    ]
                )
            )
        }

    def project_search_tool(
        self,
        effective_profile,
        context: MedicationReviewAcmPrimContext | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        if context is not None and adaptive_coverage_enabled(context):
            return search_review_kb_adaptive
        if context is not None and v7_required_investigation_count(context.v7_experiment_arm):
            return search_review_kb_v7
        return super().project_search_tool(
            effective_profile,
            context,
            state,
        )

    def tool_is_visible(
        self,
        *,
        tool_name: str,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
    ) -> bool:
        if tool_name in {
            "v7_effort_checkpoint",
            "adaptive_coverage_checkpoint",
        }:
            return False
        adaptive = adaptive_coverage_enabled(context)
        if (
            tool_name
            in {
                "extend_investigation_agenda",
                "submit_coverage_gap_assessment",
            }
            and not adaptive
        ):
            return False
        if adaptive:
            agenda = adaptive_agenda_from_state(state)
            if tool_name == "set_investigation_agenda":
                return agenda is None
            if tool_name in {"search_review_kb", "update_investigation"}:
                return agenda is not None
            if tool_name == "extend_investigation_agenda":
                return agenda is not None
            if tool_name == "submit_coverage_gap_assessment":
                if agenda is None:
                    return False
                report = build_adaptive_coverage_report(state, context)
                return not (
                    report.unprobed_investigation_ids
                    or report.missing_current_regimen_review_plan_ids
                    or report.missing_improvement_plan_ids
                    or report.unprobed_evidence_obligations
                    or report.open_investigation_ids
                    or report.unsupported_evidence_obligations
                    or report.pending_recovery_ids
                )
        if tool_name == "set_investigation_agenda":
            return bool(v7_required_investigation_count(context.v7_experiment_arm) and agenda_from_state(state) is None)
        return True

    def investigation_tools_enabled(
        self,
        *,
        effective_profile,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
    ) -> bool:
        return bool(
            adaptive_coverage_enabled(context)
            or v7_required_investigation_count(context.v7_experiment_arm)
            or super().investigation_tools_enabled(
                effective_profile=effective_profile,
                state=state,
                context=context,
            )
        )

    async def prepare_pre_final_interruption(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
        candidate_body: str,
    ) -> PreFinalInterruption | None:
        if adaptive_coverage_enabled(context):
            report = build_adaptive_coverage_report(state, context)
            if report.status != "incomplete":
                return None
            if report.executed_search_calls >= context.max_search_calls:
                return None
            previous = adaptive_checkpoint_records_from_state(state)
            if previous and previous[-1].state_fingerprint == report.state_fingerprint:
                return None
            source = f"{state.get('review_run_id') or ''}\0{report.state_fingerprint}\0{candidate_body}"
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
            checkpoint_id = f"ACMCHECK-{digest}"
            record = AdaptiveCheckpointRecord(
                checkpoint_id=checkpoint_id,
                created_at=self._utc_now(),
                state_fingerprint=report.state_fingerprint,
                incomplete_reasons=report.incomplete_reasons,
                candidate_answer_hash=hashlib.sha256(candidate_body.encode("utf-8")).hexdigest(),
            )
            notice = (
                "你刚才尝试生成最终答案，但自适应调查覆盖尚未完成。"
                "不要复述这条系统提示；请按当前确定性缺口继续使用工具。\n\n"
                + build_adaptive_investigation_memory(state, report)
            )
            return PreFinalInterruption(
                tool_name="adaptive_coverage_checkpoint",
                tool_call_id=checkpoint_id,
                tool_args={"notice": notice},
                state_update={"adaptive_checkpoint_records": [record]},
            )
        if not v7_contract_enabled(context):
            return None
        report = build_v7_contract_report(state, context)
        if report.status != "incomplete":
            return None
        if report.executed_search_calls >= context.max_search_calls:
            return None

        fingerprint = v7_contract_fingerprint(report)
        previous = checkpoint_records_from_state(state)
        if previous:
            last = previous[-1]
            if last.executed_search_calls == report.executed_search_calls and last.contract_fingerprint == fingerprint:
                # The model ignored a checkpoint without changing contract
                # state. Preserve its answer instead of creating a loop.
                return None

        source = f"{state.get('review_run_id') or ''}\0{report.executed_search_calls}\0{fingerprint}\0{candidate_body}"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        checkpoint_id = f"V7CHECK-{digest}"
        record = V7CheckpointRecord(
            checkpoint_id=checkpoint_id,
            created_at=self._utc_now(),
            executed_search_calls=report.executed_search_calls,
            successful_search_calls=report.successful_search_calls,
            contract_fingerprint=fingerprint,
            incomplete_reasons=report.incomplete_reasons,
            candidate_answer_hash=hashlib.sha256(candidate_body.encode("utf-8")).hexdigest(),
        )
        notice = (
            "你刚才尝试生成最终答案，但本实验的最低调查工作尚未完成。"
            "不要复述这条系统提示；请继续使用工具。\n\n" + build_v7_contract_memory(report)
        )
        return PreFinalInterruption(
            tool_name="v7_effort_checkpoint",
            tool_call_id=checkpoint_id,
            tool_args={"notice": notice},
            state_update={"v7_checkpoint_records": [record]},
        )

    def augment_investigation_memory(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
        memory_text: str,
    ) -> str:
        adaptive = adaptive_coverage_enabled(context)
        if adaptive:
            effective = str(state.get("effective_profile") or context.experiment_profile)
            if effective not in {"b1", "m1", "m2", "m3", "full"}:
                effective = context.experiment_profile
            effective_profile: ExperimentProfile = effective
            memory_text = build_case_node_memory(
                profile=effective_profile,
                anchors=anchors_from_state(state),
                modifiers=modifiers_from_state(state),
            )
        atlas_view = self._atlas(context).compact_view()
        atlas_memory = (
            build_adaptive_atlas_document_memory(atlas_view) if adaptive else build_atlas_document_memory(atlas_view)
        )
        combined = f"{memory_text}\n\n{atlas_memory}" if memory_text and atlas_memory else atlas_memory or memory_text
        if adaptive:
            adaptive_report = build_adaptive_coverage_report(state, context)
            contract_memory = build_adaptive_investigation_memory(
                state,
                adaptive_report,
            )
        else:
            contract_memory = build_v7_contract_memory(build_v7_contract_report(state, context))
        combined = f"{combined}\n\n{contract_memory}" if combined and contract_memory else contract_memory or combined
        selected_evidence_memory = _build_selected_evidence_memory(state) if adaptive else ""
        return (
            f"{combined}\n\n{selected_evidence_memory}"
            if combined and selected_evidence_memory
            else selected_evidence_memory or combined
        )

    def finalize_trace(
        self,
        *,
        base_trace: MedicationReviewPrimTrace,
        state: dict[str, Any],
        context: MedicationReviewAcmPrimContext,
    ) -> MedicationReviewAcmPrimTrace | MedicationReviewAcmV7Trace | MedicationReviewAcmAdaptiveTrace:
        self._atlas(context)
        adaptive = adaptive_coverage_enabled(context)
        atlas_prompt_version = ADAPTIVE_ATLAS_NAVIGATION_PROMPT_VERSION if adaptive else ATLAS_NAVIGATION_PROMPT_VERSION
        atlas_prompt_hash = ADAPTIVE_ATLAS_NAVIGATION_PROMPT_HASH if adaptive else ATLAS_NAVIGATION_PROMPT_HASH
        payload = base_trace.model_dump(
            exclude={
                "schema_version",
                "method_family",
                "method_version",
                "prompt_versions",
                "prompt_hashes",
            }
        )
        common = {
            **payload,
            "prompt_versions": {
                **base_trace.prompt_versions,
                "atlas_navigation": atlas_prompt_version,
            },
            "prompt_hashes": {
                **base_trace.prompt_hashes,
                "atlas_navigation": atlas_prompt_hash,
            },
            "atlas_snapshot": dict(state.get("atlas_snapshot") or {}),
            "atlas_document_open_records": list(state.get("atlas_document_open_records") or []),
        }
        if adaptive:
            report = build_adaptive_coverage_report(state, context)
            adaptive_common = {
                **common,
                "prompt_versions": {
                    **common["prompt_versions"],
                    "adaptive_coverage": ADAPTIVE_CONTRACT_PROMPT_VERSION,
                    "context_projection": ADAPTIVE_CONTEXT_PROJECTION_VERSION,
                },
                "prompt_hashes": {
                    **common["prompt_hashes"],
                    "adaptive_coverage": ADAPTIVE_CONTRACT_PROMPT_HASH,
                    "context_projection": ADAPTIVE_CONTEXT_PROJECTION_HASH,
                },
                "budgets": {
                    **dict(base_trace.budgets),
                    "minimum_search_calls": 0,
                    "retrieval_fetch_k": report.fetch_k,
                    "agent_visible_k": report.visible_k,
                },
            }
            return MedicationReviewAcmAdaptiveTrace(
                **adaptive_common,
                method_version=("acm-prim-rag-v8-adaptive-two-track-evidence-v5-query-focus-shadow_top25-vector"),
                adaptive_coverage_report=report,
                investigation_agenda=adaptive_agenda_from_state(state),
                investigation_meta=adaptive_meta_from_state(state),
                probe_records=adaptive_probe_records_from_state(state),
                recovery_requirements=adaptive_recoveries_from_state(state),
                gap_assessments=adaptive_gap_assessments_from_state(state),
                retrieval_records=list(state.get("adaptive_retrieval_records") or []),
                checkpoint_records=adaptive_checkpoint_records_from_state(state),
            )
        if not v7_trace_enabled(context):
            return MedicationReviewAcmPrimTrace(
                **common,
                method_version=("acm-prim-rag-v3-atlas-navigation-vector-top10"),
            )

        report = build_v7_contract_report(state, context)
        common["budgets"] = {
            **dict(base_trace.budgets),
            "minimum_search_calls": report.minimum_search_calls,
            "retrieval_fetch_k": report.fetch_k,
            "agent_visible_k": report.visible_k,
        }
        v7_common = {
            **common,
            "prompt_versions": {
                **common["prompt_versions"],
                "v7_effort_contract": V7_CONTRACT_PROMPT_VERSION,
            },
            "prompt_hashes": {
                **common["prompt_hashes"],
                "v7_effort_contract": V7_CONTRACT_PROMPT_HASH,
            },
        }
        return MedicationReviewAcmV7Trace(
            **v7_common,
            method_version=(f"acm-prim-rag-v7-{context.v7_experiment_arm}-{context.v7_retrieval_depth}-vector"),
            experiment_arm=context.v7_experiment_arm,
            retrieval_depth=context.v7_retrieval_depth,
            contract_report=report,
            investigation_agenda=agenda_from_state(state),
            probe_records=list(state.get("v7_probe_records") or []),
            retrieval_records=list(state.get("v7_retrieval_records") or []),
            checkpoint_records=checkpoint_records_from_state(state),
        )

    @staticmethod
    def _utc_now() -> str:
        from yuxi.utils.datetime_utils import utc_isoformat

        return utc_isoformat()

    @staticmethod
    def _atlas(context: MedicationReviewAcmPrimContext) -> CorpusAtlas:
        atlas = getattr(context, "_acm_prim_atlas", None)
        if not isinstance(atlas, CorpusAtlas):
            raise RuntimeError("ACM-PRIM Context 缺少已校验的 Corpus Atlas")
        return atlas

    @staticmethod
    def _atlas_snapshot(atlas: CorpusAtlas) -> dict[str, Any]:
        return {
            "schema_version": atlas.schema_version,
            "builder_version": atlas.builder_version,
            "snapshot_hash": atlas.snapshot_hash,
            "metadata_fingerprint": atlas.metadata_fingerprint,
            "source_fingerprint": atlas.source_fingerprint,
            "db_id": atlas.db_id,
            "knowledge_name": atlas.knowledge_name,
            "builder_model": atlas.builder_model,
            "prompt_versions": atlas.prompt_versions,
            "prompt_hashes": atlas.prompt_hashes,
            "document_count": len(atlas.document_cards),
            "cue_count": sum(len(value.topic_cues) for value in atlas.document_cards),
            "parameters": atlas.parameters,
        }
