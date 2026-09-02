from __future__ import annotations

from typing import Any

from yuxi import knowledge_base
from yuxi.agents.buildin.medication_review_prim.harness import (
    ReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_prim.models import (
    MedicationReviewPrimTrace,
)
from yuxi.agents.buildin.medication_review_prim.tools import (
    resolve_milvus_retriever,
)

from .atlas_memory import build_atlas_memory
from .context import (
    MedicationReviewDaPrimContext,
    profile_uses_opportunities,
    validate_da_context,
)
from .corpus_atlas import AtlasStore, CorpusAtlasBuilder
from .corpus_atlas.models import CorpusAtlas
from .corpus_atlas.router import DOCUMENT_TOP_K, RRF_K as ROUTER_RRF_K, route_case
from .models import (
    CaseRouteRecord,
    MedicationReviewDaPrimState,
    MedicationReviewDaPrimTrace,
    RetrievalOpportunity,
)
from .corpus_atlas.opportunities import (
    MAX_OPPORTUNITIES,
    MIN_SECTION_SIMILARITY,
    SECTION_TOP_R,
    build_retrieval_opportunities,
)
from .routed_retrieval import (
    FINAL_TOP_K,
    GLOBAL_TOP_K,
    PER_DOCUMENT_TOP_K,
    RRF_K as RETRIEVAL_RRF_K,
)
from .tools import search_review_kb_da, search_review_kb_da_route


class DaReviewHarnessMiddleware(ReviewHarnessMiddleware):
    state_schema = MedicationReviewDaPrimState

    async def abefore_agent(
        self,
        state: MedicationReviewDaPrimState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        context: MedicationReviewDaPrimContext = runtime.context
        validate_da_context(context)
        selection = await resolve_milvus_retriever(context)
        store = AtlasStore()
        atlas = store.load_current(selection.db_id)
        await CorpusAtlasBuilder(store=store).validate_current(atlas)
        setattr(context, "_da_prim_atlas", atlas)
        return await super().abefore_agent(state, runtime)

    async def augment_initial_state(
        self,
        *,
        state: dict[str, Any],
        update: dict[str, Any],
        runtime: Any,
    ) -> dict[str, Any]:
        context: MedicationReviewDaPrimContext = runtime.context
        validate_da_context(context)
        snapshot = dict(update.get("knowledge_base_snapshot") or {})
        db_id = str(snapshot.get("db_id") or "")
        if not db_id:
            raise ValueError("DA-PRIM 无法从知识库快照取得 db_id")
        atlas = getattr(context, "_da_prim_atlas", None)
        if not isinstance(atlas, CorpusAtlas) or atlas.db_id != db_id:
            store = AtlasStore()
            atlas = store.load_current(db_id)
            await CorpusAtlasBuilder(store=store).validate_current(atlas)
            setattr(context, "_da_prim_atlas", atlas)

        existing_route = state.get("case_route_record")
        if existing_route is not None:
            previous_profile = str(state.get("atlas_profile") or "")
            if previous_profile and previous_profile != context.atlas_profile:
                raise ValueError("同一 DA-PRIM thread 不能切换 atlas_profile，请新建会话")
            route_record = (
                existing_route
                if isinstance(existing_route, CaseRouteRecord)
                else CaseRouteRecord.model_validate(existing_route)
            )
            if route_record.atlas_snapshot_hash != atlas.snapshot_hash:
                raise ValueError("同一 DA-PRIM thread 的 Atlas 快照已变化，请新建会话")
            return {
                "atlas_profile": context.atlas_profile,
                "atlas_snapshot": self._atlas_snapshot(atlas),
            }

        merged = {**state, **update}
        computation = await route_case(
            manager=knowledge_base,
            db_id=db_id,
            atlas=atlas,
            raw_case_text=str(merged.get("raw_case_text") or ""),
            plan_anchors=list(merged.get("plan_anchors") or []),
            patient_modifiers=list(merged.get("patient_modifiers") or []),
        )
        opportunities: list[RetrievalOpportunity] = []
        if profile_uses_opportunities(context.atlas_profile):
            opportunities = build_retrieval_opportunities(
                atlas=atlas,
                computation=computation,
                plan_anchors=list(merged.get("plan_anchors") or []),
                patient_modifiers=list(merged.get("patient_modifiers") or []),
            )
        return {
            "atlas_profile": context.atlas_profile,
            "atlas_snapshot": self._atlas_snapshot(atlas),
            "case_route_record": computation.record,
            "retrieval_opportunities": opportunities,
            "adopted_opportunity_ids": [],
            "routed_retrieval_records": [],
        }

    def augment_investigation_memory(
        self,
        *,
        state: dict[str, Any],
        context: MedicationReviewDaPrimContext,
        memory_text: str,
    ) -> str:
        route_raw = state.get("case_route_record")
        if route_raw is None:
            return memory_text
        case_route = route_raw if isinstance(route_raw, CaseRouteRecord) else CaseRouteRecord.model_validate(route_raw)
        opportunities = [
            value if isinstance(value, RetrievalOpportunity) else RetrievalOpportunity.model_validate(value)
            for value in state.get("retrieval_opportunities") or []
        ]
        atlas_memory = build_atlas_memory(
            case_route=case_route,
            opportunities=opportunities,
            include_opportunities=profile_uses_opportunities(context.atlas_profile),
        )
        return f"{atlas_memory}\n\n{memory_text}" if memory_text else atlas_memory

    def finalize_trace(
        self,
        *,
        base_trace: MedicationReviewPrimTrace,
        state: dict[str, Any],
        context: MedicationReviewDaPrimContext,
    ) -> MedicationReviewDaPrimTrace:
        route_raw = state.get("case_route_record")
        if route_raw is None:
            raise ValueError("DA-PRIM 最终 Trace 缺少 case_route_record")
        payload = base_trace.model_dump(exclude={"schema_version", "method_family", "method_version"})
        return MedicationReviewDaPrimTrace(
            **payload,
            method_version=(f"da-prim-rag-v1-{context.atlas_profile}-vector-top5"),
            atlas_profile=context.atlas_profile,
            atlas_snapshot=dict(state.get("atlas_snapshot") or {}),
            case_route_record=(
                route_raw if isinstance(route_raw, CaseRouteRecord) else CaseRouteRecord.model_validate(route_raw)
            ),
            retrieval_opportunities=list(state.get("retrieval_opportunities") or []),
            adopted_opportunity_ids=list(state.get("adopted_opportunity_ids") or []),
            routed_retrieval_records=list(state.get("routed_retrieval_records") or []),
        )

    def project_search_tool(
        self,
        _effective_profile: Any,
        context: MedicationReviewDaPrimContext | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        del state
        if context is not None and profile_uses_opportunities(context.atlas_profile):
            return search_review_kb_da
        return search_review_kb_da_route

    @staticmethod
    def _atlas_snapshot(atlas: Any) -> dict[str, Any]:
        return {
            "schema_version": atlas.schema_version,
            "builder_version": atlas.builder_version,
            "snapshot_hash": atlas.snapshot_hash,
            "metadata_fingerprint": atlas.metadata_fingerprint,
            "db_id": atlas.db_id,
            "knowledge_name": atlas.knowledge_name,
            "embedding_model_id": atlas.embedding_model_id,
            "embedding_dimension": atlas.embedding_dimension,
            "document_count": len(atlas.document_cards),
            "section_count": len(atlas.section_cards),
            "parameters": atlas.parameters,
            "method_parameters": {
                "document_top_k": DOCUMENT_TOP_K,
                "section_top_r": SECTION_TOP_R,
                "max_opportunities": MAX_OPPORTUNITIES,
                "opportunity_min_similarity": MIN_SECTION_SIMILARITY,
                "global_top_k": GLOBAL_TOP_K,
                "per_document_top_k": PER_DOCUMENT_TOP_K,
                "final_top_k": FINAL_TOP_K,
                "router_rrf_k": ROUTER_RRF_K,
                "retrieval_rrf_k": RETRIEVAL_RRF_K,
            },
        }
