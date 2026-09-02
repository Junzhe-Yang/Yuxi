from __future__ import annotations

import pytest

from yuxi.agents.buildin.medication_review_da_prim.context import (
    MedicationReviewDaPrimContext,
)
from yuxi.agents.buildin.medication_review_da_prim.graph import (
    MedicationReviewDaPrimAgent,
)
from yuxi.agents.buildin.medication_review_da_prim.harness import (
    DaReviewHarnessMiddleware,
)
from yuxi.agents.buildin.medication_review_da_prim.models import (
    MedicationReviewDaPrimState,
)


def test_da_agent_has_independent_metadata_and_profiles() -> None:
    assert MedicationReviewDaPrimAgent.metadata["method_family"] == "da-prim-rag-v1"
    assert MedicationReviewDaPrimAgent.metadata["trace_schema_version"] == "6.0"
    assert MedicationReviewDaPrimAgent.context_schema is MedicationReviewDaPrimContext
    items = MedicationReviewDaPrimContext.get_configurable_items()
    assert items["atlas_profile"]["options"] == ["map", "route", "full"]
    assert "experiment_profile" not in items


@pytest.mark.asyncio
async def test_graph_keeps_agent_search_open_and_reflection_tools(
    monkeypatch,
    tmp_path,
) -> None:
    model = object()
    captured = {}

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.graph.load_chat_model",
        lambda fully_specified_name: (model if fully_specified_name == "provider/model" else None),
    )

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review_da_prim.graph.create_agent",
        fake_create_agent,
    )
    agent = object.__new__(MedicationReviewDaPrimAgent)
    agent.workdir = tmp_path

    async def get_checkpointer():
        return "checkpointer"

    agent._get_checkpointer = get_checkpointer
    context = MedicationReviewDaPrimContext(
        model="provider/model",
        knowledges=["知识库"],
        atlas_profile="route",
    )

    graph = await agent.get_graph(context=context)

    assert graph == "graph"
    assert [tool.name for tool in captured["tools"]] == [
        "search_review_kb",
        "open_review_evidence",
        "coverage_reflection",
    ]
    assert isinstance(captured["middleware"][0], DaReviewHarnessMiddleware)
    assert captured["state_schema"] is MedicationReviewDaPrimState
    assert captured["context_schema"] is MedicationReviewDaPrimContext
