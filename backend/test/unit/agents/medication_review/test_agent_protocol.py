from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from yuxi.agents.buildin.medication_review.context import MedicationReviewContext
from yuxi.agents.buildin.medication_review.graph import (
    _decision_context,
    agent_decision_node,
    route_after_extraction,
    route_after_prepare,
)
from yuxi.agents.buildin.medication_review.models import TechnicalAttempt
from yuxi.agents.buildin.medication_review.retrieval import V3RetrievalResult
from yuxi.agents.buildin.medication_review.tools import _execute_search
from yuxi.agents.buildin.medication_review.tools import AGENT_TOOLS


class _BoundModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        assert [item.name for item in tools] == [
            "search_evidence",
            "open_evidence_source",
            "finish_retrieval",
        ]
        return self

    async def ainvoke(self, _messages):
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


def test_agent_surface_contains_only_retrieval_tools():
    assert [item.name for item in AGENT_TOOLS] == [
        "search_evidence",
        "open_evidence_source",
        "finish_retrieval",
    ]


def test_run_modes_route_without_forking_the_graph():
    base = {"patient_case": {"case_id": "CASE-1"}, "plan_elements": [{"element_id": "PE001"}]}
    assert route_after_extraction({**base, "run_mode": "stop_after_plan"}) == "finalize_debug"
    assert route_after_extraction({**base, "run_mode": "full", "agenda_mode": "dynamic"}) == "build_agenda"
    assert route_after_extraction({**base, "run_mode": "full", "agenda_mode": "none"}) == "agent"
    assert route_after_prepare({"run_mode": "stop_after_retrieval"}) == "finalize_debug"
    assert route_after_prepare({"run_mode": "full", "synthesis_mode": "claims"}) == "extract_claims"
    assert route_after_prepare({"run_mode": "full", "synthesis_mode": "direct_chunks"}) == "synthesize"


def test_decision_context_exposes_short_evidence_index_and_budgets():
    context = MedicationReviewContext(knowledges=["知识库"], max_search_calls=8, max_open_calls=2)
    state = {
        "executed_query_count": 2,
        "logical_step_count": 3,
        "open_records": [{"open_id": "OP001"}],
        "evidence": [
            {
                "evidence_id": "EV001",
                "content_hash": "hash-1",
                "raw_text": "一段用于决策的证据原文",
                "source_document": "共识.md",
                "source_method": "search",
                "occurrences": [
                    {
                        "query_id": "Q001",
                        "rank": 1,
                    }
                ],
            }
        ],
    }

    value = _decision_context(state, context)

    assert value["remaining_budgets"] == {"subqueries": 6, "open": 1, "logical_steps": 9}
    assert value["evidence_index"][0]["evidence_id"] == "EV001"
    assert set(value["available_actions"]) == {
        "search_evidence",
        "open_evidence_source",
        "finish_retrieval",
    }


@pytest.mark.asyncio
async def test_model_transport_errors_do_not_consume_logical_steps(monkeypatch):
    model = _BoundModel([RuntimeError("temporary"), RuntimeError("temporary")])
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.graph.load_chat_model",
        lambda _name: model,
    )
    context = MedicationReviewContext(
        model="fake/model",
        knowledges=["知识库"],
        technical_retry_limit=1,
    )

    result = await agent_decision_node(
        {
            "review_run_id": "run-1",
            "agent_steps": [],
            "logical_step_count": 4,
            "agent_model_error_count": 0,
        },
        SimpleNamespace(context=context),
    )

    assert "logical_step_count" not in result
    assert result["technical_attempt_count"] == 2
    assert result["agent_model_error_count"] == 1
    assert result["tool_route"] == "agent"


@pytest.mark.asyncio
async def test_free_text_answer_is_protocol_error_and_consumes_one_step(monkeypatch):
    model = _BoundModel([AIMessage(content="直接给最终答案")])
    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.graph.load_chat_model",
        lambda _name: model,
    )
    context = MedicationReviewContext(model="fake/model", knowledges=["知识库"])

    result = await agent_decision_node(
        {"review_run_id": "run-1", "agent_steps": [], "logical_step_count": 0},
        SimpleNamespace(context=context),
    )

    assert result["logical_step_count"] == 1
    assert result["agent_steps"][0]["status"] == "protocol_error"
    assert result["tool_route"] == "agent"


@pytest.mark.asyncio
async def test_persistent_retrieval_transport_errors_do_not_spend_logical_steps(
    monkeypatch,
):
    async def failed_retrieval(**_kwargs):
        return V3RetrievalResult(
            candidates=[],
            attempts=[
                TechnicalAttempt(
                    attempt=1,
                    started_at="2026-01-01T00:00:00Z",
                    elapsed_ms=10,
                    status="backend_error",
                )
            ],
            knowledge_base_snapshot={},
            status="technical_failed",
            returned_count=0,
            error_type="RuntimeError",
            error_message="backend unavailable",
        )

    monkeypatch.setattr(
        "yuxi.agents.buildin.medication_review.tools.retrieve_subquery",
        failed_retrieval,
    )
    result = await _execute_search(
        args={
            "subqueries": [
                {
                    "query_text": "该患者条件下该方案是否符合来源建议？",
                    "search_reason": "核验方案",
                }
            ]
        },
        state={
            "review_run_id": "run-1",
            "patient_case": {"case_id": "CASE-1"},
            "logical_step_count": 5,
            "consecutive_tool_error_count": 2,
        },
        context=MedicationReviewContext(knowledges=["知识库"]),
    )

    assert result.updates["logical_step_count"] == 4
    assert result.updates["consecutive_tool_error_count"] == 3
    assert result.updates["degraded"] is True
    assert result.route == "prepare_evidence"
