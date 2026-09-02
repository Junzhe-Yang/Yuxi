from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_medication_review_agent_is_auto_discovered_with_expected_config(test_client, admin_headers):
    agents_response = await test_client.get("/api/chat/agent", headers=admin_headers)
    assert agents_response.status_code == 200, agents_response.text
    agents = agents_response.json().get("agents", [])
    agent = next((item for item in agents if item.get("id") == "MedicationReviewAgent"), None)

    assert agent is not None
    assert agent["name"] == "老年治疗方案合理性审查（PEA-RAG v2 实验）"
    assert agent["capabilities"] == []

    info_response = await test_client.get("/api/chat/agent/MedicationReviewAgent", headers=admin_headers)
    assert info_response.status_code == 200, info_response.text
    configurable = info_response.json()["configurable_items"]

    assert {
        "model",
        "system_prompt",
        "knowledges",
        "run_mode",
        "agenda_mode",
        "synthesis_mode",
        "diagnostic_trace",
        "retrieval_timeout_seconds",
        "max_search_calls",
        "max_open_calls",
        "max_agent_steps",
        "retrieval_top_k",
        "max_final_evidence_tokens",
        "max_review_questions",
        "max_claim_evidence",
        "max_subqueries_per_action",
        "technical_retry_limit",
        "plan_repair_limit",
    } <= configurable.keys()
    assert {
        "tools",
        "mcps",
        "skills",
        "subagents",
        "subagents_model",
        "summary_threshold",
    }.isdisjoint(configurable.keys())
    assert configurable["retrieval_top_k"]["default"] == 5
    assert configurable["max_search_calls"]["default"] == 8
    assert configurable["run_mode"]["default"] == "full"
    assert configurable["agenda_mode"]["default"] == "dynamic"
    assert configurable["synthesis_mode"]["default"] == "claims"

    d0 = next(
        (item for item in agents if item.get("id") == "MedicationReviewD0Agent"),
        None,
    )
    assert d0 is not None
