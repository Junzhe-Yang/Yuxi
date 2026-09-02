from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yuxi.agents.buildin.medication_review_d0.context import MedicationReviewD0Context
from yuxi.agents.buildin.medication_review_d0.graph import finalize_node, parse_case_node


def _structured_question(drug: str) -> str:
    return json.dumps(
        {
            "age": 70,
            "diagnoses": [],
            "medications": [
                {
                    "source_mention": drug,
                    "normalized_name": drug,
                    "normalization_source": "input",
                    "status": "current",
                }
            ],
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_parse_case_reads_only_latest_human_message_and_resets_old_trace():
    runtime = SimpleNamespace(context=MedicationReviewD0Context())
    old_trace_state = {
        "messages": [
            HumanMessage(content=_structured_question("旧药物")),
            AIMessage(content="旧回答"),
            HumanMessage(content=_structured_question("新药物")),
        ],
        "patient_case": {"case_id": "old"},
        "review_slots": [{"slot_id": "old"}],
        "query_bundles": [{"bundle_id": "old"}],
        "evidence": [{"evidence_id": "old"}],
    }

    result = await parse_case_node(old_trace_state, runtime)

    assert result["patient_case"]["medications"][0]["source_mention"] == "新药物"
    assert result["review_slots"] == []
    assert result["query_bundles"] == []
    assert result["evidence"] == []
    assert result["run_status"] == "parsed"


@pytest.mark.asyncio
async def test_finalize_always_appends_ai_message_with_json_trace():
    runtime = SimpleNamespace(context=MedicationReviewD0Context())
    state = {
        "messages": [HumanMessage(content="病例")],
        "review_run_id": "review-1",
        "run_status": "completed",
        "patient_case": {
            "case_id": "CASE-1",
            "raw_question_hash": "hash",
            "age": 70,
            "sex": None,
            "diagnoses": [],
            "medications": [],
            "renal_function": None,
            "hepatic_function": None,
            "other_labs": [],
            "clinical_risks": [],
            "allergies": [],
            "missing_information": [],
            "extraction_warnings": [],
        },
        "review_slots": [],
        "query_bundles": [],
        "retrieval_records": [],
        "evidence": [],
        "knowledge_base_snapshot": {"name": "处方知识库"},
        "agent_config_snapshot": {},
        "usage": {"query_count": 0, "status_counts": {}},
        "warnings": [],
        "errors": [],
    }

    result = await finalize_node(state, runtime)

    message = result["messages"][0]
    trace = message.additional_kwargs["medication_review_trace"]
    assert isinstance(message, AIMessage)
    assert trace["run_status"] == "completed"
    assert trace["method_version"] == "relation-coverage-a1-vector-atomic-v1"
    assert trace["usage"]["trace_bytes"] > 0
    assert "不构成临床用药结论" in message.content
