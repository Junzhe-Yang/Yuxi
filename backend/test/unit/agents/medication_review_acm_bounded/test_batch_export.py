from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _export_module():
    path = Path(__file__).resolve().parents[5] / "scripts" / "yuxi_batch_rag" / "export_rag_records.py"
    spec = importlib.util.spec_from_file_location("bounded_export_rag_records", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _batch_module():
    path = Path(__file__).resolve().parents[5] / "scripts" / "yuxi_batch_rag" / "batch_yuxi_rag.py"
    spec = importlib.util.spec_from_file_location("bounded_batch_yuxi_rag", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bounded_batch_config_freezes_controller_view_and_256k(tmp_path) -> None:
    module = _batch_module()
    input_path = tmp_path / "input.json"
    input_path.write_text('[{"question":"问题"}]', encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_url": "https://example.test",
                "auth": {"mode": "api_key", "api_key_env": "TEST_KEY"},
                "agent_id": "MedicationReviewAcmBoundedAgent",
                "input_file": input_path.name,
                "output_dir": "output",
                "variants": {
                    "bounded": {
                        "agent_config_id": 1,
                        "expected_knowledge_base_name": "知识库",
                        "expected_controller_version": "acm-bounded-controller-v1",
                        "expected_context_view_version": "acm-bounded-context-view-v1",
                        "expected_model_context_window_tokens": 262_144,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    settings = module.load_settings(config_path)
    variant = settings.variants["bounded"]

    assert variant.expected_controller_version == "acm-bounded-controller-v1"
    assert variant.expected_context_view_version == "acm-bounded-context-view-v1"
    assert variant.expected_model_context_window_tokens == 262_144


def test_trace_13_runtime_fingerprint_includes_context_capability(tmp_path) -> None:
    module = _batch_module()
    runner = object.__new__(module.BatchRunner)
    runner._manifest_path = tmp_path / "manifest.json"
    runner._manifest_lock = module.threading.Lock()
    runner._manifest_path.write_text(
        json.dumps({"runtime_fingerprints": {}}),
        encoding="utf-8",
    )
    trace = {
        "schema_version": "13.0",
        "atlas_snapshot": {"snapshot_hash": "atlas-hash"},
        "controller_version": "acm-bounded-controller-v1",
        "context_view_version": "acm-bounded-context-view-v1",
        "prompt_version": "acm-bounded-adaptive-v1",
        "model_context_window_tokens": 262_144,
        "provider_context_window_tokens": 262_144,
        "context_window_verified": False,
    }

    assert runner.validate_runtime_fingerprint({"variant": "bounded", "medication_review_trace": trace}) is None
    changed = {
        **trace,
        "provider_context_window_tokens": 131_072,
        "context_window_verified": True,
    }
    error = runner.validate_runtime_fingerprint({"variant": "bounded", "medication_review_trace": changed})

    assert error is not None
    assert error["type"] == "runtime_fingerprint_changed"


def test_trace_13_export_keeps_retrieval_and_bounded_audit_fields() -> None:
    module = _export_module()
    trace = {
        "schema_version": "13.0",
        "method_family": "acm-prim-rag-v9",
        "method_version": "bounded-v1",
        "run_status": "complete",
        "requested_profile": "full",
        "knowledge_base_snapshot": {"name": "知识库"},
        "cited_evidence_ids": ["EV-1"],
        "query_records": [
            {
                "query_id": "Q-1",
                "tool_call_id": "call-search",
                "investigation_id": "INV-1",
                "query_text": "药物 推荐剂量",
                "reason": "推荐剂量",
                "retrieval_scope": "global",
                "started_at": "2026-09-02T00:00:00Z",
                "status": "success",
                "evidence_ids": ["EV-1"],
                "new_evidence_ids": ["EV-1"],
            }
        ],
        "evidence_store": [
            {
                "evidence_id": "EV-1",
                "content_hash": "hash-1",
                "raw_text": "精确原文",
                "occurrences": [
                    {
                        "record_id": "Q-1",
                        "tool_call_id": "call-search",
                        "source_method": "search",
                        "query_text": "药物 推荐剂量",
                        "reason": "推荐剂量",
                        "shown_excerpt": "精确原文",
                        "rank": 1,
                    }
                ],
            }
        ],
        "probe_records": [
            {
                "query_id": "Q-1",
                "retrieval_intent": "source_discovery",
                "uncovered_aspect": "推荐剂量",
                "route_key": "route-1",
            }
        ],
        "tool_outcomes": [
            {
                "call_id": "call-search",
                "tool_name": "search_active_obligation",
                "transport_status": "COMPLETED",
                "semantic_outcome": "SUCCESS",
                "state_changed": True,
            }
        ],
        "directives": [{"directive_id": "DIR-1"}],
        "context_atoms": [{"atom_id": "ATOM-1"}],
        "context_manifests": [{"model_call_id": "MC-1"}],
        "citation_verification": {"status": "ready"},
        "provider_context_window_tokens": 262_144,
        "context_window_verified": True,
    }

    calls = module._trace_v13_calls(trace)
    assert calls[0]["tool_name"] == "search_active_obligation"
    assert calls[0]["semantic_outcome"] == "SUCCESS"
    assert calls[0]["retrieved_items"][0]["raw_text"] == "精确原文"

    record = module._compact_record(
        {
            "question": "问题",
            "answer": "回答",
            "retrieval_calls": [],
            "document_open_calls": [],
            "medication_review_trace": trace,
        },
        1,
    )
    assert record["bounded_context_atoms"] == [{"atom_id": "ATOM-1"}]
    assert record["bounded_citation_verification"]["status"] == "ready"
    assert record["bounded_provider_context_window_tokens"] == 262_144
