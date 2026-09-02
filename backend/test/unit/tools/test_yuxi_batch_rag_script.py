from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts" / "yuxi_batch_rag"
sys.path.insert(0, str(SCRIPT_DIR))

from batch_yuxi_rag import (  # noqa: E402
    BatchRunner,
    BatchSettings,
    Job,
    VariantConfig,
    YuxiClient,
    consume_run_events,
    extract_history_data,
    extract_sse_tool_activities,
    iter_sse_events,
    login_for_access_token,
    load_dataset,
    load_settings,
    terminal_sse_status,
)
import discover_yuxi  # noqa: E402


class FakeEventClient:
    def __init__(self):
        self.read_count = 0
        self.after_sequences: list[str] = []
        self.status_count = 0

    def read_sse_once(
        self,
        run_id: str,
        after_seq: str,
        deadline_monotonic: float | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        del run_id
        del deadline_monotonic
        self.after_sequences.append(after_seq)
        self.read_count += 1
        if self.read_count == 1:
            return ([{"event": "message", "data": {"seq": "1-0", "payload": {"status": "loading"}}}], False)
        return (
            [
                {"event": "message", "data": {"seq": "1-0", "payload": {"status": "loading"}}},
                {"event": "finished", "data": {"seq": "2-0", "payload": {"status": "finished"}}},
                {"event": "close", "data": {"last_seq": "2-0"}},
            ],
            True,
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        del run_id
        self.status_count += 1
        return {"status": "running" if self.status_count == 1 else "completed"}


class FakeYuxiClient:
    queries: list[str] = []
    threads: list[str] = []
    next_thread = 0

    def __init__(self, *args: Any, **kwargs: Any):
        del args, kwargs
        self.query = ""

    def __enter__(self) -> FakeYuxiClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback

    def create_thread(self, agent_id: str, title: str, metadata: dict[str, Any]) -> dict[str, Any]:
        del agent_id, title, metadata
        FakeYuxiClient.next_thread += 1
        thread_id = f"thread-{FakeYuxiClient.next_thread}"
        FakeYuxiClient.threads.append(thread_id)
        return {"id": thread_id}

    def create_run(
        self,
        query: str,
        agent_config_id: int,
        thread_id: str,
        request_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        del agent_config_id, thread_id, request_id, metadata
        self.query = query
        FakeYuxiClient.queries.append(query)
        return {"run_id": f"run-{len(FakeYuxiClient.queries)}"}

    def get_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "completed"}

    def get_history(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        return {
            "history": [
                {"id": 1, "type": "human", "content": self.query},
                {
                    "id": 2,
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "query_kb",
                            "args": {
                                "kb_name": "向量库",
                                "query_text": self.query + " 检索词",
                            },
                            "tool_call_result": {
                                "content": json.dumps(
                                    [
                                        {
                                            "content": "实际 chunk",
                                            "metadata": {"source": "a.md", "chunk_id": "c1"},
                                            "score": 0.9,
                                        }
                                    ],
                                    ensure_ascii=False,
                                )
                            },
                            "status": "success",
                            "error_message": None,
                        }
                    ],
                },
                {"id": 3, "type": "ai", "content": "最终回答"},
            ]
        }


class FakeRunner(BatchRunner):
    def client(self) -> FakeYuxiClient:
        return FakeYuxiClient()

    def load_or_create_manifest(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"batch_id": "batch", "input_count": len(rows)}


class FailFirstClient(FakeYuxiClient):
    def get_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "failed" if run_id == "run-1" else "completed"}


class RetryRunner(FakeRunner):
    def client(self) -> FailFirstClient:
        return FailFirstClient()


class MedicationReviewClient(FakeYuxiClient):
    trace_status = "completed"

    def get_history(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        trace = {
            "run_status": self.trace_status,
            "retrieval_records": [
                {
                    "bundle_id": "QB:MP:M001",
                    "status": "success",
                    "evidence_ids": ["evidence-1"],
                }
            ],
            "evidence": [{"evidence_id": "evidence-1", "raw_text": "片段"}],
        }
        return {
            "history": [
                {"id": 1, "type": "human", "content": self.query},
                {
                    "id": 2,
                    "type": "ai",
                    "content": "处方关系覆盖报告",
                    "extra_metadata": {
                        "additional_kwargs": {
                            "medication_review_trace": trace,
                        }
                    },
                },
            ]
        }


class MedicationReviewRunner(FakeRunner):
    def client(self) -> MedicationReviewClient:
        return MedicationReviewClient()


class MedicationReviewV2PartialClient(FakeYuxiClient):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        trace = {
            "schema_version": "2.0",
            "method_version": "pea-rag-mfull-vector-v1",
            "run_status": "partial",
            "answer_validation": {"valid": True, "errors": []},
            "search_records": [{"status": "success_empty"}],
            "usage": {
                "logical_search_count": 1,
                "technical_attempt_count": 1,
                "open_count": 0,
                "active_evidence_count": 0,
            },
        }
        return {
            "history": [
                {"id": 1, "type": "human", "content": self.query},
                {
                    "id": 2,
                    "type": "ai",
                    "content": "①【原方案要素清单】\n⑥【依据清单】",
                    "extra_metadata": {
                        "additional_kwargs": {
                            "medication_review_trace": trace,
                        }
                    },
                },
            ]
        }


class MedicationReviewV2PartialRunner(FakeRunner):
    def client(self) -> MedicationReviewV2PartialClient:
        return MedicationReviewV2PartialClient()


class MedicationReviewV3Client(FakeYuxiClient):
    run_status = "completed"
    include_final_review = True

    def get_history(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        answer = "\n".join(
            [
                "①【原方案要素清单】",
                "②【逐项判断】",
                "③【正面判断汇总】",
                "④【负面与不确定判断汇总】",
                "⑤【综合建议】",
                "⑥【依据清单】",
            ]
        )
        trace = {
            "schema_version": "3.0",
            "method_version": "pea-rag-v2-dynamic-claims-vector-v1",
            "effective_profile": "pea-rag-v2-dynamic-claims-vector-v1",
            "run_status": self.run_status,
            "run_mode": "full" if self.run_status != "debug_stopped" else "stop_after_retrieval",
            "agenda_mode": "dynamic",
            "synthesis_mode": "claims",
            "last_completed_stage": ("answer_rendered" if self.run_status != "debug_stopped" else "evidence_prepared"),
            "final_review": {"element_reviews": []} if self.include_final_review else None,
            "search_records": [],
            "evidence": [],
            "usage": {"executed_subqueries": 0},
        }
        return {
            "history": [
                {"id": 1, "type": "human", "content": self.query},
                {
                    "id": 2,
                    "type": "ai",
                    "content": answer if self.run_status != "debug_stopped" else "阶段诊断输出",
                    "extra_metadata": {"additional_kwargs": {"medication_review_trace": trace}},
                },
            ]
        }


class MedicationReviewV3Runner(FakeRunner):
    def __init__(self, settings, api_key, client):
        super().__init__(settings, api_key)
        self._client = client

    def client(self) -> MedicationReviewV3Client:
        return self._client


class MedicationReviewV4Client(FakeYuxiClient):
    run_status = "partial"

    def get_history(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        trace = {
            "schema_version": "4.0",
            "method_family": "pat-rag-v1",
            "method_version": "pat-rag-v1-m3-vector-top5",
            "experiment_profile": "m3",
            "run_status": self.run_status,
            "search_records": [{"record_id": "SEARCH-1"}],
            "open_records": [],
            "evidence_store": [],
            "budgets": {
                "executed_search_calls": 1,
                "executed_open_calls": 0,
                "technical_attempts": 1,
            },
            "usage": {"total": {"total_tokens": 100}},
        }
        return {
            "history": [
                {"id": 1, "type": "human", "content": self.query},
                {
                    "id": 2,
                    "type": "ai",
                    "content": "这是非六段格式但非空的 PAT-RAG 最终回答",
                    "extra_metadata": {
                        "additional_kwargs": {
                            "medication_review_trace": trace,
                        }
                    },
                },
            ]
        }


class MedicationReviewV4Runner(FakeRunner):
    def __init__(self, settings, api_key, client):
        super().__init__(settings, api_key)
        self._client = client

    def client(self) -> MedicationReviewV4Client:
        return self._client


class MedicationReviewV5Client(MedicationReviewV4Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"]["medication_review_trace"]
        trace.update(
            {
                "schema_version": "5.0",
                "method_family": "prim-rag-v1",
                "method_version": "prim-rag-v1-m3-vector-top5",
                "requested_profile": "m3",
                "effective_profile": "m3",
                "query_records": trace.pop("search_records"),
                "relation_investigations": [],
            }
        )
        return history


class MedicationReviewV5Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV5Client:
        return self._client


class MedicationReviewV6Client(MedicationReviewV5Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"]["medication_review_trace"]
        trace.update(
            {
                "schema_version": "6.0",
                "method_family": "da-prim-rag-v1",
                "method_version": "da-prim-rag-v1-route-vector-top5",
                "experiment_profile": "full",
                "requested_profile": "full",
                "effective_profile": "full",
                "atlas_profile": "route",
                "atlas_snapshot": {"snapshot_hash": "atlas-1"},
                "case_route_record": {
                    "atlas_snapshot_hash": "atlas-1",
                    "views": [],
                    "ranked_documents": [],
                    "map_sections": [],
                },
                "retrieval_opportunities": [],
                "adopted_opportunity_ids": [],
                "routed_retrieval_records": [],
            }
        )
        return history


class MedicationReviewV6Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV6Client:
        return self._client


class MedicationReviewV7Client(MedicationReviewV5Client):
    selector_status = "success"
    effective_profile = "full"

    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "7.0",
                "method_family": "acm-prim-rag-v1",
                "method_version": "acm-prim-rag-v1-vector-top5",
                "experiment_profile": "full",
                "requested_profile": "full",
                "effective_profile": self.effective_profile,
                "atlas_snapshot": {"snapshot_hash": "atlas-acm-1"},
                "prompt_hashes": {
                    "companion_selector": "selector-prompt-1"
                },
                "companion_selection_attempted": True,
                "companion_selection": {
                    "companion_cues": (
                        []
                        if self.selector_status == "empty"
                        else [{"companion_id": "AC01"}]
                    ),
                    "selector_audit": {
                        "status": self.selector_status,
                        "elapsed_ms": 12,
                        "usage": {"total_tokens": 34},
                    },
                },
                "companion_adoption_events": [],
                "remaining_companion_ids": [],
                "companion_cues_at_reflection": [],
            }
        )
        return history


class MedicationReviewV7Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV7Client:
        return self._client


class MedicationReviewV8Client(MedicationReviewV5Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "8.0",
                "method_family": "prim-rag-v2",
                "method_version": "prim-rag-v2-full-vector-top10",
                "experiment_profile": "full",
                "requested_profile": "full",
                "effective_profile": "full",
                "investigations": [],
                "deferred_knowledge_calls": [],
            }
        )
        return history


class MedicationReviewV8Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV8Client:
        return self._client


class MedicationReviewV9Client(MedicationReviewV7Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "9.0",
                "method_family": "acm-prim-rag-v2",
                "method_version": "acm-prim-rag-v2-vector-top10",
                "investigations": [],
                "deferred_knowledge_calls": [],
            }
        )
        return history


class MedicationReviewV9Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV9Client:
        return self._client


class MedicationReviewV10Client(MedicationReviewV8Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "10.0",
                "method_family": "acm-prim-rag-v3",
                "method_version": (
                    "acm-prim-rag-v3-atlas-navigation-vector-top10"
                ),
                "atlas_snapshot": {"snapshot_hash": "atlas-acm-3"},
                "prompt_hashes": {
                    "atlas_navigation": "navigation-prompt-1"
                },
                "atlas_document_open_records": [],
            }
        )
        return history


class MedicationReviewV10Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV10Client:
        return self._client


class MedicationReviewV11Client(MedicationReviewV10Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "11.0",
                "method_family": "acm-prim-rag-v7",
                "method_version": (
                    "acm-prim-rag-v7-a2_k2-shadow_top25-vector"
                ),
                "experiment_arm": "a2_k2",
                "retrieval_depth": "shadow_top25",
                "contract_report": {"status": "completed"},
            }
        )
        return history


class MedicationReviewV11Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV11Client:
        return self._client


class MedicationReviewV12Client(MedicationReviewV10Client):
    def get_history(self, thread_id: str) -> dict[str, Any]:
        history = super().get_history(thread_id)
        trace = history["history"][-1]["extra_metadata"]["additional_kwargs"][
            "medication_review_trace"
        ]
        trace.update(
            {
                "schema_version": "12.0",
                "method_family": "acm-prim-rag-v8",
                "method_version": (
                    "acm-prim-rag-v8-adaptive-coverage-shadow_top25-vector"
                ),
                "protocol": "adaptive_coverage",
                "retrieval_depth": "shadow_top25",
                "adaptive_coverage_report": {
                    "status": "completed",
                    "actual_investigation_count": 4,
                    "pending_recovery_ids": [],
                    "gap_assessment_status": "current",
                },
            }
        )
        return history


class MedicationReviewV12Runner(MedicationReviewV4Runner):
    def client(self) -> MedicationReviewV12Client:
        return self._client


class FakeLoginResponse:
    status_code = 200
    is_error = False
    text = ""

    def json(self) -> dict[str, Any]:
        return {
            "access_token": "jwt-token",
            "token_type": "bearer",
            "user_id": 1,
            "username": "tester",
            "user_id_login": "u001",
        }


class FakeLoginClient:
    def __init__(self, *args: Any, **kwargs: Any):
        del args, kwargs

    def __enter__(self) -> FakeLoginClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback

    def post(self, url: str, data: dict[str, str], headers: dict[str, str]) -> FakeLoginResponse:
        assert url.endswith("/api/auth/token")
        assert data == {"username": "u001", "password": "secret"}
        assert headers["Accept"] == "application/json"
        return FakeLoginResponse()


class FakeDiscoveryClient:
    def __init__(self, *args: Any, **kwargs: Any):
        del args, kwargs

    def __enter__(self) -> FakeDiscoveryClient:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback

    def get_agents(self) -> dict[str, Any]:
        return {"agents": [{"id": "ChatbotAgent", "name": "Chatbot"}]}

    def get_default_agent(self) -> dict[str, Any]:
        return {"default_agent_id": "ChatbotAgent"}

    def get_agent_configs(self, agent_id: str) -> dict[str, Any]:
        assert agent_id == "ChatbotAgent"
        return {
            "configs": [
                {"id": 7, "name": "向量配置", "description": "", "is_default": True},
                {"id": 8, "name": "图谱配置", "description": "", "is_default": False},
            ]
        }

    def get_accessible_databases(self) -> dict[str, Any]:
        return {
            "databases": [
                {"name": "向量库", "db_id": "db-vector", "description": ""},
                {"name": "图谱库", "db_id": "db-graph", "description": ""},
            ]
        }

    def get_agent_config(self, agent_id: str, config_id: int) -> dict[str, Any]:
        assert agent_id == "ChatbotAgent"
        knowledge_name = "向量库" if config_id == 7 else "图谱库"
        return {
            "config": {
                "config_json": {
                    "context": {
                        "model": "lan-llm/chat",
                        "subagents_model": "lan-llm/chat",
                        "knowledges": [knowledge_name],
                        "tools": [],
                    }
                }
            }
        }


class BatchYuxiRagTests(unittest.TestCase):
    def test_login_endpoint_returns_jwt_without_exposing_password(self) -> None:
        with patch("batch_yuxi_rag.httpx.Client", FakeLoginClient):
            result = login_for_access_token("https://example.invalid", "u001", "secret")
        self.assertEqual(result["access_token"], "jwt-token")

    def test_discovery_resolves_config_ids_models_and_database_ids(self) -> None:
        args = type(
            "Args",
            (),
            {
                "base_url": "https://example.invalid",
                "agent_id": None,
                "auth_mode": "login",
                "login_id_env": "YUXI_LOGIN_ID",
                "password_env": "YUXI_PASSWORD",
                "api_key_env": "YUXI_API_KEY",
                "timeout_seconds": 5.0,
                "insecure": False,
                "output": None,
            },
        )()
        with patch.object(discover_yuxi, "get_token", return_value=("jwt-token", {"mode": "login"})), patch.object(
            discover_yuxi, "YuxiClient", FakeDiscoveryClient
        ):
            result = discover_yuxi.discover(args)

        self.assertEqual(result["selected_agent_id"], "ChatbotAgent")
        self.assertEqual(result["agent_configs"][0]["id"], 7)
        self.assertEqual(result["agent_configs"][0]["model"], "lan-llm/chat")
        self.assertEqual(result["single_knowledge_base_recommendations"][1]["knowledge_db_id"], "db-graph")

    def test_http_client_stops_sse_on_terminal_event_and_extracts_tool_progress(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/chat/runs/run-1/events":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        b'event: loading\ndata: {"seq":"1-0","payload":{"items":['
                        b'{"msg":{"type":"AIMessageChunk","tool_calls":['
                        b'{"name":"query_kb","id":"call-1"}]}},'
                        b'{"msg":{"type":"tool","name":"query_kb",'
                        b'"tool_call_id":"call-1","content":"chunk result"}}]}}\n\n'
                        b'event: finished\ndata: {"seq":"2-0","payload":'
                        b'{"chunk":{"status":"finished"}}}\n\n'
                        b'event: close\ndata: {"last_seq":"2-0"}\n\n'
                    ),
                )
            if request.url.path == "/api/chat/runs/run-1":
                return httpx.Response(200, json={"run": {"id": "run-1", "status": "completed"}})
            return httpx.Response(404, json={"detail": "not found"})

        client = YuxiClient("https://example.invalid", "fake-key", 5)
        client.client.close()
        client.client = httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer fake-key"},
        )
        try:
            events, close_seen = client.read_sse_once("run-1", "0-0")
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["event"], "loading")
            self.assertEqual(events[1]["event"], "finished")
            self.assertFalse(close_seen)
            self.assertEqual(terminal_sse_status(events[1]), "completed")
            activities = extract_sse_tool_activities(events[0])
            self.assertEqual([activity["kind"] for activity in activities], ["call", "result"])
            self.assertEqual([activity["tool_name"] for activity in activities], ["query_kb", "query_kb"])
            self.assertEqual(client.get_run("run-1")["status"], "completed")
        finally:
            client.close()

    def test_sse_parser_and_reconnect_deduplicates_sequences(self) -> None:
        parsed = list(
            iter_sse_events(
                [
                    ": heartbeat",
                    "event: message",
                    'data: {"seq": "1-0", "payload": {"status": "loading"}}',
                    "",
                    "event: close",
                    'data: {"last_seq": "1-0"}',
                    "",
                ]
            )
        )
        self.assertEqual(parsed[0][0], "message")
        self.assertEqual(parsed[1][0], "close")

        client = FakeEventClient()
        events = consume_run_events(client, "run-1", 5, 3, 0)
        self.assertEqual([event["data"].get("seq") for event in events], ["1-0", "2-0"])
        self.assertEqual(client.after_sequences, ["0-0", "1-0"])

    def test_history_extracts_answer_and_vector_retrieval(self) -> None:
        history = {
            "history": [
                {"id": 1, "type": "human", "content": "问题"},
                {
                    "id": 2,
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "name": "query_kb",
                            "args": {"query_text": "改写问题"},
                            "tool_call_result": {"content": '[{"content": "chunk"}]'},
                            "status": "success",
                        }
                    ],
                },
                {"id": 3, "type": "ai", "content": "答案"},
            ]
        }
        result = extract_history_data(history)
        self.assertEqual(result["answer"], "答案")
        self.assertEqual(result["retrieval_status"], "called")
        self.assertEqual(result["retrieval_calls"][0]["args"]["query_text"], "改写问题")
        self.assertEqual(result["retrieval_calls"][0]["result_parsed"][0]["content"], "chunk")

    def test_history_preserves_lightrag_graph_object(self) -> None:
        graph_result = {
            "entities": [{"entity_name": "实体"}],
            "relationships": [{"src_id": "a", "tgt_id": "b"}],
            "references": ["ref"],
            "chunks": [{"content": "chunk"}],
        }
        result = extract_history_data(
            {
                "history": [
                    {
                        "type": "ai",
                        "content": "答案",
                        "tool_calls": [
                            {
                                "name": "query_kb",
                                "args": {"kb_name": "图谱库"},
                                "tool_call_result": {"content": json.dumps(graph_result)},
                                "status": "success",
                            }
                        ],
                    }
                ]
            }
        )
        parsed = result["retrieval_calls"][0]["result_parsed"]
        self.assertEqual(parsed["entities"][0]["entity_name"], "实体")
        self.assertEqual(parsed["chunks"][0]["content"], "chunk")

    def test_history_extracts_medication_review_trace_without_fake_tool_call(self) -> None:
        trace = {
            "run_status": "completed",
            "retrieval_records": [{"status": "success", "evidence_ids": ["e1"]}],
            "evidence": [{"evidence_id": "e1", "raw_text": "实际片段"}],
        }
        result = extract_history_data(
            {
                "history": [
                    {
                        "type": "ai",
                        "content": "关系覆盖报告",
                        "extra_metadata": {
                            "additional_kwargs": {
                                "medication_review_trace": trace,
                            }
                        },
                    }
                ]
            }
        )

        self.assertEqual(result["answer"], "关系覆盖报告")
        self.assertEqual(result["retrieval_status"], "called")
        self.assertEqual(result["retrieval_calls"], [])
        self.assertEqual(result["medication_review_trace"], trace)

    def test_history_recognizes_trace_v5_query_records(self) -> None:
        trace = {
            "schema_version": "5.0",
            "run_status": "completed",
            "requested_profile": "m3",
            "query_records": [
                {
                    "query_id": "Q-ONE",
                    "status": "success",
                    "evidence_ids": ["EV-0123456789ABCDEF"],
                }
            ],
        }
        result = extract_history_data(
            {
                "history": [
                    {
                        "type": "ai",
                        "content": "完整六段式答案",
                        "extra_metadata": {
                            "additional_kwargs": {
                                "medication_review_trace": trace,
                            }
                        },
                    }
                ]
            }
        )

        self.assertEqual(result["retrieval_status"], "called")
        self.assertEqual(result["medication_review_trace"], trace)

    def test_history_recognizes_v2_search_records_and_tools(self) -> None:
        trace = {
            "schema_version": "2.0",
            "method_version": "pea-rag-mfull-vector-v1",
            "run_status": "partial",
            "search_records": [{"status": "all_irrelevant"}],
        }
        result = extract_history_data(
            {
                "history": [
                    {
                        "type": "ai",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "search_evidence",
                                "args": {"query_text": "临床命题"},
                            }
                        ],
                    },
                    {
                        "type": "ai",
                        "content": "①【原方案要素清单】",
                        "extra_metadata": {
                            "additional_kwargs": {
                                "medication_review_trace": trace,
                            }
                        },
                    },
                ]
            }
        )

        self.assertEqual(result["retrieval_status"], "called")
        self.assertEqual(result["retrieval_calls"][0]["tool_name"], "search_evidence")
        self.assertEqual(result["medication_review_trace"], trace)

    def test_dataset_and_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text(json.dumps([{"question": "问题", "answer": "标准答案"}]), encoding="utf-8")
            self.assertEqual(load_dataset(input_path)[0]["question"], "问题")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "base_url": "https://example.invalid",
                        "agent_id": "ChatbotAgent",
                        "input_file": "input.json",
                        "output_dir": "output",
                        "variants": {
                            "vector": {
                                "agent_config_id": 1,
                                "expected_knowledge_base_name": "向量库",
                                "expected_run_mode": "stop_after_retrieval",
                                "expected_agenda_mode": "none",
                                "expected_synthesis_mode": "direct_chunks",
                                "expected_method_version": "method-v1",
                                "expected_effective_profile": "full",
                                "expected_acm_protocol": "adaptive_coverage",
                                "expected_v7_experiment_arm": "a2_k3",
                                "expected_v7_retrieval_depth": (
                                    "shadow_top25"
                                ),
                                "expected_max_search_calls": 50,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            settings = load_settings(config_path)
            self.assertEqual(settings.input_file, input_path)
            self.assertEqual(settings.variants["vector"].agent_config_id, 1)
            self.assertEqual(
                settings.variants["vector"].expected_run_mode,
                "stop_after_retrieval",
            )
            self.assertEqual(
                settings.variants["vector"].expected_effective_profile,
                "full",
            )
            self.assertEqual(
                settings.variants["vector"].expected_method_version,
                "method-v1",
            )
            self.assertEqual(
                settings.variants["vector"].expected_acm_protocol,
                "adaptive_coverage",
            )
            self.assertEqual(
                settings.variants["vector"].expected_v7_experiment_arm,
                "a2_k3",
            )
            self.assertEqual(
                settings.variants["vector"].expected_v7_retrieval_depth,
                "shadow_top25",
            )
            self.assertEqual(
                settings.variants["vector"].expected_max_search_calls,
                50,
            )

    def test_each_job_gets_a_new_thread_and_gold_answer_is_not_sent(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text("[]", encoding="utf-8")
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="ChatbotAgent",
                input_file=input_path,
                output_dir=root / "output",
                variants={"vector": VariantConfig("vector", 1)},
                max_attempts=1,
                write_raw_events=True,
            )
            runner = FakeRunner(settings, "fake-key")
            first = runner.run_one(
                Job("batch", 0, {"question": "问题一", "answer": "标准答案一"}, settings.variants["vector"], 1)
            )
            second = runner.run_one(
                Job("batch", 1, {"question": "问题二", "answer": "标准答案二"}, settings.variants["vector"], 1)
            )

            self.assertEqual(first["result_status"], "success")
            self.assertEqual(second["result_status"], "success")
            self.assertNotEqual(first["thread_id"], second["thread_id"])
            self.assertEqual(FakeYuxiClient.queries, ["问题一", "问题二"])
            self.assertNotIn("标准答案一", FakeYuxiClient.queries)
            self.assertNotIn("标准答案二", FakeYuxiClient.queries)
            self.assertEqual(first["retrieval_calls"][0]["result_parsed"][0]["content"], "实际 chunk")

    def test_runner_uses_medication_review_trace_status_and_preserves_trace(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"vector": VariantConfig("vector", 1, "处方知识库")},
                max_attempts=1,
            )
            runner = MedicationReviewRunner(settings, "fake-key")

            record = runner.run_one(
                Job("batch", 0, {"question": "病例", "answer": "金标准不能发送"}, settings.variants["vector"], 1)
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["retrieval_calls"], [])
        self.assertEqual(record["medication_review_trace"]["run_status"], "completed")
        self.assertNotIn("金标准不能发送", FakeYuxiClient.queries)

    def test_v2_partial_with_validated_answer_is_success_without_semantic_retry(
        self,
    ) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"pea": VariantConfig("pea", 1, "处方知识库")},
                max_attempts=2,
            )
            runner = MedicationReviewV2PartialRunner(settings, "fake-key")
            record = runner.run_one(
                Job(
                    "batch",
                    0,
                    {"question": "病例"},
                    settings.variants["pea"],
                    1,
                )
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["review_status"], "partial")
        self.assertEqual(record["method_version"], "pea-rag-mfull-vector-v1")
        self.assertEqual(record["budget_usage"]["logical_search_count"], 1)
        self.assertEqual(len(FakeYuxiClient.threads), 1)

    def test_v3_full_requires_final_review_and_all_six_sections(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"full": VariantConfig("full", 1, "处方知识库")},
                max_attempts=1,
            )
            client = MedicationReviewV3Client()
            record = MedicationReviewV3Runner(settings, "fake-key", client).run_one(
                Job("batch", 0, {"question": "病例"}, settings.variants["full"], 1)
            )
            client.include_final_review = False
            invalid = MedicationReviewV3Runner(settings, "fake-key", client).run_one(
                Job("batch", 1, {"question": "病例2"}, settings.variants["full"], 1)
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "3.0")
        self.assertEqual(record["synthesis_mode"], "claims")
        self.assertEqual(invalid["result_status"], "failed")

    def test_v4_partial_accepts_nonempty_answer_without_old_validation(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "m3",
                1,
                "处方知识库",
                expected_method_family="pat-rag-v1",
                expected_experiment_profile="m3",
                expected_trace_schema_version="4.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewLiteAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"m3": variant},
                max_attempts=1,
            )
            client = MedicationReviewV4Client()
            record = MedicationReviewV4Runner(
                settings,
                "fake-key",
                client,
            ).run_one(Job("batch", 0, {"question": "病例"}, variant, 1))

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["review_status"], "partial")
        self.assertEqual(record["trace_schema_version"], "4.0")
        self.assertEqual(record["method_family"], "pat-rag-v1")
        self.assertEqual(record["experiment_profile"], "m3")
        self.assertEqual(
            record["budget_usage"]["pat_rag"]["executed_search_calls"],
            1,
        )

    def test_v5_partial_accepts_answer_and_requested_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "m3",
                1,
                "处方知识库",
                expected_method_family="prim-rag-v1",
                expected_experiment_profile="m3",
                expected_trace_schema_version="5.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"m3": variant},
                max_attempts=1,
            )
            record = MedicationReviewV5Runner(
                settings,
                "fake-key",
                MedicationReviewV5Client(),
            ).run_one(Job("batch", 0, {"question": "病例"}, variant, 1))

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "5.0")
        self.assertEqual(record["method_family"], "prim-rag-v1")
        self.assertEqual(record["experiment_profile"], "m3")
        self.assertEqual(
            record["budget_usage"]["prim_rag"]["executed_search_calls"],
            1,
        )

    def test_v6_accepts_atlas_profile_and_exports_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "route",
                1,
                "处方知识库",
                expected_method_family="da-prim-rag-v1",
                expected_experiment_profile="full",
                expected_atlas_profile="route",
                expected_trace_schema_version="6.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewDaPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"route": variant},
                max_attempts=1,
            )
            record = MedicationReviewV6Runner(
                settings,
                "fake-key",
                MedicationReviewV6Client(),
            ).run_one(Job("batch", 0, {"question": "病例"}, variant, 1))

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "6.0")
        self.assertEqual(record["method_family"], "da-prim-rag-v1")
        self.assertEqual(record["experiment_profile"], "full")
        self.assertEqual(record["atlas_profile"], "route")
        self.assertEqual(record["atlas_snapshot_hash"], "atlas-1")
        self.assertEqual(
            record["budget_usage"]["da_prim"]["executed_search_calls"],
            1,
        )

    def test_v7_requires_successful_or_empty_companion_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "acm",
                1,
                "处方知识库",
                expected_method_family="acm-prim-rag-v1",
                expected_experiment_profile="full",
                expected_effective_profile="full",
                expected_trace_schema_version="7.0",
                require_companion_selector=True,
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm": variant},
                max_attempts=1,
            )
            runner = MedicationReviewV7Runner(
                settings,
                "fake-key",
                MedicationReviewV7Client(),
            )
            runner._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            success = runner.run_one(
                Job("batch", 0, {"question": "病例"}, variant, 1)
            )
            empty_client = MedicationReviewV7Client()
            empty_client.selector_status = "empty"
            empty = MedicationReviewV7Runner(
                settings,
                "fake-key",
                empty_client,
            ).run_one(Job("batch", 1, {"question": "病例2"}, variant, 1))
            failed_client = MedicationReviewV7Client()
            failed_client.selector_status = "failed"
            failed = MedicationReviewV7Runner(
                settings,
                "fake-key",
                failed_client,
            ).run_one(Job("batch", 2, {"question": "病例3"}, variant, 1))
            degraded_client = MedicationReviewV7Client()
            degraded_client.effective_profile = "m1"
            degraded = MedicationReviewV7Runner(
                settings,
                "fake-key",
                degraded_client,
            ).run_one(Job("batch", 3, {"question": "病例4"}, variant, 1))

        self.assertEqual(success["result_status"], "success")
        self.assertEqual(success["companion_selector_status"], "success")
        self.assertEqual(
            success["budget_usage"]["acm_prim"]["selector"]["cue_count"],
            1,
        )
        self.assertEqual(empty["result_status"], "success")
        self.assertEqual(failed["result_status"], "failed")
        self.assertEqual(failed["error"]["type"], "medication_review_trace_mismatch")
        self.assertEqual(degraded["result_status"], "failed")
        self.assertIn("effective_profile", degraded["error"]["message"])

    def test_v7_runtime_fingerprint_rejects_changed_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm": VariantConfig("acm", 1)},
            )
            runner = BatchRunner(settings, "fake-key")
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            first = {
                "variant": "acm",
                "medication_review_trace": {
                    "schema_version": "7.0",
                    "atlas_snapshot": {"snapshot_hash": "atlas-1"},
                    "prompt_hashes": {
                        "companion_selector": "prompt-1"
                    },
                },
            }
            self.assertIsNone(runner.validate_runtime_fingerprint(first))
            changed = json.loads(json.dumps(first))
            changed["medication_review_trace"]["atlas_snapshot"][
                "snapshot_hash"
            ] = "atlas-2"
            error = runner.validate_runtime_fingerprint(changed)

        self.assertEqual(error["type"], "runtime_fingerprint_changed")

    def test_v8_accepts_prim_v2_trace_and_exports_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "full",
                1,
                "处方知识库",
                expected_method_family="prim-rag-v2",
                expected_method_version="prim-rag-v2-full-vector-top10",
                expected_experiment_profile="full",
                expected_effective_profile="full",
                expected_trace_schema_version="8.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"full": variant},
                max_attempts=1,
            )
            record = MedicationReviewV8Runner(
                settings,
                "fake-key",
                MedicationReviewV8Client(),
            ).run_one(Job("batch", 0, {"question": "病例"}, variant, 1))

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "8.0")
        self.assertEqual(record["method_family"], "prim-rag-v2")
        self.assertEqual(
            record["method_version"],
            "prim-rag-v2-full-vector-top10",
        )
        self.assertEqual(record["experiment_profile"], "full")
        self.assertEqual(
            record["budget_usage"]["prim_rag"]["executed_search_calls"],
            1,
        )

    def test_v9_accepts_acm_v2_trace_and_freezes_runtime_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "acm",
                1,
                "处方知识库",
                expected_method_family="acm-prim-rag-v2",
                expected_method_version="acm-prim-rag-v2-vector-top10",
                expected_effective_profile="full",
                expected_trace_schema_version="9.0",
                require_companion_selector=True,
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm": variant},
                max_attempts=1,
            )
            runner = MedicationReviewV9Runner(
                settings,
                "fake-key",
                MedicationReviewV9Client(),
            )
            runner._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            record = runner.run_one(
                Job("batch", 0, {"question": "病例"}, variant, 1)
            )
            manifest = json.loads(
                runner._manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "9.0")
        self.assertEqual(record["method_family"], "acm-prim-rag-v2")
        self.assertEqual(
            record["method_version"],
            "acm-prim-rag-v2-vector-top10",
        )
        self.assertEqual(record["companion_selector_status"], "success")
        self.assertEqual(
            record["budget_usage"]["acm_prim"]["agent"][
                "executed_search_calls"
            ],
            1,
        )
        self.assertEqual(
            manifest["runtime_fingerprints"]["acm"]["atlas_snapshot_hash"],
            "atlas-acm-1",
        )

    def test_v10_accepts_navigation_trace_and_freezes_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "acm",
                1,
                "处方知识库",
                expected_method_family="acm-prim-rag-v3",
                expected_method_version=(
                    "acm-prim-rag-v3-atlas-navigation-vector-top10"
                ),
                expected_effective_profile="full",
                expected_trace_schema_version="10.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm": variant},
                max_attempts=1,
            )
            runner = MedicationReviewV10Runner(
                settings,
                "fake-key",
                MedicationReviewV10Client(),
            )
            runner._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            record = runner.run_one(
                Job("batch", 0, {"question": "病例"}, variant, 1)
            )
            manifest = json.loads(
                runner._manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "10.0")
        self.assertEqual(
            record["atlas_navigation_prompt_hash"],
            "navigation-prompt-1",
        )
        self.assertNotIn("selector", record["budget_usage"]["acm_prim"])
        self.assertEqual(
            record["budget_usage"]["acm_prim"]["atlas_navigation"][
                "opened_document_count"
            ],
            0,
        )
        self.assertEqual(
            manifest["runtime_fingerprints"]["acm"][
                "atlas_navigation_prompt_hash"
            ],
            "navigation-prompt-1",
        )

    def test_v11_accepts_v7_trace_and_records_contract_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "acm-v7",
                1,
                "处方知识库",
                expected_method_family="acm-prim-rag-v7",
                expected_method_version=(
                    "acm-prim-rag-v7-a2_k2-shadow_top25-vector"
                ),
                expected_effective_profile="full",
                expected_v7_experiment_arm="a2_k2",
                expected_v7_retrieval_depth="shadow_top25",
                expected_max_search_calls=50,
                expected_trace_schema_version="11.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm-v7": variant},
                max_attempts=1,
            )
            runner = MedicationReviewV11Runner(
                settings,
                "fake-key",
                MedicationReviewV11Client(),
            )
            runner._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            record = runner.run_one(
                Job("batch", 0, {"question": "病例"}, variant, 1)
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "11.0")
        self.assertEqual(record["v7_experiment_arm"], "a2_k2")
        self.assertEqual(record["v7_retrieval_depth"], "shadow_top25")
        self.assertEqual(record["v7_contract_status"], "completed")

    def test_v12_accepts_adaptive_trace_and_records_coverage_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = VariantConfig(
                "acm-adaptive",
                1,
                "处方知识库",
                expected_method_family="acm-prim-rag-v8",
                expected_method_version=(
                    "acm-prim-rag-v8-adaptive-coverage-shadow_top25-vector"
                ),
                expected_effective_profile="full",
                expected_acm_protocol="adaptive_coverage",
                expected_v7_retrieval_depth="shadow_top25",
                expected_max_search_calls=50,
                expected_trace_schema_version="12.0",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAcmPrimAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"acm-adaptive": variant},
                max_attempts=1,
            )
            runner = MedicationReviewV12Runner(
                settings,
                "fake-key",
                MedicationReviewV12Client(),
            )
            runner._manifest_path.parent.mkdir(parents=True, exist_ok=True)
            runner._manifest_path.write_text(
                json.dumps({"runtime_fingerprints": {}}),
                encoding="utf-8",
            )
            record = runner.run_one(
                Job("batch", 0, {"question": "病例"}, variant, 1)
            )

        self.assertEqual(record["result_status"], "success")
        self.assertEqual(record["trace_schema_version"], "12.0")
        self.assertEqual(record["acm_protocol"], "adaptive_coverage")
        self.assertEqual(record["v7_retrieval_depth"], "shadow_top25")
        self.assertEqual(record["adaptive_coverage_status"], "completed")
        self.assertEqual(record["adaptive_investigation_count"], 4)
        self.assertEqual(record["adaptive_pending_recovery_ids"], [])
        self.assertEqual(record["adaptive_gap_assessment_status"], "current")

    def test_v3_debug_stop_is_accepted_only_when_explicitly_enabled(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = dict(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="MedicationReviewAgent",
                input_file=root / "input.json",
                output_dir=root / "output",
                variants={"p1": VariantConfig("p1", 1, "处方知识库")},
                max_attempts=1,
            )
            client = MedicationReviewV3Client()
            client.run_status = "debug_stopped"
            accepted = MedicationReviewV3Runner(
                BatchSettings(**base, allow_debug_stopped=True),
                "fake-key",
                client,
            ).run_one(Job("batch", 0, {"question": "病例"}, base["variants"]["p1"], 1))
            rejected = MedicationReviewV3Runner(
                BatchSettings(**base, allow_debug_stopped=False),
                "fake-key",
                client,
            ).run_one(Job("batch", 1, {"question": "病例2"}, base["variants"]["p1"], 1))

        self.assertEqual(accepted["result_status"], "success")
        self.assertEqual(rejected["result_status"], "failed")

    def test_runner_writes_results_and_resumes_successes(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps([{"question": "问题一"}, {"question": "问题二"}], ensure_ascii=False),
                encoding="utf-8",
            )
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="ChatbotAgent",
                input_file=input_path,
                output_dir=root / "output",
                variants={"vector": VariantConfig("vector", 1, "向量库")},
                concurrency=2,
                max_attempts=1,
            )
            runner = FakeRunner(settings, "fake-key")
            first_counts = runner.run()
            second_counts = runner.run()

            self.assertEqual(first_counts["success"], 2)
            self.assertEqual(second_counts["scheduled"], 0)
            self.assertEqual(second_counts["skipped"], 2)
            result_lines = (root / "output" / "results" / "vector.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(result_lines), 2)
            self.assertEqual(len(set(FakeYuxiClient.threads)), 2)

    def test_failed_run_retry_uses_a_new_thread(self) -> None:
        FakeYuxiClient.queries = []
        FakeYuxiClient.threads = []
        FakeYuxiClient.next_thread = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text("[]", encoding="utf-8")
            settings = BatchSettings(
                config_path=root / "config.json",
                base_url="https://example.invalid",
                auth_mode="api_key",
                api_key_env="YUXI_API_KEY",
                login_id_env=None,
                password_env=None,
                agent_id="ChatbotAgent",
                input_file=input_path,
                output_dir=root / "output",
                variants={"vector": VariantConfig("vector", 1, "向量库")},
                max_attempts=2,
            )
            runner = RetryRunner(settings, "fake-key")
            record = runner.run_one(Job("batch", 0, {"question": "问题"}, settings.variants["vector"], 1))

            self.assertEqual(record["result_status"], "success")
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(len(FakeYuxiClient.threads), 2)
            self.assertNotEqual(FakeYuxiClient.threads[0], FakeYuxiClient.threads[1])
