import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "yuxi_batch_rag" / "export_rag_records.py"
SPEC = importlib.util.spec_from_file_location("yuxi_batch_rag_export", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class YuxiBatchRagExportTests(unittest.TestCase):
    def test_exports_vector_retrieval_and_reference_documents(self):
        source = {
            "job_key": "vector:000000",
            "row_index": 0,
            "variant": "vector",
            "question": "问题一",
            "answer": "回答一",
            "retrieval_status": "called",
            "input_record": {"case_id": "case-1", "documents": ["doc-a"]},
            "retrieval_calls": [
                {
                    "message_id": 10,
                    "tool_call_id": "call-1",
                    "tool_name": "query_kb",
                    "args": {"kb_name": "向量库", "query_text": "改写问题一"},
                    "status": "success",
                    "result_raw": '[{"content":"chunk-a"}]',
                    "result_parsed": [
                        {
                            "content": "chunk-a",
                            "metadata": {"file_id": "file-a", "chunk_id": "chunk-a"},
                            "score": 0.91,
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")

            count, call_count, item_count = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual((count, call_count, item_count), (1, 1, 1))
            self.assertEqual(exported[0]["question"], "问题一")
            self.assertEqual(exported[0]["response"], "回答一")
            self.assertEqual(exported[0]["reference_documents"], ["doc-a"])
            call = exported[0]["retrieval_calls"][0]
            self.assertEqual(call["call_index"], 1)
            self.assertEqual(call["query_text"], "改写问题一")
            self.assertEqual(call["retrieved_items"][0]["metadata"]["chunk_id"], "chunk-a")
            self.assertEqual(call["retrieval_result"][0]["score"], 0.91)

    def test_exports_lightrag_chunks_and_preserves_graph_result(self):
        graph_result = {
            "entities": [{"entity_name": "实体一"}],
            "relationships": [{"src_id": "a", "tgt_id": "b"}],
            "references": ["ref-1"],
            "chunks": [{"content": "chunk-1", "chunk_id": "c-1"}],
        }
        source = {
            "question": "问题二",
            "answer": "回答二",
            "retrieval_calls": [
                {
                    "args": {"kb_name": "图谱库", "query_text": "问题二"},
                    "result_parsed": graph_result,
                    "result_raw": json.dumps(graph_result, ensure_ascii=False),
                    "status": "success",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")

            MODULE.export_rag_records(input_path, output_path)
            call = json.loads(output_path.read_text(encoding="utf-8"))[0]["retrieval_calls"][0]

            self.assertEqual(call["retrieved_items"], graph_result["chunks"])
            self.assertEqual(call["retrieval_result"]["entities"], graph_result["entities"])
            self.assertEqual(call["retrieval_result"]["relationships"], graph_result["relationships"])
            self.assertEqual(call["retrieval_result"]["references"], graph_result["references"])

    def test_exports_search_and_original_document_open_in_execution_order(self):
        search_call = {
            "message_id": 10,
            "tool_call_id": "search-1",
            "tool_name": "query_kb",
            "args": {"kb_name": "向量库", "query_text": "检索问题"},
            "status": "success",
            "result_parsed": [
                {
                    "content": "召回片段",
                    "metadata": {"file_id": "file-1", "chunk_id": "chunk-1"},
                }
            ],
        }
        open_call = {
            "message_id": 11,
            "tool_call_id": "open-1",
            "tool_name": "open_kb_document",
            "args": {"resource_id": "db-1", "file_id": "file-1", "line": 20},
            "status": "success",
            "result_parsed": {
                "resource_id": "db-1",
                "file_id": "file-1",
                "start_line": 20,
                "end_line": 30,
                "content": "    20\t原文内容",
            },
        }
        source = {
            "question": "问题",
            "answer": "回答",
            "retrieval_calls": [search_call],
            "document_open_calls": [open_call],
            "all_tool_calls": [search_call, open_call],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 2, 2))
        self.assertEqual(
            [call["tool_name"] for call in exported["retrieval_calls"]],
            ["query_kb", "open_kb_document"],
        )
        self.assertEqual(len(exported["search_calls"]), 1)
        self.assertEqual(len(exported["document_open_calls"]), 1)
        self.assertEqual(
            exported["opened_evidence"][0]["metadata"]["file_id"],
            "file-1",
        )
        self.assertEqual(
            exported["opened_evidence"][0]["metadata"]["start_line"],
            20,
        )

    def test_exports_pea_rag_v2_search_records_from_trace(self):
        source = {
            "question": "病例问题",
            "answer": "六段式回答",
            "retrieval_status": "called",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "2.0",
                "knowledge_base_snapshot": {"name": "处方知识库"},
                "search_records": [
                    {
                        "intent": {
                            "query_id": "Q001",
                            "query_text": "老年患者使用异烟肼是否适宜",
                            "target_element_ids": ["PE001"],
                            "target_review_ids": ["RT001"],
                            "intended_evidence_role": "support",
                            "search_reason": "核验适应证",
                        },
                        "status": "success",
                        "candidate_evidence_ids": ["EV-1"],
                        "active_evidence_ids": ["EV-1"],
                        "new_requirements_closed": ["PE001:indication"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "EV-1",
                        "raw_text": "直接证据",
                        "source_document": "共识.md",
                        "occurrences": [{"query_id": "Q001", "rank": 1}],
                    }
                ],
                "evidence_assessments": [
                    {
                        "evidence_id": "EV-1",
                        "bindings": [{"element_id": "PE001", "relevance": "direct"}],
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            call = json.loads(output_path.read_text(encoding="utf-8"))[0]["retrieval_calls"][0]

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(call["tool_name"], "search_evidence")
            self.assertEqual(call["query_text"], "老年患者使用异烟肼是否适宜")
            self.assertEqual(call["retrieved_items"][0]["evidence_id"], "EV-1")
            self.assertTrue(call["retrieved_items"][0]["active"])
            self.assertEqual(
                call["retrieved_items"][0]["assessment"]["evidence_id"],
                "EV-1",
            )

    def test_exports_trace_v3_selection_rank_and_claims(self):
        source = {
            "question": "病例问题",
            "answer": "六段式回答",
            "trace_schema_version": "3.0",
            "run_mode": "full",
            "agenda_mode": "dynamic",
            "synthesis_mode": "claims",
            "effective_profile": "pea-rag-v2-dynamic-claims-vector-v1",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "3.0",
                "knowledge_base_snapshot": {"name": "处方知识库"},
                "search_records": [
                    {
                        "subquery": {
                            "query_id": "Q001",
                            "query_text": "该患者条件下该方案的剂量是否合适？",
                            "linked_element_ids": ["PE001"],
                            "search_reason": "核验剂量",
                        },
                        "status": "success",
                        "evidence_ids": ["EV001"],
                        "new_evidence_ids": ["EV001"],
                        "duplicate_ratio": 0,
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "EV001",
                        "content_hash": "hash-1",
                        "raw_text": "直接来源证据",
                        "source_document": "共识.md",
                        "occurrences": [{"query_id": "Q001", "rank": 2}],
                    }
                ],
                "evidence_selection": {"selected_evidence_ids": ["EV001"]},
                "evidence_claims": [
                    {
                        "claim_id": "CL001",
                        "evidence_id": "EV001",
                        "source_span": "直接来源证据",
                        "statement": "来源陈述",
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]
            item = exported["retrieval_calls"][0]["retrieved_items"][0]

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(item["rank"], 2)
            self.assertTrue(item["selected"])
            self.assertEqual(item["claims"][0]["claim_id"], "CL001")
            self.assertEqual(exported["synthesis_mode"], "claims")

    def test_exports_pat_rag_trace_v4_full_evidence_and_shown_excerpt(self):
        evidence_id = "EV-0123456789ABCDEF"
        source = {
            "question": "病例问题",
            "answer": f"逐项回答，依据：[{evidence_id}]",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "4.0",
                "method_family": "pat-rag-v1",
                "method_version": "pat-rag-v1-m3-vector-top5",
                "experiment_profile": "m3",
                "run_status": "partial",
                "knowledge_base_snapshot": {"name": "处方知识库"},
                "plan_anchors": [{"element_id": "PE001"}],
                "search_records": [
                    {
                        "record_id": "SEARCH-1",
                        "tool_call_id": "call-1",
                        "query_text": "该方案在当前患者条件下是否适用？",
                        "reason": "核验适用性",
                        "focus_element_ids": ["PE001"],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [evidence_id],
                        "new_evidence_ids": [evidence_id],
                        "attempts": [],
                    }
                ],
                "open_records": [],
                "evidence_store": [
                    {
                        "evidence_id": evidence_id,
                        "content_hash": "hash-1",
                        "raw_text": "必须完整保留的原始片段",
                        "source_document": "共识.md",
                        "file_id": "file-1",
                        "chunk_id": "chunk-1",
                        "chunk_index": 7,
                        "occurrences": [
                            {
                                "record_id": "SEARCH-1",
                                "shown_excerpt": "展示给模型的局部窗口",
                                "excerpt_start": 20,
                                "excerpt_end": 32,
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
                "coverage_report": {"missing_after_patch": []},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]
            item = exported["retrieval_calls"][0]["retrieved_items"][0]

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(exported["method_family"], "pat-rag-v1")
            self.assertEqual(exported["experiment_profile"], "m3")
            self.assertEqual(exported["review_status"], "partial")
            self.assertEqual(item["raw_text"], "必须完整保留的原始片段")
            self.assertEqual(item["shown_excerpt"], "展示给模型的局部窗口")
            self.assertTrue(item["cited_in_answer"])
            self.assertEqual(
                exported["retrieved_evidence"][0]["raw_text"],
                "必须完整保留的原始片段",
            )

    def test_exports_prim_rag_trace_v5_query_and_relation_fields(self):
        evidence_id = "EV-0123456789ABCDEF"
        source = {
            "question": "病例问题",
            "answer": f"逐项回答，依据：[{evidence_id}]",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "5.0",
                "method_family": "prim-rag-v1",
                "method_version": "prim-rag-v1-m3-vector-top5",
                "requested_profile": "m3",
                "effective_profile": "m3",
                "run_status": "completed",
                "knowledge_base_snapshot": {"name": "处方知识库"},
                "plan_anchors": [{"element_id": "PE001"}],
                "patient_modifiers": [{"modifier_id": "PM001"}],
                "query_records": [
                    {
                        "query_id": "Q-ONE",
                        "tool_call_id": "call-1",
                        "relation_id": "RI-ONE",
                        "query_text": "当前患者条件是否改变方案甲？",
                        "reason": "调查适用性",
                        "focus_plan_ids": ["PE001"],
                        "focus_modifier_ids": ["PM001"],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [evidence_id],
                        "new_evidence_ids": [evidence_id],
                        "attempts": [],
                    }
                ],
                "relation_investigations": [
                    {
                        "relation_id": "RI-ONE",
                        "relation_question": "患者条件是否改变方案甲？",
                    }
                ],
                "open_records": [],
                "evidence_store": [
                    {
                        "evidence_id": evidence_id,
                        "raw_text": "完整原始片段",
                        "occurrences": [
                            {
                                "record_id": "Q-ONE",
                                "shown_excerpt": "局部窗口",
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
                "coverage_report": {},
                "reflection_report": {"enabled": False},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]
            call = exported["retrieval_calls"][0]

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(exported["experiment_profile"], "m3")
            self.assertEqual(call["query_id"], "Q-ONE")
            self.assertEqual(call["relation_id"], "RI-ONE")
            self.assertEqual(call["args"]["focus_modifier_ids"], ["PM001"])
            self.assertEqual(
                exported["relation_investigations"][0]["relation_id"],
                "RI-ONE",
            )

    def test_exports_da_prim_trace_v6_routing_fields(self):
        evidence_id = "EV-DA0123456789AB"
        routed = {
            "retrieval_record_id": "RR-ONE",
            "query_id": "Q-ONE",
            "opportunity_id": "OP-ONE",
            "strategy": "routed",
            "effective_documents": [{"file_id": "file-1", "rank": 1}],
            "global_candidates": [],
            "local_candidates": [],
            "fused_candidates": [],
        }
        source = {
            "question": "病例问题",
            "answer": f"逐项回答，依据：[{evidence_id}]",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "6.0",
                "method_family": "da-prim-rag-v1",
                "method_version": "da-prim-rag-v1-full-vector-top5",
                "requested_profile": "full",
                "effective_profile": "full",
                "atlas_profile": "full",
                "atlas_snapshot": {"snapshot_hash": "atlas-1"},
                "run_status": "completed",
                "plan_anchors": [{"element_id": "PE001"}],
                "patient_modifiers": [{"modifier_id": "PM001"}],
                "query_records": [
                    {
                        "query_id": "Q-ONE",
                        "tool_call_id": "call-1",
                        "query_text": "方案与患者条件",
                        "reason": "调查关系",
                        "focus_plan_ids": ["PE001"],
                        "focus_modifier_ids": ["PM001"],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [evidence_id],
                        "new_evidence_ids": [evidence_id],
                        "attempts": [],
                    }
                ],
                "case_route_record": {
                    "atlas_snapshot_hash": "atlas-1",
                    "ranked_documents": [{"file_id": "file-1", "rank": 1}],
                },
                "retrieval_opportunities": [{"opportunity_id": "OP-ONE", "file_id": "file-1"}],
                "adopted_opportunity_ids": ["OP-ONE"],
                "routed_retrieval_records": [routed],
                "relation_investigations": [],
                "open_records": [],
                "evidence_store": [
                    {
                        "evidence_id": evidence_id,
                        "raw_text": "完整原始片段",
                        "occurrences": [
                            {
                                "record_id": "Q-ONE",
                                "shown_excerpt": "局部窗口",
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
                "coverage_report": {},
                "reflection_report": {"enabled": True},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]
            call = exported["retrieval_calls"][0]

            self.assertEqual(counts, (1, 1, 1))
            self.assertEqual(exported["atlas_profile"], "full")
            self.assertEqual(exported["atlas_snapshot_hash"], "atlas-1")
            self.assertEqual(exported["adopted_opportunity_ids"], ["OP-ONE"])
            self.assertEqual(call["args"]["opportunity_id"], "OP-ONE")
            self.assertEqual(
                call["routed_retrieval"]["retrieval_record_id"],
                "RR-ONE",
            )

    def test_exports_acm_trace_v7_without_mixing_suggestions_into_retrieval(self):
        evidence_id = "EV-ACM0123456789"
        selection = {
            "trigger_type": "after_query",
            "created_after_query_id": "Q-ONE",
            "observed_query_ids": ["Q-ONE"],
            "companion_cues": [
                {
                    "companion_id": "AC01",
                    "question_hint": "是否还应调查监测要求？",
                    "linked_plan_ids": ["PE001"],
                    "linked_modifier_ids": [],
                    "atlas_cue_ids": ["AT-ONE"],
                    "suggested_doc_ids": ["suggested-file"],
                    "novelty_explanation": "首轮未涉及监测。",
                }
            ],
            "selector_audit": {
                "status": "success",
                "elapsed_ms": 10,
            },
        }
        source = {
            "question": "病例问题",
            "answer": f"逐项回答，依据：[{evidence_id}]",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "7.0",
                "method_family": "acm-prim-rag-v1",
                "method_version": "acm-prim-rag-v1-vector-top5",
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "atlas_snapshot": {"snapshot_hash": "atlas-acm-1"},
                "plan_anchors": [{"element_id": "PE001"}],
                "patient_modifiers": [],
                "query_records": [
                    {
                        "query_id": "Q-ONE",
                        "tool_call_id": "call-1",
                        "query_text": "患者条件下方案是否适用？",
                        "reason": "调查适用性",
                        "focus_plan_ids": ["PE001"],
                        "focus_modifier_ids": [],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [evidence_id],
                        "new_evidence_ids": [evidence_id],
                        "attempts": [],
                        "atlas_companion_ids": ["AC01"],
                        "rejected_atlas_companion_ids": ["AC02"],
                        "invalid_atlas_companion_ids": [],
                    }
                ],
                "relation_investigations": [],
                "open_records": [],
                "evidence_store": [
                    {
                        "evidence_id": evidence_id,
                        "raw_text": "真实检索片段",
                        "source_document": "真实文档.md",
                        "file_id": "retrieved-file",
                        "occurrences": [
                            {
                                "record_id": "Q-ONE",
                                "shown_excerpt": "真实检索片段",
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
                "coverage_report": {},
                "reflection_report": {
                    "enabled": True,
                    "triggered": True,
                    "trigger_reason": "remaining_companion_cues",
                },
                "companion_selection": selection,
                "companion_adoption_events": [
                    {
                        "query_id": "Q-ONE",
                        "tool_call_id": "call-1",
                        "requested_companion_ids": ["AC01"],
                        "accepted_companion_ids": ["AC01"],
                        "rejected_companion_ids": [],
                        "invalid_companion_ids": [],
                    }
                ],
                "remaining_companion_ids": [],
                "companion_cues_at_reflection": ["AC01"],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 1))
        self.assertEqual(
            exported["retrieval_calls"][0]["retrieved_items"][0]["file_id"],
            "retrieved-file",
        )
        self.assertEqual(
            exported["retrieval_calls"][0]["args"]["atlas_companion_ids"],
            ["AC01"],
        )
        self.assertEqual(
            exported["retrieval_calls"][0]["rejected_atlas_companion_ids"],
            ["AC02"],
        )
        self.assertEqual(
            exported["companion_cues"][0]["suggested_doc_ids"],
            ["suggested-file"],
        )
        self.assertEqual(exported["atlas_snapshot_hash"], "atlas-acm-1")
        self.assertNotEqual(
            exported["retrieval_calls"][0]["retrieved_items"][0]["file_id"],
            exported["companion_cues"][0]["suggested_doc_ids"][0],
        )

    def test_exports_prim_v8_investigation_and_document_scope(self):
        evidence_id = "EV-V8DOCUMENT0001"
        source = {
            "question": "病例问题",
            "answer": f"逐项回答 [{evidence_id}]",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "8.0",
                "method_family": "prim-rag-v2",
                "method_version": "prim-rag-v2-full-vector-top5",
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "query_records": [
                    {
                        "query_id": "Q-V8",
                        "tool_call_id": "call-v8",
                        "investigation_id": "INV-V8",
                        "query_text": "剂量 给药间隔",
                        "reason": "文档内补查",
                        "retrieval_scope": "document",
                        "file_id": "file-guideline",
                        "focus_plan_ids": ["PE001"],
                        "focus_modifier_ids": [],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [evidence_id],
                        "new_evidence_ids": [evidence_id],
                        "attempts": [],
                    }
                ],
                "investigations": [
                    {
                        "investigation_id": "INV-V8",
                        "question": "剂量是否需要调整？",
                        "status": "answered",
                        "selected_evidence_ids": [evidence_id],
                    }
                ],
                "deferred_knowledge_calls": [
                    {
                        "tool_call_id": "call-deferred",
                        "tool_name": "search_review_kb",
                        "reason": "same_model_turn",
                        "created_at": "2026-01-01T00:00:01Z",
                    }
                ],
                "evidence_store": [
                    {
                        "evidence_id": evidence_id,
                        "raw_text": "文档内目标片段",
                        "file_id": "file-guideline",
                        "occurrences": [
                            {
                                "record_id": "Q-V8",
                                "shown_excerpt": "文档内目标片段",
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "open_records": [],
                "cited_evidence_ids": [evidence_id],
                "coverage_report": {},
                "reflection_report": {"enabled": True},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 1))
        call = exported["retrieval_calls"][0]
        self.assertEqual(call["investigation_id"], "INV-V8")
        self.assertEqual(call["retrieval_scope"], "document")
        self.assertEqual(call["args"]["file_id"], "file-guideline")
        self.assertEqual(exported["investigations"][0]["status"], "answered")
        self.assertEqual(
            exported["deferred_knowledge_calls"][0]["reason"],
            "same_model_turn",
        )

    def test_exports_acm_v9_cue_to_investigation_chain(self):
        source = {
            "question": "病例问题",
            "answer": "有边界的回答",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "9.0",
                "method_family": "acm-prim-rag-v2",
                "method_version": "acm-prim-rag-v2-vector-top5",
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "atlas_snapshot": {"snapshot_hash": "atlas-v2"},
                "query_records": [
                    {
                        "query_id": "Q-V9",
                        "tool_call_id": "call-v9",
                        "investigation_id": "INV-V9",
                        "query_text": "监测 要求",
                        "reason": "采用 Atlas 线索",
                        "retrieval_scope": "document",
                        "file_id": "file-atlas",
                        "focus_plan_ids": ["PE001"],
                        "focus_modifier_ids": [],
                        "atlas_companion_ids": ["AC01"],
                        "rejected_atlas_companion_ids": [],
                        "invalid_atlas_companion_ids": [],
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success_empty",
                        "evidence_ids": [],
                        "new_evidence_ids": [],
                        "attempts": [],
                    }
                ],
                "investigations": [
                    {
                        "investigation_id": "INV-V9",
                        "question": "是否还应核验监测要求？",
                        "origin": "atlas",
                        "status": "open",
                        "atlas_companion_ids": ["AC01"],
                        "candidate_file_ids": ["file-atlas"],
                    }
                ],
                "evidence_store": [],
                "open_records": [],
                "coverage_report": {},
                "reflection_report": {"enabled": True},
                "companion_selection": {
                    "companion_cues": [],
                    "selector_audit": {"status": "empty"},
                },
                "companion_adoption_events": [
                    {
                        "query_id": "Q-V9",
                        "tool_call_id": "call-v9",
                        "investigation_id": "INV-V9",
                        "accepted_companion_ids": ["AC01"],
                    }
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 0))
        self.assertEqual(
            exported["retrieval_calls"][0]["args"]["atlas_companion_ids"],
            ["AC01"],
        )
        self.assertEqual(
            exported["companion_adoption_events"][0]["investigation_id"],
            "INV-V9",
        )
        self.assertEqual(exported["atlas_snapshot_hash"], "atlas-v2")

    def test_reports_invalid_json_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                '{"question":"问题一","answer":"回答一"}\nnot-json\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(MODULE.ExportError, "line 2"):
                MODULE.export_rag_records(input_path, output_path)

    def test_exports_acm_v10_navigation_without_companion_fields(self):
        source = {
            "question": "病例问题",
            "answer": "回答",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "10.0",
                "method_family": "acm-prim-rag-v3",
                "method_version": (
                    "acm-prim-rag-v3-atlas-navigation-vector-top10"
                ),
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "atlas_snapshot": {"snapshot_hash": "atlas-v3"},
                "atlas_document_open_records": [
                    {
                        "record_id": "AOPEN-1",
                        "tool_call_id": "open-1",
                        "doc_id": "file-1",
                        "title": "共识",
                        "reason": "查看方案",
                        "topic_count": 12,
                    }
                ],
                "query_records": [
                    {
                        "query_id": "Q-V10",
                        "tool_call_id": "search-1",
                        "investigation_id": "INV-1",
                        "query_text": "治疗方案 调整",
                        "reason": "核验地图提示",
                        "retrieval_scope": "document",
                        "file_id": "file-1",
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success_empty",
                        "evidence_ids": [],
                        "new_evidence_ids": [],
                        "attempts": [],
                    }
                ],
                "evidence_store": [],
                "open_records": [],
                "investigations": [],
                "coverage_report": {},
                "reflection_report": {},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 0))
        self.assertEqual(exported["atlas_snapshot_hash"], "atlas-v3")
        self.assertEqual(
            exported["atlas_document_open_records"][0]["doc_id"],
            "file-1",
        )
        call = exported["retrieval_calls"][0]
        self.assertEqual(call["file_id"], "file-1")
        self.assertNotIn("atlas_companion_ids", call["args"])

    def test_prefers_trace_open_evidence_without_duplicating_history_call(self):
        search_evidence_id = "EV-SEARCH"
        opened_evidence_id = "EV-OPENED"
        source = {
            "question": "病例问题",
            "answer": "回答",
            "retrieval_calls": [],
            "document_open_calls": [
                {
                    "tool_call_id": "database-tool-row-1",
                    "tool_name": "open_review_evidence",
                    "args": {"evidence_id": search_evidence_id},
                    "status": "success",
                    "result_raw": "已打开相邻原文",
                    "result_parsed": None,
                }
            ],
            "all_tool_calls": [
                {
                    "tool_call_id": "database-tool-row-1",
                    "tool_name": "open_review_evidence",
                    "args": {"evidence_id": search_evidence_id},
                    "status": "success",
                    "result_raw": "已打开相邻原文",
                    "result_parsed": None,
                }
            ],
            "medication_review_trace": {
                "schema_version": "10.0",
                "query_records": [
                    {
                        "query_id": "Q-1",
                        "tool_call_id": "provider-search-call-1",
                        "query_text": "检索问题",
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success",
                        "evidence_ids": [search_evidence_id],
                    }
                ],
                "open_records": [
                    {
                        "record_id": "OPEN-1",
                        "tool_call_id": "provider-open-call-1",
                        "parent_evidence_id": search_evidence_id,
                        "started_at": "2026-01-01T00:00:01Z",
                        "status": "success",
                        "evidence_ids": [opened_evidence_id],
                    }
                ],
                "evidence_store": [
                    {
                        "evidence_id": search_evidence_id,
                        "raw_text": "检索片段",
                        "occurrences": [{"record_id": "Q-1", "rank": 1}],
                    },
                    {
                        "evidence_id": opened_evidence_id,
                        "raw_text": "打开的原文片段",
                        "occurrences": [{"record_id": "OPEN-1", "rank": 1}],
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 2, 2))
        self.assertEqual(len(exported["document_open_calls"]), 1)
        self.assertEqual(
            exported["opened_evidence"][0]["raw_text"],
            "打开的原文片段",
        )

    def test_exports_acm_v11_contract_and_shadow_rankings(self):
        source = {
            "question": "病例问题",
            "answer": "回答",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "11.0",
                "method_family": "acm-prim-rag-v7",
                "method_version": (
                    "acm-prim-rag-v7-a2_k2-shadow_top25-vector"
                ),
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "experiment_arm": "a2_k2",
                "retrieval_depth": "shadow_top25",
                "contract_report": {
                    "status": "completed",
                    "successful_search_calls": 6,
                },
                "investigation_agenda": {
                    "agenda_id": "AGENDA-1",
                    "required_count": 2,
                },
                "probe_records": [
                    {
                        "query_id": "Q-V11",
                        "investigation_id": "INV-1",
                        "probe_pass": "initial_probe",
                        "status": "success",
                    }
                ],
                "retrieval_records": [
                    {
                        "record_id": "VR-Q-V11",
                        "query_id": "Q-V11",
                        "fetch_k": 25,
                        "visible_k": 10,
                        "candidates": [
                            {"rank": 11, "chunk_id": "chunk-11"}
                        ],
                    }
                ],
                "checkpoint_records": [],
                "atlas_snapshot": {"snapshot_hash": "atlas-v3"},
                "atlas_document_open_records": [],
                "query_records": [
                    {
                        "query_id": "Q-V11",
                        "tool_call_id": "search-1",
                        "investigation_id": "INV-1",
                        "query_text": "治疗方案 调整",
                        "reason": "初始探查",
                        "retrieval_scope": "global",
                        "started_at": "2026-01-01T00:00:00Z",
                        "status": "success_empty",
                        "evidence_ids": [],
                        "new_evidence_ids": [],
                        "attempts": [],
                    }
                ],
                "evidence_store": [],
                "open_records": [],
                "investigations": [],
                "coverage_report": {},
                "reflection_report": {},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 0))
        self.assertEqual(exported["v7_experiment_arm"], "a2_k2")
        self.assertEqual(
            exported["v7_contract_report"]["successful_search_calls"],
            6,
        )
        self.assertEqual(
            exported["v7_retrieval_records"][0]["candidates"][0][
                "chunk_id"
            ],
            "chunk-11",
        )

    def test_exports_acm_v12_adaptive_coverage_state(self):
        source = {
            "question": "病例问题",
            "answer": "回答",
            "retrieval_calls": [],
            "medication_review_trace": {
                "schema_version": "12.0",
                "method_family": "acm-prim-rag-v8",
                "method_version": (
                    "acm-prim-rag-v8-adaptive-coverage-shadow_top25-vector"
                ),
                "requested_profile": "full",
                "effective_profile": "full",
                "run_status": "completed",
                "protocol": "adaptive_coverage",
                "retrieval_depth": "shadow_top25",
                "adaptive_coverage_report": {
                    "status": "completed",
                    "actual_investigation_count": 4,
                    "executed_search_calls": 7,
                },
                "investigation_agenda": {
                    "agenda_id": "AGENDA-ACM-1",
                    "revision": 2,
                },
                "investigation_meta": [
                    {
                        "investigation_id": "INV-1",
                        "resolved_aspects": ["适用性"],
                    }
                ],
                "probe_records": [
                    {
                        "query_id": "Q-V12",
                        "investigation_id": "INV-1",
                        "retrieval_intent": "source_discovery",
                        "uncovered_aspect": "缺少替代治疗方案来源",
                        "route_key": "source_discovery:global",
                        "redundant": False,
                        "status": "success_empty",
                    }
                ],
                "recovery_requirements": [],
                "gap_assessments": [
                    {
                        "assessment_id": "GAP-1",
                        "material_gap_found": False,
                    }
                ],
                "retrieval_records": [],
                "checkpoint_records": [],
                "atlas_snapshot": {"snapshot_hash": "atlas-v3"},
                "atlas_document_open_records": [],
                "query_records": [
                    {
                        "query_id": "Q-V12",
                        "tool_call_id": "search-1",
                        "investigation_id": "INV-1",
                        "query_text": "治疗方案 调整",
                        "reason": "来源发现",
                        "retrieval_scope": "global",
                        "started_at": "2026-08-25T00:00:00Z",
                        "status": "success_empty",
                        "evidence_ids": [],
                        "new_evidence_ids": [],
                        "attempts": [],
                    }
                ],
                "evidence_store": [],
                "open_records": [],
                "investigations": [],
                "coverage_report": {},
                "reflection_report": {},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "results.jsonl"
            output_path = Path(directory) / "rag_records.json"
            input_path.write_text(
                json.dumps(source, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            counts = MODULE.export_rag_records(input_path, output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(counts, (1, 1, 0))
        self.assertEqual(exported["acm_protocol"], "adaptive_coverage")
        self.assertEqual(exported["adaptive_coverage_status"], "completed")
        self.assertEqual(
            exported["adaptive_coverage_report"]["actual_investigation_count"],
            4,
        )
        self.assertEqual(
            exported["adaptive_investigation_agenda"]["revision"],
            2,
        )
        self.assertEqual(
            exported["adaptive_probe_records"][0]["retrieval_intent"],
            "source_discovery",
        )
        self.assertEqual(
            exported["search_calls"][0]["retrieval_intent"],
            "source_discovery",
        )
        self.assertEqual(
            exported["search_calls"][0]["args"]["uncovered_aspect"],
            "缺少替代治疗方案来源",
        )
        self.assertEqual(
            exported["search_calls"][0]["route_key"],
            "source_discovery:global",
        )
