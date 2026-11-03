"""End-to-end ontology execution tests without LLM or external ML packages."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ontology_v3 import graph_nodes
from ontology_v3.service import OntologyService


def run(coro):
    return asyncio.run(coro)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = OntologyService(Path(self.tmp.name) / "ops.sqlite")
        self.store.seed_demo()
        self.patcher = patch.object(graph_nodes, "_store", lambda: self.store)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def pipeline(self, query, risk=0.82):
        request = graph_nodes.detect_ontology_request(query)
        self.assertIsNotNone(request)
        state = {"ontology_equipment_id": request["equipment_id"],
                 "ontology_action_requested": request["action_requested"]}
        async def flow():
            for node in (graph_nodes.ontology_lookup_node, graph_nodes.ontology_predict_node, graph_nodes.ontology_policy_node):
                state.update(await node(state))
            if graph_nodes.route_after_ontology_policy(state) == "propose":
                state.update(await graph_nodes.ontology_propose_node(state))
            state.update(await graph_nodes.ontology_respond_node(state))
            state.update(await graph_nodes.ontology_verify_node(state))
        with patch.object(graph_nodes, "predict_observation", return_value={"failure_probability": risk, "risk_level": "HIGH"}):
            run(flow())
        return state

    def test_multi_step_risk_to_pending_to_human_approval(self):
        result = self.pipeline("EQ-001 설비 고장 위험을 분석하고 점검 작업을 제안해줘")
        self.assertTrue(result["validation_passed"])
        self.assertEqual(len(result["ontology_predictions"]), 2)
        self.assertEqual(len(result["ontology_trace"]), 5)
        request = result["ontology_proposal"]
        self.assertEqual(request["status"], "PENDING")
        self.assertEqual(self.store.equipment_context("EQ-001")["maintenance_tasks"], [])
        done = self.store.approve(request["id"], "demo:manager-1")
        self.assertEqual(done["status"], "APPROVED")
        self.assertEqual(len(OntologyService(self.store.path).equipment_context("EQ-001")["maintenance_tasks"]), 1)

    def test_risk_analysis_only_never_writes(self):
        state = self.pipeline("EQ-001 설비 고장 위험 알려줘", risk=0.99)
        self.assertFalse(state["ontology_proposal"] if "ontology_proposal" in state else False)
        self.assertEqual(self.store.list_requests(), [])

    def test_low_risk_policy_refuses_proposal(self):
        state = self.pipeline("EQ-002 점검 작업 생성 요청", risk=0.13)
        self.assertFalse(state.get("ontology_proposal"))
        self.assertEqual(self.store.list_requests(), [])
        self.assertTrue(state["validation_passed"])

    def test_missing_model_fails_closed(self):
        with patch.object(graph_nodes, "predict_observation", side_effect=FileNotFoundError("model missing")):
            req = graph_nodes.detect_ontology_request("EQ-001 설비 점검 생성해줘")
            state = {"ontology_equipment_id":req["equipment_id"], "ontology_action_requested": req["action_requested"]}
            for node in (graph_nodes.ontology_lookup_node, graph_nodes.ontology_predict_node, graph_nodes.ontology_policy_node, graph_nodes.ontology_respond_node, graph_nodes.ontology_verify_node):
                state.update(run(node(state)))
        self.assertFalse(state["validation_passed"])
        self.assertEqual(self.store.list_requests(), [])
        self.assertIn("MODEL_UNAVAILABLE", state["answer"])

    def test_non_ontology_queries_are_unchanged(self):
        self.assertIsNone(graph_nodes.detect_ontology_request("Product Type L의 고장 확률 예측해줘"))
        self.assertIsNone(graph_nodes.detect_ontology_request("AI4I HDF 정의는 무엇이야?"))
        self.assertIsNone(graph_nodes.detect_ontology_request("EQ-001"))

    def test_unknown_equipment_fails_closed(self):
        state = self.pipeline("EQ-999 설비 위험 분석 후 점검 제안해줘")
        self.assertFalse(state["validation_passed"])
        self.assertEqual(self.store.list_requests(), [])


if __name__ == "__main__":
    unittest.main()
