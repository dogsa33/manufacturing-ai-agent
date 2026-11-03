"""No APIs, models, network, or optional packages required."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from ontology_v3.service import OntologyService, RuleViolation


class OntologyV3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "operations.sqlite"
        self.store = OntologyService(self.db)
        self.assertTrue(self.store.seed_demo())

    def tearDown(self):
        self.tmp.cleanup()

    def test_idempotent_seed_and_link_graph(self):
        self.assertFalse(self.store.seed_demo())
        equipment = self.store.equipment_context("EQ-001")
        self.assertEqual(len(equipment["lots"]), 2)
        self.assertEqual(len(equipment["observations"]), 2)
        self.assertEqual(equipment["data_origin"], "SYNTHETIC_DEMO")
        self.assertEqual(self.store.traverse("ProductionLot", "LOT-001", "producedOn", reverse=True)[0]["id"], "EQ-001")

    def test_proposal_requires_human_approval_and_persists(self):
        req = self.store.propose_maintenance("EQ-001", "Investigate high vibration event")
        self.assertEqual(req["status"], "PENDING")
        self.assertEqual(self.store.equipment_context("EQ-001")["maintenance_tasks"], [])
        reopened = OntologyService(self.db)
        result = reopened.approve(req["id"], "demo:manager-1")
        self.assertEqual(result["status"], "APPROVED")
        self.assertEqual(len(reopened.equipment_context("EQ-001")["maintenance_tasks"]), 1)
        self.assertTrue(OntologyService(self.db).get_object("MaintenanceTask", result["task_id"]))
        self.assertTrue(any(a["code"] == "TASK_CREATED" for a in reopened.audit_history()))

    def test_duplicate_proposals_and_approvals_are_idempotent(self):
        req = self.store.propose_maintenance("EQ-001", "Inspect condition and monitor wear")
        dup = self.store.propose_maintenance("EQ-001", "Inspect condition and monitor wear")
        self.assertEqual(req["id"], dup["id"])
        a = self.store.approve(req["id"], "demo:manager-1")
        b = self.store.approve(req["id"], "demo:manager-1")
        self.assertEqual(a["task_id"], b["task_id"])
        self.assertTrue(b["idempotent"])
        with self.assertRaises(RuleViolation) as ctx:
            self.store.propose_maintenance("EQ-001", "Propose another identical task")
        self.assertEqual(ctx.exception.code, "EXISTING_TASK")

    def test_inactive_and_missing_equipment_blocked(self):
        for equipment_id, expected in [("EQ-003", "INACTIVE_EQUIPMENT"), ("EQ-XXX", "NOT_FOUND")]:
            with self.assertRaises(RuleViolation) as ctx:
                self.store.propose_maintenance(equipment_id, "Inspect equipment before continuing")
            self.assertEqual(ctx.exception.code, expected)

    def test_no_unauthorized_approval(self):
        req = self.store.propose_maintenance("EQ-002", "Inspection task for trial review")
        with self.assertRaises(RuleViolation) as ctx:
            self.store.approve(req["id"], "agent")
        self.assertEqual(ctx.exception.code, "UNAUTHORIZED_ROLE")
        self.assertEqual(self.store.list_requests("PENDING")[0]["id"], req["id"])
        self.assertTrue(any(a["code"] == "UNAUTHORIZED_ROLE" for a in self.store.audit_history()))

    def test_human_rejection_no_task_and_cannot_approve(self):
        req = self.store.propose_maintenance("EQ-002", "Inspection task for trial review")
        self.store.reject(req["id"], "demo:manager-1", "Not needed")
        self.assertEqual(self.store.equipment_context("EQ-002")["maintenance_tasks"], [])
        with self.assertRaises(RuleViolation) as ctx:
            self.store.approve(req["id"], "demo:manager-1")
        self.assertEqual(ctx.exception.code, "NOT_PENDING")

    def test_invalid_link_and_direction(self):
        with self.assertRaises(RuleViolation):
            self.store.traverse("Equipment", "EQ-001", "hasObservation")
        with self.assertRaises(RuleViolation):
            self.store.traverse("Equipment", "EQ-001", "madeUpLink")


if __name__ == "__main__":
    unittest.main()
