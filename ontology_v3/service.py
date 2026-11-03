"""Operational ontology runtime with SQLite transactions and auditable Actions.

The seed is synthetic, identities are DEMO ONLY, and there is no authentication.
Never expose approval/rejection as LLM-callable tools in a real system.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .schema import ACTION_TYPES, LINK_TYPES, OBJECT_TYPES, FUNCTION_TYPES, OBJECT_AUTHORITY

DEFAULT_DB = Path(__file__).resolve().parents[1] / "database" / "ontology_operations.sqlite"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RuleViolation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OntologyService:
    """Local prototype store; SQLite owns objects, links, requests and audit."""

    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    type TEXT NOT NULL,
                    id TEXT NOT NULL,
                    props TEXT NOT NULL,
                    PRIMARY KEY(type, id)
                );
                CREATE TABLE IF NOT EXISTS links (
                    name TEXT NOT NULL,
                    from_type TEXT NOT NULL,
                    from_id TEXT NOT NULL,
                    to_type TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    PRIMARY KEY(name, from_type, from_id, to_type, to_id),
                    FOREIGN KEY(from_type, from_id) REFERENCES objects(type,id),
                    FOREIGN KEY(to_type, to_id) REFERENCES objects(type,id)
                );
                CREATE TABLE IF NOT EXISTS action_requests (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('PENDING','APPROVED','REJECTED')),
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    review_note TEXT,
                    result_object_id TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    code TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_action_requests_target
                    ON action_requests(target_id, status);
            """)

    @staticmethod
    def _obj(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else {
            "type": row["type"], "id": row["id"],
            "properties": json.loads(row["props"]),
        }

    def _add_object(self, con: sqlite3.Connection, typ: str, oid: str,
                    props: dict[str, Any]) -> None:
        if typ not in OBJECT_TYPES:
            raise RuleViolation("UNKNOWN_TYPE", f"Unknown object type: {typ}")
        missing = set(OBJECT_TYPES[typ]) - set(props)
        if missing:
            raise RuleViolation("INVALID_PROPERTIES", f"Missing properties: {sorted(missing)}")
        if props.get("id") != oid:
            raise RuleViolation("INVALID_ID", "id property must match object id")
        con.execute("INSERT INTO objects(type,id,props) VALUES(?,?,?)",
                    (typ, oid, json.dumps(props, ensure_ascii=False, allow_nan=False)))

    def _add_link(self, con: sqlite3.Connection, name: str,
                  source_id: str, target_id: str) -> None:
        if name not in LINK_TYPES:
            raise RuleViolation("UNKNOWN_LINK", f"Unknown link: {name}")
        source_type, target_type = LINK_TYPES[name]
        con.execute("""INSERT INTO links(name,from_type,from_id,to_type,to_id)
                      VALUES(?,?,?,?,?)""",
                    (name, source_type, source_id, target_type, target_id))

    def seed_demo(self) -> bool:
        """Seed synthetic equipment-lot-observation relations, *never* real MES.

        Runs once per database. Observation values are illustrative, not AI4I
        source records. AI4I-trained predictor can be applied separately.
        """
        equipments = [
            ("EQ-001", "Press Alpha", "LINE-A", "ACTIVE"),
            ("EQ-002", "Press Beta", "LINE-B", "ACTIVE"),
            ("EQ-003", "Press Gamma", "LINE-C", "INACTIVE"),
        ]
        lots = [
            ("LOT-001", "EQ-001", "L", "2026-10-20"),
            ("LOT-002", "EQ-001", "M", "2026-10-21"),
            ("LOT-003", "EQ-002", "H", "2026-10-21"),
        ]
        observations = [
            ("OBS-001", "LOT-001", "L", 301.0, 310.5, 1300.0, 65.0, 200.0),
            ("OBS-002", "LOT-002", "M", 298.5, 309.5, 1450.0, 40.0, 80.0),
            ("OBS-003", "LOT-003", "H", 299.5, 310.2, 1500.0, 48.0, 110.0),
        ]
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT 1 FROM objects LIMIT 1").fetchone():
                return False
            for eid, name, line, status in equipments:
                self._add_object(con, "Equipment", eid,
                                 dict(id=eid, name=name, line=line, status=status))
            for lid, equipment_id, product_type, date in lots:
                self._add_object(con, "ProductionLot", lid,
                                 dict(id=lid, product_type=product_type, production_date=date))
                self._add_link(con, "producedOn", equipment_id, lid)
            for oid, lot_id, typ, air, proc, speed, torque, wear in observations:
                self._add_object(con, "MachineObservation", oid, dict(
                    id=oid, product_type=typ, air_temperature=air,
                    process_temperature=proc, rotational_speed=speed,
                    torque=torque, tool_wear=wear, observed_at="2026-10-22T00:00:00Z"))
                self._add_link(con, "hasObservation", lot_id, oid)
        return True

    def get_object(self, typ: str, oid: str) -> dict[str, Any] | None:
        if typ not in OBJECT_TYPES:
            raise RuleViolation("UNKNOWN_TYPE", f"Unknown object type: {typ}")
        with self._connect() as con:
            return self._obj(con.execute(
                "SELECT * FROM objects WHERE type=? AND id=?", (typ, oid)).fetchone())

    def list_objects(self, typ: str) -> list[dict[str, Any]]:
        if typ not in OBJECT_TYPES:
            raise RuleViolation("UNKNOWN_TYPE", f"Unknown object type: {typ}")
        with self._connect() as con:
            return [self._obj(row) for row in con.execute(
                "SELECT * FROM objects WHERE type=? ORDER BY id", (typ,))]

    def traverse(self, typ: str, oid: str, link_name: str,
                 reverse: bool = False) -> list[dict[str, Any]]:
        if link_name not in LINK_TYPES:
            raise RuleViolation("UNKNOWN_LINK", f"Unknown link: {link_name}")
        source_type, target_type = LINK_TYPES[link_name]
        expected = target_type if reverse else source_type
        if typ != expected:
            raise RuleViolation("INVALID_DIRECTION", f"{link_name} requires {expected}")
        if self.get_object(typ, oid) is None:
            raise RuleViolation("NOT_FOUND", f"{typ}:{oid} not found")
        column = "to_id" if reverse else "from_id"
        join_type = source_type if reverse else target_type
        join_column = "from_id" if reverse else "to_id"
        with self._connect() as con:
            query = ("SELECT o.* FROM links AS l JOIN objects AS o "
                     f"ON o.type=? AND o.id=l.{join_column} "
                     f"WHERE l.name=? AND l.{column}=? ORDER BY o.id")
            return [self._obj(row) for row in con.execute(
                query, (join_type, link_name, oid))]

    def equipment_context(self, equipment_id: str) -> dict[str, Any]:
        eq = self.get_object("Equipment", equipment_id)
        if not eq:
            raise RuleViolation("NOT_FOUND", "Equipment does not exist")
        lots = self.traverse("Equipment", equipment_id, "producedOn")
        observations = [obs for lot in lots for obs in self.traverse(
            "ProductionLot", lot["id"], "hasObservation")]
        tasks = self.traverse("Equipment", equipment_id, "hasMaintenanceTask")
        return {"equipment": eq, "lots": lots, "observations": observations,
                "maintenance_tasks": tasks, "data_origin": "SYNTHETIC_DEMO"}

    @staticmethod
    def _audit(con: sqlite3.Connection, request_id: str | None, action: str,
               actor: str, outcome: str, code: str, details: str) -> None:
        con.execute("""INSERT INTO audit_log
                    (request_id,action,actor,outcome,code,details,timestamp)
                    VALUES(?,?,?,?,?,?,?)""",
                    (request_id, action, actor, outcome, code, details, now_utc()))

    @staticmethod
    def _validate_reason(reason: str) -> str:
        reason = reason.strip()
        if not 10 <= len(reason) <= 500:
            raise RuleViolation("INVALID_REASON", "Reason length must be 10–500 chars")
        return reason

    def _validate_equipment(self, con: sqlite3.Connection,
                            equipment_id: str) -> None:
        row = con.execute("SELECT * FROM objects WHERE type='Equipment' AND id=?",
                          (equipment_id,)).fetchone()
        if not row:
            raise RuleViolation("NOT_FOUND", f"Equipment {equipment_id} not found")
        if json.loads(row["props"])["status"] != "ACTIVE":
            raise RuleViolation("INACTIVE_EQUIPMENT", "Equipment is inactive")
        open_task = con.execute("""SELECT o.id FROM links AS l JOIN objects AS o
            ON o.type='MaintenanceTask' AND o.id=l.to_id
            WHERE l.name='hasMaintenanceTask' AND l.from_id=?
                AND json_extract(o.props, '$.status') IN ('OPEN','IN_PROGRESS')
            LIMIT 1""", (equipment_id,)).fetchone()
        if open_task:
            raise RuleViolation("EXISTING_TASK", "An open maintenance task already exists")

    def propose_maintenance(self, equipment_id: str, reason: str,
                            requested_by: str = "agent") -> dict[str, Any]:
        """Only creates PENDING proposal; no maintenance task is created."""
        reason = self._validate_reason(reason)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._validate_equipment(con, equipment_id)
            row = con.execute("""SELECT * FROM action_requests
                WHERE target_id=? AND status='PENDING' LIMIT 1""", (equipment_id,)).fetchone()
            if row:
                return dict(row)  # idempotent per equipment while pending
            request_id = "REQ-" + uuid4().hex[:12].upper()
            con.execute("""INSERT INTO action_requests
                (id,action,target_id,reason,requested_by,status,created_at)
                VALUES(?,?,?,?,?,'PENDING',?)""",
                (request_id, "createMaintenanceTask", equipment_id, reason,
                 requested_by, now_utc()))
            self._audit(con, request_id, "createMaintenanceTask", requested_by,
                        "PROPOSED", "PENDING_REVIEW", reason)
            return dict(con.execute("SELECT * FROM action_requests WHERE id=?",
                                    (request_id,)).fetchone())

    def _audit_rejection(self, request_id: str, actor: str, code: str,
                         message: str) -> None:
        with self._connect() as con:
            self._audit(con, request_id, "createMaintenanceTask", actor,
                        "DENIED", code, message)

    def approve(self, request_id: str, reviewer: str) -> dict[str, Any]:
        """Called ONLY by UI/operator path. DEMO reviewer identity isn't verified."""
        reviewer = reviewer.strip()
        if not reviewer.startswith("demo:manager-"):
            self._audit_rejection(request_id, reviewer, "UNAUTHORIZED_ROLE", "Reviewer role denied")
            raise RuleViolation("UNAUTHORIZED_ROLE", "Only demo managers can approve")
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("SELECT * FROM action_requests WHERE id=?", (request_id,)).fetchone()
                if not row:
                    raise RuleViolation("REQUEST_NOT_FOUND", "Unknown request")
                if row["status"] == "APPROVED":
                    return {"request_id": request_id, "status": "APPROVED",
                            "task_id": row["result_object_id"], "idempotent": True}
                if row["status"] != "PENDING":
                    raise RuleViolation("NOT_PENDING", "Request is already rejected")
                self._validate_equipment(con, row["target_id"])
                task_id = "TASK-" + uuid4().hex[:12].upper()
                self._add_object(con, "MaintenanceTask", task_id, {
                    "id": task_id, "status": "OPEN", "reason": row["reason"],
                    "created_at": now_utc(), "request_id": request_id,
                })
                self._add_link(con, "hasMaintenanceTask", row["target_id"], task_id)
                con.execute("""UPDATE action_requests SET status='APPROVED',
                    reviewed_at=?,reviewed_by=?,result_object_id=? WHERE id=?""",
                    (now_utc(), reviewer, task_id, request_id))
                self._audit(con, request_id, "createMaintenanceTask", reviewer,
                            "APPLIED", "TASK_CREATED", task_id)
                return {"request_id": request_id, "status": "APPROVED",
                        "task_id": task_id, "idempotent": False}
        except RuleViolation as exc:
            self._audit_rejection(request_id, reviewer, exc.code, str(exc))
            raise

    def reject(self, request_id: str, reviewer: str, note: str) -> dict[str, Any]:
        reviewer = reviewer.strip()
        if not reviewer.startswith("demo:manager-"):
            self._audit_rejection(request_id, reviewer, "UNAUTHORIZED_ROLE", "Reviewer role denied")
            raise RuleViolation("UNAUTHORIZED_ROLE", "Only demo managers can reject")
        if not note.strip():
            raise RuleViolation("MISSING_NOTE", "Rejection note is required")
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM action_requests WHERE id=?", (request_id,)).fetchone()
            if not row:
                raise RuleViolation("REQUEST_NOT_FOUND", "Unknown request")
            if row["status"] != "PENDING":
                raise RuleViolation("NOT_PENDING", "Request is not pending")
            con.execute("""UPDATE action_requests SET status='REJECTED',
                reviewed_at=?,reviewed_by=?,review_note=? WHERE id=?""",
                (now_utc(), reviewer, note.strip(), request_id))
            self._audit(con, request_id, "createMaintenanceTask", reviewer,
                        "REJECTED", "HUMAN_REJECTED", note.strip())
            return {"request_id": request_id, "status": "REJECTED"}

    def list_requests(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in {"PENDING", "APPROVED", "REJECTED"}:
            raise RuleViolation("BAD_STATUS", "Invalid status")
        with self._connect() as con:
            if status:
                rows = con.execute("SELECT * FROM action_requests WHERE status=? ORDER BY created_at DESC", (status,))
            else:
                rows = con.execute("SELECT * FROM action_requests ORDER BY created_at DESC")
            return [dict(row) for row in rows]

    def audit_history(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 1000))
        with self._connect() as con:
            return [dict(row) for row in con.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]

    @staticmethod
    def describe_schema() -> dict[str, Any]:
        return {"objects": OBJECT_TYPES, "links": LINK_TYPES, "functions": FUNCTION_TYPES,
                "actions": ACTION_TYPES, "state_authority": OBJECT_AUTHORITY,
                "implementation": "Educational reference, not Palantir Foundry"}
