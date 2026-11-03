"""Human review workbench for an educational operational ontology prototype."""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ontology_v3 import OntologyService, RuleViolation

st.set_page_config(page_title="Ontology Operations | V3", layout="wide")
st.title("Manufacturing Operations | Ontology V3")
st.warning(
    "DEMO ONLY: all Equipment/LOT/Observation mappings are synthetic. "
    "The demo reviewer role is self-declared; no real authentication or ERP write-back exists."
)
store = OntologyService()
store.seed_demo()

st.subheader("1. Objects and Links")
equipment = store.list_objects("Equipment")
selected = st.selectbox("Equipment", [o["id"] for o in equipment])
context = store.equipment_context(selected)
c1, c2 = st.columns(2)
with c1:
    st.write("Equipment", context["equipment"])
    st.write("Linked ProductionLots", context["lots"])
with c2:
    st.write("Linked MachineObservations", context["observations"])
    st.write("MaintenanceTasks", context["maintenance_tasks"])

st.subheader("2. Create a proposal (no state change yet)")
with st.form("new_proposal"):
    reason = st.text_area("Rationale / evidence", placeholder="Describe evidence and proposed inspection")
    proposed = st.form_submit_button("Submit for review")
    if proposed:
        try:
            proposal = store.propose_maintenance(selected, reason, requested_by="demo:operator")
            st.info(f"Proposal {proposal['id']} is {proposal['status']}; no work task created yet.")
        except (RuleViolation, ValueError) as exc:
            st.error(str(exc))

st.subheader("3. Human review")
st.caption("Role selection here demonstrates the workflow only; it is NOT secure access control.")
reviewer = st.selectbox("Reviewer (demo)", ["demo:manager-1", "demo:manager-2"])
requests = store.list_requests("PENDING")
if not requests:
    st.info("No pending proposals")
else:
    for req in requests:
        with st.expander(f"{req['id']} | {req['target_id']}", expanded=True):
            st.write({k: req[k] for k in ("reason", "requested_by", "created_at", "status")})
            note = st.text_input("Rejection explanation", key=f"note_{req['id']}")
            yes, no = st.columns(2)
            with yes:
                if st.button("Approve & create MaintenanceTask", key=f"ok_{req['id']}"):
                    try:
                        result = store.approve(req["id"], reviewer)
                        st.success(str(result))
                        st.rerun()
                    except RuleViolation as exc:
                        st.error(f"Blocked ({exc.code}): {exc}")
            with no:
                if st.button("Reject", key=f"no_{req['id']}"):
                    try:
                        store.reject(req["id"], reviewer, note)
                        st.rerun()
                    except RuleViolation as exc:
                        st.error(f"Blocked ({exc.code}): {exc}")

st.subheader("4. Action history")
st.dataframe(store.list_requests(), use_container_width=True)
st.dataframe(store.audit_history(), use_container_width=True)
st.caption("Do not use this simulated authorization or data for production operations.")
