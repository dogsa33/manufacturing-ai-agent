# Ontology V3 — technical design and validation

## Goal

The original V2 is an analysis-centered `LangGraph + MCP + RAG + ML` assistant. V3 adds a multi-step **operational decision workflow** while keeping all original routes intact. This is a synthetic demo, not a Palantir Foundry/OSDK implementation and not a production MES integration.

## LangGraph routes

```text
START -> begin_turn -> router
  ├─ tool   -> check_tool_arguments -> execute_tool -> generate_tool_answer -> V2 validator
  ├─ rag    -> run_rag -> V2 validator
  ├─ direct -> direct_answer -> V2 validator
  └─ ontology -> ontology_lookup -> ontology_predict -> ontology_policy
                                              ├─[eligible + user explicitly requests Action]
                                              │    -> ontology_propose -> ontology_respond
                                              └─[otherwise] ------------> ontology_respond
                                          -> ontology_verify -> END
```

A deterministic parser intercepts messages containing an **explicit** synthetic `EQ-001` format ID and a manufacturing context term. Other queries remain on original V2 routing. The parser makes no LLM call and never treats a request to inspect or predict as permission to change state.

## Data and model

- Object types: `Equipment`, `ProductionLot`, `MachineObservation`, `MaintenanceTask`.
- Link types: `producedOn`, `hasObservation`, `hasMaintenanceTask`.
- Functions: `equipmentContext`, `maxPredictedFailureRisk` (implemented as workflow computations).
- Action: `createMaintenanceTask` with preconditions and an audited PENDING -> APPROVED / REJECTED lifecycle.
- Data: only **synthetic** equipment/LOT/observation records seeded into local SQLite. These records are **not** the source AI4I dataset. No LGES data or customer IP is imported.
- Model: reuses the repo's actual `src.prediction_tool.predict_failure` (Random Forest) when model and packages are available; no invented probability values.
- Storage: original DuckDB and LangGraph SQLite remain untouched; new `database/ontology_operations.sqlite` persists domain operations and audit. This is separate from dialogue memory and survives restart.

## Action security / design

- Ontology MCP tools expose only read operations. The deterministic LangGraph policy node alone proposes PENDING Actions when an explicit request meets the demo risk threshold.
- No `approve` or `reject` is registered as an MCP tool; these methods exist only on the separate Streamlit demonstration page.
- On approval, service checks equipment ACTIVE, absence of an OPEN or IN_PROGRESS task, and valid pending request, within `BEGIN IMMEDIATE` transaction; then creates the `MaintenanceTask`, link, and audit entry.
- Duplicated pending proposals return the same request ID; duplicated approvals return the existing task ID.
- Role label `demo:manager-1` is **self-declared**, not authenticated. Do not put this prototype on an untrusted network or connect to operational data/systems.
- There is no real source-of-truth write-back. Production usage would require authenticated identities, per-object authorization, policy-enforced server-side approvals, idempotency keys, source ownership, stable transactions, migrations, and operational observability.

## Verification

Run `python -m unittest discover -s tests -p "test_*.py" -v`. Tests cover:

1. Object/link traversal and synthetic provenance
2. Approved action persistence across restart
3. Proposal and approval idempotence
4. Inactive or non-existent equipment blocking
5. Unprivileged/rejected action blocking and audit
6. Invalid link/direction handling
7. End-to-end workflow nodes with a stubbed deterministic RF output
8. No actions without explicit user request
9. Low-risk policy fails closed
10. Missing model fails closed
11. Normal V2 routes not intercepted
12. Unknown equipment fails closed
13. Idempotent installer, original V2 anchor preservation, syntax validation and fail-safe behavior

**Testing boundary:** offline tests use a stubbed predictor and synthetically emulated base-file snippets. They do not establish that the real existing V2 app, model dependencies, MCP subprocesses, or Streamlit version work end-to-end. Run `python scripts/run_v3_smoke.py` and the original V2 route tests in your local V2 environment after applying the overlay to validate real integration.

## Deployment Strategist portfolio explanation

> Designed an ontology-inspired operational model for synthetic manufacturing objects and relationships. Integrated graph traversal, existing ML predictions, a deterministic business-rule policy, human-reviewed Actions, durable state and an auditable execution trail into an existing LangGraph agent, without altering the previous Tool/RAG routes.

Never claim that the system is an official Palantir implementation or operates on employer/customer production data.
