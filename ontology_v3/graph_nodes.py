"""Deterministic, auditable LangGraph nodes for the synthetic Ontology V3.

The LLM routes normal V2 queries as before. Ontology requests identified by an
explicit equipment ID follow a fixed graph that cannot approve Actions.
The production graph's LLM validator is intentionally NOT trusted for writes:
Action preconditions and final verification are deterministic.
"""
from __future__ import annotations

import re
from typing import Any

from .service import OntologyService, RuleViolation

HIGH_RISK_THRESHOLD = 0.70
_ID_PATTERN = re.compile(r"(?<![A-Z0-9])EQ[-_ ]?(\d{1,5})(?![A-Z0-9])", re.IGNORECASE)
_ACTION_TERMS = ("제안", "생성", "등록", "요청", "만들", "신청", "propose", "create", "request", "schedule")
_MAINTENANCE_TERMS = ("점검", "정비", "보전", "maintenance", "inspection", "task", "작업")
_CONTEXT_TERMS = ("설비", "연결", "관계", "온톨로지", "lot", "로트", "위험", "예측", "고장", "분석", "ontology", "equipment", "machine", "prediction", "risk", "observation")


def detect_ontology_request(query: str) -> dict[str, Any] | None:
    """Require an explicit synthetic equipment ID to avoid hijacking V2 queries."""
    if not isinstance(query, str):
        return None
    match = _ID_PATTERN.search(query)
    normalized = query.lower()
    if not match or not any(term in normalized for term in _CONTEXT_TERMS + _MAINTENANCE_TERMS):
        return None
    equipment_id = f"EQ-{int(match.group(1)):03d}"
    wants_action = any(s in normalized for s in _ACTION_TERMS) and any(s in normalized for s in _MAINTENANCE_TERMS)
    return {"equipment_id": equipment_id, "action_requested": wants_action}


def _store() -> OntologyService:
    store = OntologyService()
    store.seed_demo()
    return store


def predict_observation(properties: dict[str, Any]) -> dict[str, Any]:
    """Use V2's trained RF with the same input units and thresholds."""
    from src.prediction_tool import predict_failure
    return predict_failure(
        product_type=properties["product_type"],
        air_temperature=properties["air_temperature"],
        process_temperature=properties["process_temperature"],
        rotational_speed=properties["rotational_speed"],
        torque=properties["torque"],
        tool_wear=properties["tool_wear"],
    )


async def ontology_lookup_node(state: dict[str, Any]) -> dict[str, Any]:
    equipment_id = state.get("ontology_equipment_id", "")
    try:
        context = _store().equipment_context(equipment_id)
        return {"ontology_context": context, "ontology_error": "",
                "ontology_trace": ["lookup:objects-and-links"]}
    except (RuleViolation, OSError, ValueError) as exc:
        return {"ontology_context": {}, "ontology_error": f"{type(exc).__name__}: {exc}",
                "ontology_trace": ["lookup:failed"]}


async def ontology_predict_node(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("ontology_error"):
        return {"ontology_predictions": [], "ontology_trace": state.get("ontology_trace", []) + ["predict:skipped"]}
    observations = state.get("ontology_context", {}).get("observations", [])
    results = []
    try:
        for obj in observations:
            result = predict_observation(obj["properties"])
            probability = result.get("failure_probability")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError("Prediction probability must be numeric and in [0,1]")
            results.append({"observation_id": obj["id"], "probability": float(probability),
                            "risk_level": result.get("risk_level"), "model_result": result})
        return {"ontology_predictions": results,
                "ontology_trace": state.get("ontology_trace", []) + ["predict:existing-random-forest"]}
    except (ImportError, FileNotFoundError, KeyError, ValueError, RuntimeError, OSError) as exc:
        return {"ontology_predictions": [],
                "ontology_error": f"MODEL_UNAVAILABLE: {type(exc).__name__}: {exc}",
                "ontology_trace": state.get("ontology_trace", []) + ["predict:failed"]}


async def ontology_policy_node(state: dict[str, Any]) -> dict[str, Any]:
    predictions = state.get("ontology_predictions", [])
    highest = max((r["probability"] for r in predictions), default=None)
    eligible = (not state.get("ontology_error") and highest is not None and highest >= HIGH_RISK_THRESHOLD)
    decision = {
        "action_requested": bool(state.get("ontology_action_requested")),
        "eligible_for_proposal": bool(eligible),
        "maximum_predicted_risk": highest,
        "threshold": HIGH_RISK_THRESHOLD,
        "human_approval_required": True,
    }
    return {"ontology_decision": decision,
            "ontology_trace": state.get("ontology_trace", []) + ["policy:deterministic"]}


def route_after_ontology_policy(state: dict[str, Any]) -> str:
    decision = state.get("ontology_decision", {})
    if decision.get("action_requested") and decision.get("eligible_for_proposal"):
        return "propose"
    return "respond"


async def ontology_propose_node(state: dict[str, Any]) -> dict[str, Any]:
    """Create PENDING request only. This node has no approve/reject capability."""
    if route_after_ontology_policy(state) != "propose":
        return {"ontology_error": "POLICY_BLOCKED: no eligible proposal"}
    equipment_id = state["ontology_equipment_id"]
    max_risk = state["ontology_decision"]["maximum_predicted_risk"]
    observation_ids = [p["observation_id"] for p in state.get("ontology_predictions", [])]
    reason = (f"Synthetic demo: model estimated max failure probability {max_risk:.2%} "
              f"for {equipment_id} with observations {', '.join(observation_ids)}. "
              "Request technician inspection; model estimate is not a physical diagnosis.")
    try:
        proposal = _store().propose_maintenance(equipment_id, reason, requested_by="agent:ontology-v3")
        return {"ontology_proposal": proposal,
                "ontology_trace": state.get("ontology_trace", []) + ["action:PENDING-created"]}
    except (RuleViolation, OSError, ValueError) as exc:
        return {"ontology_proposal": {}, "ontology_error": f"ACTION_BLOCKED: {type(exc).__name__}: {exc}",
                "ontology_trace": state.get("ontology_trace", []) + ["action:blocked"]}


async def ontology_respond_node(state: dict[str, Any]) -> dict[str, Any]:
    equipment_id = state.get("ontology_equipment_id", "")
    ctx = state.get("ontology_context", {})
    predictions = state.get("ontology_predictions", [])
    decision = state.get("ontology_decision", {})
    proposal = state.get("ontology_proposal", {})
    error = state.get("ontology_error", "")
    lines = [f"[Ontology V3 | {equipment_id}]", "데이터: 합성 제조 데모 (실제 LGES 설비/LOT 아님)"]
    if ctx:
        lines += [f"연결된 생산 LOT: {', '.join(o['id'] for o in ctx.get('lots', [])) or '없음'}",
                  f"측정 기록: {', '.join(o['id'] for o in ctx.get('observations', [])) or '없음'}",
                  f"등록된 점검 작업: {len(ctx.get('maintenance_tasks', []))}건"]
    for item in predictions:
        lines.append(f"- {item['observation_id']}: 모델 추정 고장 확률 {item['probability']:.2%}")
    if decision.get("maximum_predicted_risk") is not None:
        lines.append(f"위험 판단 기준: {HIGH_RISK_THRESHOLD:.0%} (데모 정책, 실제 현장 기준 아님)")
    if proposal:
        lines.append(f"점검 제안: {proposal['id']} / PENDING — 아직 작업은 생성되지 않았습니다.")
        lines.append("별도 Ontology Operations 화면에서 담당자가 승인해야 실제 점검 작업이 생성됩니다.")
    elif decision.get("action_requested") and not decision.get("eligible_for_proposal") and not error:
        lines.append("점검 작업 제안은 생성하지 않았습니다. 모델 위험도가 데모 정책 기준에 미달했습니다.")
    if error:
        lines.append(f"실행 제한: {error}")
        lines.append("검증된 결과가 없으면 점검 제안을 실행하지 않습니다.")
    return {"answer": "\n".join(lines),
            "tool_result": {"context": ctx, "predictions": predictions, "decision": decision,
                            "proposal": proposal, "error": error,
                            "data_origin": "SYNTHETIC_DEMO"},
            "ontology_trace": state.get("ontology_trace", []) + ["respond:deterministic"]}


async def ontology_verify_node(state: dict[str, Any]) -> dict[str, Any]:
    """Verify postconditions against persisted DB, separate from V2 LLM validator."""
    proposal = state.get("ontology_proposal", {})
    if proposal:
        try:
            actual = next((r for r in _store().list_requests() if r["id"] == proposal["id"]), None)
            if actual is None or actual["status"] != "PENDING":
                return {"validation_passed": False, "validation_feedback": "Proposal postcondition failed",
                        "answer": "Ontology 제안의 저장 상태를 확인하지 못했습니다. 관리자 화면에서 상태를 확인하세요."}
            tasks = _store().equipment_context(state["ontology_equipment_id"])["maintenance_tasks"]
            if any(t["properties"].get("request_id") == proposal["id"] for t in tasks):
                return {"validation_passed": False, "validation_feedback": "Task was created before approval",
                        "answer": "승인 전 상태 변경이 탐지되었습니다. 작업 확인이 필요합니다."}
        except (RuleViolation, OSError, ValueError) as exc:
            return {"validation_passed": False, "validation_feedback": f"Postcondition error: {exc}",
                    "answer": "Ontology 실행 결과 검증에 실패했습니다."}
    return {"validation_passed": not bool(state.get("ontology_error")),
            "validation_feedback": state.get("ontology_error", "") or "Deterministic ontology checks passed"}
