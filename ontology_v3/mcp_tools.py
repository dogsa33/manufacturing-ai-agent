"""Read-only Ontology MCP tools for V3.

No write tools are exposed to V2's generic LLM router. Maintenance proposals
are created only by the deterministic Ontology LangGraph policy branch, and
approval/rejection are exclusively controlled by the operator demo UI.
"""
from __future__ import annotations
from typing import Any
from .service import OntologyService, RuleViolation


def register_ontology_tools(mcp: Any) -> None:
    @mcp.tool()
    def ontology_schema() -> dict[str, Any]:
        """Describe manufacturing Ontology object, link and Action definitions.
        Read-only. Use this tool to understand allowed ontology relations.
        """
        return OntologyService.describe_schema()

    @mcp.tool()
    def ontology_list_equipment() -> dict[str, Any]:
        """List Equipment objects in the synthetic demo Ontology.
        Do not claim these are real production equipment.
        """
        store = OntologyService()
        store.seed_demo()
        return {"equipment": store.list_objects("Equipment"), "data_origin": "SYNTHETIC_DEMO"}

    @mcp.tool()
    def ontology_equipment_context(equipment_id: str) -> dict[str, Any]:
        """Follow Equipment -> ProductionLot -> MachineObservation links.
        Return the equipment's existing maintenance tasks and evidence.
        equipment_id example: EQ-001. All relation data is synthetic.
        """
        store = OntologyService()
        store.seed_demo()
        return store.equipment_context(equipment_id)

    @mcp.tool()
    def ontology_predict_equipment_risk(equipment_id: str) -> dict[str, Any]:
        """Predict failure risk for synthetic observations linked to Equipment.
        Uses the existing AI4I-trained Random Forest prediction tool and
        returns model predictions, NOT measured real-world probabilities.
        """
        store = OntologyService()
        store.seed_demo()
        context = store.equipment_context(equipment_id)
        from src.prediction_tool import predict_failure
        predictions = []
        for item in context["observations"]:
            p = item["properties"]
            result = predict_failure(
                product_type=p["product_type"],
                air_temperature=p["air_temperature"],
                process_temperature=p["process_temperature"],
                rotational_speed=p["rotational_speed"],
                torque=p["torque"],
                tool_wear=p["tool_wear"],
            )
            predictions.append({"observation_id": item["id"], "prediction": result})
        return {"equipment_id": equipment_id, "predictions": predictions,
                "source": "existing RF model + synthetic measurements",
                "data_origin": "SYNTHETIC_DEMO"}

    @mcp.tool()
    def ontology_list_maintenance_requests() -> dict[str, Any]:
        """Read existing Action proposals and review statuses. READ ONLY."""
        store = OntologyService()
        return {"requests": store.list_requests()}
