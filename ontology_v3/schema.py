"""Explicit business ontology definitions for the *synthetic* manufacturing demo.

This is an independent educational implementation inspired by operational
ontology patterns; it is NOT Palantir Foundry or an implementation of OSDK.
"""
from __future__ import annotations

OBJECT_TYPES = {
    "Equipment": ("id", "name", "line", "status"),
    "ProductionLot": ("id", "product_type", "production_date"),
    "MachineObservation": (
        "id", "product_type", "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear", "observed_at",
    ),
    "MaintenanceTask": ("id", "status", "reason", "created_at", "request_id"),
}

# Link direction and allowed endpoint types are validated when saving links.
LINK_TYPES = {
    "producedOn": ("Equipment", "ProductionLot"),
    "hasObservation": ("ProductionLot", "MachineObservation"),
    "hasMaintenanceTask": ("Equipment", "MaintenanceTask"),
}

ACTION_TYPES = {
    "createMaintenanceTask": {
        "target": "Equipment",
        "preconditions": [
            "Equipment exists",
            "Equipment is active",
            "No existing open maintenance task for equipment",
            "Human reviewer approved a pending proposal",
        ],
        "effect": "Create MaintenanceTask and hasMaintenanceTask link",
    }
}


# Business functions are deterministic, not arbitrary LLM-generated SQL.
FUNCTION_TYPES = {
    "equipmentContext": {"input": "Equipment.id", "output": "linked Lots, Observations and MaintenanceTasks"},
    "maxPredictedFailureRisk": {"input": "MachineObservation observations",
                                "output": "maximum RF probability",
                                "policy_threshold": 0.70},
}

OBJECT_AUTHORITY = {
    "Equipment": "synthetic source-backed demo",
    "ProductionLot": "synthetic source-backed demo",
    "MachineObservation": "synthetic source-backed demo",
    "MaintenanceTask": "ontology-owned action state",
}
