"""Smoke test the fully installed LangGraph V3 using the existing Agent dependencies.

Requires a working V2 environment including the trained RF model, but needs no
API call for the Ontology route itself.
"""
from __future__ import annotations

import asyncio
import uuid
from graph.manufacturing_graph import manufacturing_graph


async def main():
    for query in ("EQ-001 설비의 연결된 LOT와 고장 위험을 분석해줘",
                  "EQ-001 설비 고장 위험 분석 후 점검 작업을 제안해줘"):
        result = await manufacturing_graph.ainvoke(
            {"query": query},
            config={"configurable": {"thread_id": "ontology-v3-smoke-" + uuid.uuid4().hex}},
        )
        print("\nQUERY:", query)
        print("ROUTE:", result.get("route"))
        print("VALIDATED:", result.get("validation_passed"))
        print("TRACE:", result.get("ontology_trace"))
        print("ANSWER:", result.get("answer"))
        assert result.get("route") == "ontology"
        assert "lookup:objects-and-links" in result.get("ontology_trace", [])
        assert "respond:deterministic" in result.get("ontology_trace", [])
    print("\nCheck the 'Ontology Operations' Streamlit page for pending approval.")


if __name__ == "__main__":
    asyncio.run(main())
