"""Installer integration tests: patch original-style files and preserve V2 paths."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from install_v3 import main, plan

GRAPH_FIXTURE = '''from langgraph.graph import StateGraph, START, END
# ============================================================
# Configuration
# ============================================================

async def begin_turn(state):
    return {
        "route": "",
        "answer": "",
    }

async def route_query(state):
    print("\\n[NODE] router")
    query = state["query"]
    return {"route": "direct"}

def choose_route(state):
    route = state.get("route", "direct")
    if route == "tool":
        return "tool"
    if route == "rag":
        return "rag"
    return "direct"

def build_manufacturing_graph(checkpointer=None):
    builder = StateGraph(dict)
    builder.add_node("router", route_query)
    builder.add_node("begin_turn", begin_turn)
    # START → Router
    builder.add_edge(
        START,
        "begin_turn",
    )
    builder.add_conditional_edges("router", choose_route, {
            "tool": "execute_tool",
            "rag": "rag",
            "direct": "direct",
    })
    return builder.compile(checkpointer=checkpointer)
'''

STATE_FIXTURE = '''from typing import Any, TypedDict
class ManufacturingState(TypedDict, total=False):
    query: str
    # ========================================================
    # Final Answer
    # ========================================================
    answer: str
'''

SERVER_FIXTURE = '''from mcp.server import MCPServer
mcp = MCPServer("Manufacturing Analysis Agent Tools")
@mcp.tool()
def process_summary():
    return {"original": True}

if __name__ == "__main__":
    mcp.run()
'''


class InstallerTests(unittest.TestCase):
    def make_root(self, path: Path) -> None:
        for folder in ("mcp_server", "graph", "ontology_v3"):
            (path / folder).mkdir()
        (path / "ontology_v3/graph_nodes.py").write_text("# mock nodes", encoding="utf-8")
        (path / "graph/manufacturing_graph.py").write_text(GRAPH_FIXTURE, encoding="utf-8")
        (path / "graph/manufacturing_state.py").write_text(STATE_FIXTURE, encoding="utf-8")
        (path / "mcp_server/server.py").write_text(SERVER_FIXTURE, encoding="utf-8")

    def test_all_paths_patched_preserved_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_root(root)
            main(["--root", str(root), "--check"])
            self.assertNotIn("ontology_lookup", (root / "graph/manufacturing_graph.py").read_text())
            main(["--root", str(root)])
            graph = (root / "graph/manufacturing_graph.py").read_text()
            state = (root / "graph/manufacturing_state.py").read_text()
            server = (root / "mcp_server/server.py").read_text()
            self.assertIn('"direct": "direct"', graph)
            self.assertIn('"ontology": "ontology_lookup"', graph)
            self.assertIn('builder.add_edge("ontology_verify", END)', graph)
            self.assertIn("ontology_trace: list[str]", state)
            self.assertIn("def process_summary()", server)
            self.assertIn("register_ontology_tools(mcp)", server)
            self.assertIn(".v3_install_backups/", (root / ".gitignore").read_text())
            for path in ("graph/manufacturing_graph.py", "graph/manufacturing_state.py", "mcp_server/server.py"):
                compile((root / path).read_text(), path, "exec")
                self.assertTrue((root / ".v3_install_backups" / path).exists())
            before = {p: text for p, text in plan(root).items()}
            main(["--root", str(root)])
            self.assertEqual(before, plan(root))

    def test_broken_anchors_prevent_all_mutations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_root(root)
            (root / "graph/manufacturing_graph.py").write_text("# incompatible revision", encoding="utf-8")
            original = (root / "mcp_server/server.py").read_text()
            with self.assertRaises(SystemExit):
                main(["--root", str(root)])
            self.assertEqual((root / "mcp_server/server.py").read_text(), original)
            self.assertFalse((root / ".v3_install_backups").exists())


if __name__ == "__main__":
    unittest.main()
