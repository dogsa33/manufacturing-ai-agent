"""MCP catalog contains only Ontology reads; LLM cannot invoke Action writes."""
from __future__ import annotations
import unittest
from ontology_v3.mcp_tools import register_ontology_tools

class FakeMCP:
    def __init__(self):
        self.tools = {}
    def tool(self):
        def register(func):
            self.tools[func.__name__] = func
            return func
        return register

class MCPContractTests(unittest.TestCase):
    def test_read_only_ontology_tools(self):
        mcp = FakeMCP()
        register_ontology_tools(mcp)
        names = set(mcp.tools)
        self.assertIn('ontology_equipment_context', names)
        self.assertIn('ontology_predict_equipment_risk', names)
        self.assertIn('ontology_list_maintenance_requests', names)
        self.assertNotIn('ontology_propose_maintenance', names)
        self.assertTrue(all(x not in name for name in names for x in ('approve', 'reject', 'create_task', 'write_back')))

if __name__ == '__main__':
    unittest.main()
