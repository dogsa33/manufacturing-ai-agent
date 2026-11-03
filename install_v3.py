"""Idempotent installer for the published main branch of Manufacturing AI Agent.

Safely adds Ontology V3 branches to LangGraph while keeping Tool/RAG/Direct
and existing source content. Run *after* git switch -c on a fresh branch.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

REGISTRATION = '''\n# V3: Read-only Ontology tools and PENDING maintenance proposals.\nfrom ontology_v3.mcp_tools import register_ontology_tools\nregister_ontology_tools(mcp)\n'''
IMPORTS = '''from ontology_v3.graph_nodes import (\n    detect_ontology_request, ontology_lookup_node, ontology_predict_node,\n    ontology_policy_node, route_after_ontology_policy, ontology_propose_node,\n    ontology_respond_node, ontology_verify_node,\n)\n\n'''
NODES = '''    # V3 Ontology branch: deterministic multi-step flow and postcondition verifier.\n    builder.add_node("ontology_lookup", ontology_lookup_node)\n    builder.add_node("ontology_predict", ontology_predict_node)\n    builder.add_node("ontology_policy", ontology_policy_node)\n    builder.add_node("ontology_propose", ontology_propose_node)\n    builder.add_node("ontology_respond", ontology_respond_node)\n    builder.add_node("ontology_verify", ontology_verify_node)\n    builder.add_edge("ontology_lookup", "ontology_predict")\n    builder.add_edge("ontology_predict", "ontology_policy")\n    builder.add_conditional_edges("ontology_policy", route_after_ontology_policy,\n                                  {"propose": "ontology_propose", "respond": "ontology_respond"})\n    builder.add_edge("ontology_propose", "ontology_respond")\n    builder.add_edge("ontology_respond", "ontology_verify")\n    builder.add_edge("ontology_verify", END)\n\n'''


def replace_once(code: str, needle: str, replacement: str, label: str) -> str:
    count = code.count(needle)
    if count != 1:
        raise ValueError(f"Cannot safely patch {label}: expected one anchor, got {count}")
    return code.replace(needle, replacement, 1)


def patch_server(code: str) -> str:
    if "register_ontology_tools(mcp)" in code:
        return code
    if "mcp = MCPServer(" not in code:
        raise ValueError("MCPServer declaration missing")
    anchor = '\nif __name__ == "__main__":'
    code = replace_once(code, anchor, "\n" + REGISTRATION + anchor, "MCP server")
    ast.parse(code)
    return code


def patch_graph(code: str) -> str:
    if "builder.add_node(\"ontology_lookup\"" in code:
        return code
    marker = "# ============================================================\n# Configuration"
    code = replace_once(code, marker, IMPORTS + marker, "imports")
    router_head = '    print("\\n[NODE] router")\n'
    check = '''\n    # V3: An explicit synthetic Equipment ID enters a fixed Ontology workflow.\n    ontology_request = detect_ontology_request(state.get("query", ""))\n    if ontology_request is not None:\n        return {\n            "route": "ontology",\n            "ontology_equipment_id": ontology_request["equipment_id"],\n            "ontology_action_requested": ontology_request["action_requested"],\n            "ontology_context": {}, "ontology_predictions": [],\n            "ontology_decision": {}, "ontology_proposal": {},\n            "ontology_error": "", "ontology_trace": [],\n            "retry_count": 0,\n        }\n'''
    code = replace_once(code, router_head, router_head + check, "router")
    begin = '        "route": "",\n'
    code = replace_once(code, begin, begin + '''        "ontology_equipment_id": "",\n        "ontology_action_requested": False,\n        "ontology_context": {}, "ontology_predictions": [],\n        "ontology_decision": {}, "ontology_proposal": {},\n        "ontology_error": "", "ontology_trace": [],\n''', "begin-turn reset")
    chooser = '    if route == "rag":\n        return "rag"\n'
    code = replace_once(code, chooser, chooser + '\n    if route == "ontology":\n        return "ontology"\n', "conditional route")
    start = '    builder.add_edge(\n        START,\n'
    code = replace_once(code, start, NODES + start, "graph nodes/edges")
    mapping = '            "direct": "direct",\n'
    code = replace_once(code, mapping, mapping + '            "ontology": "ontology_lookup",\n', "router edges")
    ast.parse(code)
    return code


def patch_state(code: str) -> str:
    if "ontology_equipment_id:" in code:
        return code
    marker = '    # ========================================================\n    # Final Answer\n'
    addition = '''    # V3 Ontology evidence and human-reviewed Action proposal\n    ontology_equipment_id: str\n    ontology_action_requested: bool\n    ontology_context: dict[str, Any]\n    ontology_predictions: list[dict[str, Any]]\n    ontology_decision: dict[str, Any]\n    ontology_proposal: dict[str, Any]\n    ontology_error: str\n    ontology_trace: list[str]\n\n'''
    code = replace_once(code, marker, addition + marker, "LangGraph state")
    ast.parse(code)
    return code


def plan(root: Path) -> dict[Path, str]:
    required = {
        Path("mcp_server/server.py"): patch_server,
        Path("graph/manufacturing_graph.py"): patch_graph,
        Path("graph/manufacturing_state.py"): patch_state,
    }
    output = {}
    for path, update in required.items():
        full = root / path
        if not full.is_file():
            raise FileNotFoundError(f"Required original file not found: {path}")
        output[full] = update(full.read_text(encoding="utf-8"))
    ignore = root / ".gitignore"
    text = ignore.read_text(encoding="utf-8") if ignore.exists() else ""
    rule = "database/ontology_operations.sqlite*"
    if rule not in text.splitlines():
        text = text.rstrip("\n") + "\n\n# Ontology V3 runtime SQLite DB\n" + rule + "\n"
    output[ignore] = text
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate patch compatibility without writing files")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not (root / "ontology_v3/graph_nodes.py").is_file():
        raise SystemExit("Extract V3 overlay at repository root before installation")
    try:
        changes = plan(root)
    except (FileNotFoundError, ValueError, SyntaxError) as exc:
        raise SystemExit(f"V3 installer stopped without changes: {exc}")
    if args.check:
        print("Compatible base source: all patch anchors and Python syntax verified")
        return
    backups = root / ".v3_install_backups"
    for source, updated in changes.items():
        original = source.read_text(encoding="utf-8") if source.exists() else ""
        if original == updated:
            continue
        backup = backups / source.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        source.write_text(updated, encoding="utf-8")
        print(f"Patched: {source.relative_to(root)} (original saved to .v3_install_backups)")
    gitignore = root / ".gitignore"
    backups_rule = ".v3_install_backups/"
    txt = gitignore.read_text(encoding="utf-8")
    if backups_rule not in txt.splitlines():
        gitignore.write_text(txt.rstrip("\n") + "\n" + backups_rule + "\n", encoding="utf-8")
    print("V3 installed. Existing Tool/RAG/Direct graph remains intact.")


if __name__ == "__main__":
    main()
