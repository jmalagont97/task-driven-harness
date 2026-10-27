"""test_agent_scoped_server.py

Tests that each debate agent sees only its own tools: the scoping of code/agent_mcp_server.py, its call budget, the
debate flows' wiring (one server per node, keyed by the node id) and a real MCP client over stdio.

    .venv/bin/python -m unittest tests.test_agent_scoped_server -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import agent_mcp_server  # noqa: E402
import tool_registry  # noqa: E402

AGENTS_DIR = REPO_ROOT / "declarations" / "agents"
FLOWS_DIR = REPO_ROOT / "declarations" / "flows"
SCOPES = {
    "abalone_sex_agent_a": {"abalone_sex_random_forest", "abalone_sex_maturity_rules"},
    "abalone_sex_agent_b": {"abalone_sex_logistic_regression", "abalone_sex_bivariate_profile"},
    "abalone_age_agent_a": {"abalone_age_gradient_boosting", "abalone_age_shell_weight_lookup"},
    "abalone_age_agent_b": {"abalone_age_ridge_regression", "abalone_age_allometric_formula"},
}
EXPERTS = {
    "article_nash_1994_agent": "nash_et_al_1994_tr48",
    "article_prince_1988a_agent": "prince_et_al_1988_ageing",
    "article_prince_1988b_agent": "prince_et_al_1988_marbiol",
    "article_mcshane_1992_agent": "mcshane_smith_1992",
}
SPECIMEN = {"length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245,
            "viscera_weight": 0.101, "shell_weight": 0.15, "number_of_rings": 15}
ARTIFACTS_READY = (REPO_ROOT / "artifacts" / "models" / "abalone" / "metrics.json").is_file()


def agent(agent_id: str) -> dict:
    return tool_registry.read_declaration(AGENTS_DIR / f"{agent_id}.json")


class TestScope(unittest.TestCase):
    def test_each_tool_agent_is_scoped_to_its_two_tools(self):
        for agent_id, expected in SCOPES.items():
            with self.subTest(agent=agent_id):
                declarations, bindings = agent_mcp_server.scoped_declarations(agent(agent_id))
                self.assertEqual(set(declarations), expected)
                lists = sorted(binding["binding_list"] for binding in bindings.values())
                self.assertEqual(lists, ["expert_tools", "tools"], "one ML tool and one expert tool")

    def test_the_two_agents_of_a_task_share_no_tool(self):
        self.assertFalse(SCOPES["abalone_sex_agent_a"] & SCOPES["abalone_sex_agent_b"])
        self.assertFalse(SCOPES["abalone_age_agent_a"] & SCOPES["abalone_age_agent_b"])

    def test_unknown_tool_id_stops_the_server(self):
        broken = agent("abalone_sex_agent_a")
        broken["tools"][0]["tool_id"] = "abalone_sex_random_forrest"
        with self.assertRaises(agent_mcp_server.ScopeError) as caught:
            agent_mcp_server.scoped_declarations(broken)
        self.assertIn("abalone_sex_random_forrest", str(caught.exception))
        empty = agent("abalone_debate_consolidator_agent")
        with self.assertRaises(agent_mcp_server.ScopeError):
            agent_mcp_server.scoped_declarations(empty)

    def test_description_adds_the_scope_note_and_usage_hint(self):
        declarations, bindings = agent_mcp_server.scoped_declarations(agent("abalone_age_agent_b"))
        text = agent_mcp_server.scoped_description(declarations["abalone_age_ridge_regression"], bindings["abalone_age_ridge_regression"], sorted(declarations))
        self.assertIn("you can call only abalone_age_allometric_formula, abalone_age_ridge_regression", text)
        self.assertIn("How to use it in your role: Purpose:", text)

    def test_call_budget(self):
        budget = agent_mcp_server.CallBudget({"a": 2, "b": None})
        self.assertEqual([budget.take("a") for _ in range(3)], [True, True, False])
        self.assertTrue(all(budget.take("b") for _ in range(10)))


class TestDebateFlows(unittest.TestCase):
    def test_flows_wire_one_server_per_participant(self):
        metadata_ids = {d["id"] for d in json.loads((REPO_ROOT / "artifacts/guides/guides_abalone/rag/metadata.json").read_text())["documents"]}
        for task in ("sex", "age"):
            flow = tool_registry.read_declaration(FLOWS_DIR / f"abalone_{task}_debate_flow.json")
            with self.subTest(task=task):
                nodes = flow["nodes"]
                participants = [n for n in nodes if n != flow["terminal_node"]]
                self.assertEqual(participants, [f"abalone_{task}_agent_a", f"abalone_{task}_agent_b", *EXPERTS])
                self.assertEqual(set(flow["mcp_servers"]), set(participants))
                for node_id, node in nodes.items():
                    self.assertEqual(tool_registry.read_declaration(REPO_ROOT / node["agent_declaration"])["id"], node_id)
                for expert, document_id in EXPERTS.items():
                    args = flow["mcp_servers"][expert]["args"]
                    self.assertEqual(args[:1] + args[-2:], ["code/article_mcp_server.py", "--document-id", document_id])
                    self.assertIn(document_id, metadata_ids)
                for tool_agent in participants[:2]:
                    self.assertEqual(flow["mcp_servers"][tool_agent]["args"][-1], f"declarations/agents/{tool_agent}.json")

                written = [node["blackboard_permissions"]["write_section"] for node in nodes.values()]
                self.assertEqual(len(written), len(set(written)), "two nodes write the same section")
                declared = {s["id"] for s in flow["blackboard_sections"]} | {"problem_spec", "final_decision"}
                self.assertTrue(set(written) <= declared)
                self.assertNotIn("debate_state", written)
                self.assertNotIn("debate_ledger", written)
                loops = [e for e in flow["edges"] if e.get("loop_controls")]
                self.assertEqual(len(loops), 1)
                self.assertEqual((loops[0]["from"], loops[0]["to"]), (participants[-1], participants[0]))

    def test_experts_bind_only_the_article_tools(self):
        for expert in EXPERTS:
            declaration = agent(expert)
            with self.subTest(agent=expert):
                self.assertEqual([b["tool_id"] for b in declaration["tools"]], ["article_search_article", "article_get_section"])
                self.assertEqual(declaration["expert_tools"], [])
                self.assertTrue(declaration["role"]["permissions"]["can_use_rag"])
                self.assertTrue(declaration["knowledge"]["rag"]["enabled"])

    def test_agent_output_formats_are_valid_json_schema(self):
        from jsonschema import Draft202012Validator

        for path in AGENTS_DIR.glob("*.json"):
            declaration = tool_registry.read_declaration(path)
            if declaration["model"].get("provider") != "ollama":
                continue
            with self.subTest(agent=path.name):
                Draft202012Validator.check_schema(declaration["prompt"]["output_format"]["schema"])
                self.assertEqual(declaration["model"]["model_name"], "gemma4:e4b")
                self.assertFalse(declaration["runtime"]["adapters"]["ollama"]["think"])


def mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return ARTIFACTS_READY


@unittest.skipUnless(mcp_available(), "needs the mcp package and the trained Abalone artifacts")
class TestScopedServerOverStdio(unittest.IsolatedAsyncioTestCase):
    async def test_lists_and_calls_only_its_tools_within_budget(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.shared.exceptions import MCPError

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(CODE_DIR / "agent_mcp_server.py"), "--agent", "declarations/agents/abalone_sex_agent_a.json", "--log-level", "WARNING"],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                self.assertEqual(set(tools), SCOPES["abalone_sex_agent_a"])
                self.assertIn("Scope: in this session you can call only", tools["abalone_sex_random_forest"].description)

                for _ in range(2):
                    result = await session.call_tool("abalone_sex_random_forest", {"specimens": [SPECIMEN]})
                    self.assertFalse(result.is_error)
                exhausted = await session.call_tool("abalone_sex_random_forest", {"specimens": [SPECIMEN]})
                self.assertTrue(exhausted.is_error)
                self.assertIn("budget exhausted", exhausted.structured_content["error"]["message"])

                with self.assertRaises(MCPError):
                    await session.call_tool("abalone_sex_logistic_regression", {"specimens": [SPECIMEN]})


if __name__ == "__main__":
    unittest.main()
