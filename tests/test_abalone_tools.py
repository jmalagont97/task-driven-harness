"""test_abalone_tools.py

Tests for the eight Abalone tools: declarations, input validation, executors, the frozen split,
the plan's reference metrics and the MCP server. Run from the repository root after training:

    .venv/bin/python code/tools/abalone/train_abalone_tools.py
    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from tools.abalone import common  # noqa: E402

SERVER = "abalone_tools"
SEX_TOOLS = (
    "abalone_sex_random_forest",
    "abalone_sex_logistic_regression",
    "abalone_sex_maturity_rules",
    "abalone_sex_bivariate_profile",
)
AGE_ML_TOOLS = ("abalone_age_gradient_boosting", "abalone_age_ridge_regression")
AGE_EXPERT_TOOLS = ("abalone_age_allometric_formula", "abalone_age_shell_weight_lookup")
AGE_TOOLS = AGE_ML_TOOLS + AGE_EXPERT_TOOLS
ALL_TOOLS = SEX_TOOLS + AGE_TOOLS

# Hold-out specimen used in the plan's comparison table: female, 12 rings, never seen in training.
HOLDOUT = {
    "length": 0.475, "diameter": 0.365, "height": 0.13, "whole_weight": 0.4805,
    "shucked_weight": 0.1905, "viscera_weight": 0.114, "shell_weight": 0.1475,
}
METRICS_PATH = REPO_ROOT / "artifacts" / "models" / "abalone" / "metrics.json"
ARTIFACTS_READY = METRICS_PATH.is_file()
needs_artifacts = unittest.skipUnless(ARTIFACTS_READY, f"train the tools first: {common.TRAIN_COMMAND}")

DECLARATIONS = tool_registry.load_tool_declarations(server=SERVER)


def call(name: str, arguments) -> dict:
    return tool_registry.invoke(DECLARATIONS[name], arguments)


def specimen(name: str, **overrides) -> dict:
    """A valid specimen for the tool, with optional field overrides (None removes the field)."""
    base = {**HOLDOUT, "number_of_rings": 12} if name in SEX_TOOLS else {**HOLDOUT, "sex": "F"}
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def trained_with_plan_versions() -> bool:
    if not ARTIFACTS_READY:
        return False
    versions = json.loads(METRICS_PATH.read_text())["library_versions"]
    return versions["scikit-learn"] == "1.4.2" and versions["numpy"].startswith("1.26")


class TestDeclarations(unittest.TestCase):
    def test_eight_tools_exposed_on_the_abalone_server(self):
        self.assertEqual(set(DECLARATIONS), set(ALL_TOOLS))

    def test_example_template_still_parses_and_is_not_exposed(self):
        example = tool_registry.read_declaration(tool_registry.TOOLS_DIR / "example_tool.json")
        self.assertEqual(example["id"], "execute_python")
        self.assertNotIn(example["name"], tool_registry.load_tool_declarations())

    def test_envelope_identity_matches_declaration(self):
        for name, declaration in DECLARATIONS.items():
            with self.subTest(tool=name):
                envelope = call(name, {"specimens": []})  # rejected before any artifact is read
                self.assertEqual(envelope["tool_id"], declaration["id"])
                self.assertEqual(envelope["version"], declaration["version"])
                self.assertEqual(envelope["disclaimer"], declaration["disclaimer"])

    def test_input_limits_match_the_executors(self):
        for name, declaration in DECLARATIONS.items():
            with self.subTest(tool=name):
                specimens = declaration["input_schema"]["properties"]["specimens"]
                self.assertEqual(specimens["maxItems"], declaration["config"]["max_specimens"])
                properties = specimens["items"]["properties"]
                for field, (low, high) in common.MEASUREMENT_LIMITS.items():
                    self.assertEqual((properties[field]["minimum"], properties[field]["maximum"]), (low, high))
                if name in SEX_TOOLS:
                    rings = properties["number_of_rings"]
                    self.assertEqual((rings["minimum"], rings["maximum"]), common.RINGS_LIMITS)

    def test_annotations_describe_read_only_closed_world_tools(self):
        for name, declaration in DECLARATIONS.items():
            with self.subTest(tool=name):
                self.assertEqual(
                    declaration["mcp"]["annotations"],
                    {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
                )
                self.assertEqual(declaration["side_effects"], "read")
                self.assertTrue(declaration["deterministic"])


INVALID_CASES = [
    # (label, arguments builder, fragment expected in the error message)
    ("millimetres", lambda n: {"specimens": [specimen(n, length=95)]}, "divide by 200"),
    ("grams", lambda n: {"specimens": [specimen(n, whole_weight=135.4)]}, "grams"),
    ("zero height", lambda n: {"specimens": [specimen(n, height=0)]}, "height of 0"),
    ("empty list", lambda n: {"specimens": []}, "non-empty"),
    ("too many", lambda n: {"specimens": [specimen(n)] * 51}, "limit is 50"),
    ("extra argument", lambda n: {"specimens": [specimen(n)], "verbose": True}, "Unexpected argument"),
    ("missing field", lambda n: {"specimens": [specimen(n, shell_weight=None)]}, "missing"),
    ("string number", lambda n: {"specimens": [specimen(n, diameter="0.4")]}, "must be a number"),
    ("boolean number", lambda n: {"specimens": [specimen(n, diameter=True)]}, "must be a number"),
    ("not an object", lambda n: {"specimens": ["abalone"]}, "must be an object"),
    ("target field", lambda n: {"specimens": [specimen(n, **({"sex": "F"} if n in SEX_TOOLS else {"number_of_rings": 9}))]},
     "what this tool predicts"),
]
SEX_ONLY_CASES = [
    ("fractional rings", lambda n: {"specimens": [specimen(n, number_of_rings=12.5)]}, "whole number"),
    ("zero rings", lambda n: {"specimens": [specimen(n, number_of_rings=0)]}, "outside"),
]
AGE_ONLY_CASES = [("bad sex", lambda n: {"specimens": [specimen(n, sex="X")]}, "must be one of")]


class TestInputValidation(unittest.TestCase):
    def cases(self, name):
        return INVALID_CASES + (SEX_ONLY_CASES if name in SEX_TOOLS else AGE_ONLY_CASES)

    def test_invalid_inputs_return_actionable_input_errors(self):
        for name in ALL_TOOLS:
            for label, build, fragment in self.cases(name):
                with self.subTest(tool=name, case=label):
                    envelope = call(name, build(name))
                    self.assertFalse(envelope["ok"])
                    self.assertEqual(envelope["error"]["type"], "input_error")
                    self.assertIn(fragment, envelope["error"]["message"])
                    self.assertEqual(tool_registry.output_problems(DECLARATIONS[name], envelope), [])

    def test_executors_reject_everything_the_input_schema_rejects(self):
        from jsonschema import Draft202012Validator

        for name in ALL_TOOLS:
            validator = Draft202012Validator(DECLARATIONS[name]["input_schema"])
            for label, build, _ in self.cases(name):
                with self.subTest(tool=name, case=label):
                    arguments = build(name)
                    self.assertFalse(validator.is_valid(arguments), "case should also break the JSON Schema")
                    self.assertFalse(call(name, arguments)["ok"])

    def test_ml_age_tools_need_a_known_sex(self):
        for name in AGE_ML_TOOLS:
            with self.subTest(tool=name):
                envelope = call(name, {"specimens": [specimen(name, sex="unknown")]})
                self.assertEqual(envelope["error"]["type"], "input_error")
                self.assertIn("call a sex tool first", envelope["error"]["message"])

    def test_whole_float_ring_count_is_accepted_like_json_schema_does(self):
        rows = common.parse_specimens({"specimens": [specimen(SEX_TOOLS[0], number_of_rings=12.0)]}, target="sex")
        self.assertEqual(rows[0]["number_of_rings"], 12)


@needs_artifacts
class TestExecutors(unittest.TestCase):
    def test_examples_validate_and_emit_expected_flags(self):
        for name, declaration in DECLARATIONS.items():
            for example in declaration["examples"]:
                with self.subTest(tool=name):
                    envelope = call(name, example["input"])
                    self.assertTrue(envelope["ok"], envelope.get("error"))
                    self.assertEqual(tool_registry.output_problems(declaration, envelope), [])
                    self.assertEqual(envelope["flags"], sorted(example["expected_flags"]))

    def test_same_input_same_output(self):
        for name in ALL_TOOLS:
            with self.subTest(tool=name):
                arguments = {"specimens": [specimen(name)]}
                self.assertEqual(call(name, arguments), call(name, copy.deepcopy(arguments)))

    def test_batches_keep_input_order(self):
        for name in ALL_TOOLS:
            with self.subTest(tool=name):
                specimens = [specimen(name), specimen(name, shell_weight=0.4, whole_weight=1.2, diameter=0.5), specimen(name)]
                envelope = call(name, {"specimens": specimens})
                self.assertEqual([p["index"] for p in envelope["result"]["predictions"]], [0, 1, 2])
                self.assertEqual(envelope["result"]["predictions"][0] | {"index": 2}, envelope["result"]["predictions"][2])

    def test_sex_probabilities_sum_to_one(self):
        for name in ("abalone_sex_random_forest", "abalone_sex_logistic_regression", "abalone_sex_bivariate_profile"):
            with self.subTest(tool=name):
                prediction = call(name, {"specimens": [specimen(name)]})["result"]["predictions"][0]
                self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, delta=1e-3)
                self.assertAlmostEqual(prediction["p_adult"], prediction["probabilities"]["M"] + prediction["probabilities"]["F"], delta=1e-3)

    def test_age_intervals_contain_the_prediction(self):
        for name in AGE_TOOLS:
            with self.subTest(tool=name):
                prediction = call(name, {"specimens": [specimen(name)]})["result"]["predictions"][0]
                interval = prediction["interval_80"]
                self.assertLessEqual(interval["low"], prediction["predicted_rings"])
                self.assertLessEqual(prediction["predicted_rings"], interval["high"])
                self.assertGreaterEqual(interval["low"], 1.0)
                self.assertAlmostEqual(prediction["predicted_age_years"], prediction["predicted_rings"] + 1.5, places=2)

    def test_infant_rule_fires_and_does_not_abstain(self):
        small = specimen(SEX_TOOLS[2], length=0.3, diameter=0.22, height=0.07, whole_weight=0.15,
                         shucked_weight=0.06, viscera_weight=0.03, shell_weight=0.05, number_of_rings=5)
        prediction = call("abalone_sex_maturity_rules", {"specimens": [small]})["result"]["predictions"][0]
        self.assertEqual((prediction["verdict"], prediction["deciding_rule"], prediction["abstained"]), ("I", "R1", False))
        self.assertIsNone(prediction["adult_base_rates"])
        self.assertIsNotNone(prediction["fired_rules"][0]["train_precision"])

    def test_adult_verdict_abstains_on_male_vs_female(self):
        prediction = call("abalone_sex_maturity_rules", {"specimens": [specimen(SEX_TOOLS[2])]})["result"]["predictions"][0]
        self.assertEqual((prediction["verdict"], prediction["abstained"], prediction["sex_candidates"]), ("ADULT", True, ["M", "F"]))

    def test_changed_rules_mark_training_stats_as_stale(self):
        declaration = copy.deepcopy(DECLARATIONS["abalone_sex_maturity_rules"])
        declaration["config"]["rules"][0]["conditions"][0]["value"] = 0.19
        envelope = tool_registry.invoke(declaration, {"specimens": [specimen(SEX_TOOLS[2])]})
        self.assertIn("rule_stats_stale", envelope["flags"])
        self.assertTrue(envelope["result"]["model_card"]["stale"])
        self.assertIsNone(envelope["result"]["predictions"][0]["fired_rules"][0]["train_precision"])
        self.assertEqual(tool_registry.output_problems(declaration, envelope), [])

    def test_expert_age_tools_accept_unknown_sex(self):
        for name in AGE_EXPERT_TOOLS:
            with self.subTest(tool=name):
                envelope = call(name, {"specimens": [specimen(name, sex="unknown")]})
                self.assertTrue(envelope["ok"], envelope.get("error"))
                self.assertIn("sex_unknown", envelope["result"]["predictions"][0]["flags"])

    def test_tail_region_flag_for_very_heavy_shells(self):
        heavy = specimen(AGE_EXPERT_TOOLS[0], length=0.8, diameter=0.63, height=0.22, whole_weight=2.6,
                         shucked_weight=1.0, viscera_weight=0.5, shell_weight=1.2, sex="M")
        prediction = call("abalone_age_allometric_formula", {"specimens": [heavy]})["result"]["predictions"][0]
        self.assertGreater(prediction["predicted_rings"], 15)
        self.assertIn("tail_region", prediction["flags"])
        self.assertIn("out_of_distribution", prediction["flags"])

    def test_sparse_lookup_cells_are_flagged(self):
        params = json.loads((REPO_ROOT / "artifacts/models/abalone/expert_params.json").read_text())["age_shell_weight_lookup"]
        edges = params["edges"]
        min_n = DECLARATIONS["abalone_age_shell_weight_lookup"]["config"]["min_cell_n"]
        sex, cell = next((s, c) for s in common.SEX_CLASSES for c in params["cells"][s] if 0 < c["n"] < min_n)
        low = 0.0 if cell["bin"] == 0 else edges[cell["bin"] - 1]
        weight = edges[cell["bin"]] if cell["bin"] < len(edges) else low + 0.05
        probe = specimen(AGE_EXPERT_TOOLS[1], sex=sex, shell_weight=weight)
        prediction = call("abalone_age_shell_weight_lookup", {"specimens": [probe]})["result"]["predictions"][0]
        self.assertEqual(prediction["cell"]["bin"], cell["bin"])
        self.assertIn("sparse_cell", prediction["flags"])

    def test_missing_artifact_is_reported_not_raised(self):
        declaration = copy.deepcopy(DECLARATIONS["abalone_sex_random_forest"])
        declaration["config"]["model_path"] = "artifacts/models/abalone/does_not_exist.joblib"
        envelope = tool_registry.invoke(declaration, {"specimens": [specimen(SEX_TOOLS[0])]})
        self.assertEqual(envelope["error"]["type"], "artifact_missing")
        self.assertIn("train_abalone_tools.py", envelope["error"]["message"])

    def test_broken_config_becomes_a_runtime_error_envelope(self):
        for name in ALL_TOOLS:
            with self.subTest(tool=name):
                declaration = copy.deepcopy(DECLARATIONS[name])
                declaration["config"] = {}
                envelope = tool_registry.invoke(declaration, {"specimens": [specimen(name)]})
                self.assertFalse(envelope["ok"])
                self.assertEqual(envelope["error"]["type"], "runtime_error")
                self.assertEqual(tool_registry.output_problems(declaration, envelope), [])


@needs_artifacts
class TestSplit(unittest.TestCase):
    def setUp(self):
        self.split = common.load_split()
        self.frame = common.load_dataset()

    def test_split_is_disjoint_and_complete(self):
        train, test = set(self.split["train_indices"]), set(self.split["test_indices"])
        excluded = {row["index"] for row in self.split["excluded_rows"]}
        self.assertFalse(train & test)
        self.assertEqual(len(train) + len(test) + len(excluded), len(self.frame))
        self.assertTrue((self.frame.loc[sorted(excluded), "height"] <= 0).all())

    def test_split_regenerates_identically(self):
        fresh = common.make_split(self.frame, self.split["dataset_sha256"])
        self.assertEqual(fresh["train_indices"], self.split["train_indices"])
        self.assertEqual(fresh["test_indices"], self.split["test_indices"])

    def test_plan_example_specimen_is_held_out(self):
        match = self.frame
        for field, value in HOLDOUT.items():
            match = match[match[field] == value]
        match = match[(match["sex"] == "F") & (match["number_of_rings"] == 12)]
        self.assertEqual(len(match), 1)
        self.assertIn(int(match.index[0]), self.split["test_indices"])


# Hold-out figures published in the plan (fixed split, seed 42, scikit-learn 1.4.2).
PLAN_METRICS = {
    "abalone_sex_random_forest": {"accuracy": 0.5437, "macro_f1": 0.5388, "infant_vs_adult_accuracy": 0.8228, "mf_accuracy_among_adults": 0.5166},
    "abalone_sex_logistic_regression": {"accuracy": 0.5569, "macro_f1": 0.5451, "infant_vs_adult_accuracy": 0.8000, "mf_accuracy_among_adults": 0.5509},
    "abalone_sex_maturity_rules": {"infant_vs_adult_accuracy": 0.8120},
    "abalone_sex_bivariate_profile": {"accuracy": 0.5269, "macro_f1": 0.5262, "infant_vs_adult_accuracy": 0.7904},
    "abalone_age_gradient_boosting": {"mae": 1.5140, "rmse": 2.1847, "r2": 0.5545, "within_2_rings": 0.7533, "interval_coverage": 0.8012},
    "abalone_age_ridge_regression": {"mae": 1.6507, "rmse": 2.5446, "r2": 0.3956, "within_2_rings": 0.7317, "interval_coverage": 0.7964},
    "abalone_age_allometric_formula": {"mae": 1.7484, "rmse": 2.5099, "r2": 0.4120, "within_2_rings": 0.7030, "interval_coverage": 0.8263},
    "abalone_age_shell_weight_lookup": {"mae": 1.7353, "rmse": 2.6525, "r2": 0.3433, "within_2_rings": 0.7976, "interval_coverage": 0.8491},
}


@unittest.skipUnless(trained_with_plan_versions(), "plan figures were measured with scikit-learn 1.4.2 / numpy 1.26")
class TestPlanReference(unittest.TestCase):
    """Phase 2 acceptance: the trained tools reproduce the plan's hold-out metrics within 0.005."""

    def test_metrics_match_the_plan(self):
        tools = json.loads(METRICS_PATH.read_text())["tools"]
        for tool_id, expected in PLAN_METRICS.items():
            for metric, value in expected.items():
                with self.subTest(tool=tool_id, metric=metric):
                    self.assertAlmostEqual(tools[tool_id]["holdout_metrics"][metric], value, delta=0.005)

    def test_comparison_table_values(self):
        def first(name):
            return call(name, {"specimens": [specimen(name)]})["result"]["predictions"][0]

        forest = first("abalone_sex_random_forest")
        self.assertEqual(forest["predicted_sex"], "F")
        self.assertAlmostEqual(forest["probabilities"]["F"], 0.466, delta=0.01)
        self.assertEqual(first("abalone_sex_logistic_regression")["predicted_sex"], "I")
        self.assertEqual(first("abalone_sex_bivariate_profile")["predicted_sex"], "M")
        self.assertEqual(first("abalone_sex_maturity_rules")["verdict"], "ADULT")
        expected_age = {
            "abalone_age_gradient_boosting": (9.15, 6.83, 11.48),
            "abalone_age_ridge_regression": (9.83, 7.46, 12.19),
            "abalone_age_allometric_formula": (9.37, 6.99, 12.54),
            "abalone_age_shell_weight_lookup": (9.0, 7.0, 12.9),
        }
        for name, (rings, low, high) in expected_age.items():
            with self.subTest(tool=name):
                prediction = first(name)
                self.assertAlmostEqual(prediction["predicted_rings"], rings, delta=0.05)
                self.assertAlmostEqual(prediction["interval_80"]["low"], low, delta=0.05)
                self.assertAlmostEqual(prediction["interval_80"]["high"], high, delta=0.05)


ML_TOOLS = {"abalone_sex_random_forest", "abalone_sex_logistic_regression", *AGE_ML_TOOLS}
AGENT_FILES = ("abalone_sex_agent.json", "abalone_age_agent.json", "abalone_consolidator_agent.json")
FLOW_PATH = REPO_ROOT / "declarations" / "flows" / "abalone_tools_flow.json"


class TestAgentsAndFlow(unittest.TestCase):
    """Phase 6: the agent and flow declarations reference only things that exist."""

    def setUp(self):
        self.agents = {
            path: tool_registry.read_declaration(REPO_ROOT / "declarations" / "agents" / path) for path in AGENT_FILES
        }
        self.flow = tool_registry.read_declaration(FLOW_PATH)

    def test_agent_bindings_use_exposed_tools_in_the_right_list(self):
        for path, agent in self.agents.items():
            with self.subTest(agent=path):
                for binding in agent["tools"]:
                    self.assertIn(binding["tool_id"], ML_TOOLS)
                    self.assertEqual(binding["access"], "mcp")
                for binding in agent["expert_tools"]:
                    self.assertIn(binding["tool_id"], set(ALL_TOOLS) - ML_TOOLS)
                    self.assertTrue(agent["role"]["permissions"]["can_use_expert_tools"])
                if agent["role"]["type"] == "specialist":
                    self.assertGreaterEqual(len(agent["tools"]), 1)
                    self.assertGreaterEqual(len(agent["expert_tools"]), 1)

    def test_agent_output_formats_are_valid_json_schema(self):
        from jsonschema import Draft202012Validator

        for path, agent in self.agents.items():
            with self.subTest(agent=path):
                Draft202012Validator.check_schema(agent["prompt"]["output_format"]["schema"])

    def test_flow_references_existing_agents_and_sections(self):
        nodes = self.flow["nodes"]
        self.assertIn(self.flow["entry_node"], nodes)
        self.assertIn(self.flow["terminal_node"], nodes)
        for node_id, node in nodes.items():
            with self.subTest(node=node_id):
                agent = tool_registry.read_declaration(REPO_ROOT / node["agent_declaration"])
                self.assertEqual(agent["id"], node_id)
        for edge in self.flow["edges"]:
            self.assertIn(edge["from"], nodes)
            self.assertIn(edge["to"], nodes)

        blackboard = (REPO_ROOT / "blackboard.md").read_text(encoding="utf-8")
        known = {line.split("SECTION:")[1].strip(" ->") for line in blackboard.splitlines() if line.startswith("<!-- SECTION:")}
        known |= {section["id"] for section in self.flow["blackboard_sections"]}
        written = [node["blackboard_permissions"]["write_section"] for node in nodes.values()]
        self.assertEqual(len(written), len(set(written)), "two nodes write the same section")
        for node in nodes.values():
            permissions = node["blackboard_permissions"]
            self.assertIn(permissions["write_section"], known)
            for section in permissions["read_sections"]:
                if section != "*":
                    self.assertIn(section, known)

    def test_flow_and_mcp_json_launch_the_same_server(self):
        mcp_json = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["abalone_tools"]
        flow_entry = self.flow["mcp_servers"]["abalone_tools"]
        self.assertEqual((mcp_json["command"], mcp_json["args"]), (flow_entry["command"], flow_entry["args"]))


def mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return ARTIFACTS_READY


@unittest.skipUnless(mcp_available(), "needs the mcp package (requirements-abalone-tools.txt) and trained artifacts")
class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    """Phase 5 acceptance, through a real MCP client talking to code/mcp_server.py over stdio."""

    async def test_list_and_call_over_stdio(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.shared.exceptions import MCPError

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(CODE_DIR / "mcp_server.py"), "--server", SERVER],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = (await session.list_tools()).tools
                self.assertEqual({tool.name for tool in listed}, set(ALL_TOOLS))
                for tool in listed:
                    declaration = DECLARATIONS[tool.name]
                    self.assertEqual(tool.title, declaration["mcp"]["title"])
                    self.assertEqual(tool.description, declaration["prompt_context"])
                    self.assertEqual(tool.input_schema, declaration["input_schema"])
                    self.assertEqual(tool.output_schema, declaration["output_schema"])
                    self.assertTrue(tool.annotations.read_only_hint)
                    self.assertFalse(tool.annotations.destructive_hint)

                for name in ALL_TOOLS:
                    with self.subTest(tool=name):
                        result = await session.call_tool(name, {"specimens": [specimen(name)]})
                        self.assertFalse(result.is_error)
                        self.assertEqual(json.loads(result.content[0].text), result.structured_content)
                        self.assertEqual(result.structured_content, call(name, {"specimens": [specimen(name)]}))

                failed = await session.call_tool("abalone_sex_random_forest", {"specimens": [specimen(SEX_TOOLS[0], length=95)]})
                self.assertTrue(failed.is_error)
                self.assertEqual(failed.structured_content["error"]["type"], "input_error")
                self.assertIn("divide by 200", json.loads(failed.content[0].text)["error"]["message"])

                with self.assertRaises(MCPError):
                    await session.call_tool("abalone_does_not_exist", {})


if __name__ == "__main__":
    unittest.main()
