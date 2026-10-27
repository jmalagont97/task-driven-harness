"""test_forced_sex.py

Tests for the forced-class sex debate (abalone_sex_forced_debate_flow.json), in which the tool agents and the arbiter
must answer M, F or I: the declarations, a whole scripted debate, the arbiter fallback that picks a class from the
tools' probabilities, and the num_ctx override used to fit gemma4:31b on the GPU.

    .venv/bin/python -m unittest tests.test_forced_sex -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.flow_runner import FlowRun, RunOptions  # noqa: E402
from runtime.ollama_chat import ChatReply, OllamaChat  # noqa: E402
from runtime.problem_statement import problems_from_csv  # noqa: E402
from tests.test_debate_protocol import SMOKE_SEX, FakeGateway, ScriptedLLM, sex_position, turn  # noqa: E402

AGENTS = REPO_ROOT / "declarations" / "agents"
FORCED_FLOW = REPO_ROOT / "declarations" / "flows" / "abalone_sex_forced_debate_flow.json"
FORCED_AGENTS = ("abalone_sex_forced_agent_a", "abalone_sex_forced_agent_b", "abalone_debate_forced_consolidator_agent")


def load(agent_id: str) -> dict:
    return json.loads((AGENTS / f"{agent_id}.json").read_text(encoding="utf-8"))


def sex_call_enum(agent: dict) -> list[str]:
    result = agent["prompt"]["output_format"]["schema"]["properties"]["result"]["properties"]
    holder = result["position"]["properties"] if "position" in result else result
    return holder["sex_call"]["enum"]


class TestForcedDeclarations(unittest.TestCase):
    def test_forced_agents_only_accept_a_class(self):
        for agent_id in FORCED_AGENTS:
            self.assertEqual(sex_call_enum(load(agent_id)), ["M", "F", "I"], agent_id)

    def test_standard_agents_keep_adult_undetermined(self):
        for agent_id in ("abalone_sex_agent_a", "abalone_sex_agent_b", "abalone_debate_consolidator_agent"):
            self.assertIn("adult_undetermined", sex_call_enum(load(agent_id)), agent_id)

    def test_forced_flow_uses_only_forced_tool_agents_and_arbiter(self):
        flow = json.loads(FORCED_FLOW.read_text(encoding="utf-8"))
        self.assertEqual(flow["entry_node"], "abalone_sex_forced_agent_a")
        self.assertEqual(flow["terminal_node"], "abalone_debate_forced_consolidator_agent")
        for node_id, node in flow["nodes"].items():
            self.assertEqual(Path(node["agent_declaration"]).stem, node_id)
        self.assertEqual(flow["mcp_servers"]["abalone_sex_forced_agent_a"]["args"][-1], "declarations/agents/abalone_sex_forced_agent_a.json")
        self.assertNotIn("abalone_sex_agent_a", json.dumps(flow))

    def test_forced_agents_bind_the_same_tools(self):
        for letter in ("a", "b"):
            standard, forced = load(f"abalone_sex_agent_{letter}"), load(f"abalone_sex_forced_agent_{letter}")
            for key in ("tools", "expert_tools"):
                self.assertEqual([b["tool_id"] for b in standard[key]], [b["tool_id"] for b in forced[key]])


class ForcedLLM(ScriptedLLM):
    """The scripted sex debate of test_debate_protocol, with every adult_undetermined replaced by M and the claim ids
    renamed to the forced agents' codes."""

    def __init__(self, agent: dict):
        super().__init__(agent)
        self.code = self.code.replace("_FORCED", "")  # SEX_FORCED_A -> SEX_A, as the script names it

    async def chat(self, messages, tools=None, schema=None, max_tokens=None):
        reply = await super().chat(messages, tools, schema, max_tokens)
        if schema is None:
            return reply
        content = reply.content.replace('"adult_undetermined"', '"M"').replace("-SEX_A-", "-SEX_FORCED_A-").replace("-SEX_B-", "-SEX_FORCED_B-")
        return ChatReply(content, reply.tool_calls, reply.stats)


class TestForcedDebate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runs = Path(tempfile.mkdtemp(prefix="forced_run_"))

    async def asyncTearDown(self):
        shutil.rmtree(self.runs, ignore_errors=True)

    async def test_forced_debate_answers_a_class(self):
        problem = problems_from_csv(SMOKE_SEX)[0]
        options = RunOptions(runs_dir=self.runs, llm_factory=ForcedLLM, gateway_factory=FakeGateway, echo=lambda _: None)
        result = await FlowRun(FORCED_FLOW, problem, options).run()
        final = result.final["result"]
        self.assertEqual((final["answer"], final["sex_call"], final["forced_choice"]), ("M", "M", "M"))
        self.assertFalse(final.get("arbiter_fallback", False))
        self.assertEqual(result.debate["round"], 3)
        self.assertTrue(result.debate["converged"])
        discussion = (result.run_dir / "discussion.md").read_text()
        self.assertIn("### SEX_FORCED_A", discussion)
        self.assertNotIn("adult_undetermined", discussion)

    def test_fallback_picks_the_class_the_tools_favour(self):
        problem = problems_from_csv(SMOKE_SEX)[0]
        run = FlowRun(FORCED_FLOW, problem, RunOptions(runs_dir=self.runs, echo=lambda _: None))
        run.debate.record_turn("abalone_sex_forced_agent_a", turn(sex_position("adult", "M"), [{"text": "adult", "evidence": ["rf"]}]))
        run.debate.record_turn("abalone_sex_forced_agent_b", turn(sex_position("adult", "F"), [{"text": "adult", "evidence": ["lr"]}]))
        envelope = {"ok": True, "result": {"predictions": [{"probabilities": {"M": 0.3, "F": 0.45, "I": 0.25}}]}}
        run.records = [{"tool_calls": [{"tool": "abalone_sex_random_forest", "ok": True, "result": envelope}]},
                       {"tool_calls": [{"tool": "abalone_sex_logistic_regression", "ok": True, "result": envelope}]}]
        final = run.fallback_decision("arbiter broke")["result"]
        run.transcript.close()
        self.assertEqual((final["sex_call"], final["forced_choice"], final["maturity"]), ("F", "F", "adult"))
        self.assertTrue(final["arbiter_fallback"])

    def test_fallback_keeps_an_infant_an_infant(self):
        problem = problems_from_csv(SMOKE_SEX)[0]
        run = FlowRun(FORCED_FLOW, problem, RunOptions(runs_dir=self.runs, echo=lambda _: None))
        run.debate.record_turn("abalone_sex_forced_agent_a", turn(sex_position("infant", "I"), [{"text": "infant", "evidence": ["rules"]}]))
        run.debate.record_turn("abalone_sex_forced_agent_b", turn(sex_position("infant", "M"), [{"text": "infant", "evidence": ["lr"]}]))
        final = run.fallback_decision("arbiter broke")["result"]
        run.transcript.close()
        self.assertEqual(final["sex_call"], "I")


class TestNumCtxOverride(unittest.TestCase):
    def test_override_replaces_the_declared_context(self):
        agent = load("abalone_sex_forced_agent_a")
        self.assertEqual(OllamaChat(agent).options["num_ctx"], agent["runtime"]["adapters"]["ollama"]["num_ctx"])
        self.assertEqual(OllamaChat(agent, "gemma4:31b", num_ctx_override=20480).options["num_ctx"], 20480)


if __name__ == "__main__":
    unittest.main()
