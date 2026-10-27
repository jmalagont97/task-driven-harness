"""test_debate_protocol.py

Tests for the debate runtime without any model server: the Debate rules (rounds, claim and challenge ids, caps,
abstentions, skips, agreement, the extra cycle, unanswered challenges), the flow-condition evaluator, and a whole sex
debate through FlowRun with a scripted LLM and fake tool gateways.

    .venv/bin/python -m unittest tests.test_debate_protocol -v
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime.agent_runner import cited_chunk_ids, repair, sanitize_citations  # noqa: E402
from runtime.debate import Debate, Participant, participant_code  # noqa: E402
from runtime.flow_runner import FlowRun, RunOptions, consistent_sex_answer, evaluate  # noqa: E402
from runtime.ollama_chat import ChatReply  # noqa: E402
from runtime.problem_statement import problems_from_csv  # noqa: E402

SEX_FLOW = REPO_ROOT / "declarations" / "flows" / "abalone_sex_debate_flow.json"
SMOKE_SEX = REPO_ROOT / "evals" / "abalone_debate" / "smoke_sex.csv"


def sex_position(maturity="adult", call="adult_undetermined"):
    return {"maturity": maturity, "sex_call": call, "reliability": "medium", "rationale": "test"}


def turn(position=None, claims=(), reviews=(), responses=(), status="completed", expectation=None):
    result = {"claims": list(claims), "reviews": list(reviews), "responses": list(responses), "markdown_summary": "summary"}
    if position is not None:
        result["position"] = position
    if expectation is not None:
        result["expectation"] = expectation
    return {"status": status, "result": result, "warnings": []}


def small_debate(task="sex", max_cycles=2) -> Debate:
    return Debate(task, [
        Participant("abalone_sex_agent_a", "SEX_A", "tool_agent", "A", "tools"),
        Participant("abalone_sex_agent_b", "SEX_B", "tool_agent", "B", "tools"),
        Participant("article_nash_1994_agent", "NASH_1994", "expert", "Nash", "article"),
        Participant("article_mcshane_1992_agent", "MCSHANE_1992", "expert", "McShane", "article"),
    ], max_cycles=max_cycles)


class TestDebateRules(unittest.TestCase):
    def test_participant_codes(self):
        self.assertEqual(participant_code("abalone_sex_agent_a"), "SEX_A")
        self.assertEqual(participant_code("article_nash_1994_agent"), "NASH_1994")
        self.assertEqual(participant_code("article_prince_1988a_agent"), "PRINCE_1988A")

    def test_round_one_claims_get_ids_and_abstention_silences_an_expert(self):
        debate = small_debate()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), [{"text": "adult", "evidence": ["rules: DEFAULT"]}, {"text": "no evidence", "evidence": []}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position(), [{"text": "adult too", "evidence": ["lr: p_adult 0.9"]}]))
        debate.record_turn("article_nash_1994_agent", turn(None, [{"text": "mature at 91 mm", "evidence": ["nash_et_al_1994_tr48:095"]}], expectation="adult"))
        debate.record_turn("article_mcshane_1992_agent", turn(None, status="abstained"))
        self.assertEqual(sorted(debate.claims), ["R1-NASH_1994-1", "R1-SEX_A-1", "R1-SEX_B-1"])
        self.assertTrue(debate.finish_pass())
        self.assertEqual(debate.kind, "challenges")
        self.assertFalse(debate.needs_turn("article_mcshane_1992_agent"))
        self.assertTrue(debate.needs_turn("article_nash_1994_agent"))

    def test_reviews_are_capped_checked_and_create_challenges(self):
        debate = small_debate()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), [{"text": "a", "evidence": ["x"]}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position("infant", "I"), [{"text": "b1", "evidence": ["y"]}, {"text": "b2", "evidence": ["y"]}]))
        debate.record_turn("article_nash_1994_agent", turn(None, [{"text": "n", "evidence": ["nash_et_al_1994_tr48:001"]}]))
        debate.record_turn("article_mcshane_1992_agent", turn(None, [{"text": "m", "evidence": ["mcshane_smith_1992:001"]}]))
        debate.finish_pass()
        warnings = debate.record_turn("article_nash_1994_agent", turn(None, reviews=[
            {"claim_id": "R1-NASH_1994-1", "verdict": "confirm", "reason": "own", "evidence": ["nash_et_al_1994_tr48:001"]},
            {"claim_id": "R1-SEX_B-1", "verdict": "challenge", "reason": "too big for an infant", "evidence": ["nash_et_al_1994_tr48:001"]},
            {"claim_id": "R1-SEX_B-2", "verdict": "challenge", "reason": "no evidence", "evidence": []},
            {"claim_id": "R9-NOPE-1", "verdict": "challenge", "reason": "unknown", "evidence": ["nash_et_al_1994_tr48:001"]},
            {"claim_id": "R1-SEX_A-1", "verdict": "confirm", "reason": "agree", "evidence": ["nash_et_al_1994_tr48:001"]},
            {"claim_id": "R1-MCSHANE_1992-1", "verdict": "confirm", "reason": "agree", "evidence": ["nash_et_al_1994_tr48:001"]},
            {"claim_id": "R1-SEX_B-2", "verdict": "challenge", "reason": "fourth review", "evidence": ["nash_et_al_1994_tr48:001"]},
        ]))
        self.assertEqual(list(debate.challenges), ["C1"])
        self.assertEqual(debate.challenges["C1"].target, "abalone_sex_agent_b")
        self.assertEqual(len([r for c in debate.claims.values() for r in c.reviews if r["by"] == "article_nash_1994_agent"]), 3)
        self.assertTrue(any("own claim" in w for w in warnings))
        self.assertTrue(any("without evidence" in w for w in warnings))
        self.assertTrue(any("unknown claim" in w for w in warnings))
        self.assertTrue(any("more than 3" in w for w in warnings))

    def test_ids_written_with_brackets_are_recognised(self):
        debate = small_debate()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), [{"text": "a", "evidence": ["x"]}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position(), [{"text": "b", "evidence": ["y"]}]))
        debate.finish_pass()
        warnings = debate.record_turn("abalone_sex_agent_b", turn(sex_position(), reviews=[{"claim_id": "[r1-sex_a-1]", "verdict": "challenge", "reason": "r", "evidence": ["y"]}]))
        self.assertEqual(warnings, [])
        self.assertEqual(debate.challenges["C1"].claim_id, "R1-SEX_A-1")
        debate.finish_pass()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), responses=[{"challenge_id": " [C1] ", "action": "defend", "text": "d", "evidence": ["x"]}]))
        self.assertEqual(debate.challenges["C1"].status, "defended")

    def test_replies_close_the_loop_and_unanswered_is_marked(self):
        debate = small_debate()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), [{"text": "a", "evidence": ["x"]}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position(), [{"text": "b", "evidence": ["y"]}]))
        debate.finish_pass()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), reviews=[{"claim_id": "R1-SEX_B-1", "verdict": "challenge", "reason": "r", "evidence": ["x"]}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position(), reviews=[{"claim_id": "R1-SEX_A-1", "verdict": "challenge", "reason": "r", "evidence": ["y"]}]))
        self.assertTrue(debate.finish_pass())
        self.assertEqual(debate.kind, "replies")
        self.assertFalse(debate.needs_turn("article_nash_1994_agent"))
        self.assertTrue(debate.needs_turn("abalone_sex_agent_a"))
        warnings = debate.record_turn("abalone_sex_agent_a", turn(sex_position(), claims=[{"text": "new", "evidence": ["x"]}],
                                                                   responses=[{"challenge_id": "C2", "action": "defend", "text": "d", "evidence": ["x"]}]))
        self.assertTrue(any("only accepted in round 1" in w for w in warnings))
        self.assertFalse(debate.finish_pass(), "positions agree, so no extra cycle")
        self.assertEqual(debate.challenges["C2"].status, "defended")
        self.assertEqual(debate.challenges["C1"].status, "unanswered")
        self.assertTrue(debate.converged())

    def test_extra_cycle_only_on_disagreement_with_a_defended_dispute(self):
        debate = small_debate()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), [{"text": "a", "evidence": ["x"]}]))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position("infant", "I"), [{"text": "b", "evidence": ["y"]}]))
        debate.finish_pass()
        debate.record_turn("abalone_sex_agent_b", turn(sex_position("infant", "I"), reviews=[{"claim_id": "R1-SEX_A-1", "verdict": "challenge", "reason": "r", "evidence": ["y"]}]))
        debate.finish_pass()
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), responses=[{"challenge_id": "C1", "action": "defend", "text": "d", "evidence": ["x"]}]))
        self.assertTrue(debate.finish_pass())
        self.assertEqual((debate.round, debate.kind), (4, "challenges"))
        self.assertEqual(debate.disputed_claims, {"R1-SEX_A-1"})
        self.assertFalse(debate.needs_turn("article_mcshane_1992_agent"))
        debate.record_turn("abalone_sex_agent_b", turn(sex_position("infant", "I"), reviews=[{"claim_id": "R1-SEX_A-1", "verdict": "challenge", "reason": "again", "evidence": ["y"]}]))
        self.assertTrue(debate.finish_pass())
        debate.record_turn("abalone_sex_agent_a", turn(sex_position(), responses=[{"challenge_id": "C2", "action": "defend", "text": "still", "evidence": ["x"]}]))
        self.assertFalse(debate.finish_pass(), "at most one extra cycle")
        self.assertFalse(debate.converged())
        self.assertIn("| R1-SEX_A-1 | SEX_A |", debate.ledger_markdown())

    def test_age_agreement_needs_close_estimates_and_overlapping_intervals(self):
        debate = Debate("age", [Participant("abalone_age_agent_a", "AGE_A", "tool_agent", "A", ""), Participant("abalone_age_agent_b", "AGE_B", "tool_agent", "B", "")])
        position = lambda r, lo, hi: {"rings": r, "rings_low": lo, "rings_high": hi, "reliability": "medium", "rationale": ""}
        debate.record_turn("abalone_age_agent_a", turn(position(11, 9, 13)))
        debate.record_turn("abalone_age_agent_b", turn(position(11.8, 10, 14)))
        self.assertTrue(debate.converged())
        debate.record_turn("abalone_age_agent_b", turn(position(12.5, 10, 15)))
        self.assertFalse(debate.converged())

    def test_conditions(self):
        context = {"debate": {"next_round_needed": True}, "cycle_count": 1, "max_cycle_iterations": 4, "problem_spec": {"task": "sex"}}
        self.assertTrue(evaluate("debate.next_round_needed == true AND cycle_count < max_cycle_iterations", context))
        self.assertFalse(evaluate("debate.next_round_needed == false", context))
        self.assertTrue(evaluate("problem_spec.task in ['sex', 'age']", context))
        with self.assertRaises(ValueError):
            evaluate("__import__('os')", context)

    def test_repair_drops_only_malformed_items(self):
        from jsonschema import Draft202012Validator

        schema = json.loads((REPO_ROOT / "declarations/agents/abalone_sex_agent_a.json").read_text())["prompt"]["output_format"]["schema"]
        validator = Draft202012Validator(schema)
        answer = turn(sex_position(), claims=[{"text": "kept", "evidence": ["rf: 0.9"]}, {"text": "no evidence", "evidence": []}])
        fixed, notes = repair(answer, validator)
        self.assertEqual([c["text"] for c in fixed["result"]["claims"]], ["kept"])
        self.assertTrue(notes)
        broken_position = turn({"maturity": "adult"}, claims=[])
        self.assertEqual(repair(broken_position, validator), (broken_position, []), "errors outside the arrays are not repaired")

    def test_arbiter_forced_choice_follows_a_definite_call(self):
        self.assertIsNone(consistent_sex_answer({"result": {"maturity": "adult", "sex_call": "adult_undetermined", "forced_choice": "M"}}))
        self.assertIsNone(consistent_sex_answer({"result": {"maturity": "adult", "sex_call": "adult_undetermined", "forced_choice": "I"}}),
                          "the most likely single class may be I while an adult is more likely")
        self.assertIn("must equal", consistent_sex_answer({"result": {"maturity": "adult", "sex_call": "M", "forced_choice": "F"}}))

    def test_citation_sanitizer_keeps_only_retrieved_chunks(self):
        output = turn(None, [{"text": "ok", "evidence": ["nash_et_al_1994_tr48:001"]}, {"text": "made up", "evidence": ["nash_et_al_1994_tr48:999"]}])
        self.assertEqual(cited_chunk_ids(output), {"nash_et_al_1994_tr48:001", "nash_et_al_1994_tr48:999"})
        warnings = sanitize_citations(output, {"nash_et_al_1994_tr48:001"})
        self.assertEqual([c["text"] for c in output["result"]["claims"]], ["ok"])
        self.assertTrue(warnings)
        reply = turn(None, responses=[{"challenge_id": "C1", "action": "defend", "text": "other article", "evidence": ["mcshane_smith_1992:002"]}])
        sanitize_citations(reply, {"nash_et_al_1994_tr48:001"})
        self.assertEqual([(r["challenge_id"], r["action"]) for r in reply["result"]["responses"]], [("C1", "clarify")])


# ---------------------------------------------------------------------------
# A whole debate with a scripted model
# ---------------------------------------------------------------------------

ROUND = re.compile(r"Round (\d+): (\w+)\. You are (\S+)")


class FakeGateway:
    def __init__(self, node_id: str, agent: dict):
        self.node_id = node_id
        self.expert = agent["role"]["permissions"]["can_use_rag"]
        self.document = {"article_nash_1994_agent": "nash_et_al_1994_tr48", "article_prince_1988a_agent": "prince_et_al_1988_ageing",
                         "article_prince_1988b_agent": "prince_et_al_1988_marbiol", "article_mcshane_1992_agent": "mcshane_smith_1992"}.get(node_id)
        names = ["search_article", "get_section"] if self.expert else [b["tool_id"] for k in ("tools", "expert_tools") for b in agent[k]]
        self.tools = [SimpleNamespace(name=n, description=n, input_schema={"type": "object"}) for n in names]

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        if self.expert:
            envelope = {"ok": True, "result": {"results": [{"chunk_id": f"{self.document}:00{i}", "text": "passage"} for i in (1, 2)]}}
        else:
            envelope = {"ok": True, "result": {"predictions": [{"index": 0, "predicted_sex": "M", "p_adult": 0.9}]}}
        return envelope, json.dumps(envelope), False


def scripted_output(code: str, round_number: int, attempt: int) -> dict:
    """What each participant says in each round of the scripted sex debate."""
    if code == "DEBATE_CONSOLIDATOR":
        return {"status": "completed", "warnings": [], "result": {
            "answer": "adult_undetermined", "sex_call": "adult_undetermined", "maturity": "adult", "forced_choice": "M", "reliability": "medium",
            "debate_outcome": "resolved_by_evidence", "rationale": "B conceded to Nash.",
            "participants": [{"participant": "SEX_B", "final_position": "adult", "weight": "accepted", "why": "conceded with evidence"}],
            "markdown_summary": "Final: adult_undetermined."}}
    if round_number == 1:
        return {
            "SEX_A": turn(sex_position(), [{"text": "No infant rule fired", "evidence": ["abalone_sex_maturity_rules: DEFAULT"]}]),
            "SEX_B": turn(sex_position("infant", "I"), [{"text": "Nearest profile is I", "evidence": ["abalone_sex_bivariate_profile: nearest I"]}]),
            "NASH_1994": turn(None, [{"text": "91 mm is above every L50", "evidence": ["nash_et_al_1994_tr48:001"]}], expectation="adult [nash_et_al_1994_tr48:001]"),
            "PRINCE_1988A": turn(None, status="abstained", expectation=""),
            # First attempt cites a chunk its tools never returned; the runner must ask again.
            "PRINCE_1988B": turn(None, [{"text": "maturity from 90 mm", "evidence": ["prince_et_al_1988_marbiol:999" if attempt == 0 else "prince_et_al_1988_marbiol:002"]}], expectation="adult"),
            "MCSHANE_1992": turn(None, [{"text": "rings are not annual", "evidence": ["mcshane_smith_1992:001"]}], expectation="ring counts are uncertain"),
        }[code]
    if round_number == 2:
        return {
            "SEX_A": turn(sex_position(), reviews=[{"claim_id": "R1-SEX_B-1", "verdict": "challenge", "reason": "rings 15 > 9", "evidence": ["abalone_sex_maturity_rules: DEFAULT"]}]),
            "SEX_B": turn(sex_position("infant", "I"), reviews=[{"claim_id": "R1-SEX_A-1", "verdict": "challenge", "reason": "profile says I", "evidence": ["abalone_sex_bivariate_profile: nearest I"]}]),
            "NASH_1994": turn(None, reviews=[{"claim_id": "R1-SEX_B-1", "verdict": "challenge", "reason": "too large for an infant", "evidence": ["nash_et_al_1994_tr48:001"]},
                                             {"claim_id": "R1-SEX_A-1", "verdict": "confirm", "reason": "consistent", "evidence": ["nash_et_al_1994_tr48:002"]}], expectation=""),
            "PRINCE_1988B": turn(None, reviews=[{"claim_id": "R1-NASH_1994-1", "verdict": "confirm", "reason": "agrees", "evidence": ["prince_et_al_1988_marbiol:002"]}], expectation=""),
            "MCSHANE_1992": turn(None, reviews=[{"claim_id": "R1-SEX_A-1", "verdict": "challenge", "reason": "rule uses rings as age", "evidence": ["mcshane_smith_1992:001"]}], expectation=""),
        }[code]
    return {
        "SEX_A": turn(sex_position(), responses=[{"challenge_id": "C2", "action": "defend", "text": "rules fit this dataset", "evidence": ["abalone_sex_maturity_rules: DEFAULT"]},
                                                 {"challenge_id": "C4", "action": "clarify", "text": "the rule counts rings, not years", "evidence": []}]),
        "SEX_B": turn(sex_position(), responses=[{"challenge_id": "C1", "action": "concede", "text": "rings rule out infant", "evidence": []},
                                                 {"challenge_id": "C3", "action": "concede", "text": "Nash is right", "evidence": []}]),
    }[code]


class ScriptedLLM:
    def __init__(self, agent: dict):
        self.code = participant_code(agent["id"])
        self.answers = 0

    async def chat(self, messages, tools=None, schema=None, max_tokens=None):
        system = messages[0]["content"]
        match = ROUND.search(system)
        round_number = int(match.group(1)) if match else 0
        if schema is None:
            called = {m.get("tool_name") for m in messages if m["role"] == "tool"}
            names = [t["function"]["name"] for t in tools or []]
            if match and match.group(2) == "OPINIONS" and not called:
                wanted = ["search_article"] if "search_article" in names else names
                args = {"query": "maturity size"} if "search_article" in names else {"specimens": [{}]}
                return ChatReply("", [{"function": {"name": n, "arguments": args}} for n in wanted], {"wall_s": 0})
            return ChatReply("ready", [], {"wall_s": 0})
        rejected = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("Your answer was rejected"))
        self.answers += 1
        return ChatReply(json.dumps(scripted_output(self.code, round_number, rejected)), [], {"wall_s": 0, "prompt_tokens": 10, "output_tokens": 5})


class TestFlowRunWithScriptedModel(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runs = Path(tempfile.mkdtemp(prefix="debate_run_"))

    async def asyncTearDown(self):
        shutil.rmtree(self.runs, ignore_errors=True)

    async def test_full_sex_debate(self):
        problem = problems_from_csv(SMOKE_SEX)[0]
        options = RunOptions(runs_dir=self.runs, llm_factory=ScriptedLLM, gateway_factory=FakeGateway, echo=lambda _: None)
        result = await FlowRun(SEX_FLOW, problem, options).run()

        self.assertEqual(result.final["result"]["answer"], "adult_undetermined")
        debate = result.debate
        self.assertEqual(debate["round"], 3)
        self.assertTrue(debate["converged"], "B conceded and ended with A's position")
        self.assertEqual(debate["abstained"], ["PRINCE_1988A"])
        statuses = {c["challenge_id"]: c["status"] for row in debate["ledger"] for c in row["challenges"]}
        self.assertEqual(statuses, {"C1": "conceded", "C2": "defended", "C3": "conceded", "C4": "clarified"})
        # 6 opinions + 5 reviews (the abstainer is skipped) + 2 replies + the arbiter
        self.assertEqual(result.stats["turns"], 14)
        self.assertEqual(result.stats["skipped"], 5)

        b = result.final["result"]["participants"][0]
        self.assertEqual(b["participant"], "SEX_B")
        self.assertEqual({c["challenge_id"] for c in b["challenges_received"]}, {"C1", "C3"})

        run_dir = result.run_dir
        blackboard = (run_dir / "blackboard.md").read_text()
        self.assertIn("### Round 3 (replies)", blackboard)
        self.assertIn("- length: 91 mm", blackboard)
        self.assertIn("| R1-SEX_A-1 | SEX_A |", blackboard)
        self.assertIn("Final: adult_undetermined.", blackboard)
        events = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]
        sex_a = [e for e in events if e.get("event") == "turn" and e.get("label") == "R1-SEX_A"][0]
        self.assertEqual({c["tool"] for c in sex_a["tool_calls"]}, {"abalone_sex_random_forest", "abalone_sex_maturity_rules"})
        for call in sex_a["tool_calls"]:
            self.assertEqual(call["corrected_from"], {"specimens": [{}]}, "the scripted model sends an empty specimen")
            self.assertEqual(call["arguments"], {"specimens": [problem.specimen()]})
        self.assertTrue(any("replaced by the exact values" in w for w in sex_a["warnings"]))
        prince_b = [e for e in events if e.get("event") == "turn" and e.get("label") == "R1-PRINCE_1988B"][0]
        self.assertTrue(any("did not return" in w for w in prince_b["warnings"]))
        self.assertEqual(prince_b["output"]["result"]["claims"][0]["evidence"], ["prince_et_al_1988_marbiol:002"])
        self.assertTrue((run_dir / "final_decision.json").is_file())

        discussion = (run_dir / "discussion.md").read_text()
        for fragment in ("## Problem as the agents saw it", "- length: 91 mm", "## Round 1: opinions", "## Round 3: replies",
                         "Claim [R1-NASH_1994-1]", "Challenge [R1-SEX_A-1]", "Reply concede [C3]", "## Debate ledger",
                         "## Final decision (arbiter)", "Tool: `search_article` query \"maturity size\"", "Abstained"):
            self.assertIn(fragment, discussion)
        index = [json.loads(line) for line in (self.runs / "discussions_index.jsonl").read_text().splitlines()]
        self.assertEqual(len(index), 1)
        self.assertEqual((index[0]["kind"], index[0]["task"], index[0]["answer"], index[0]["completed"]), ("debate", "sex", "adult_undetermined", True))


class BrokenArbiterLLM(ScriptedLLM):
    """Scripted participants, but an arbiter that never writes valid JSON."""

    async def chat(self, messages, tools=None, schema=None, max_tokens=None):
        if self.code == "DEBATE_CONSOLIDATOR" and schema is not None:
            return ChatReply('{"status": "completed", "result": {"answer": "cut off', [], {"wall_s": 0})
        return await super().chat(messages, tools, schema, max_tokens)


class TestArbiterFallback(unittest.IsolatedAsyncioTestCase):
    async def test_failed_arbiter_keeps_the_debate_with_a_low_reliability_answer(self):
        runs = Path(tempfile.mkdtemp(prefix="debate_fallback_"))
        try:
            problem = problems_from_csv(SMOKE_SEX)[0]
            options = RunOptions(runs_dir=runs, llm_factory=BrokenArbiterLLM, gateway_factory=FakeGateway, echo=lambda _: None)
            result = await FlowRun(SEX_FLOW, problem, options).run()
            final = result.final["result"]
            self.assertTrue(final["arbiter_fallback"])
            self.assertEqual((final["sex_call"], final["maturity"], final["reliability"]), ("adult_undetermined", "adult", "low"))
            discussion = (result.run_dir / "discussion.md").read_text()
            self.assertIn("arbiter failed", discussion)
            self.assertIn("Arbiter fallback: True", discussion)
        finally:
            shutil.rmtree(runs, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
