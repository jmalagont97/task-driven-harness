"""flow_runner.py

Walk a debate flow (declarations/flows/abalone_*_debate_flow.json) for one problem.

Responsibilities:
- Start one MCP server per node from the flow's mcp_servers (the key is the node id) and close them at the end;
  each server's stderr goes to the run folder.
- Keep the run's blackboard: the repository's blackboard.md as template plus the flow's blackboard_sections, written to
  artifacts/runs/<run>/blackboard.md. The repository's blackboard.md is never modified.
- Walk the graph: run a node when the debate says it has work in this round (otherwise pass without calling the model),
  follow edges by evaluating their conditions, take a loop edge while its continue_condition holds and
  cycle_count < max_cycle_iterations, and fall back to the terminal node when no edge applies or a guard is exhausted
  (global_max_turns counts model turns; max_visits counts a node's turns).
- Write the runtime-owned sections: problem_spec, debate_state (rendered for each participant before its turn) and
  debate_ledger (before the terminal node).
- Log every model call, tool call (with its result) and decision to transcript.jsonl, and save the debate state and the
  readable discussion (discussion.md) after every turn; each finished run adds a line to discussions_index.jsonl.

Conditions are evaluated by a small safe evaluator (comparisons, AND/OR/NOT, names and attributes); it supports the
expressions used by the flows in this repository and refuses anything else.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tool_registry
from runtime.agent_runner import AgentRunner, AgentState, MCPGateway, ToolGateway, TurnRequest, TurnResult
from runtime.debate import Debate, Participant, participant_code
from runtime.discussion import append_index, render_discussion
from runtime.ollama_chat import OllamaChat
from runtime.problem_statement import Problem

REPO_ROOT = tool_registry.REPO_ROOT
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
BLACKBOARD_TEMPLATE = REPO_ROOT / "blackboard.md"
SECTION = re.compile(r"<!-- SECTION: (\w+) -->\n(.*?)<!-- END_SECTION: \1 -->", re.DOTALL)
RUNTIME_SECTIONS = ("problem_spec", "debate_state", "debate_ledger")
QUESTION_STATE = (
    "Mode: standalone question. There is no debate and no specimen. Answer the question in problem_spec from your "
    "article only: call search_article first, put each finding in 'claims' with its chunk_ids as evidence, write "
    "'answer_markdown' in the language of the question with a [chunk_id] after every statement, and set 'coverage'. "
    "If the article does not cover the question, set coverage to 'not_covered' and answer only that the article does "
    "not cover this topic. Leave 'reviews' and 'responses' empty."
)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def evaluate(expression: str, context: dict[str, Any]) -> bool:
    text = re.sub(r"\bAND\b", "and", expression)
    text = re.sub(r"\bOR\b", "or", text)
    text = re.sub(r"\bNOT\b", "not", text)
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    return bool(_eval(ast.parse(text, mode="eval").body, context))


def _eval(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, context) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, context)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, context)
            ok = {
                ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
                ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
            }.get(type(op))
            if ok is None or not ok(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unknown name {node.id!r} in a flow condition.")
        return context[node.id]
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, context)
        if isinstance(base, dict) and node.attr in base:
            return base[node.attr]
        raise ValueError(f"Unknown attribute {node.attr!r} in a flow condition.")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, context) for e in node.elts]
    raise ValueError(f"Unsupported expression in a flow condition: {ast.dump(node)[:80]}")


# ---------------------------------------------------------------------------
# Blackboard of one run
# ---------------------------------------------------------------------------

class RunBlackboard:
    def __init__(self, template: str, extra_sections: list[dict[str, Any]]):
        self.header = template.split("<!-- SECTION:", 1)[0].rstrip() + "\n"
        self.order: list[str] = []
        self.sections: dict[str, dict[str, Any]] = {}
        for match in SECTION.finditer(template):
            body = match.group(2).strip()
            title, _, rest = body.partition("\n")
            self.order.append(match.group(1))
            self.sections[match.group(1)] = {"title": title.lstrip("# ").strip(), "content": "", "placeholder": rest.strip()}
        for section in extra_sections:
            anchor = section.get("insert_after")
            position = self.order.index(anchor) + 1 if anchor in self.order else len(self.order)
            self.order.insert(position, section["id"])
            self.sections[section["id"]] = {"title": section["title"], "content": "", "placeholder": section.get("placeholder", "")}

    def write(self, section_id: str, content: str) -> None:
        self.sections[section_id]["content"] = content.strip()

    def append(self, section_id: str, content: str) -> None:
        current = self.sections[section_id]["content"]
        self.sections[section_id]["content"] = (current + "\n\n" + content.strip()).strip()

    def read(self, section_id: str) -> str:
        return self.sections[section_id]["content"]

    def filled(self, exclude: tuple[str, ...] = ()) -> dict[str, str]:
        return {sid: self.sections[sid]["content"] for sid in self.order if self.sections[sid]["content"] and sid not in exclude}

    def render(self) -> str:
        parts = [self.header]
        for sid in self.order:
            section = self.sections[sid]
            body = section["content"] or section["placeholder"]
            parts.append(f"<!-- SECTION: {sid} -->\n## {section['title']}\n{body}\n<!-- END_SECTION: {sid} -->\n")
        return "\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Running a flow
# ---------------------------------------------------------------------------

@dataclass
class RunOptions:
    model: str | None = None
    think: bool | None = None
    num_ctx: int | None = None
    runs_dir: Path = RUNS_DIR
    llm_factory: Callable[[dict[str, Any]], Any] | None = None
    gateway_factory: Callable[[str, dict[str, Any]], ToolGateway | None] | None = None
    echo: Callable[[str], None] = print


@dataclass
class RunResult:
    run_dir: Path
    final: dict[str, Any] | None
    debate: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)


def tool_name_map() -> dict[str, str]:
    return {d["id"]: name for name, d in tool_registry.load_tool_declarations().items()}


def server_params(entry: dict[str, Any]) -> Any:
    from mcp.client.stdio import StdioServerParameters

    command = entry["command"]
    if command.startswith("."):
        command = str(REPO_ROOT / command)
    return StdioServerParameters(command=command, args=list(entry.get("args", [])), cwd=str(REPO_ROOT))


def participant_card(agent: dict[str, Any], names: dict[str, str]) -> tuple[str, str]:
    """(kind, card) shown to the other participants."""
    if (agent.get("role") or {}).get("permissions", {}).get("can_use_rag"):
        return "expert", agent.get("description", "")
    tools = [
        f"{names.get(b['tool_id'], b['tool_id'])} ({b.get('usage_hint', '')})"
        for key in ("tools", "expert_tools") for b in agent.get(key) or [] if b.get("enabled", True)
    ]
    return "tool_agent", "tools: " + "; ".join(tools)


def turn_markdown(debate: Debate, node_id: str, round_number: int, kind: str, output: dict[str, Any]) -> str:
    result = output.get("result") or {}
    lines = [f"### Round {round_number} ({kind})"]
    if output.get("status") == "abstained":
        lines.append("Abstained: the article does not cover this specimen.")
    if isinstance(result.get("position"), dict):
        lines.append("Position: " + debate.position_text(result["position"]))
    if result.get("expectation"):
        lines.append(f"Expectation: {result['expectation']}")
    for claim in debate.claims.values():
        if claim.by == node_id and claim.round == round_number:
            lines.append(f"- [{claim.claim_id}] {claim.text} (evidence: {', '.join(claim.evidence)})")
    for review in result.get("reviews") or []:
        lines.append(f"- {review.get('verdict')} [{review.get('claim_id')}]: {review.get('reason')} (evidence: {', '.join(review.get('evidence') or [])})")
    for response in result.get("responses") or []:
        lines.append(f"- {response.get('action')} [{response.get('challenge_id')}]: {response.get('text')} (evidence: {', '.join(response.get('evidence') or [])})")
    if result.get("markdown_summary"):
        lines.append(str(result["markdown_summary"]).strip())
    return "\n".join(lines)


def consistent_sex_answer(answer: dict[str, Any]) -> str | None:
    """When the arbiter gives a definite sex_call, forced_choice must be that class.

    forced_choice may differ from maturity: with P(M) = 0.3, P(F) = 0.3 and P(I) = 0.4 the single most likely class is I
    while an adult is more likely (0.6), and both readings are right."""
    result = answer.get("result") or {}
    forced, call = result.get("forced_choice"), result.get("sex_call")
    if call in ("M", "F", "I") and forced != call:
        return f"forced_choice {forced} must equal sex_call {call} when sex_call is a class."
    return None


def kickoff(round_number: int, kind: str, participant_kind: str, required: list[str]) -> str:
    """The first user message of a turn: what to do first, so a small model calls its tools without being nudged."""
    if kind == "opinions" and participant_kind == "tool_agent":
        return (f"Round {round_number} ({kind}). Call {' and '.join(required)} now, each once, with the dataset-unit values "
                "from problem_spec as one specimen. Call the tools before writing anything.")
    if kind == "opinions":
        return (f"Round {round_number} ({kind}). Call search_article now with a few English key words about what your article "
                "says on a specimen like this one or on the tools' assumptions (for example size at maturity, ageing, growth). "
                "Search before writing anything.")
    return (f"Round {round_number} ({kind}). Follow 'What to do in this round' in debate_state. Call a tool only if you need "
            "new evidence for this round; otherwise reply 'ready'.")


class RunContext:
    """What every run needs: the flow, its agents, a run folder, the transcript and the totals."""

    def __init__(self, flow_path: Path, options: RunOptions, run_name: str):
        self.flow_path = flow_path
        self.flow = tool_registry.read_declaration(flow_path)
        self.options = options
        self.nodes = self.flow["nodes"]
        self.terminal = self.flow["terminal_node"]
        self.agents = {nid: tool_registry.read_declaration(REPO_ROOT / node["agent_declaration"]) for nid, node in self.nodes.items()}
        self.names = tool_name_map()
        self.run_dir = options.runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.transcript = (self.run_dir / "transcript.jsonl").open("a", encoding="utf-8")
        self.states = {nid: AgentState() for nid in self.nodes}
        self.stats: dict[str, Any] = {"turns": 0, "skipped": 0, "llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "wall_s": 0.0}

    def log(self, event: dict[str, Any]) -> None:
        self.transcript.write(json.dumps({"ts": round(time.time(), 3), **event}, ensure_ascii=False, default=str) + "\n")
        self.transcript.flush()

    def llm_for(self, agent: dict[str, Any]) -> Any:
        if self.options.llm_factory:
            return self.options.llm_factory(agent)
        return OllamaChat(agent, self.options.model, self.options.think, self.options.num_ctx)

    def allowed_sex_calls(self) -> list[str]:
        """The sex_call values the arbiter's declared schema accepts; a forced-class flow leaves out adult_undetermined."""
        properties = self.agents[self.terminal]["prompt"]["output_format"]["schema"]["properties"]["result"]["properties"]
        return list((properties.get("sex_call") or {}).get("enum") or [])

    def most_likely_sex(self, maturity: str) -> str:
        """The class with the largest summed probability over every probabilistic tool result in the debate."""
        totals: Counter = Counter()
        for record in self.records:
            for call in record.get("tool_calls") or []:
                result = call.get("result")
                body = result.get("result") if isinstance(result, dict) else None
                for prediction in (body or {}).get("predictions") or []:
                    totals.update({k: float(v) for k, v in (prediction.get("probabilities") or {}).items() if k in ("M", "F", "I")})
        classes = ("I",) if maturity == "infant" else ("M", "F") if maturity == "adult" else ("M", "F", "I")
        return max(classes, key=lambda cls: totals[cls])

    def fallback_decision(self, reason: str) -> dict[str, Any]:
        """A low-reliability decision from the tool agents' final positions, used only when the arbiter fails."""
        positions = list(self.debate.final_positions().values())
        result: dict[str, Any] = {"reliability": "low", "debate_outcome": "unresolved", "arbiter_fallback": True,
                                  "rationale": f"The arbiter failed ({reason[:200]}); the runtime combined the tool agents' final positions.",
                                  "participants": [], "markdown_summary": "Arbiter failed; runtime fallback from the tool agents' positions."}
        if self.problem.task == "sex":
            calls = [p.get("sex_call") for p in positions]
            maturities = [p.get("maturity") for p in positions]
            maturity = maturities[0] if maturities and all(m == maturities[0] for m in maturities) else "uncertain"
            call = calls[0] if calls and all(c == calls[0] for c in calls) else ("I" if maturity == "infant" else "adult_undetermined" if maturity == "adult" else "undetermined")
            if call not in self.allowed_sex_calls():
                call = self.most_likely_sex(maturity)
            result.update(answer=call, sex_call=call, maturity=maturity, forced_choice=call if call in ("M", "F", "I") else None)
        else:
            rings = [float(p["rings"]) for p in positions if p.get("rings") is not None]
            if rings:
                result.update(rings=round(statistics.median(rings), 2),
                              rings_low=min(float(p.get("rings_low", p["rings"])) for p in positions if p.get("rings") is not None),
                              rings_high=max(float(p.get("rings_high", p["rings"])) for p in positions if p.get("rings") is not None))
                result["answer"] = f"{result['rings']} rings ({result['rings_low']}-{result['rings_high']})"
            else:
                result["answer"] = "undetermined"
        return {"status": "completed", "result": result, "warnings": ["arbiter_fallback"]}

    def account(self, stats: dict[str, Any]) -> None:
        self.stats["turns"] += 1
        for key in ("llm_calls", "prompt_tokens", "output_tokens"):
            self.stats[key] += stats.get(key) or 0
        self.stats["wall_s"] = round(self.stats["wall_s"] + (stats.get("wall_s") or 0), 1)

    async def open_gateways(self, stack: AsyncExitStack, node_ids: list[str]) -> dict[str, ToolGateway | None]:
        gateways: dict[str, ToolGateway | None] = {}
        servers = self.flow.get("mcp_servers") or {}
        for node_id in node_ids:
            if self.options.gateway_factory:
                gateways[node_id] = self.options.gateway_factory(node_id, self.agents[node_id])
                continue
            entry = servers.get(node_id)
            if entry is None:
                gateways[node_id] = None
                continue
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client

            errlog = stack.enter_context((self.run_dir / f"mcp_{node_id}.log").open("w", encoding="utf-8"))
            read, write = await stack.enter_async_context(stdio_client(server_params(entry), errlog=errlog))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            gateways[node_id] = MCPGateway(session)
        return gateways


class FlowRun(RunContext):
    def __init__(self, flow_path: Path, problem: Problem, options: RunOptions):
        super().__init__(flow_path, options, f"{problem.task}-row{problem.row_number}")
        self.problem = problem
        self.blackboard = RunBlackboard(BLACKBOARD_TEMPLATE.read_text(encoding="utf-8"), self.flow.get("blackboard_sections", []))
        participants = []
        for nid in self.nodes:
            if nid == self.terminal:
                continue
            kind, card = participant_card(self.agents[nid], self.names)
            participants.append(Participant(nid, participant_code(nid), kind, self.agents[nid]["name"], card))
        loops = [e["loop_controls"] for e in self.flow["edges"] if e.get("loop_controls")]
        # max_cycle_iterations counts returns to the first speaker: 2 -> 3 rounds (one cycle), 4 -> 5 rounds (two cycles).
        max_cycles = 2 if loops and loops[0]["max_cycle_iterations"] >= 4 else 1
        self.debate = Debate(problem.task, participants, max_cycles=max_cycles)
        self.records: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def save(self) -> None:
        (self.run_dir / "blackboard.md").write_text(self.blackboard.render(), encoding="utf-8")
        (self.run_dir / "debate_state.json").write_text(self.debate.to_json(), encoding="utf-8")
        meta = {
            "title": f"Debate ({self.problem.task}), CSV row {self.problem.row_number}",
            "run_folder": str(self.run_dir),
            "model": self.options.model or "declared per agent",
            "flow": self.flow_path.name,
            "started": self.started_at,
            "truth_hidden_from_agents": self.problem.truth or None,
            "rounds_so_far": self.debate.round,
            "tool_agents_agree": self.debate.converged(),
        }
        ledger = self.blackboard.read("debate_ledger") or self.debate.ledger_markdown()
        try:
            text = render_discussion(meta, self.problem.statement, self.records, ledger, self.final)
        except Exception as exc:  # noqa: BLE001 - the record must never stop a debate; the raw transcript still has everything
            self.log({"event": "discussion_error", "error": repr(exc)})
            text = f"# Discussion could not be rendered ({exc!r}); see transcript.jsonl.\n"
        (self.run_dir / "discussion.md").write_text(text, encoding="utf-8")

    def next_node(self, current: str) -> str:
        edges = [e for e in self.flow["edges"] if e["from"] == current]
        if any(e.get("loop_controls") for e in edges):
            self.debate.finish_pass()
        context = {
            "debate": {"next_round_needed": self.debate.next_round_needed, "round": self.debate.round, "converged": self.debate.converged()},
            "problem_spec": {"task": self.problem.task},
        }
        for edge in edges:
            loop = edge.get("loop_controls")
            if loop:
                local = {**context, "cycle_count": self.cycle_counts[edge["edge_id"]], "max_cycle_iterations": loop["max_cycle_iterations"]}
                if evaluate(loop["continue_condition"], local):
                    self.cycle_counts[edge["edge_id"]] += 1
                    self.log({"event": "edge", "edge": edge["edge_id"], "cycle_count": self.cycle_counts[edge["edge_id"]]})
                    return edge["to"]
                continue
            condition = edge.get("transition_condition")
            if condition is None or evaluate(condition, context):
                self.log({"event": "edge", "edge": edge["edge_id"]})
                return edge["to"]
        self.log({"event": "fallback", "from": current, "to": self.terminal})
        return self.terminal

    async def run_participant(self, node_id: str, gateway: ToolGateway | None) -> None:
        agent = self.agents[node_id]
        participant = self.debate.participants[node_id]
        round_number, kind = self.debate.round, self.debate.kind
        self.blackboard.write("debate_state", self.debate.render_state(node_id))
        permissions = self.nodes[node_id]["blackboard_permissions"]
        inputs = {sid: self.blackboard.read(sid) for sid in permissions["read_sections"] if sid != "*"}
        required: set[str] = set()
        if kind == "opinions":
            bound = [self.names.get(b["tool_id"], b["tool_id"]) for key in ("tools", "expert_tools") for b in agent.get(key) or [] if b.get("enabled", True)]
            required = set(bound) if participant.kind == "tool_agent" else {"search_article"}
        label = f"R{round_number}-{participant.code}"
        self.options.echo(f"  {label} ({kind}) ...")
        pinned = {}
        if participant.kind == "tool_agent" and self.problem.values:
            specimen = {"specimens": [self.problem.specimen()]}
            pinned = {self.names.get(b["tool_id"], b["tool_id"]): specimen for key in ("tools", "expert_tools") for b in agent.get(key) or []}
        start = kickoff(round_number, kind, participant.kind, sorted(required))
        addressed = self.debate.open_challenges_for(node_id) if kind == "replies" else []
        if addressed:
            listed = "; ".join(f"{c.challenge_id} by {self.debate.participants[c.by].code} on [{c.claim_id}]: {c.reason}" for c in addressed)
            start = (f"Round {round_number} (replies). Answer each challenge addressed to you, one item in 'responses' per challenge: "
                     f"{listed}. Use defend (with evidence), concede (and correct your position) or clarify. Call a tool only if you need new evidence.")
        request = TurnRequest(agent, inputs, label, required, [], self.names, start,
                              stop_after_required=kind == "opinions" and participant.kind == "tool_agent", pinned_arguments=pinned,
                              required_responses=[c.challenge_id for c in addressed])
        runner = AgentRunner(self.llm_for(agent), gateway, self.states[node_id], self.log)
        try:
            result = await runner.run(request)
        except Exception as exc:  # noqa: BLE001 - one failed turn must not lose a long debate
            # A failed turn says nothing; an expert that fails its opening turn is treated as abstaining.
            status = "abstained" if kind == "opinions" and participant.kind == "expert" else "failed"
            output = {"status": status, "result": {"claims": [], "reviews": [], "responses": [], "markdown_summary": f"Turn failed: {exc}"}, "warnings": [str(exc)]}
            result = TurnResult(output, [f"Turn failed and was skipped: {exc}"], [], {})
            self.log({"event": "turn_error", "node": node_id, "label": label, "error": repr(exc)})
        warnings = result.warnings + self.debate.record_turn(node_id, result.output)
        self.records.append({
            "round_heading": f"Round {round_number}: {kind}",
            "code": participant.code,
            "name": participant.label,
            "label": label,
            "output": result.output,
            "tool_calls": result.tool_calls,
            "warnings": warnings,
            "claim_ids": [c.claim_id for c in self.debate.claims.values() if c.by == node_id and c.round == round_number],
            "error": next((w for w in warnings if w.startswith("Turn failed")), None),
        })
        self.blackboard.append(permissions["write_section"], turn_markdown(self.debate, node_id, round_number, kind, result.output))
        self.account(result.stats)
        self.log({"event": "turn", "node": node_id, "label": label, "output": result.output, "warnings": warnings, "tool_calls": result.tool_calls, "stats": result.stats})
        for warning in warnings:
            self.options.echo(f"    warning: {warning}")

    async def run_terminal(self) -> dict[str, Any]:
        agent = self.agents[self.terminal]
        self.blackboard.write("debate_ledger", self.debate.ledger_markdown())
        readable = self.nodes[self.terminal]["blackboard_permissions"]["read_sections"]
        inputs = self.blackboard.filled(exclude=("debate_state",)) if "*" in readable else {sid: self.blackboard.read(sid) for sid in readable}
        self.options.echo("  consolidator ...")
        runner = AgentRunner(self.llm_for(agent), None, self.states[self.terminal], self.log)
        structured = ["sex_call", "maturity", "forced_choice"] if self.problem.task == "sex" else ["rings", "rings_low", "rings_high"]
        start = (f"Write the final decision for the {self.problem.task} task now, from debate_ledger. "
                 f"Fill {', '.join(structured)} as well as answer. Keep every text field short.")
        try:
            result = await runner.run(TurnRequest(agent, inputs, "FINAL", set(), structured, self.names, start,
                                                  validators=[consistent_sex_answer] if self.problem.task == "sex" else []))
        except Exception as exc:  # noqa: BLE001 - a failed arbiter must not lose a finished debate
            self.log({"event": "arbiter_error", "error": repr(exc)})
            result = TurnResult(self.fallback_decision(str(exc)), [f"Arbiter failed; runtime fallback used: {exc}"], [], {})
        output = result.output
        ledger = self.debate.ledger()
        for entry in (output.get("result") or {}).get("participants") or []:
            code = entry.get("participant")
            entry["challenges_received"] = [
                {**challenge, "claim_id": row["claim_id"]} for row in ledger if row["by"] == code for challenge in row["challenges"]
            ]
        self.blackboard.write(self.nodes[self.terminal]["blackboard_permissions"]["write_section"], (output.get("result") or {}).get("markdown_summary", ""))
        self.final = output
        self.account(result.stats)
        self.log({"event": "turn", "node": self.terminal, "label": "FINAL", "output": output, "warnings": result.warnings, "stats": result.stats})
        return output

    async def run(self) -> RunResult:
        started = time.perf_counter()
        guards = self.flow.get("global_guards") or {}
        max_turns = int(guards.get("global_max_turns", 32))
        self.cycle_counts: Counter[str] = Counter()
        visits: Counter[str] = Counter()
        self.blackboard.write("problem_spec", self.problem.statement)
        (self.run_dir / "problem.md").write_text(self.problem.statement, encoding="utf-8")
        self.log({"event": "start", "flow": str(self.flow_path), "task": self.problem.task, "row": self.problem.row_number, "model": self.options.model})
        final = None
        try:
            async with AsyncExitStack() as stack:
                gateways = await self.open_gateways(stack, [n for n in self.nodes if n != self.terminal])
                current = self.flow["entry_node"]
                while current != self.terminal:
                    if self.debate.needs_turn(current):
                        if self.stats["turns"] >= max_turns or visits[current] >= int(self.nodes[current].get("max_visits", 99)):
                            self.log({"event": "guard", "node": current, "turns": self.stats["turns"], "visits": visits[current]})
                            break
                        await self.run_participant(current, gateways[current])
                        visits[current] += 1
                        self.save()
                    else:
                        self.stats["skipped"] += 1
                        self.log({"event": "skip", "node": current, "round": self.debate.round})
                    current = self.next_node(current)
                final = await self.run_terminal()
        finally:
            self.stats["elapsed_s"] = round(time.perf_counter() - started, 1)
            self.save()
            summary = {"problem": {"task": self.problem.task, "row": self.problem.row_number, "truth": self.problem.truth},
                       "final": final, "debate": self.debate.snapshot(), "stats": self.stats, "model": self.options.model}
            (self.run_dir / "final_decision.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            result = (final or {}).get("result") or {}
            append_index(self.options.runs_dir / "discussions_index.jsonl", {
                "kind": "debate", "run": str(self.run_dir), "discussion": str(self.run_dir / "discussion.md"),
                "task": self.problem.task, "row": self.problem.row_number, "truth": self.problem.truth,
                "answer": result.get("answer"), "forced_choice": result.get("forced_choice"), "rings": result.get("rings"),
                "reliability": result.get("reliability"), "debate_outcome": result.get("debate_outcome"),
                "converged": self.debate.converged(), "rounds": self.debate.round, "turns": self.stats["turns"],
                "completed": final is not None, "model": self.options.model, "started": self.started_at, "elapsed_s": self.stats.get("elapsed_s"),
            })
            self.log({"event": "end", "stats": self.stats})
            self.transcript.close()
        return RunResult(self.run_dir, final, self.debate.snapshot(), self.stats)


async def run_question(flow_path: Path, node_id: str, question: str, options: RunOptions) -> RunResult:
    """Ask one article expert a standalone question, using the server wiring declared in a debate flow."""
    run = RunContext(flow_path, options, f"question-{participant_code(node_id).lower()}")
    if node_id not in run.nodes or node_id not in (run.flow.get("mcp_servers") or {}):
        raise ValueError(f"{node_id!r} is not a node with an MCP server in {flow_path.name}.")
    started = time.perf_counter()
    result = None
    try:
        async with AsyncExitStack() as stack:
            gateway = (await run.open_gateways(stack, [node_id]))[node_id]
            agent = run.agents[node_id]
            inputs = {"problem_spec": question, "debate_state": QUESTION_STATE}
            runner = AgentRunner(run.llm_for(agent), gateway, run.states[node_id], run.log)
            start = "Call search_article now with a few English key words for the question in problem_spec. Search before writing anything."
            result = await runner.run(TurnRequest(agent, inputs, "QUESTION", {"search_article"}, ["answer_markdown", "coverage"], run.names, start))
            run.account(result.stats)
            run.log({"event": "turn", "node": node_id, "label": "QUESTION", "output": result.output, "warnings": result.warnings,
                     "tool_calls": result.tool_calls, "stats": result.stats})
    finally:
        run.stats["elapsed_s"] = round(time.perf_counter() - started, 1)
        run.transcript.close()
    summary = {"question": question, "node": node_id, "output": result.output, "warnings": result.warnings,
               "retrieved": sorted(run.states[node_id].retrieved), "tool_calls": result.tool_calls, "stats": run.stats, "model": options.model}
    (run.run_dir / "answer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    record = {"round_heading": "Question", "code": participant_code(node_id), "name": run.agents[node_id]["name"], "label": "QUESTION",
              "output": result.output, "tool_calls": result.tool_calls, "warnings": result.warnings, "claim_ids": [], "error": None}
    meta = {"title": f"Question to {participant_code(node_id)}", "run_folder": str(run.run_dir), "model": options.model or "declared per agent"}
    (run.run_dir / "discussion.md").write_text(render_discussion(meta, question, [record], None, None), encoding="utf-8")
    append_index(options.runs_dir / "discussions_index.jsonl", {
        "kind": "question", "run": str(run.run_dir), "discussion": str(run.run_dir / "discussion.md"), "node": node_id,
        "question": question, "coverage": (result.output.get("result") or {}).get("coverage"), "model": options.model,
    })
    return RunResult(run.run_dir, result.output, {}, run.stats)
