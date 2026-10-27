"""agent_runner.py

One agent turn: render the declared prompt, let the model call its MCP tools, then get a validated JSON answer.

Responsibilities:
- Render the agent's Mustache template ({{role}}, {{task}}, {{inputs}}, {{rules}}, {{tools}}, {{output_format}}) with
  the blackboard sections the node may read.
- Run the tool loop through the agent's own MCP server (a ToolGateway), at most runtime.max_iterations model calls,
  enforcing each binding's max_calls_per_task for the whole run and the tools a round requires before answering.
- Ask for the final answer with the output_format schema as Ollama's `format`, validate it with jsonschema and retry
  up to guardrails.output_validation.max_retries times.
- Citation check for every agent with can_use_rag: each chunk_id it cites must be one its own tools returned in this
  run. One corrective retry, then evidence that is not a retrieved chunk_id is removed, and items left without
  evidence are dropped.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from runtime.debate import normalize_id
from runtime.ollama_chat import ChatReply, mcp_tool_to_ollama

CHUNK_ID = re.compile(r"\b[a-z0-9_]+:\d{3}\b")
TOOL_RESULT_CHARS = 6000
# Tool-calling replies are short; a cap keeps a chatty reply from costing a full answer's worth of CPU time.
TOOL_LOOP_MAX_TOKENS = 256
KICKOFF = "Take your turn now. Follow 'What to do in this round' in debate_state, or answer the question in problem_spec."
OUTPUT_FORMAT_NOTE = (
    "One JSON object. Its exact JSON Schema comes in the last message, when you write your answer for this turn; "
    "until then, call your tools if the round needs them."
)
FINAL_INSTRUCTION = (
    "Now write your answer for this turn as one JSON object that matches this JSON Schema. Use only evidence that "
    "appeared in the tool results above or in the blackboard. Do not call tools.\n"
)


class LLM(Protocol):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply: ...


class ToolGateway(Protocol):
    async def list_tools(self) -> list[Any]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str, bool]: ...


class MCPGateway:
    """ToolGateway over an initialised mcp.ClientSession."""

    def __init__(self, session: Any):
        self.session = session
        self._tools: list[Any] | None = None

    async def list_tools(self) -> list[Any]:
        if self._tools is None:
            self._tools = list((await self.session.list_tools()).tools)
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any] | None, str, bool]:
        from mcp.shared.exceptions import MCPError

        try:
            result = await self.session.call_tool(name, arguments)
        except MCPError as exc:
            return None, json.dumps({"ok": False, "error": {"type": "unknown_tool", "message": str(exc)}}), True
        text = result.content[0].text if result.content else ""
        return result.structured_content, text, bool(result.is_error)


@dataclass
class AgentState:
    """What an agent accumulates over a run: call counts per tool and the chunk_ids its tools returned."""

    calls: dict[str, int] = field(default_factory=dict)
    retrieved: set[str] = field(default_factory=set)
    tool_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TurnRequest:
    agent: dict[str, Any]
    inputs: dict[str, str]
    label: str
    required_tools: set[str] = field(default_factory=set)
    extra_required: list[str] = field(default_factory=list)
    tool_names: dict[str, str] = field(default_factory=dict)  # binding tool_id -> MCP tool name
    kickoff: str = KICKOFF
    stop_after_required: bool = False  # go straight to the answer once every required tool was called
    # Exact arguments for these tools (the specimen from problem_spec). A small model can mistype a value when it copies
    # the measurements; the runtime sends the exact ones and records the correction.
    pinned_arguments: dict[str, dict[str, Any]] = field(default_factory=dict)
    required_responses: list[str] = field(default_factory=list)  # challenge ids this turn must answer
    validators: list[Any] = field(default_factory=list)  # callables(answer) -> problem text or None; one retry each turn


@dataclass
class TurnResult:
    output: dict[str, Any]
    warnings: list[str]
    tool_calls: list[dict[str, Any]]
    stats: dict[str, Any]


def output_schema(agent: dict[str, Any], extra_required: list[str] | None = None) -> dict[str, Any]:
    schema = copy.deepcopy(agent["prompt"]["output_format"]["schema"])
    if extra_required:
        result = schema["properties"]["result"]
        result["required"] = list(dict.fromkeys([*result.get("required", []), *extra_required]))
    return schema


def bindings(agent: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for key in ("tools", "expert_tools") for b in agent.get(key) or [] if b.get("enabled", True)]


def render_tools(agent: dict[str, Any], tool_names: dict[str, str]) -> str:
    section = agent["prompt"].get("tools") or {}
    if section.get("render") == "none" or not bindings(agent):
        return "No tools: you only read the blackboard."
    lines = [section.get("description", "")]
    for binding in bindings(agent):
        name = tool_names.get(binding["tool_id"], binding["tool_id"])
        lines.append(f"- {name}: {binding.get('usage_hint', '')} (at most {binding.get('max_calls_per_task', 'unlimited')} calls in this task)")
    policy = section.get("usage_policy") or []
    if policy:
        lines.append("Usage policy:")
        lines += [f"- {rule}" for rule in policy]
    return "\n".join(line for line in lines if line)


def render_prompt(agent: dict[str, Any], inputs: dict[str, str], tool_names: dict[str, str]) -> str:
    """The declared template with every placeholder filled. The JSON Schema itself is sent with the final instruction,
    so the tool-calling exchanges carry a shorter prompt and share it as a cached prefix with the answer call."""
    prompt = agent["prompt"]
    values = {
        "role": prompt.get("role", ""),
        "task": prompt.get("task", ""),
        "inputs": "\n\n".join(f"### {name}\n{content.strip()}" for name, content in inputs.items()) or "(empty)",
        "rules": "\n".join(f"- {rule}" for rule in prompt.get("rules", [])),
        "tools": render_tools(agent, tool_names),
        "output_format": OUTPUT_FORMAT_NOTE,
    }
    text = prompt["template"]
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def collect_retrieved(envelope: dict[str, Any] | None) -> set[str]:
    result = (envelope or {}).get("result") or {}
    found = {r["chunk_id"] for r in result.get("results", []) if isinstance(r, dict) and "chunk_id" in r}
    found |= {c["chunk_id"] for c in result.get("chunks", []) if isinstance(c, dict) and "chunk_id" in c}
    return found


def cited_chunk_ids(output: dict[str, Any]) -> set[str]:
    return set(CHUNK_ID.findall(json.dumps(output.get("result") or {}, ensure_ascii=False)))


REPAIRABLE = ("claims", "reviews", "responses")


def repair(candidate: Any, validator: Draft202012Validator) -> tuple[Any, list[str]]:
    """Drop the items of claims, reviews or responses that break the schema, when they are the only errors.

    A malformed item (for example a claim with no evidence) is discarded by the debate rules anyway; dropping it keeps
    the rest of the turn instead of losing it."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("result"), dict):
        return candidate, []
    errors = list(validator.iter_errors(candidate))
    bad: dict[str, set[int]] = {}
    for error in errors:
        path = list(error.absolute_path)
        if len(path) < 3 or path[0] != "result" or path[1] not in REPAIRABLE or not isinstance(path[2], int):
            return candidate, []
        bad.setdefault(path[1], set()).add(path[2])
    fixed = copy.deepcopy(candidate)
    notes = []
    for key, indexes in bad.items():
        items = fixed["result"].get(key) or []
        fixed["result"][key] = [item for i, item in enumerate(items) if i not in indexes]
        notes.append(f"{len(indexes)} malformed item(s) in {key} were dropped.")
    return (fixed, notes) if validator.is_valid(fixed) else (candidate, [])


def sanitize_citations(output: dict[str, Any], retrieved: set[str]) -> list[str]:
    """Keep only evidence that is a retrieved chunk_id; drop claims and reviews left without evidence."""
    result = output.get("result") or {}
    warnings = []
    for key in ("claims", "reviews", "responses"):
        kept = []
        for item in result.get(key) or []:
            evidence = [e for e in item.get("evidence") or [] if CHUNK_ID.fullmatch(str(e).strip()) and str(e).strip() in retrieved]
            removed = len(item.get("evidence") or []) - len(evidence)
            if removed:
                warnings.append(f"{removed} citation(s) in {key} were not chunk_ids returned by this agent's tools and were removed.")
            item["evidence"] = evidence
            if key == "responses" and not evidence and item.get("action") == "defend":
                item["action"] = "clarify"
                warnings.append(f"A defence of {item.get('challenge_id')} had no valid citation and was recorded as a clarification.")
            if evidence or (key == "responses" and item.get("action") in ("concede", "clarify")):
                kept.append(item)
            else:
                warnings.append(f"An item in {key} was dropped because none of its evidence was a retrieved chunk_id.")
        if key in result:
            result[key] = kept
    return warnings


class AgentRunner:
    def __init__(self, llm: LLM, gateway: ToolGateway | None, state: AgentState, log: Callable[[dict[str, Any]], None]):
        self.llm = llm
        self.gateway = gateway
        self.state = state
        self.log = log

    async def call_tool(self, request: TurnRequest, name: str, arguments: dict[str, Any], limits: dict[str, int | None]) -> str:
        used = self.state.calls.get(name, 0)
        limit = limits.get(name)
        if name not in limits:
            text = json.dumps({"ok": False, "error": {"type": "unknown_tool", "message": f"{name} is not one of your tools: {sorted(limits)}."}})
            self.state.tool_log.append({"turn": request.label, "tool": name, "arguments": arguments, "ok": False, "error": "unknown_tool"})
            return text
        if limit is not None and used >= limit:
            text = json.dumps({"ok": False, "error": {"type": "budget_exhausted", "message": f"{name} allows {limit} calls in this task and all were used. Answer with the evidence you have."}})
            self.state.tool_log.append({"turn": request.label, "tool": name, "arguments": arguments, "ok": False, "error": "budget_exhausted"})
            return text
        self.state.calls[name] = used + 1
        entry: dict[str, Any] = {"turn": request.label, "tool": name, "arguments": arguments}
        pinned = request.pinned_arguments.get(name)
        if pinned is not None and arguments != pinned:
            entry["corrected_from"] = arguments
            entry["arguments"] = arguments = pinned
        structured, text, is_error = await self.gateway.call_tool(name, arguments)
        self.state.retrieved |= collect_retrieved(structured)
        entry["ok"] = not is_error
        if is_error and structured:
            entry["error"] = (structured.get("error") or {}).get("type")
        entry["result"] = structured if structured is not None else text[:TOOL_RESULT_CHARS]
        self.state.tool_log.append(entry)
        if "corrected_from" in entry:
            text = ("Note: the specimen values you sent did not match problem_spec; the tool was called with the exact "
                    f"problem_spec values {json.dumps(arguments)}.\n") + text
        return text[:TOOL_RESULT_CHARS] + ("... [truncated]" if len(text) > TOOL_RESULT_CHARS else "")

    async def run(self, request: TurnRequest) -> TurnResult:
        agent = request.agent
        schema = output_schema(agent, request.extra_required)
        system = render_prompt(agent, request.inputs, request.tool_names)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": request.kickoff}]
        limits = {request.tool_names.get(b["tool_id"], b["tool_id"]): b.get("max_calls_per_task") for b in bindings(agent)}
        tools = [mcp_tool_to_ollama(t) for t in await self.gateway.list_tools()] if self.gateway and limits else []
        max_iterations = int((agent.get("runtime") or {}).get("max_iterations") or 4)
        stats = {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "wall_s": 0.0}
        calls_this_turn: list[str] = []
        warnings: list[str] = []

        def account(reply: ChatReply, phase: str) -> None:
            stats["llm_calls"] += 1
            stats["prompt_tokens"] += reply.stats.get("prompt_tokens") or 0
            stats["output_tokens"] += reply.stats.get("output_tokens") or 0
            stats["wall_s"] = round(stats["wall_s"] + (reply.stats.get("wall_s") or 0), 2)
            self.log({"event": "llm_call", "agent": agent["id"], "turn": request.label, "phase": phase, **reply.stats,
                      "tool_calls": [c.get("function", {}).get("name") for c in reply.tool_calls]})

        if tools:
            nudges = 0
            for _ in range(max_iterations):
                reply = await self.llm.chat(messages, tools=tools, max_tokens=TOOL_LOOP_MAX_TOKENS)
                account(reply, "tools")
                if reply.tool_calls:
                    messages.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls})
                    for call in reply.tool_calls:
                        function = call.get("function") or {}
                        name = function.get("name", "")
                        arguments = parse_arguments(function.get("arguments"))
                        text = await self.call_tool(request, name, arguments, limits)
                        calls_this_turn.append(name)
                        messages.append({"role": "tool", "tool_name": name, "content": text})
                    if request.stop_after_required and request.required_tools <= set(calls_this_turn):
                        break
                    continue
                missing = sorted(request.required_tools - set(calls_this_turn))
                if missing and nudges < 2:
                    nudges += 1
                    messages.append({"role": "assistant", "content": reply.content})
                    messages.append({"role": "user", "content": f"Before answering, call {', '.join(missing)}: this round requires it."})
                    continue
                if reply.content:
                    messages.append({"role": "assistant", "content": reply.content})
                break
            missing = sorted(request.required_tools - set(calls_this_turn))
            if missing:
                warnings.append(f"Required tool(s) not called this turn: {missing}.")
            for entry in self.state.tool_log:
                if entry["turn"] == request.label and "corrected_from" in entry:
                    warnings.append(f"{entry['tool']}: the model's arguments differed from problem_spec and were replaced by the exact values.")

        messages.append({"role": "user", "content": FINAL_INSTRUCTION + json.dumps(schema, separators=(",", ":"))})
        validation = (agent.get("guardrails") or {}).get("output_validation") or {}
        attempts = 1 + int(validation.get("max_retries", 2))
        rag = bool(((agent.get("role") or {}).get("permissions") or {}).get("can_use_rag"))
        validator = Draft202012Validator(schema)
        output: dict[str, Any] | None = None
        fallback: dict[str, Any] | None = None  # schema-valid, failed only the citation check
        citation_retry_used = False
        responses_retry_used = False
        validators_retry_used = False
        for _ in range(attempts):
            reply = await self.llm.chat(messages, schema=schema)
            account(reply, "answer")
            try:
                candidate = json.loads(reply.content)
            except json.JSONDecodeError as exc:
                candidate, problem = None, (f"not valid JSON ({exc}); it was probably cut off by the length limit: keep every text "
                                            "field to one or two sentences and do not repeat earlier claims")
            else:
                candidate, notes = repair(candidate, validator)
                warnings += notes
                errors = sorted(validator.iter_errors(candidate), key=lambda e: list(e.absolute_path))
                problem = "; ".join(f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors[:5]) or None
            if problem is None and request.validators and not validators_retry_used:
                found = [text for check in request.validators if (text := check(candidate))]
                if found:
                    validators_retry_used = True
                    fallback = fallback or candidate
                    problem = " ".join(found)
            if problem is None and request.required_responses and not responses_retry_used:
                answered = {normalize_id(r.get("challenge_id")) for r in (candidate.get("result") or {}).get("responses") or []}
                missing = [cid for cid in request.required_responses if cid not in answered]
                if missing:
                    responses_retry_used = True
                    fallback = fallback or candidate
                    problem = (f"You must answer every challenge addressed to you; missing: {missing}. Add one item in 'responses' "
                               "per challenge, with action defend (and evidence), concede or clarify.")
            if problem is None and rag:
                invalid = sorted(cited_chunk_ids(candidate) - self.state.retrieved)
                if invalid and not citation_retry_used:
                    citation_retry_used = True
                    fallback = candidate
                    problem = (
                        f"You cited chunk_ids your tools did not return in this task: {invalid}. Cite only these: "
                        f"{sorted(self.state.retrieved) or 'none (then abstain or say the article does not cover the topic)'}."
                    )
            if problem is None:
                output = candidate
                break
            warnings.append(f"Answer rejected ({problem[:300]}); retrying.")
            self.log({"event": "answer_rejected", "agent": agent["id"], "turn": request.label, "problem": problem, "answer": reply.content[:4000]})
            messages.append({"role": "assistant", "content": reply.content})
            messages.append({"role": "user", "content": f"Your answer was rejected: {problem}. Send the corrected JSON object."})
        output = output if output is not None else fallback
        if output is None:
            raise RuntimeError(f"{agent['id']} produced no valid answer for {request.label} after {attempts} attempts.")
        if rag:
            warnings += sanitize_citations(output, self.state.retrieved)
        turn_calls = [entry for entry in self.state.tool_log if entry["turn"] == request.label]
        return TurnResult(output, warnings, turn_calls, stats)
