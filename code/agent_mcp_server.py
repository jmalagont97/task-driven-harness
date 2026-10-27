"""agent_mcp_server.py

MCP server scoped to one agent (stdio transport).

Responsibilities:
- Read an agent declaration and expose only the MCP tools it binds in `tools` and `expert_tools` (enabled bindings
  with access "mcp"). Every other tool is unknown to this server, so the agent cannot list or call it.
- Refuse to start when a binding names a tool that no declaration exposes, so a typo never leaves an agent silently
  without a tool.
- Enforce each binding's max_calls_per_task for the lifetime of the process; the runtime starts one process per run.
- Append a scope note and the binding's usage_hint to each tool description: the agent learns which tools it can call
  and what each one is for in its role.
- Log every call (and every call to a tool outside the scope) as one JSON line on stderr; the runtime stores it.

Tool contracts, envelope checks and result conversion come from code/mcp_server.py and code/tool_registry.py.

Run from the repository root:
    .venv/bin/python code/agent_mcp_server.py --agent declarations/agents/abalone_sex_agent_a.json
    .venv/bin/python code/agent_mcp_server.py --agent declarations/agents/abalone_sex_agent_a.json --check
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

import mcp_server
import tool_registry

logger = logging.getLogger("agent_mcp_server")

SERVER_VERSION = "0.1.0"
BINDING_LISTS = ("tools", "expert_tools")


class ScopeError(ValueError):
    """The agent declaration cannot be served: no MCP bindings, or a binding without an exposed declaration."""


def load_agent(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = tool_registry.REPO_ROOT / path
    return tool_registry.read_declaration(path)


def mcp_bindings(agent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """tool_id -> binding for every enabled MCP binding in tools and expert_tools."""
    bindings: dict[str, dict[str, Any]] = {}
    for list_name in BINDING_LISTS:
        for binding in agent.get(list_name) or []:
            if binding.get("enabled", True) and binding.get("access") == "mcp":
                bindings[binding["tool_id"]] = {**binding, "binding_list": list_name}
    return bindings


def scoped_declarations(agent: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """(tool name -> declaration, tool name -> binding) for the tools this agent may call."""
    bindings = mcp_bindings(agent)
    if not bindings:
        raise ScopeError(f"Agent {agent.get('id')!r} binds no enabled MCP tools; there is nothing to serve.")
    exposed = {declaration["id"]: declaration for declaration in tool_registry.load_tool_declarations().values()}
    missing = sorted(set(bindings) - set(exposed))
    if missing:
        raise ScopeError(
            f"Agent {agent.get('id')!r} binds {missing}, but no declaration in declarations/tools exposes a tool with "
            "that id over MCP. Fix the tool_id or expose the tool."
        )
    declarations = {exposed[tool_id]["name"]: exposed[tool_id] for tool_id in bindings}
    by_name = {exposed[tool_id]["name"]: binding for tool_id, binding in bindings.items()}
    return declarations, by_name


def scoped_description(declaration: dict[str, Any], binding: dict[str, Any], callable_names: list[str]) -> str:
    scope = (
        f"Scope: in this session you can call only {', '.join(callable_names)}. Any other tool named above belongs to "
        "another agent; its opinion reaches you through the blackboard, not through a call."
    )
    hint = binding.get("usage_hint")
    return f"{declaration['prompt_context']}\n\n{scope}" + (f"\nHow to use it in your role: {hint}" if hint else "")


class CallBudget:
    """Calls left per tool for the lifetime of the server (one run)."""

    def __init__(self, limits: dict[str, int | None]):
        self.limits = dict(limits)
        self.used = dict.fromkeys(limits, 0)

    def take(self, name: str) -> bool:
        limit = self.limits.get(name)
        if limit is not None and self.used.get(name, 0) >= limit:
            return False
        self.used[name] = self.used.get(name, 0) + 1
        return True


def log_event(**fields: Any) -> None:
    logger.info(json.dumps({"ts": round(time.time(), 3), **fields}, ensure_ascii=False, default=str))


async def call_checked(declaration: dict[str, Any], arguments: dict[str, Any], executor: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run an executor in a worker thread and check its envelope against output_schema, as mcp_server does."""
    if config is not None:
        declaration = {**declaration, "config": config}
    call = functools.partial(tool_registry.invoke, declaration, arguments, executor)
    envelope = await anyio.to_thread.run_sync(call)
    problems = tool_registry.output_problems(declaration, envelope)
    if problems:
        logger.error("%s returned output that breaks its output_schema: %s", declaration["name"], problems)
        envelope = tool_registry.error_envelope(
            declaration,
            "runtime_error",
            "The tool produced output that does not match its output_schema: " + "; ".join(problems[:5]),
        )
    return envelope


def build_server(agent_path: str | Path) -> Server:
    agent = load_agent(agent_path)
    declarations, bindings = scoped_declarations(agent)
    names = sorted(declarations)
    executors = {name: tool_registry.resolve_entrypoint(d["mcp"]["entrypoint"]) for name, d in declarations.items()}
    tools = [
        mcp_server.to_mcp_tool(declarations[name]).model_copy(
            update={"description": scoped_description(declarations[name], bindings[name], names)}
        )
        for name in names
    ]
    budget = CallBudget({name: bindings[name].get("max_calls_per_task") for name in names})

    async def on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        declaration = declarations.get(params.name)
        if declaration is None:
            log_event(event="scope_violation", agent=agent["id"], tool=params.name)
            raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {params.name}")
        if not budget.take(params.name):
            log_event(event="budget_exhausted", agent=agent["id"], tool=params.name)
            envelope = tool_registry.error_envelope(
                declaration,
                "input_error",
                f"Call budget exhausted: {params.name} allows {budget.limits[params.name]} calls per task. "
                "Answer with the evidence you already have.",
            )
            return mcp_server.to_call_result(envelope)
        started = time.perf_counter()
        envelope = await call_checked(declaration, params.arguments or {}, executors[params.name])
        log_event(
            event="tool_call",
            agent=agent["id"],
            tool=params.name,
            ok=envelope.get("ok"),
            error=(envelope.get("error") or {}).get("type"),
            flags=envelope.get("flags"),
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return mcp_server.to_call_result(envelope)

    description = (agent.get("prompt", {}).get("tools") or {}).get("description") or ""
    instructions = f"Tools for agent {agent['id']}: {', '.join(names)}. {description}".strip()
    return Server(
        f"agent:{agent['id']}",
        version=SERVER_VERSION,
        title=f"Task-Driven Harness tools scoped to {agent['id']}",
        instructions=instructions,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve only the MCP tools one agent declares (stdio).")
    parser.add_argument("--agent", required=True, help="Path to the agent declaration (relative to the repository root).")
    parser.add_argument("--check", action="store_true", help="Validate the scope, list the tools and exit.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    # stdout carries the protocol; logs go to stderr.
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(), format="%(message)s")

    if args.check:
        agent = load_agent(args.agent)
        declarations, bindings = scoped_declarations(agent)
        for name in sorted(declarations):
            binding = bindings[name]
            print(f"{name:36s} {binding['binding_list']:13s} max_calls_per_task={binding.get('max_calls_per_task')}")
        print(f"{len(declarations)} tools in the scope of {agent['id']!r}")
        return 0

    server = build_server(args.agent)

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
