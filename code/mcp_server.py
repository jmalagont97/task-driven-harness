"""mcp_server.py

Generic MCP server for the harness (stdio transport).

Responsibilities:
- Expose every declaration in declarations/tools/ whose mcp.expose is true and whose mcp.server matches --server.
  Tool contracts (name, title, description, input and output schemas, annotations) come from the JSON
  declarations; this module never defines a contract itself.
- Dispatch tools/call to the declared Python entrypoint in a worker thread.
- Check every envelope against the tool's output_schema before it reaches the client.
- Report tool failures inside the result (isError: true) so the calling LLM can read them and correct itself;
  only an unknown tool name is a protocol error.

Run from the repository root:
    .venv/bin/python code/mcp_server.py --server abalone_tools
    .venv/bin/python code/mcp_server.py --server abalone_tools --check   # validate and list, then exit
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from typing import Any

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

import tool_registry

logger = logging.getLogger("mcp_server")

SERVER_VERSION = "0.1.0"
INSTRUCTIONS = {
    "abalone_tools": (
        "Abalone tools. Four estimate sex (M = male, F = female, I = infant) and four estimate age as a ring count "
        "(age in years = rings + 1.5). For each task two tools are ML models and two are expert tools (rules or simple "
        "statistics): consult at least one of each and report where they disagree. All measurements use dataset units, "
        "the original millimetres or grams divided by 200. Infant vs adult is reliable; male vs female is near chance, so "
        "honour the mf_ambiguous flag. Age tools return 80% intervals: report them, not only the point estimate."
    ),
}


def to_mcp_tool(declaration: dict[str, Any]) -> types.Tool:
    mcp = declaration["mcp"]
    return types.Tool(
        name=declaration["name"],
        title=mcp["title"],
        description=declaration["prompt_context"],
        input_schema=declaration["input_schema"],
        output_schema=declaration["output_schema"],
        annotations=types.ToolAnnotations.model_validate(mcp.get("annotations", {})),
    )


def to_call_result(envelope: dict[str, Any]) -> types.CallToolResult:
    """The same envelope as structured content and as serialized text, for clients that read either."""
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(envelope, ensure_ascii=False))],
        structured_content=envelope,
        is_error=not envelope.get("ok", False),
    )


def build_server(server_name: str) -> Server:
    declarations = tool_registry.load_tool_declarations(server=server_name)
    if not declarations:
        raise SystemExit(f"No declarations expose tools for MCP server {server_name!r}.")
    executors = {name: tool_registry.resolve_entrypoint(d["mcp"]["entrypoint"]) for name, d in declarations.items()}
    tools = [to_mcp_tool(declaration) for declaration in declarations.values()]

    async def on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        declaration = declarations.get(params.name)
        if declaration is None:
            raise MCPError(code=INVALID_PARAMS, message=f"Unknown tool: {params.name}")
        call = functools.partial(tool_registry.invoke, declaration, params.arguments or {}, executors[params.name])
        envelope = await anyio.to_thread.run_sync(call)
        problems = tool_registry.output_problems(declaration, envelope)
        if problems:
            logger.error("%s returned output that breaks its output_schema: %s", params.name, problems)
            envelope = tool_registry.error_envelope(
                declaration,
                "runtime_error",
                "The tool produced output that does not match its output_schema: " + "; ".join(problems[:5]),
            )
        return to_call_result(envelope)

    return Server(
        server_name,
        version=SERVER_VERSION,
        title=f"Task-Driven Harness tools ({server_name})",
        instructions=INSTRUCTIONS.get(server_name),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve declared harness tools over MCP (stdio).")
    parser.add_argument("--server", default="abalone_tools", help="Value of mcp.server to expose.")
    parser.add_argument("--check", action="store_true", help="Validate the declarations, list the tools and exit.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    # stdout carries the protocol; logs go to stderr.
    logging.basicConfig(stream=sys.stderr, level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    if args.check:
        declarations = tool_registry.load_tool_declarations(server=args.server)
        for name, declaration in declarations.items():
            print(f"{name:36s} {declaration['mcp']['title']}")
        print(f"{len(declarations)} tools valid for MCP server {args.server!r}")
        return 0

    server = build_server(args.server)

    async def serve() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
