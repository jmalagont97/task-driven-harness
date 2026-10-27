"""ollama_chat.py

Async client for the Ollama chat API (/api/chat), configured from an agent declaration.

Responsibilities:
- Map the declaration's model block onto Ollama: model_name, temperature, top_p, seed, stop, max_tokens (num_predict),
  and the provider-specific settings in runtime.adapters.ollama (num_ctx, think, keep_alive).
- Send tool definitions and return tool calls, or ask for a structured answer with a JSON Schema in `format`.
- Retry connection failures up to connection.max_retries and report token counts and durations for the run log.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    """The model server failed or answered with an error."""


@dataclass
class ChatReply:
    content: str
    tool_calls: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)
    thinking: str = ""


class OllamaChat:
    def __init__(self, agent: dict[str, Any], model_override: str | None = None, think_override: bool | None = None,
                 num_ctx_override: int | None = None):
        model = agent["model"]
        if model.get("provider") != "ollama":
            raise ValueError(f"Agent {agent['id']} uses provider {model.get('provider')!r}; this runtime only speaks to Ollama.")
        adapters = ((agent.get("runtime") or {}).get("adapters") or {}).get("ollama") or {}
        connection = model.get("connection") or {}
        parameters = model.get("parameters") or {}
        self.agent_id = agent["id"]
        self.model = model_override or model["model_name"]
        self.url = (connection.get("base_url") or "http://localhost:11434").rstrip("/") + "/api/chat"
        self.timeout_s = float(connection.get("timeout_s") or 600)
        self.max_retries = int(connection.get("max_retries") or 0)
        self.think = adapters.get("think", False) if think_override is None else think_override
        self.keep_alive = adapters.get("keep_alive", "15m")
        options: dict[str, Any] = {"num_ctx": int(num_ctx_override or adapters.get("num_ctx", 16384))}
        for key, option in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed"), ("max_tokens", "num_predict")):
            if parameters.get(key) is not None:
                options[option] = parameters[key]
        if parameters.get("stop"):
            options["stop"] = parameters["stop"]
        self.options = options

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> ChatReply:
        options = dict(self.options)
        if max_tokens is not None:
            options["num_predict"] = min(max_tokens, options.get("num_predict", max_tokens))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if tools:
            body["tools"] = tools
        if schema is not None:
            body["format"] = schema
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(self.url, json=body)
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(2 * (attempt + 1))
                continue
            data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            if response.status_code != 200:
                raise LLMError(f"Ollama answered {response.status_code} for {self.model}: {data.get('error', response.text[:300])}")
            message = data.get("message") or {}
            stats = {
                "model": self.model,
                "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "prompt_s": round((data.get("prompt_eval_duration") or 0) / 1e9, 2),
                "output_s": round((data.get("eval_duration") or 0) / 1e9, 2),
                "wall_s": round(time.perf_counter() - started, 2),
            }
            return ChatReply(message.get("content") or "", message.get("tool_calls") or [], stats, message.get("thinking") or "")
        raise LLMError(f"Ollama is not reachable at {self.url} after {self.max_retries + 1} attempt(s): {last_error}")


def mcp_tool_to_ollama(tool: Any) -> dict[str, Any]:
    """An MCP tool (from list_tools) as an Ollama/OpenAI-style function definition."""
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema},
    }
