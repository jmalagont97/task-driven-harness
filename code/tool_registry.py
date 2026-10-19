"""tool_registry.py

Bridge between tool declarations (declarations/tools/*.json) and their Python executors.

Responsibilities:
- Read declaration files (full-line // comments, as used by the example templates, are ignored).
- Validate the declarations exposed over MCP: required keys, MCP tool-name rules, JSON Schemas,
  envelope contract, annotation consistency and a resolvable entrypoint.
- Resolve "code/<path>.py::<function>" entrypoints to callables.
- Invoke an executor with a copy of its declared config and check the envelope against output_schema.

Used by code/mcp_server.py and code/tools/abalone/train_abalone_tools.py. wake_up.py can import it when the
harness bootstrap is implemented.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
TOOLS_DIR = REPO_ROOT / "declarations" / "tools"

TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
REQUIRED_KEYS = (
    "schema_version", "id", "name", "kind", "version", "description", "prompt_context",
    "input_schema", "output_schema", "config", "deterministic", "side_effects", "references", "mcp",
)
SIDE_EFFECTS = {"none", "read", "write", "external"}
ANNOTATION_KEYS = {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
ENVELOPE_REQUIRED = {"ok", "tool_id", "version"}

Executor = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class DeclarationError(ValueError):
    """One or more exposed declarations break the contract; the message lists every problem."""


def read_declaration(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("//")]
    return json.loads("\n".join(lines))


def iter_declarations(tools_dir: Path = TOOLS_DIR) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(tools_dir.glob("*.json")):
        yield path, read_declaration(path)


def load_tool_declarations(server: str | None = None, tools_dir: Path = TOOLS_DIR) -> dict[str, dict[str, Any]]:
    """Validated declarations with mcp.expose = true (optionally only those of one MCP server), keyed by name."""
    declarations: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for path, declaration in iter_declarations(tools_dir):
        mcp = declaration.get("mcp") or {}
        if not mcp.get("expose") or (server is not None and mcp.get("server") != server):
            continue
        issues = validate_declaration(declaration)
        if declaration.get("name") in declarations:
            issues.append(f"duplicate tool name {declaration['name']!r}")
        if issues:
            problems.extend(f"{path.relative_to(REPO_ROOT)}: {issue}" for issue in issues)
            continue
        declarations[declaration["name"]] = declaration
    if problems:
        raise DeclarationError("Invalid tool declarations:\n- " + "\n- ".join(problems))
    return declarations


def validate_declaration(declaration: dict[str, Any]) -> list[str]:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    issues = [f"missing key {key!r}" for key in REQUIRED_KEYS if key not in declaration]
    if issues:
        return issues
    if declaration["kind"] != "tool":
        issues.append("kind must be 'tool'")
    if not TOOL_NAME_PATTERN.match(str(declaration["name"])):
        issues.append("name must be 1-128 characters from [A-Za-z0-9_.-] (MCP tool-name rules)")
    for key in ("input_schema", "output_schema"):
        schema = declaration[key]
        if not isinstance(schema, dict) or schema.get("type") != "object":
            issues.append(f"{key} must be a JSON Schema whose type is 'object'")
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            issues.append(f"{key} is not valid JSON Schema: {exc.message}")
    if isinstance(declaration["output_schema"], dict):
        missing = ENVELOPE_REQUIRED - set(declaration["output_schema"].get("required", []))
        if missing:
            issues.append(f"output_schema must require the envelope fields {sorted(ENVELOPE_REQUIRED)}")
    if declaration["side_effects"] not in SIDE_EFFECTS:
        issues.append(f"side_effects must be one of {sorted(SIDE_EFFECTS)}")
    if not isinstance(declaration["config"], dict):
        issues.append("config must be an object")

    mcp = declaration["mcp"]
    for key in ("server", "title", "entrypoint"):
        if not isinstance(mcp.get(key), str) or not mcp[key].strip():
            issues.append(f"mcp.{key} must be a non-empty string")
    annotations = mcp.get("annotations", {})
    unknown = set(annotations) - ANNOTATION_KEYS
    if unknown:
        issues.append(f"mcp.annotations has unknown keys {sorted(unknown)}")
    if any(not isinstance(value, bool) for value in annotations.values()):
        issues.append("mcp.annotations values must be booleans")
    if annotations.get("readOnlyHint") and declaration["side_effects"] not in {"none", "read"}:
        issues.append("readOnlyHint is true but side_effects says the tool writes")
    if annotations.get("readOnlyHint") and annotations.get("destructiveHint"):
        issues.append("a read-only tool cannot be destructive")
    if annotations.get("openWorldHint") is False and declaration["side_effects"] == "external":
        issues.append("openWorldHint is false but side_effects is 'external'")
    if isinstance(mcp.get("entrypoint"), str):
        try:
            resolve_entrypoint(mcp["entrypoint"])
        except Exception as exc:  # noqa: BLE001 - report every resolution problem the same way
            issues.append(f"mcp.entrypoint cannot be resolved: {exc}")
    return issues


def resolve_entrypoint(spec: str) -> Executor:
    """Resolve "code/<path>.py::<function>" to a callable.

    The repository's code/ folder cannot be imported as a package: Python's standard library already has a
    module named `code`. The file path is therefore mapped to a module name relative to code/
    (code/tools/abalone/sex_models.py -> tools.abalone.sex_models), with code/ on sys.path.
    """
    file_part, separator, function_name = spec.partition("::")
    if not separator or not function_name:
        raise ValueError(f"entrypoint {spec!r} must look like 'code/<path>.py::<function>'")
    path = (REPO_ROOT / file_part).resolve()
    if path.suffix != ".py" or CODE_DIR not in path.parents or not path.is_file():
        raise ValueError(f"{file_part!r} is not a Python file inside code/")
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    module_name = ".".join(path.relative_to(CODE_DIR).with_suffix("").parts)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"{module_name} has no function {function_name!r}")
    return function


def invoke(declaration: dict[str, Any], arguments: Any, executor: Executor | None = None) -> dict[str, Any]:
    """Run a tool with a copy of its declared config. Always returns an envelope."""
    executor = executor or resolve_entrypoint(declaration["mcp"]["entrypoint"])
    try:
        return executor(copy.deepcopy(arguments), copy.deepcopy(declaration["config"]))
    except Exception as exc:  # noqa: BLE001 - executors should not raise; this keeps callers alive if one does
        return error_envelope(declaration, "runtime_error", f"{type(exc).__name__}: {exc}")


def output_problems(declaration: dict[str, Any], envelope: Any) -> list[str]:
    """Every place where an envelope breaks the tool's output_schema (empty when it conforms)."""
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(declaration["output_schema"])
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(envelope)
    ]


def error_envelope(declaration: dict[str, Any], error_type: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool_id": declaration["id"],
        "version": declaration["version"],
        "flags": [],
        "warnings": [],
        "error": {"type": error_type, "message": message},
        "disclaimer": declaration.get("disclaimer"),
    }
