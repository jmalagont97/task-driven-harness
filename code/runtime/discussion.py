"""discussion.py

The saved record of every debate and question.

Responsibilities:
- render_discussion: one readable Markdown file per run (discussion.md) with the problem as the agents saw it and,
  round by round, every speaker's tool calls (arguments and what the tools returned), position or expectation,
  claims with evidence, reviews, replies and warnings; then the ledger and the arbiter's decision. It is rewritten
  after every turn, so an interrupted run keeps everything said until then.
- append_index: one JSON line per finished run in <runs_dir>/discussions_index.jsonl, so every discussion can be found
  again by task, row, answer or outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime.debate import normalize_id

PREDICTION_FIELDS = (
    "predicted_sex", "verdict", "deciding_rule", "abstained", "probabilities", "p_adult", "mf_margin", "nearest_profile",
    "predicted_rings", "predicted_age_years", "interval_80", "median_rings", "n", "formula", "flags",
)


def tool_line(call: dict[str, Any]) -> str:
    name, arguments, result = call.get("tool", "?"), call.get("arguments") or {}, call.get("result")
    prefix = f"`{name}`"
    if call.get("corrected_from") is not None:
        prefix += " (arguments corrected to the problem_spec values)"
    if not call.get("ok", False):
        error = (result or {}).get("error") if isinstance(result, dict) else None
        detail = (error or {}).get("message") or call.get("error") or "failed"
        return f"{prefix} failed: {str(detail)[:300]}"
    body = (result or {}).get("result") if isinstance(result, dict) else None
    if not isinstance(body, dict):
        return f"{prefix} returned: {str(result)[:300]}"
    if "results" in body:
        hits = ", ".join(f"{r.get('chunk_id')} ({r.get('score', '?')})" for r in body["results"])
        below = " **below_threshold**" if body.get("below_threshold") else ""
        return f"{prefix} query \"{arguments.get('query', '')}\" -> {hits or 'no passages'}{below}"
    if "chunks" in body:
        ids = ", ".join(str(chunk.get("chunk_id")) for chunk in body["chunks"])
        return f"{prefix} section {body.get('section_id')} part {body.get('part')}/{body.get('parts_total')} -> {ids}"
    predictions = body.get("predictions") or []
    if predictions:
        fields = {key: predictions[0][key] for key in PREDICTION_FIELDS if key in predictions[0]}
        return f"{prefix} -> {json.dumps(fields, ensure_ascii=False)}"
    return f"{prefix} -> {json.dumps(body, ensure_ascii=False)[:300]}"


def evidence(items: list[Any]) -> str:
    return ", ".join(str(item) for item in items) or "none"


def turn_section(record: dict[str, Any]) -> list[str]:
    output = record.get("output") or {}
    result = output.get("result") or {}
    lines = [f"### {record['code']} ({record.get('name', '')})", ""]
    if record.get("error"):
        lines.append(f"Turn failed: {record['error']}")
    for call in record.get("tool_calls") or []:
        lines.append(f"- Tool: {tool_line(call)}")
    if output.get("status") == "abstained":
        lines.append("- Abstained: the article does not cover this specimen.")
    position = result.get("position")
    if isinstance(position, dict):
        lines.append(f"- Position: {json.dumps(position, ensure_ascii=False)}")
    if result.get("expectation"):
        lines.append(f"- Expectation: {result['expectation']}")
    for claim_id, claim in zip(record.get("claim_ids") or [], result.get("claims") or []):
        lines.append(f"- Claim [{claim_id}]: {claim.get('text')} (evidence: {evidence(claim.get('evidence') or [])})")
    for review in result.get("reviews") or []:
        lines.append(f"- {str(review.get('verdict', '')).capitalize()} [{normalize_id(review.get('claim_id'))}]: {review.get('reason')} (evidence: {evidence(review.get('evidence') or [])})")
    for response in result.get("responses") or []:
        lines.append(f"- Reply {response.get('action')} [{normalize_id(response.get('challenge_id'))}]: {response.get('text')} (evidence: {evidence(response.get('evidence') or [])})")
    if result.get("answer_markdown"):
        lines += ["", "Answer:", "", str(result["answer_markdown"]).strip(), "", f"Coverage: {result.get('coverage')}"]
    if result.get("markdown_summary"):
        lines += ["", f"Summary: {str(result['markdown_summary']).strip()}"]
    for warning in record.get("warnings") or []:
        lines.append(f"- Runtime warning: {warning}")
    lines.append("")
    return lines


def render_discussion(meta: dict[str, Any], statement: str, records: list[dict[str, Any]], ledger: str | None, final: dict[str, Any] | None) -> str:
    lines = [f"# {meta['title']}", ""]
    for key, value in meta.items():
        if key != "title" and value not in (None, "", {}):
            lines.append(f"- {key.replace('_', ' ').capitalize()}: {value}")
    lines += ["", "## Problem as the agents saw it", "", "```text", statement.strip(), "```", ""]
    current = None
    for record in records:
        heading = record.get("round_heading")
        if heading and heading != current:
            lines += [f"## {heading}", ""]
            current = heading
        lines += turn_section(record)
    if ledger:
        lines += ["## Debate ledger", "", ledger.strip(), ""]
    if final:
        result = final.get("result") or {}
        lines += ["## Final decision (arbiter)", ""]
        for key in ("answer", "forced_choice", "sex_call", "maturity", "rings", "rings_low", "rings_high", "reliability", "debate_outcome", "arbiter_fallback"):
            if result.get(key) not in (None, ""):
                lines.append(f"- {key.replace('_', ' ').capitalize()}: {result[key]}")
        if result.get("rationale"):
            lines += ["", str(result["rationale"]).strip(), ""]
        for entry in result.get("participants") or []:
            received = "; ".join(
                f"{c.get('challenge_id')} by {c.get('by')} ({c.get('status')}): {c.get('reason')}" for c in entry.get("challenges_received") or []
            ) or "no challenges"
            lines.append(f"- {entry.get('participant')}: {entry.get('weight')}. {entry.get('why')} Challenges received: {received}")
        lines.append("")
    return "\n".join(lines)


def append_index(index_path: Path, record: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
