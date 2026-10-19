"""age_experts.py

Non-ML executors for age prediction (ring count; age in years is about rings + 1.5):
- abalone_age_allometric_formula: rings = a * shell_weight^b per sex, with a lognormal 80 % interval.
- abalone_age_shell_weight_lookup: empirical table (shell_weight decile x sex) -> median ring count and P10-P90.

Both accept sex = "unknown", which switches to parameters estimated on all training specimens.
Parameters live in expert_params.json; the pure functions here are reused by train_abalone_tools.py.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from statistics import NormalDist
from typing import Any

from .common import (
    SEX,
    SEX_CLASSES,
    UNKNOWN_SEX,
    ToolSpec,
    age_record,
    load_json_artifact,
    parse_specimens,
    run_tool,
    soft_flags,
    tool_metrics,
    train_ranges,
    union_flags,
)

ALLOMETRIC_FORMULA = ToolSpec(
    "abalone_age_allometric_formula",
    "0.1.0",
    "Allometric rule of thumb based on dried shell weight and sex only; age in years is about rings + 1.5.",
)
SHELL_WEIGHT_LOOKUP = ToolSpec(
    "abalone_age_shell_weight_lookup",
    "0.1.0",
    "Empirical summary of similar training specimens (same sex, same shell-weight decile); age in years is about rings + 1.5.",
)

EXPERT_SEX_VALUES = (*SEX_CLASSES, UNKNOWN_SEX)
Z_80 = NormalDist().inv_cdf(0.9)  # central 80 % of a normal distribution on the log scale


def predict_allometric_formula(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(ALLOMETRIC_FORMULA, arguments, config, _allometric_formula)


def predict_shell_weight_lookup(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(SHELL_WEIGHT_LOOKUP, arguments, config, _shell_weight_lookup)


# ---------------------------------------------------------------------------
# Allometric formula
# ---------------------------------------------------------------------------

def allometric_predict(params: dict[str, Any], sex: str, shell_weight: float) -> tuple[float, float, float, dict[str, Any]]:
    """(rings, low, high, coefficients) for one specimen; the interval is multiplicative on the ring scale."""
    group = params["groups"][sex]
    rings = group["a"] * shell_weight ** group["b"]
    spread = math.exp(Z_80 * group["sigma_log"])
    return rings, rings / spread, rings * spread, group


def _allometric_formula(arguments: dict[str, Any], config: dict[str, Any]):
    rows = parse_specimens(arguments, target="age", sex_values=EXPERT_SEX_VALUES, max_specimens=config["max_specimens"])
    params = load_json_artifact(config["params_path"])["age_allometric_formula"]
    metrics = tool_metrics(config, ALLOMETRIC_FORMULA.tool_id)
    ranges = train_ranges(config)

    predictions = []
    for index, row in enumerate(rows):
        weight = row["shell_weight"]
        rings, low, high, group = allometric_predict(params, row[SEX], weight)
        flags = soft_flags(row, ranges) + (["sex_unknown"] if row[SEX] == UNKNOWN_SEX else [])
        record = age_record(index, rings, low, high, "lognormal residuals", tail_rings_above=config["tail_rings_above"], flags=flags)
        record["formula"] = (
            f"rings = {group['a']:.3f} * shell_weight^{group['b']:.3f} = {group['a']:.3f} * {weight:g}^{group['b']:.3f} = {rings:.2f}"
        )
        record["coefficients"] = {
            "group": row[SEX],
            "a": round(group["a"], 4),
            "b": round(group["b"], 4),
            "sigma_log": round(group["sigma_log"], 4),
        }
        predictions.append(record)

    holdout = metrics["holdout_metrics"]
    warnings = [
        f"Uses shell_weight and sex only. Hold-out MAE {holdout['mae']:.2f} rings; "
        f"the 80% interval covered {holdout['interval_coverage']:.1%} of test specimens."
    ]
    model_card = {
        "algorithm": metrics["algorithm"],
        "trained_on": metrics["trained_on"],
        "coefficients": {
            name: {key: round(value, 4) if isinstance(value, float) else value for key, value in group.items()}
            for name, group in params["groups"].items()
        },
        "interval": {"method": "lognormal residuals", "level": 0.8},
        "holdout_metrics": holdout,
    }
    return {"predictions": predictions, "model_card": model_card}, union_flags(predictions), warnings


# ---------------------------------------------------------------------------
# Shell-weight lookup table
# ---------------------------------------------------------------------------

def lookup_bin(edges: list[float], shell_weight: float) -> int:
    """Index of the decile bin; bins are closed on the right like pandas.cut(include_lowest=True)."""
    return bisect_left(edges, shell_weight)


def lookup_cell(params: dict[str, Any], sex: str, shell_weight: float) -> tuple[dict[str, Any], str, bool]:
    """(cell, group used, fell back to all sexes). Falls back when the sex-specific cell is empty."""
    position = lookup_bin(params["edges"], shell_weight)
    cell = params["cells"][sex][position]
    if cell["n"] > 0:
        return cell, sex, False
    return params["cells"][UNKNOWN_SEX][position], UNKNOWN_SEX, True


def _shell_weight_lookup(arguments: dict[str, Any], config: dict[str, Any]):
    rows = parse_specimens(arguments, target="age", sex_values=EXPERT_SEX_VALUES, max_specimens=config["max_specimens"])
    params = load_json_artifact(config["params_path"])["age_shell_weight_lookup"]
    metrics = tool_metrics(config, SHELL_WEIGHT_LOOKUP.tool_id)
    ranges = train_ranges(config)
    edges = params["edges"]

    predictions = []
    for index, row in enumerate(rows):
        cell, group, fallback = lookup_cell(params, row[SEX], row["shell_weight"])
        flags = soft_flags(row, ranges)
        if row[SEX] == UNKNOWN_SEX:
            flags.append("sex_unknown")
        if fallback:
            flags.append("empty_cell_fallback")
        if cell["n"] < config["min_cell_n"]:
            flags.append("sparse_cell")
        record = age_record(
            index,
            cell["median"],
            cell["p10"],
            cell["p90"],
            "empirical P10-P90",
            tail_rings_above=config["tail_rings_above"],
            flags=flags,
        )
        position = cell["bin"]
        record["cell"] = {
            "sex_group": group,
            "bin": position,
            "shell_weight_range": [0.0 if position == 0 else edges[position - 1], edges[position] if position < len(edges) else None],
            "n": cell["n"],
        }
        predictions.append(record)

    sparse = sum(1 for sex in SEX_CLASSES for cell in params["cells"][sex] if cell["n"] < config["min_cell_n"])
    holdout = metrics["holdout_metrics"]
    warnings = [
        f"Predictions move in whole or half rings and ignore every variable except shell_weight and sex. "
        f"Hold-out MAE {holdout['mae']:.2f} rings; P10-P90 covered {holdout['interval_coverage']:.1%} of test specimens."
    ]
    model_card = {
        "algorithm": metrics["algorithm"],
        "trained_on": metrics["trained_on"],
        "edges": edges,
        "min_cell_n": config["min_cell_n"],
        "sparse_cells": sparse,
        "holdout_metrics": holdout,
    }
    return {"predictions": predictions, "model_card": model_card}, union_flags(predictions), warnings
