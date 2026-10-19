"""sex_experts.py

Non-ML executors for sex prediction:
- abalone_sex_maturity_rules: transparent infant-vs-adult rules read from the declaration's config.rules.
  It abstains on male vs female by design.
- abalone_sex_bivariate_profile: Bayes rule over bivariate normal class profiles of (diameter, number_of_rings),
  whose parameters live in expert_params.json.

The pure functions (validate_rules, rule_fires, maturity_verdict, profile_scores) are reused by
train_abalone_tools.py, so training statistics and live answers come from the same code.
"""

from __future__ import annotations

import json
import math
import operator
from typing import Any

import numpy as np

from .common import (
    MEASUREMENTS,
    RINGS,
    SEX_CLASSES,
    TRAIN_COMMAND,
    ToolSpec,
    load_json_artifact,
    parse_specimens,
    run_tool,
    sex_call,
    soft_flags,
    tool_metrics,
    train_ranges,
    union_flags,
)

MATURITY_RULES = ToolSpec(
    "abalone_sex_maturity_rules",
    "0.1.0",
    "Expert heuristic for infant vs adult from shell size and ring count; it never claims male or female.",
)
BIVARIATE_PROFILE = ToolSpec(
    "abalone_sex_bivariate_profile",
    "0.1.0",
    "Bayes-rule estimate from two variables (diameter, number_of_rings); male and female profiles overlap almost completely.",
)

OPERATORS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}
RULE_FEATURES = (*MEASUREMENTS, RINGS)
RULE_VERDICTS = ("I", "ADULT")


def predict_maturity_rules(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(MATURITY_RULES, arguments, config, _maturity_rules)


def predict_bivariate_profile(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(BIVARIATE_PROFILE, arguments, config, _bivariate_profile)


# ---------------------------------------------------------------------------
# Maturity rules
# ---------------------------------------------------------------------------

def validate_rules(rules: list[dict[str, Any]], default: dict[str, Any]) -> None:
    """Raise ValueError naming the first broken entry of config.rules / config.default."""
    if not isinstance(rules, list) or not rules:
        raise ValueError("config.rules must be a non-empty list.")
    ids = set()
    for i, rule in enumerate(rules):
        where = f"config.rules[{i}]"
        for key in ("id", "verdict", "confidence", "text", "conditions"):
            if key not in rule:
                raise ValueError(f"{where} is missing {key!r}.")
        if rule["id"] in ids or rule["id"] == default.get("id"):
            raise ValueError(f"{where} repeats the id {rule['id']!r}.")
        ids.add(rule["id"])
        if rule["verdict"] not in RULE_VERDICTS:
            raise ValueError(f"{where}.verdict must be one of {list(RULE_VERDICTS)}; this tool never claims M or F.")
        if not rule["conditions"]:
            raise ValueError(f"{where}.conditions must not be empty.")
        for j, condition in enumerate(rule["conditions"]):
            if condition.get("feature") not in RULE_FEATURES:
                raise ValueError(f"{where}.conditions[{j}].feature must be one of {list(RULE_FEATURES)}.")
            if condition.get("op") not in OPERATORS:
                raise ValueError(f"{where}.conditions[{j}].op must be one of {list(OPERATORS)}.")
            if isinstance(condition.get("value"), bool) or not isinstance(condition.get("value"), (int, float)):
                raise ValueError(f"{where}.conditions[{j}].value must be a number.")
    for key in ("id", "verdict", "confidence", "text"):
        if key not in default:
            raise ValueError(f"config.default is missing {key!r}.")
    if default["verdict"] not in RULE_VERDICTS:
        raise ValueError(f"config.default.verdict must be one of {list(RULE_VERDICTS)}.")


def rule_fires(rule: dict[str, Any], row: dict[str, Any]) -> bool:
    return all(OPERATORS[c["op"]](row[c["feature"]], c["value"]) for c in rule["conditions"])


def maturity_verdict(rules: list[dict[str, Any]], default: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """The first rule that fires decides; with none, the default does. Returns (verdict, deciding-first entries)."""
    fired = [rule for rule in rules if rule_fires(rule, row)]
    deciding = fired or [default]
    return deciding[0]["verdict"], deciding


def rules_fingerprint(rules: list[dict[str, Any]], default: dict[str, Any]) -> str:
    return json.dumps({"rules": rules, "default": default}, sort_keys=True)


def _maturity_rules(arguments: dict[str, Any], config: dict[str, Any]):
    rules, default = config["rules"], config["default"]
    validate_rules(rules, default)
    rows = parse_specimens(arguments, target="sex", max_specimens=config["max_specimens"])
    metrics = tool_metrics(config, MATURITY_RULES.tool_id)
    ranges = train_ranges(config)
    stale = metrics["rules_fingerprint"] != rules_fingerprint(rules, default)
    stats = {} if stale else metrics["rule_stats"]

    predictions = []
    for index, row in enumerate(rows):
        verdict, deciding = maturity_verdict(rules, default, row)
        adult = verdict == "ADULT"
        predictions.append({
            "index": index,
            "verdict": verdict,
            "sex_candidates": ["M", "F"] if adult else ["I"],
            "abstained": adult,
            "confidence": deciding[0]["confidence"],
            "deciding_rule": deciding[0]["id"],
            "fired_rules": [
                {
                    "id": rule["id"],
                    "text": rule["text"],
                    "verdict": rule["verdict"],
                    "train_precision": stats.get(rule["id"], {}).get("train_precision"),
                    "train_coverage": stats.get(rule["id"], {}).get("train_coverage"),
                }
                for rule in deciding
            ],
            "adult_base_rates": metrics["adult_base_rates"] if adult else None,
            "flags": soft_flags(row, ranges),
        })

    warnings = ["This tool separates infants from adults only; for adults it abstains on male vs female by design."]
    tool_flags = []
    if stale:
        tool_flags.append("rule_stats_stale")
        warnings.append(
            "config.rules changed after training, so rule precision and hold-out figures are unavailable. "
            f"Re-run: {TRAIN_COMMAND}"
        )
    model_card = {
        "algorithm": metrics["algorithm"],
        "rules": [{key: rule[key] for key in ("id", "text", "verdict", "confidence")} for rule in [*rules, default]],
        "trained_on": metrics["trained_on"],
        "holdout_metrics": None if stale else metrics["holdout_metrics"],
        "stale": stale,
    }
    return {"predictions": predictions, "model_card": model_card}, union_flags(predictions, *tool_flags), warnings


# ---------------------------------------------------------------------------
# Bivariate profile
# ---------------------------------------------------------------------------

def chi2_2dof_quantile(quantile: float) -> float:
    """Quantile of the chi-square distribution with 2 degrees of freedom (closed form)."""
    return -2.0 * math.log(1.0 - quantile)


def profile_scores(params: dict[str, Any], row: dict[str, Any]) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """Posterior P(sex | x) from priors and bivariate normal likelihoods, plus squared Mahalanobis distances."""
    x = np.array([row[name] for name in params["features"]], dtype=float)
    classes = list(params["classes"])
    log_posterior, distances = [], {}
    for cls in classes:
        profile = params["classes"][cls]
        mean, cov = np.asarray(profile["mean"], dtype=float), np.asarray(profile["cov"], dtype=float)
        diff = x - mean
        distances[cls] = float(diff @ np.linalg.solve(cov, diff))
        _, log_det = np.linalg.slogdet(cov)
        log_posterior.append(math.log(profile["prior"]) - 0.5 * distances[cls] - 0.5 * log_det - math.log(2 * math.pi))
    log_posterior = np.array(log_posterior)
    proba = np.exp(log_posterior - log_posterior.max())
    return classes, proba / proba.sum(), distances


def _bivariate_profile(arguments: dict[str, Any], config: dict[str, Any]):
    rows = parse_specimens(arguments, target="sex", max_specimens=config["max_specimens"])
    params = load_json_artifact(config["params_path"])["sex_bivariate_profile"]
    metrics = tool_metrics(config, BIVARIATE_PROFILE.tool_id)
    ranges = train_ranges(config)
    threshold = chi2_2dof_quantile(config["out_of_profile_quantile"])

    predictions = []
    for index, row in enumerate(rows):
        classes, proba, distances = profile_scores(params, row)
        call, flags = sex_call(
            classes,
            proba,
            mf_ambiguity_margin=config["mf_ambiguity_margin"],
            maturity_uncertainty_band=config["maturity_uncertainty_band"],
        )
        nearest = min(distances, key=distances.get)
        if distances[nearest] > threshold:
            flags.append("out_of_profile")
        predictions.append({
            "index": index,
            **call,
            "nearest_profile": nearest,
            "mahalanobis_sq": {cls: round(distances[cls], 3) for cls in SEX_CLASSES},
            "flags": flags + soft_flags(row, ranges),
        })

    holdout = metrics["holdout_metrics"]
    warnings = [
        f"Male vs female accuracy among adults is {holdout['mf_accuracy_among_adults']:.1%} on hold-out data "
        f"(always answering the most common adult sex scores {holdout['mf_majority_baseline']:.1%})."
    ]
    model_card = {
        "algorithm": metrics["algorithm"],
        "features": list(params["features"]),
        "trained_on": metrics["trained_on"],
        "profiles": {
            cls: {"mean": [round(v, 4) for v in params["classes"][cls]["mean"]], "prior": round(params["classes"][cls]["prior"], 4)}
            for cls in SEX_CLASSES
        },
        "out_of_profile_threshold": round(threshold, 3),
        "holdout_metrics": holdout,
    }
    return {"predictions": predictions, "model_card": model_card}, union_flags(predictions), warnings
