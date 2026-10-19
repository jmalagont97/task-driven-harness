"""age_models.py

ML executors for age prediction (ring count; age in years is about rings + 1.5):
- abalone_age_gradient_boosting: HistGradientBoosting on log1p(rings), the strongest of the four age tools.
- abalone_age_ridge_regression: linear baseline with per-specimen contributions.

Both return an 80 % cross-conformal interval whose half-width was estimated at training time from
out-of-fold residuals. Nothing is trained at call time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    SEX_CLASSES,
    ToolSpec,
    age_features,
    age_record,
    load_model_artifact,
    parse_specimens,
    run_tool,
    soft_flags,
    tool_metrics,
    top_contributions,
    train_ranges,
    union_flags,
)

DISCLAIMER = (
    "Statistical estimate of the ring count from shell morphometrics; age in years is about rings + 1.5. "
    "The 80% interval holds on average over specimens, not for every single specimen."
)
GRADIENT_BOOSTING = ToolSpec("abalone_age_gradient_boosting", "0.1.0", DISCLAIMER)
RIDGE_REGRESSION = ToolSpec("abalone_age_ridge_regression", "0.1.0", DISCLAIMER)


def predict_gradient_boosting(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(GRADIENT_BOOSTING, arguments, config, _gradient_boosting)


def predict_ridge_regression(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(RIDGE_REGRESSION, arguments, config, _ridge_regression)


def predict_rings(bundle: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Model output mapped back to rings (undoing the log1p target transform when the bundle used it)."""
    raw = bundle["model"].predict(features[list(bundle["features"])])
    return np.expm1(raw) if bundle["target_transform"] == "log1p" else np.asarray(raw, dtype=float)


def _gradient_boosting(arguments: dict[str, Any], config: dict[str, Any]):
    bundle, metrics, _, predictions = _score(GRADIENT_BOOSTING, arguments, config)
    result = {"predictions": predictions, "model_card": _model_card(bundle, metrics)}
    return result, union_flags(predictions), _warnings(metrics)


def _ridge_regression(arguments: dict[str, Any], config: dict[str, Any]):
    bundle, metrics, frame, predictions = _score(RIDGE_REGRESSION, arguments, config)
    scaler, ridge = bundle["model"][0], bundle["model"][-1]
    features = list(bundle["features"])
    standardized = scaler.transform(frame[features])
    for prediction, raw, z in zip(predictions, frame[features].to_numpy(), standardized):
        prediction["contributions"] = top_contributions(features, raw, ridge.coef_ * z)
    model_card = {**_model_card(bundle, metrics), "alpha": round(float(ridge.alpha_), 6)}
    return {"predictions": predictions, "model_card": model_card}, union_flags(predictions), _warnings(metrics)


def _score(spec: ToolSpec, arguments: dict[str, Any], config: dict[str, Any]):
    rows = parse_specimens(arguments, target="age", sex_values=SEX_CLASSES, max_specimens=config["max_specimens"])
    bundle = load_model_artifact(config["model_path"])
    metrics = tool_metrics(config, spec.tool_id)
    ranges = train_ranges(config)
    frame = age_features(pd.DataFrame(rows))
    rings = predict_rings(bundle, frame)
    half_width = bundle["interval"]["half_width_rings"]
    predictions = [
        age_record(
            index,
            value,
            value - half_width,
            value + half_width,
            "cross-conformal",
            tail_rings_above=config["tail_rings_above"],
            flags=soft_flags(row, ranges),
        )
        for index, (row, value) in enumerate(zip(rows, rings))
    ]
    return bundle, metrics, frame, predictions


def _model_card(bundle: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": metrics["algorithm"],
        "features": list(bundle["features"]),
        "trained_on": metrics["trained_on"],
        "interval": bundle["interval"],
        "holdout_metrics": metrics["holdout_metrics"],
    }


def _warnings(metrics: dict[str, Any]) -> list[str]:
    holdout = metrics["holdout_metrics"]
    old = holdout["mae_by_age_group"].get("Old (16+)")
    tail = f" Error grows in the tail: MAE {old:.2f} rings for specimens with 16+ rings." if old is not None else ""
    return [f"Hold-out MAE is {holdout['mae']:.2f} rings and the 80% interval covered {holdout['interval_coverage']:.1%} of test specimens.{tail}"]
