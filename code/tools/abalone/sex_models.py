"""sex_models.py

ML executors for sex prediction (M = male, F = female, I = infant):
- abalone_sex_random_forest: non-linear opinion with feature importances.
- abalone_sex_logistic_regression: linear, better calibrated, with per-specimen contributions.

Both load a model bundle written by train_abalone_tools.py; nothing is trained at call time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .common import (
    ToolSpec,
    load_model_artifact,
    parse_specimens,
    run_tool,
    sex_call,
    sex_features,
    soft_flags,
    tool_metrics,
    top_contributions,
    train_ranges,
    union_flags,
)

DISCLAIMER = (
    "Statistical estimate from shell morphometrics, not a biological sexing method. "
    "Male vs female separation is near chance on hold-out data."
)
RANDOM_FOREST = ToolSpec("abalone_sex_random_forest", "0.1.0", DISCLAIMER)
LOGISTIC_REGRESSION = ToolSpec("abalone_sex_logistic_regression", "0.1.0", DISCLAIMER)


def predict_random_forest(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(RANDOM_FOREST, arguments, config, _random_forest)


def predict_logistic_regression(arguments: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return run_tool(LOGISTIC_REGRESSION, arguments, config, _logistic_regression)


def _random_forest(arguments: dict[str, Any], config: dict[str, Any]):
    bundle, metrics, _, predictions = _score(RANDOM_FOREST, arguments, config)
    result = {"predictions": predictions, "model_card": _model_card(bundle, metrics)}
    return result, union_flags(predictions), _warnings(metrics)


def _logistic_regression(arguments: dict[str, Any], config: dict[str, Any]):
    bundle, metrics, frame, predictions = _score(LOGISTIC_REGRESSION, arguments, config)
    pipeline = base_estimator(bundle["model"])
    scaler, logistic = pipeline[0], pipeline[-1]
    features = list(bundle["features"])
    standardized = scaler.transform(frame[features])
    class_index = {cls: i for i, cls in enumerate(logistic.classes_)}
    for prediction, raw, z in zip(predictions, frame[features].to_numpy(), standardized):
        coefficients = logistic.coef_[class_index[prediction["predicted_sex"]]]
        prediction["contributions"] = top_contributions(features, raw, coefficients * z)
    result = {"predictions": predictions, "model_card": _model_card(bundle, metrics)}
    return result, union_flags(predictions), _warnings(metrics)


def _score(spec: ToolSpec, arguments: dict[str, Any], config: dict[str, Any]):
    rows = parse_specimens(arguments, target="sex", max_specimens=config["max_specimens"])
    bundle = load_model_artifact(config["model_path"])
    metrics = tool_metrics(config, spec.tool_id)
    ranges = train_ranges(config)
    frame = sex_features(pd.DataFrame(rows))
    model = bundle["model"]
    proba = model.predict_proba(frame[list(bundle["features"])])
    classes = [str(cls) for cls in model.classes_]
    predictions = []
    for index, (row, p) in enumerate(zip(rows, proba)):
        call, flags = sex_call(
            classes,
            p,
            mf_ambiguity_margin=config["mf_ambiguity_margin"],
            maturity_uncertainty_band=config["maturity_uncertainty_band"],
        )
        predictions.append({"index": index, **call, "flags": flags + soft_flags(row, ranges)})
    return bundle, metrics, frame, predictions


def base_estimator(model: Any) -> Any:
    """The fitted estimator behind an optional CalibratedClassifierCV wrapper."""
    calibrated = getattr(model, "calibrated_classifiers_", None)
    return calibrated[0].estimator if calibrated else model


def _model_card(bundle: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": metrics["algorithm"],
        "features": list(bundle["features"]),
        "trained_on": metrics["trained_on"],
        "top_features": metrics["top_features"],
        "holdout_metrics": metrics["holdout_metrics"],
    }


def _warnings(metrics: dict[str, Any]) -> list[str]:
    holdout = metrics["holdout_metrics"]
    return [
        f"Male vs female accuracy among adults is {holdout['mf_accuracy_among_adults']:.1%} on hold-out data "
        f"(always answering the most common adult sex scores {holdout['mf_majority_baseline']:.1%}); "
        "treat M/F calls as weak evidence and rely on p_adult for infant vs adult."
    ]
