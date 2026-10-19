"""train_abalone_tools.py

Offline training for the eight Abalone tools. Run from the repository root:

    .venv/bin/python code/tools/abalone/train_abalone_tools.py

Steps:
1. Freeze the train/test split in artifacts/datasets/abalone_dataset/splits.json (reused when it already exists).
2. Fit the four ML models on the train split, estimate their cross-conformal intervals and check calibration.
3. Estimate the parameters of the four expert tools on the train split.
4. Evaluate all eight tools once on the test split and write expert_params.json and metrics.json.

Artifact paths and the expert rule thresholds come from the tools' declarations (declarations/tools/abalone_*.json).
The test split is only used for the final evaluation, never to choose a model or a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

import tool_registry  # noqa: E402
from tools.abalone import common  # noqa: E402
from tools.abalone.age_experts import allometric_predict, lookup_bin, lookup_cell  # noqa: E402
from tools.abalone.age_models import predict_rings  # noqa: E402
from tools.abalone.common import RINGS, SEED, SEX, SEX_CLASSES, UNKNOWN_SEX  # noqa: E402
from tools.abalone.sex_experts import maturity_verdict, profile_scores, rule_fires, rules_fingerprint, validate_rules  # noqa: E402
from tools.abalone.sex_models import base_estimator  # noqa: E402

TOOL_IDS = (
    "abalone_sex_random_forest",
    "abalone_sex_logistic_regression",
    "abalone_sex_maturity_rules",
    "abalone_sex_bivariate_profile",
    "abalone_age_gradient_boosting",
    "abalone_age_ridge_regression",
    "abalone_age_allometric_formula",
    "abalone_age_shell_weight_lookup",
)
CV_FOLDS = 5
INTERVAL_LEVEL = 0.8
CALIBRATION_ECE_THRESHOLD = 0.05  # calibrate a sex model when its out-of-fold ECE on the train split is above this
ECE_BINS = 10
OOD_QUANTILES = (0.005, 0.995)
LOOKUP_BINS = 10
PROFILE_FEATURES = ["diameter", RINGS]

SEX_MODELS = {
    "abalone_sex_random_forest": (
        RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=42)",
    ),
    "abalone_sex_logistic_regression": (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "StandardScaler + multinomial LogisticRegression(lbfgs, C=1.0, max_iter=3000)",
    ),
}
AGE_MODELS = {
    "abalone_age_gradient_boosting": (
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED),
        "log1p",
        "HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42) on log1p(rings)",
    ),
    "abalone_age_ridge_regression": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "identity",
        "StandardScaler + RidgeCV(alphas=logspace(-3, 3, 25)) on raw rings",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the eight Abalone tools.")
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Discard the frozen splits.json and create it again (only needed if the dataset changed).",
    )
    args = parser.parse_args(argv)

    configs = load_configs()
    frame = common.load_dataset()
    split, reused = freeze_split(frame, resplit=args.resplit)
    train, test = common.split_frames(frame, split)
    fingerprint = common.split_fingerprint(split)
    trained_on = f"splits.json train split: {len(train)} rows, seed {split['seed']}, fingerprint {fingerprint}"
    print(f"Split {'reused' if reused else 'created'}: {len(train)} train / {len(test)} test rows, "
          f"{len(split['excluded_rows'])} excluded, fingerprint {fingerprint}")

    ranges = {
        name: [round(float(np.quantile(train[name], OOD_QUANTILES[0])), 6), round(float(np.quantile(train[name], OOD_QUANTILES[1])), 6)]
        for name in (*common.MEASUREMENTS, RINGS)
    }
    train_flags = [common.soft_flags(row, ranges) for row in train.to_dict("records")]

    tools: dict[str, dict[str, Any]] = {}
    tools.update(train_sex_models(train, test, configs, trained_on))
    tools.update(train_age_models(train, test, configs, trained_on))
    params = {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "split_fingerprint": fingerprint,
        "sex_bivariate_profile": fit_bivariate_profile(train),
        "age_allometric_formula": fit_allometric_formula(train),
        "age_shell_weight_lookup": fit_shell_weight_lookup(train),
    }
    tools.update(evaluate_maturity_rules(train, test, configs["abalone_sex_maturity_rules"], trained_on))
    tools.update(evaluate_bivariate_profile(train, test, params["sex_bivariate_profile"], configs["abalone_sex_bivariate_profile"], trained_on))
    tools.update(evaluate_allometric_formula(train, test, params["age_allometric_formula"], trained_on))
    tools.update(evaluate_shell_weight_lookup(train, test, params["age_shell_weight_lookup"], configs["abalone_age_shell_weight_lookup"], trained_on))

    metrics = {
        "schema_version": "1.0.0",
        "generated_at": now(),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "path": split["dataset_path"],
            "sha256": split["dataset_sha256"],
            "rows": int(len(frame)),
        },
        "split": {
            "path": str(common.SPLIT_PATH.relative_to(common.REPO_ROOT)),
            "fingerprint": fingerprint,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "excluded_rows": [row["index"] for row in split["excluded_rows"]],
        },
        "train_profile": {
            "ranges": ranges,
            "quantiles": list(OOD_QUANTILES),
            "out_of_distribution_rate": rate(["out_of_distribution" in flags for flags in train_flags]),
            "inconsistent_weights_rate": rate(["inconsistent_weights" in flags for flags in train_flags]),
        },
        "tools": {tool_id: tools[tool_id] for tool_id in TOOL_IDS},
    }
    shared = configs[TOOL_IDS[0]]
    write_json(common.REPO_ROOT / configs["abalone_sex_bivariate_profile"]["params_path"], params)
    write_json(common.REPO_ROOT / shared["metrics_path"], metrics)
    print_summary(metrics)
    return 0


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_configs() -> dict[str, dict[str, Any]]:
    """The config block of every Abalone declaration; they must share one metrics file and one params file."""
    declarations = {d.get("id"): d for _, d in tool_registry.iter_declarations()}
    missing = [tool_id for tool_id in TOOL_IDS if tool_id not in declarations]
    if missing:
        raise SystemExit(f"Missing declarations in declarations/tools/: {missing}")
    configs = {tool_id: declarations[tool_id]["config"] for tool_id in TOOL_IDS}
    if len({config["metrics_path"] for config in configs.values()}) != 1:
        raise SystemExit("Every Abalone declaration must use the same config.metrics_path.")
    if len({config["params_path"] for config in configs.values() if "params_path" in config}) != 1:
        raise SystemExit("Every Abalone expert declaration must use the same config.params_path.")
    return configs


def freeze_split(frame: pd.DataFrame, *, resplit: bool) -> tuple[dict[str, Any], bool]:
    sha = common.file_sha256(common.DATASET_PATH)
    if common.SPLIT_PATH.is_file() and not resplit:
        split = common.load_split()
        if split["dataset_sha256"] != sha:
            raise SystemExit(
                "The dataset changed since splits.json was frozen. Re-run with --resplit to freeze a new split "
                "(every metric will change)."
            )
        return split, True
    split = common.make_split(frame, sha)
    write_json(common.SPLIT_PATH, split)
    return split, False


# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------

def train_sex_models(train: pd.DataFrame, test: pd.DataFrame, configs: dict[str, Any], trained_on: str) -> dict[str, Any]:
    x_train, x_test = common.sex_features(train), common.sex_features(test)
    y_train, y_test = train[SEX].to_numpy(), test[SEX].to_numpy()
    results = {}
    for tool_id, (estimator, algorithm) in SEX_MODELS.items():
        config = configs[tool_id]
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
        oof = cross_val_predict(clone(estimator), x_train, y_train, cv=folds, method="predict_proba")
        ece_train = expected_calibration_error(oof, y_train, sorted(set(y_train)))
        model = clone(estimator).fit(x_train, y_train)
        calibrated = ece_train > CALIBRATION_ECE_THRESHOLD
        if calibrated:
            model = CalibratedClassifierCV(clone(estimator), method="sigmoid", cv=CV_FOLDS).fit(x_train, y_train)
            algorithm += " + sigmoid calibration"
        classes = [str(cls) for cls in model.classes_]
        proba = model.predict_proba(x_test)
        predicted = np.asarray(classes)[proba.argmax(axis=1)]
        bundle = {
            "model": model,
            "features": list(common.SEX_FEATURES),
            "classes": classes,
            "trained_on": trained_on,
            "sklearn_version": sklearn.__version__,
        }
        save_model(config["model_path"], bundle)
        results[tool_id] = {
            "family": "ml",
            "algorithm": algorithm,
            "features": list(common.SEX_FEATURES),
            "trained_on": trained_on,
            "top_features": top_features(model, list(common.SEX_FEATURES)),
            "calibration": {
                "ece_train_out_of_fold": round(ece_train, 4),
                "threshold": CALIBRATION_ECE_THRESHOLD,
                "calibrated": calibrated,
            },
            "holdout_metrics": sex_holdout(y_train, y_test, predicted, proba, classes, config["mf_ambiguity_margin"]),
        }
    return results


def train_age_models(train: pd.DataFrame, test: pd.DataFrame, configs: dict[str, Any], trained_on: str) -> dict[str, Any]:
    x_train, x_test = common.age_features(train), common.age_features(test)
    y_train, y_test = train[RINGS].to_numpy(dtype=float), test[RINGS].to_numpy(dtype=float)
    results = {}
    for tool_id, (estimator, transform, algorithm) in AGE_MODELS.items():
        target = np.log1p(y_train) if transform == "log1p" else y_train
        folds = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
        oof = cross_val_predict(clone(estimator), x_train, target, cv=folds)
        oof = np.expm1(oof) if transform == "log1p" else oof
        half_width = conformal_half_width(np.abs(y_train - oof))
        bundle = {
            "model": clone(estimator).fit(x_train, target),
            "features": list(common.AGE_FEATURES),
            "target_transform": transform,
            "interval": {"method": "cross-conformal", "level": INTERVAL_LEVEL, "half_width_rings": round(half_width, 4)},
            "trained_on": trained_on,
            "sklearn_version": sklearn.__version__,
        }
        save_model(configs[tool_id]["model_path"], bundle)
        predicted = predict_rings(bundle, x_test)
        half = bundle["interval"]["half_width_rings"]
        results[tool_id] = {
            "family": "ml",
            "algorithm": algorithm,
            "features": list(common.AGE_FEATURES),
            "trained_on": trained_on,
            "interval": bundle["interval"],
            "holdout_metrics": age_holdout(y_train, y_test, predicted, predicted - half, predicted + half),
        }
    return results


def top_features(model: Any, features: list[str], k: int = 3) -> list[str]:
    estimator = base_estimator(model)
    final = estimator[-1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "feature_importances_"):
        weights = final.feature_importances_
    else:  # standardized coefficients: their size across classes
        weights = np.linalg.norm(final.coef_, axis=0)
    return [features[i] for i in np.argsort(weights)[::-1][:k]]


def conformal_half_width(abs_residuals: np.ndarray, level: float = INTERVAL_LEVEL) -> float:
    """Split-conformal quantile with the finite-sample correction ceil((n + 1) * level) / n."""
    n = len(abs_residuals)
    return float(np.quantile(abs_residuals, min(1.0, math.ceil((n + 1) * level) / n), method="higher"))


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, classes: list[str], bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    confidence = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


# ---------------------------------------------------------------------------
# Expert tools
# ---------------------------------------------------------------------------

def evaluate_maturity_rules(train: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any], trained_on: str) -> dict[str, Any]:
    rules, default = config["rules"], config["default"]
    validate_rules(rules, default)
    train_rows = train.to_dict("records")
    truth_train = np.where(train[SEX] == "I", "I", "ADULT")
    stats, fired_any = {}, np.zeros(len(train_rows), dtype=bool)
    for rule in rules:
        fires = np.array([rule_fires(rule, row) for row in train_rows])
        fired_any |= fires
        stats[rule["id"]] = {
            "train_coverage": rate(fires),
            "train_precision": rate(truth_train[fires] == rule["verdict"]) if fires.any() else None,
        }
    stats[default["id"]] = {
        "train_coverage": rate(~fired_any),
        "train_precision": rate(truth_train[~fired_any] == default["verdict"]) if (~fired_any).any() else None,
    }

    truth_test = np.where(test[SEX] == "I", "I", "ADULT")
    verdicts = np.array([maturity_verdict(rules, default, row)[0] for row in test.to_dict("records")])
    infant_pred, infant_true = verdicts == "I", truth_test == "I"
    adults = train[train[SEX] != "I"][SEX]
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(train[["shell_weight", RINGS]], truth_train)
    return {
        "abalone_sex_maturity_rules": {
            "family": "expert",
            "algorithm": "Expert maturity rules on shell size and ring count (first rule that fires decides); abstains on male vs female",
            "trained_on": trained_on,
            "rules_fingerprint": rules_fingerprint(rules, default),
            "rule_stats": stats,
            "adult_base_rates": {sex: rate(adults == sex) for sex in ("M", "F")},
            "reference_tree": export_text(tree, feature_names=["shell_weight", RINGS]),
            "holdout_metrics": {
                "infant_vs_adult_accuracy": rate(verdicts == truth_test),
                "infant_precision": rate(infant_true[infant_pred]),
                "infant_recall": rate(infant_pred[infant_true]),
                "n_test": int(len(test)),
            },
        }
    }


def fit_bivariate_profile(train: pd.DataFrame) -> dict[str, Any]:
    classes = {}
    for sex in SEX_CLASSES:
        subset = train[train[SEX] == sex][PROFILE_FEATURES].to_numpy(dtype=float)
        classes[sex] = {
            "mean": subset.mean(axis=0).tolist(),
            "cov": np.cov(subset.T).tolist(),
            "prior": len(subset) / len(train),
            "n": int(len(subset)),
        }
    return {"features": PROFILE_FEATURES, "classes": classes}


def evaluate_bivariate_profile(train, test, params, config, trained_on) -> dict[str, Any]:
    scored = [profile_scores(params, row) for row in test.to_dict("records")]
    classes = scored[0][0]
    proba = np.vstack([p for _, p, _ in scored])
    predicted = np.asarray(classes)[proba.argmax(axis=1)]
    return {
        "abalone_sex_bivariate_profile": {
            "family": "expert",
            "algorithm": "Bivariate normal class profiles of (diameter, number_of_rings) combined with class priors by Bayes' rule",
            "features": PROFILE_FEATURES,
            "trained_on": trained_on,
            "holdout_metrics": sex_holdout(
                train[SEX].to_numpy(), test[SEX].to_numpy(), predicted, proba, classes, config["mf_ambiguity_margin"]
            ),
        }
    }


def fit_allometric_formula(train: pd.DataFrame) -> dict[str, Any]:
    groups = {sex: fit_power_law(train[train[SEX] == sex]) for sex in SEX_CLASSES}
    groups[UNKNOWN_SEX] = fit_power_law(train)
    return {"feature": "shell_weight", "groups": groups}


def fit_power_law(subset: pd.DataFrame) -> dict[str, Any]:
    log_weight, log_rings = np.log(subset["shell_weight"].to_numpy()), np.log(subset[RINGS].to_numpy(dtype=float))
    b, log_a = np.polyfit(log_weight, log_rings, 1)
    residuals = log_rings - (log_a + b * log_weight)
    return {"a": float(np.exp(log_a)), "b": float(b), "sigma_log": float(residuals.std(ddof=2)), "n": int(len(subset))}


def evaluate_allometric_formula(train, test, params, trained_on) -> dict[str, Any]:
    outputs = np.array([allometric_predict(params, sex, weight)[:3] for sex, weight in zip(test[SEX], test["shell_weight"])])
    return {
        "abalone_age_allometric_formula": {
            "family": "expert",
            "algorithm": "Allometric power law rings = a * shell_weight^b fitted per sex by least squares on log-log scale",
            "trained_on": trained_on,
            "holdout_metrics": age_holdout(
                train[RINGS].to_numpy(dtype=float), test[RINGS].to_numpy(dtype=float), outputs[:, 0], outputs[:, 1], outputs[:, 2]
            ),
        }
    }


def fit_shell_weight_lookup(train: pd.DataFrame) -> dict[str, Any]:
    quantiles = np.quantile(train["shell_weight"], np.linspace(0, 1, LOOKUP_BINS + 1))[1:-1]
    edges = [round(float(edge), 6) for edge in quantiles]
    bins = np.array([lookup_bin(edges, weight) for weight in train["shell_weight"]])
    rings = train[RINGS].to_numpy(dtype=float)
    cells = {}
    for group in (*SEX_CLASSES, UNKNOWN_SEX):
        in_group = np.ones(len(train), dtype=bool) if group == UNKNOWN_SEX else (train[SEX] == group).to_numpy()
        cells[group] = [cell_stats(rings[in_group & (bins == k)], k) for k in range(LOOKUP_BINS)]
    return {"feature": "shell_weight", "edges": edges, "cells": cells}


def cell_stats(values: np.ndarray, position: int) -> dict[str, Any]:
    if len(values) == 0:
        return {"bin": position, "n": 0, "median": None, "p10": None, "p90": None}
    return {
        "bin": position,
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def evaluate_shell_weight_lookup(train, test, params, config, trained_on) -> dict[str, Any]:
    cells = [lookup_cell(params, sex, weight)[0] for sex, weight in zip(test[SEX], test["shell_weight"])]
    sparse = sum(1 for sex in SEX_CLASSES for cell in params["cells"][sex] if cell["n"] < config["min_cell_n"])
    return {
        "abalone_age_shell_weight_lookup": {
            "family": "expert",
            "algorithm": "Empirical table: shell_weight decile x sex -> median ring count and P10-P90 of training specimens",
            "trained_on": trained_on,
            "sparse_cells": sparse,
            "holdout_metrics": age_holdout(
                train[RINGS].to_numpy(dtype=float),
                test[RINGS].to_numpy(dtype=float),
                np.array([cell["median"] for cell in cells]),
                np.array([cell["p10"] for cell in cells]),
                np.array([cell["p90"] for cell in cells]),
            ),
        }
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sex_holdout(y_train, y_true, y_pred, proba, classes, mf_margin) -> dict[str, Any]:
    """Hold-out metrics for a sex tool. Baselines answer the most common (adult) class of the train split."""
    y_train, y_true, y_pred = np.asarray(y_train), np.asarray(y_true), np.asarray(y_pred)
    adult_true, adult_pred = y_true != "I", y_pred != "I"
    both_adult = adult_true & adult_pred
    p_male, p_female = proba[:, classes.index("M")], proba[:, classes.index("F")]
    majority = pd.Series(y_train).value_counts().idxmax()
    adult_majority = pd.Series(y_train[y_train != "I"]).value_counts().idxmax()
    return {
        "accuracy": rate(y_pred == y_true),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "infant_vs_adult_accuracy": rate(adult_true == adult_pred),
        "majority_baseline": rate(y_true == majority),
        "mf_accuracy_among_adults": rate(y_pred[both_adult] == y_true[both_adult]),
        "mf_majority_baseline": rate(y_true[adult_true] == adult_majority),
        "mf_ambiguous_rate": rate(np.abs(p_male - p_female)[adult_pred] < mf_margin),
        "expected_calibration_error": round(expected_calibration_error(proba, y_true, classes), 4),
        "per_class_recall": {sex: rate(y_pred[y_true == sex] == sex) for sex in SEX_CLASSES},
        "n_test": int(len(y_true)),
    }


def age_holdout(y_train, y_true, y_pred, low, high) -> dict[str, Any]:
    """Hold-out metrics for an age tool; the interval is clipped at 1 ring exactly as the tools report it."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    low, high = np.maximum(np.asarray(low, dtype=float), 1.0), np.asarray(high, dtype=float)
    errors = np.abs(y_true - y_pred)
    groups = np.array([common.age_group(value) for value in y_true])
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "within_2_rings": rate(errors <= 2),
        "interval_coverage": rate((y_true >= low) & (y_true <= high)),
        "interval_mean_width": round(float(np.mean(high - low)), 4),
        "median_baseline_mae": round(float(np.mean(np.abs(y_true - np.median(y_train)))), 4),
        "mae_by_age_group": {label: round(float(errors[groups == label].mean()), 4) for _, label in common.AGE_GROUPS if (groups == label).any()},
        "n_test": int(len(y_true)),
    }


def rate(values: Any) -> float:
    return round(float(np.mean(np.asarray(values, dtype=float))), 4)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_model(relative: str, bundle: dict[str, Any]) -> None:
    path = common.REPO_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def print_summary(metrics: dict[str, Any]) -> None:
    print(f"\n{'tool':34s} {'family':7s} hold-out metrics")
    for tool_id, entry in metrics["tools"].items():
        m = entry["holdout_metrics"]
        if "mae" in m:
            text = f"MAE {m['mae']:.3f} | RMSE {m['rmse']:.3f} | R2 {m['r2']:.3f} | <=2 rings {m['within_2_rings']:.3f} | 80% coverage {m['interval_coverage']:.3f}"
        elif "accuracy" in m:
            text = f"acc {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f} | infant-vs-adult {m['infant_vs_adult_accuracy']:.3f} | M/F among adults {m['mf_accuracy_among_adults']:.3f}"
        else:
            text = f"infant-vs-adult {m['infant_vs_adult_accuracy']:.3f} | infant precision {m['infant_precision']:.3f} | infant recall {m['infant_recall']:.3f}"
        print(f"{tool_id:34s} {entry['family']:7s} {text}")
    print(f"\nWrote {metrics['split']['path']}, artifacts/models/abalone/*.joblib, expert_params.json and metrics.json")


if __name__ == "__main__":
    raise SystemExit(main())
