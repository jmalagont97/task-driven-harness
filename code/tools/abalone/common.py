"""common.py

Shared utilities for the Abalone tools (declarations/tools/abalone_*.json).

Responsibilities:
- Locate the dataset, the frozen train/test split and the trained artifacts.
- Validate tool arguments (units, ranges, required fields) with messages an LLM can act on.
- Build the feature frames shared by training and inference, so both use the same columns in the same order.
- Wrap every executor so it returns the repository's uniform envelope and never raises.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "train" / "abalone_dataset.csv"
SPLIT_PATH = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "splits.json"
TRAIN_COMMAND = ".venv/bin/python code/tools/abalone/train_abalone_tools.py"

SEED = 42
TEST_SIZE = 0.2

MEASUREMENTS = ("length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight")
LENGTHS = ("length", "diameter", "height")
RINGS = "number_of_rings"
SEX = "sex"
SEX_CLASSES = ("M", "F", "I")
UNKNOWN_SEX = "unknown"
# Ratio feature -> numerator column; the denominator is always whole_weight.
RATIOS = {"shell_ratio": "shell_weight", "shucked_ratio": "shucked_weight", "viscera_ratio": "viscera_weight"}
# Column order is part of the model contract: training and inference must build the same frame.
SEX_FEATURES = [*MEASUREMENTS, RINGS, *RATIOS]
AGE_FEATURES = [*MEASUREMENTS, "sex_F", "sex_I", "sex_M", *RATIOS]

# Accepted ranges in dataset units (original mm or g divided by 200). Keep in sync with every input_schema.
MEASUREMENT_LIMITS = {
    "length": (0.0005, 1.0),
    "diameter": (0.0005, 1.0),
    "height": (0.0005, 1.5),
    "whole_weight": (0.0005, 4.0),
    "shucked_weight": (0.0005, 2.0),
    "viscera_weight": (0.0005, 1.0),
    "shell_weight": (0.0005, 1.5),
}
RINGS_LIMITS = (1, 40)
UNIT_EXAMPLES = {"millimetres": "95 mm -> 0.475", "grams": "135.4 g -> 0.677"}

RINGS_TO_YEARS = 1.5
# Upper ring bound (inclusive) -> label; same bins as the EDA notebook.
AGE_GROUPS = ((7.0, "Young (1-7)"), (10.0, "Young Adult (8-10)"), (15.0, "Adult (11-15)"), (math.inf, "Old (16+)"))
WEIGHT_PARTS = ("shucked_weight", "viscera_weight", "shell_weight")
INCONSISTENT_WEIGHT_FACTOR = 1.1


class InputError(ValueError):
    """The arguments break the tool's input contract; the message says how to fix the call."""


class ArtifactMissing(FileNotFoundError):
    """A trained artifact is missing or out of date; the message says how to rebuild it."""


@dataclass(frozen=True)
class ToolSpec:
    """Identity stamped on every envelope. Must match the tool's declaration (checked by the tests)."""

    tool_id: str
    version: str
    disclaimer: str | None


# A computation returns (result, flags, warnings) or raises InputError / ArtifactMissing.
Computation = Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], list[str], list[str]]]


def run_tool(spec: ToolSpec, arguments: Any, config: dict[str, Any] | None, compute: Computation) -> dict[str, Any]:
    """Run a computation and always return the uniform envelope; never raises."""
    try:
        result, flags, warnings = compute(arguments, config or {})
    except InputError as exc:
        return error_envelope(spec, "input_error", str(exc))
    except ArtifactMissing as exc:
        return error_envelope(spec, "artifact_missing", str(exc))
    except Exception as exc:  # noqa: BLE001 - an executor must never crash the orchestrator
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return error_envelope(spec, "runtime_error", f"{detail}\n{traceback.format_exc(limit=4)}")
    return {
        "ok": True,
        "tool_id": spec.tool_id,
        "version": spec.version,
        "result": result,
        "flags": flags,
        "warnings": warnings,
        "disclaimer": spec.disclaimer,
    }


def error_envelope(spec: ToolSpec, error_type: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool_id": spec.tool_id,
        "version": spec.version,
        "flags": [],
        "warnings": [],
        "error": {"type": error_type, "message": message},
        "disclaimer": spec.disclaimer,
    }


def union_flags(predictions: Iterable[dict[str, Any]], *extra: str) -> list[str]:
    """Top-level flags: every per-specimen flag plus tool-level ones, sorted for deterministic output."""
    found = {flag for prediction in predictions for flag in prediction.get("flags", [])}
    return sorted(found.union(extra))


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def parse_specimens(
    arguments: Any,
    *,
    target: str,
    sex_values: tuple[str, ...] = SEX_CLASSES,
    max_specimens: int = 50,
) -> list[dict[str, Any]]:
    """Validate {"specimens": [...]} and return clean rows.

    `target` is what the tool predicts: "sex" (specimens carry number_of_rings) or "age" (specimens carry sex).
    """
    if not isinstance(arguments, dict):
        raise InputError('Arguments must be an object of the form {"specimens": [...]}.')
    unexpected = sorted(set(arguments) - {"specimens"})
    if unexpected:
        raise InputError(f"Unexpected argument(s) {unexpected}; the only argument is 'specimens'.")
    specimens = arguments.get("specimens")
    if not isinstance(specimens, list) or not specimens:
        raise InputError("'specimens' must be a non-empty list of specimen objects.")
    if len(specimens) > max_specimens:
        raise InputError(
            f"'specimens' has {len(specimens)} items; the limit is {max_specimens} per call. Split the request into batches."
        )

    known_field, target_field = (RINGS, SEX) if target == "sex" else (SEX, RINGS)
    required = [*MEASUREMENTS, known_field]
    rows = []
    for i, specimen in enumerate(specimens):
        where = f"specimens[{i}]"
        if not isinstance(specimen, dict):
            raise InputError(f"{where} must be an object, got {type(specimen).__name__}.")
        missing = [name for name in required if name not in specimen]
        if missing:
            raise InputError(f"{where} is missing {missing}. Every specimen needs {required}.")
        unexpected = sorted(set(specimen) - set(required))
        if unexpected:
            hint = f" '{target_field}' is what this tool predicts, so leave it out." if target_field in unexpected else ""
            raise InputError(f"{where} has unexpected field(s) {unexpected}.{hint} Expected exactly {required}.")
        row: dict[str, Any] = {name: _measurement(specimen[name], f"{where}.{name}", name) for name in MEASUREMENTS}
        if target == "sex":
            row[RINGS] = _rings(specimen[RINGS], f"{where}.{RINGS}")
        else:
            row[SEX] = _sex(specimen[SEX], f"{where}.{SEX}", sex_values)
        rows.append(row)
    return rows


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be a number, got {type(value).__name__} {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise InputError(f"{where} must be a finite number, got {value!r}.")
    return value


def _measurement(value: Any, where: str, name: str) -> float:
    value = _number(value, where)
    low, high = MEASUREMENT_LIMITS[name]
    if value > high:
        unit = "millimetres" if name in LENGTHS else "grams"
        raise InputError(
            f"{where} = {value:g} exceeds {high:g}. Values look like {unit}; divide by 200 to use dataset units "
            f"(e.g. {UNIT_EXAMPLES[unit]})."
        )
    if value < low:
        hint = " A height of 0 is a measurement error in the source data." if name == "height" else ""
        raise InputError(f"{where} = {value:g} is below the minimum {low:g}; measurements must be positive.{hint}")
    return value


def _rings(value: Any, where: str) -> int:
    value = _number(value, where)
    if not value.is_integer():
        raise InputError(f"{where} must be a whole number of rings, got {value:g}.")
    low, high = RINGS_LIMITS
    if not low <= value <= high:
        raise InputError(f"{where} = {value:g} is outside the accepted range {low}-{high} rings.")
    return int(value)


def _sex(value: Any, where: str, allowed: tuple[str, ...]) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    hint = ""
    if value == UNKNOWN_SEX:
        hint = (
            " This model needs a known sex: call a sex tool first, or use abalone_age_allometric_formula or"
            " abalone_age_shell_weight_lookup, which accept 'unknown'."
        )
    raise InputError(f"{where} must be one of {list(allowed)}, got {value!r}.{hint}")


def soft_flags(row: dict[str, Any], train_ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Warnings that do not reject the input: odd weights, or values in the extreme 1 % of the training data."""
    flags = []
    if sum(row[name] for name in WEIGHT_PARTS) > INCONSISTENT_WEIGHT_FACTOR * row["whole_weight"]:
        flags.append("inconsistent_weights")
    if any(not low <= row[name] <= high for name, (low, high) in train_ranges.items() if name in row):
        flags.append("out_of_distribution")
    return flags


# ---------------------------------------------------------------------------
# Features and shared result fields
# ---------------------------------------------------------------------------

def sex_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for the sex task: 7 measurements, ring count and 3 weight ratios."""
    features = frame[[*MEASUREMENTS, RINGS]].copy()
    for name, numerator in RATIOS.items():
        features[name] = frame[numerator] / frame["whole_weight"]
    return features[SEX_FEATURES]


def age_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for the age task: 7 measurements, one-hot sex and 3 weight ratios."""
    features = frame[list(MEASUREMENTS)].copy()
    for sex in ("F", "I", "M"):
        features[f"sex_{sex}"] = (frame[SEX] == sex).astype(float)
    for name, numerator in RATIOS.items():
        features[name] = frame[numerator] / frame["whole_weight"]
    return features[AGE_FEATURES]


def sex_call(
    classes: list[str],
    proba: Iterable[float],
    *,
    mf_ambiguity_margin: float,
    maturity_uncertainty_band: float,
) -> tuple[dict[str, Any], list[str]]:
    """Two-level reading of a probability vector: infant vs adult first, then male vs female."""
    proba = np.asarray(list(proba), dtype=float)
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    predicted = classes[int(np.argmax(proba))]
    p_adult = probabilities["M"] + probabilities["F"]
    mf_margin = abs(probabilities["M"] - probabilities["F"])
    flags = []
    if predicted != "I" and mf_margin < mf_ambiguity_margin:
        flags.append("mf_ambiguous")
    if abs(p_adult - 0.5) < maturity_uncertainty_band:
        flags.append("maturity_uncertain")
    call = {
        "predicted_sex": predicted,
        "probabilities": {cls: round(probabilities[cls], 4) for cls in SEX_CLASSES},
        "p_adult": round(p_adult, 4),
        "mf_margin": round(mf_margin, 4),
    }
    return call, flags


def age_group(rings: float) -> str:
    for upper, label in AGE_GROUPS:
        if rings <= upper:
            return label
    return AGE_GROUPS[-1][1]


def age_record(
    index: int,
    rings: float,
    low: float,
    high: float,
    method: str,
    *,
    tail_rings_above: float,
    flags: list[str],
) -> dict[str, Any]:
    """Fields every age tool returns for one specimen. The ring count is never below 1."""
    return {
        "index": index,
        "predicted_rings": round(float(rings), 2),
        "predicted_age_years": round(float(rings) + RINGS_TO_YEARS, 2),
        "interval_80": {"low": round(max(float(low), 1.0), 2), "high": round(float(high), 2), "method": method},
        "age_group": age_group(rings),
        "flags": [*flags, "tail_region"] if rings > tail_rings_above else list(flags),
    }


def top_contributions(features: list[str], raw_values: Iterable[float], contributions: Iterable[float], k: int = 3) -> list[dict[str, Any]]:
    """The k largest signed contributions (coefficient x standardized value), largest magnitude first."""
    items = sorted(zip(features, raw_values, contributions), key=lambda item: -abs(item[2]))[:k]
    return [{"feature": name, "value": round(float(value), 4), "contribution": round(float(c), 4)} for name, value, c in items]


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def artifact_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    if REPO_ROOT not in path.parents:
        raise ValueError(f"Artifact path {relative!r} must stay inside the repository.")
    if not path.is_file():
        raise ArtifactMissing(f"Artifact not found: {relative}. Train the tools first: {TRAIN_COMMAND}")
    return path


def load_json_artifact(relative: str) -> dict[str, Any]:
    path = artifact_path(relative)
    return _read_json(str(path), path.stat().st_mtime_ns)


def load_model_artifact(relative: str) -> dict[str, Any]:
    """Load a model bundle written by train_abalone_tools.py. Never point this at files from elsewhere:
    joblib files can execute code when loaded."""
    path = artifact_path(relative)
    return _read_joblib(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _read_json(path: str, _mtime_ns: int) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _read_joblib(path: str, _mtime_ns: int) -> dict[str, Any]:
    import joblib

    return joblib.load(path)


def tool_metrics(config: dict[str, Any], tool_id: str) -> dict[str, Any]:
    metrics = load_json_artifact(config["metrics_path"])
    try:
        return metrics["tools"][tool_id]
    except KeyError:
        raise ArtifactMissing(f"{config['metrics_path']} has no entry for {tool_id}. Re-run: {TRAIN_COMMAND}") from None


def train_ranges(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    metrics = load_json_artifact(config["metrics_path"])
    return {name: (low, high) for name, (low, high) in metrics["train_profile"]["ranges"].items()}


# ---------------------------------------------------------------------------
# Dataset and frozen split
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactMissing(f"Dataset not found at {path}. See artifacts/datasets/abalone_dataset/abalone.py.")
    return pd.read_csv(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_split(frame: pd.DataFrame, dataset_sha256: str) -> dict[str, Any]:
    """Stratified 80/20 split by sex with seed 42, after dropping rows with a physically impossible height."""
    from sklearn.model_selection import train_test_split

    excluded = frame.index[frame["height"] <= 0]
    clean = frame.drop(index=excluded)
    train, test = train_test_split(clean, test_size=TEST_SIZE, random_state=SEED, stratify=clean[SEX])
    return {
        "schema_version": "1.0.0",
        "dataset_path": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "dataset_sha256": dataset_sha256,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "stratify": SEX,
        "note": (
            "Indices are 0-based CSV row positions (header excluded), stored in the order train_test_split returned "
            "them. Keep that order: the Random Forest bootstrap depends on it."
        ),
        "excluded_rows": [{"index": int(i), "reason": "height <= 0 (physically impossible)"} for i in excluded],
        "train_indices": [int(i) for i in train.index],
        "test_indices": [int(i) for i in test.index],
    }


def load_split(path: Path = SPLIT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactMissing(f"Split not found at {path}. Create it with: {TRAIN_COMMAND}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_fingerprint(split: dict[str, Any]) -> str:
    payload = json.dumps({"train": split["train_indices"], "test": split["test_indices"]}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_frames(frame: pd.DataFrame, split: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test rows, in the stored order."""
    return frame.loc[split["train_indices"]], frame.loc[split["test_indices"]]
