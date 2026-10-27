"""final_report.py

Final analysis of a finished debate evaluation (evals/abalone_debate/run_debate_eval.py).

It reads <output>/results.jsonl (held-out test rows only) and every debate's debate_state.json, re-runs the eight tools
on the same rows (they are deterministic) and writes <output>/final_report.json and final_report.md with:
- sex: 3-class accuracy and macro-F1 of the debate's forced_choice, of a hybrid class (the debate's maturity plus the
  tools' soft vote between M and F for adults), of the tools-only soft vote and of each tool alone; infant vs adult
  accuracy of each; how often the debate commits to M or F and how accurate it is when it does;
- age: MAE, RMSE, R², RMSLE, share within 1 and 2 rings and 80%-range coverage of the debate, the baseline and each
  tool alone, overall and by ring group;
- paired tests: exact McNemar for sex, bootstrap 95% intervals of MAE differences for age;
- calibration: accuracy (sex) and MAE and coverage (age) by the reliability label the arbiter gave;
- debate value: how often the final answer differs from the tool agents' round-1 positions and whether that helped;
- debate dynamics: agreement, rounds, challenge outcomes, expert abstentions and time.

Run from the repository root:
    .venv/bin/python evals/abalone_debate/final_report.py --output artifacts/runs/evals/debate_full
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from runtime.problem_statement import problems_from_csv  # noqa: E402

DATASET = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "train" / "abalone_dataset.csv"
SEX_ML = ("abalone_sex_random_forest", "abalone_sex_logistic_regression", "abalone_sex_bivariate_profile")
SEX_TOOLS = (*SEX_ML, "abalone_sex_maturity_rules")
AGE_TOOLS = ("abalone_age_gradient_boosting", "abalone_age_ridge_regression", "abalone_age_allometric_formula", "abalone_age_shell_weight_lookup")
RING_GROUPS = (("1-7", 1, 7), ("8-10", 8, 10), ("11-15", 11, 15), ("16+", 16, 99))
SEED = 42


def tool_predictions(task: str, rows: list[int]) -> dict[str, dict[int, dict]]:
    tools = tool_registry.load_tool_declarations(server="abalone_tools")
    problems = problems_from_csv(DATASET, task, rows)
    names = SEX_TOOLS if task == "sex" else AGE_TOOLS
    out: dict[str, dict[int, dict]] = {name: {} for name in names}
    for start in range(0, len(problems), 50):
        batch = problems[start : start + 50]
        specimens = [p.specimen() for p in batch]
        for name in names:
            predictions = tool_registry.invoke(tools[name], {"specimens": specimens})["result"]["predictions"]
            for problem, prediction in zip(batch, predictions):
                out[name][problem.row_number] = prediction
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification(truth: list[str], predicted: list[str | None]) -> dict:
    accuracy = sum(p == t for p, t in zip(predicted, truth)) / len(truth)
    f1s = []
    for cls in ("M", "F", "I"):
        tp = sum(p == cls and t == cls for p, t in zip(predicted, truth))
        fp = sum(p == cls and t != cls for p, t in zip(predicted, truth))
        fn = sum(p != cls and t == cls for p, t in zip(predicted, truth))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"accuracy_3_class": round(accuracy, 4), "macro_f1": round(sum(f1s) / 3, 4)}


def maturity_accuracy(truth: list[str], maturity: list[str | None]) -> float:
    return round(sum((m == "infant") == (t == "I") and m in ("infant", "adult") for m, t in zip(maturity, truth)) / len(truth), 4)


def regression(truth: list[float], predicted: list[float], low: list[float] | None = None, high: list[float] | None = None) -> dict:
    t, p = np.asarray(truth, float), np.asarray(predicted, float)
    errors = p - t
    metrics = {
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "r2": round(float(1 - np.sum(errors ** 2) / np.sum((t - t.mean()) ** 2)), 4),
        "rmsle": round(float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(t)) ** 2))), 4),
        "within_1_ring": round(float(np.mean(np.abs(errors) <= 1)), 4),
        "within_2_rings": round(float(np.mean(np.abs(errors) <= 2)), 4),
    }
    if low is not None and high is not None:
        metrics["range_coverage"] = round(float(np.mean((np.asarray(low) <= t) & (t <= np.asarray(high)))), 4)
        metrics["mean_range_width"] = round(float(np.mean(np.asarray(high) - np.asarray(low))), 3)
    return metrics


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))
    c = sum(y and not x for x, y in zip(a_correct, b_correct))
    n = b + c
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n if n else 1.0
    return {"only_first_right": b, "only_second_right": c, "p_value": round(min(1.0, 2 * tail), 5)}


def bootstrap_mae_difference(errors_a: np.ndarray, errors_b: np.ndarray, resamples: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    diff = np.abs(errors_a) - np.abs(errors_b)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"mae_difference": round(float(diff.mean()), 4), "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def load(output: Path) -> list[dict]:
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("completed") and r["split"] == "test"]
    for record in rows:
        state_path = REPO_ROOT / record["run"] / "debate_state.json"
        record["state"] = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    return rows


def sex_section(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["row"])
    truth = [r["truth"] for r in rows]
    preds = tool_predictions("sex", [r["row"] for r in rows])
    soft, hybrid = [], []
    for record in rows:
        totals = Counter()
        for name in SEX_ML:
            totals.update(preds[name][record["row"]]["probabilities"])
        soft.append(max(("M", "F", "I"), key=lambda cls: totals[cls]))
        maturity = record["debate"].get("maturity")
        if maturity == "infant":
            hybrid.append("I")
        elif maturity == "adult":
            hybrid.append("M" if totals["M"] >= totals["F"] else "F")
        else:
            hybrid.append(soft[-1])
    debate_forced = [r["debate"].get("forced_choice") for r in rows]
    debate_maturity = [r["debate"].get("maturity") for r in rows]
    methods = {
        "debate_forced_choice": (debate_forced, debate_maturity),
        "debate_hybrid": (hybrid, debate_maturity),
        "tools_soft_vote": (soft, ["infant" if s == "I" else "adult" for s in soft]),
    }
    for name in SEX_ML:
        classes = [preds[name][r["row"]]["predicted_sex"] for r in rows]
        methods[name] = (classes, ["infant" if c == "I" else "adult" for c in classes])
    rules = ["infant" if preds["abalone_sex_maturity_rules"][r["row"]]["verdict"] == "I" else "adult" for r in rows]
    table = {name: {**classification(truth, classes), "infant_vs_adult_accuracy": maturity_accuracy(truth, maturity)}
             for name, (classes, maturity) in methods.items()}
    table["abalone_sex_maturity_rules"] = {"infant_vs_adult_accuracy": maturity_accuracy(truth, rules)}
    committed = [(r["debate"].get("sex_call"), t) for r, t in zip(rows, truth) if r["debate"].get("sex_call") in ("M", "F")]
    reliability = {}
    for label in ("high", "medium", "low"):
        subset = [(m, t) for r, m, t in zip(rows, debate_maturity, truth) if r.get("reliability") == label]
        if subset:
            reliability[label] = {"n": len(subset), "infant_vs_adult_accuracy": maturity_accuracy([t for _, t in subset], [m for m, _ in subset])}
    correct = lambda classes: [c == t for c, t in zip(classes, truth)]  # noqa: E731
    mature_correct = lambda maturity: [(m == "infant") == (t == "I") and m in ("infant", "adult") for m, t in zip(maturity, truth)]  # noqa: E731
    return {
        "n": len(rows),
        "class_counts": dict(Counter(truth)),
        "methods": table,
        "debate_sex_call_counts": dict(Counter(r["debate"].get("sex_call") for r in rows)),
        "debate_commits_to_m_or_f": {"n": len(committed), "accuracy": round(sum(c == t for c, t in committed) / len(committed), 4) if committed else None},
        "tests": {
            "hybrid_vs_soft_vote_3_class": mcnemar(correct(hybrid), correct(soft)),
            "forced_vs_soft_vote_3_class": mcnemar(correct(debate_forced), correct(soft)),
            "debate_vs_soft_vote_maturity": mcnemar(mature_correct(debate_maturity), mature_correct(["infant" if s == "I" else "adult" for s in soft])),
        },
        "by_reliability": reliability,
    }


def age_section(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["row"])
    truth = np.asarray([float(r["truth"]) for r in rows])
    preds = tool_predictions("age", [r["row"] for r in rows])
    debate = np.asarray([float(r["debate"]["rings"]) for r in rows])
    base = np.asarray([float(r["baseline"]["rings"]) for r in rows])
    methods = {
        "debate": regression(truth, debate, [r["debate"]["rings_low"] for r in rows], [r["debate"]["rings_high"] for r in rows]),
        "tools_median": regression(truth, base, [r["baseline"]["rings_low"] for r in rows], [r["baseline"]["rings_high"] for r in rows]),
    }
    tool_arrays = {}
    for name in AGE_TOOLS:
        values = [preds[name][r["row"]] for r in rows]
        tool_arrays[name] = np.asarray([v["predicted_rings"] for v in values], float)
        methods[name] = regression(truth, tool_arrays[name], [v["interval_80"]["low"] for v in values], [v["interval_80"]["high"] for v in values])
    groups = {}
    for label, low, high in RING_GROUPS:
        mask = (truth >= low) & (truth <= high)
        if mask.any():
            groups[label] = {"n": int(mask.sum()),
                             "debate_mae": round(float(np.mean(np.abs(debate[mask] - truth[mask]))), 3),
                             "tools_median_mae": round(float(np.mean(np.abs(base[mask] - truth[mask]))), 3),
                             "gradient_boosting_mae": round(float(np.mean(np.abs(tool_arrays["abalone_age_gradient_boosting"][mask] - truth[mask]))), 3),
                             "debate_bias": round(float(np.mean(debate[mask] - truth[mask])), 3)}
    reliability = {}
    for label in ("high", "medium", "low"):
        idx = [i for i, r in enumerate(rows) if r.get("reliability") == label]
        if idx:
            covered = [rows[i]["debate"]["rings_low"] <= truth[i] <= rows[i]["debate"]["rings_high"] for i in idx]
            reliability[label] = {"n": len(idx), "mae": round(float(np.mean(np.abs(debate[idx] - truth[idx]))), 3), "range_coverage": round(sum(covered) / len(idx), 3)}
    return {
        "n": len(rows),
        "methods": methods,
        "by_ring_group": groups,
        "tests": {
            "debate_minus_tools_median": bootstrap_mae_difference(debate - truth, base - truth),
            "debate_minus_gradient_boosting": bootstrap_mae_difference(debate - truth, tool_arrays["abalone_age_gradient_boosting"] - truth),
        },
        "by_reliability": reliability,
    }


def debate_value(rows: list[dict]) -> dict:
    """Did the debate change the round-1 answer of the tool agents, and did that help?"""
    sex_changed = sex_helped = sex_hurt = 0
    age_changed = age_helped = age_hurt = 0
    for record in rows:
        positions = record["state"].get("positions") or {}
        first = [p[0] for p in positions.values() if isinstance(p, list) and p]
        if len(first) < 2:
            continue
        if record["task"] == "sex":
            before = {p.get("maturity") for p in first}
            if len(before) == 1:
                start = before.pop()
                end = record["debate"].get("maturity")
                if end != start:
                    sex_changed += 1
                    right = lambda m: (m == "infant") == (record["truth"] == "I") and m in ("infant", "adult")  # noqa: E731
                    sex_helped += right(end) and not right(start)
                    sex_hurt += right(start) and not right(end)
        else:
            start = statistics.mean(float(p["rings"]) for p in first if p.get("rings") is not None)
            end = float(record["debate"]["rings"])
            if abs(end - start) >= 0.5:
                age_changed += 1
                age_helped += abs(end - record["truth"]) < abs(start - record["truth"])
                age_hurt += abs(end - record["truth"]) > abs(start - record["truth"])
    return {"sex_maturity_changed_after_round_1": sex_changed, "sex_change_helped": sex_helped, "sex_change_hurt": sex_hurt,
            "age_estimate_moved_half_ring_or_more": age_changed, "age_move_helped": age_helped, "age_move_hurt": age_hurt}


def dynamics(rows: list[dict]) -> dict:
    n = len(rows)
    challenges = sum((Counter(r["challenges"]) for r in rows), Counter())
    seconds = [r["elapsed_s"] for r in rows]
    return {
        "debates": n,
        "tool_agents_agree_at_end": round(sum(bool(r["converged"]) for r in rows) / n, 4),
        "rounds": dict(Counter(r["rounds"] for r in rows)),
        "challenges": dict(challenges),
        "challenges_per_debate": round(sum(challenges.values()) / n, 2),
        "conceded_share": round(challenges.get("conceded", 0) / max(1, sum(challenges.values())), 4),
        "abstention_rate_per_expert": {code: round(count / n, 4) for code, count in Counter(c for r in rows for c in r["abstained"]).items()},
        "outcomes": dict(Counter(r["debate_outcome"] for r in rows)),
        "reliability": dict(Counter(r["reliability"] for r in rows)),
        "arbiter_fallbacks": sum(bool(r.get("arbiter_fallback")) for r in rows),
        "seconds_per_debate": {"mean": round(statistics.mean(seconds), 1), "median": round(statistics.median(seconds), 1)},
        "runtime_versions": sorted({r.get("runtime_version") for r in rows}),
    }


def markdown(report: dict) -> str:
    sex, age = report["sex"], report["age"]
    lines = ["# Final report: Abalone panel debate", "", f"Held-out test split: {sex['n']} rows for sex, {age['n']} rows for age.", "",
             "## Sex", "", "| Method | 3-class accuracy | Macro-F1 | Infant vs adult |", "| --- | --- | --- | --- |"]
    for name, m in sex["methods"].items():
        lines.append(f"| {name} | {m.get('accuracy_3_class', '-')} | {m.get('macro_f1', '-')} | {m.get('infant_vs_adult_accuracy', '-')} |")
    lines += ["", f"Debate sex calls: {sex['debate_sex_call_counts']}; commits to M or F: {sex['debate_commits_to_m_or_f']}.",
              f"Tests: {json.dumps(sex['tests'])}", f"By reliability: {json.dumps(sex['by_reliability'])}", "",
              "## Age", "", "| Method | MAE | RMSE | R2 | RMSLE | within 1 | within 2 | 80% coverage | mean width |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in age["methods"].items():
        lines.append(f"| {name} | {m['mae']} | {m['rmse']} | {m['r2']} | {m['rmsle']} | {m['within_1_ring']} | {m['within_2_rings']} | {m.get('range_coverage', '-')} | {m.get('mean_range_width', '-')} |")
    lines += ["", "| Rings | n | Debate MAE | Tools median MAE | Gradient boosting MAE | Debate bias |", "| --- | --- | --- | --- | --- | --- |"]
    for label, g in age["by_ring_group"].items():
        lines.append(f"| {label} | {g['n']} | {g['debate_mae']} | {g['tools_median_mae']} | {g['gradient_boosting_mae']} | {g['debate_bias']} |")
    lines += ["", f"Tests: {json.dumps(age['tests'])}", f"By reliability: {json.dumps(age['by_reliability'])}", "",
              "## Debate", "", f"Value: {json.dumps(report['debate_value'])}", "", f"Dynamics: {json.dumps(report['dynamics'])}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final analysis of a finished debate evaluation.")
    parser.add_argument("--output", default="artifacts/runs/evals/debate_full")
    args = parser.parse_args(argv)
    output = REPO_ROOT / args.output
    rows = load(output)
    report = {
        "sex": sex_section([r for r in rows if r["task"] == "sex"]),
        "age": age_section([r for r in rows if r["task"] == "age"]),
        "debate_value": debate_value(rows),
        "dynamics": dynamics(rows),
    }
    (output / "final_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "final_report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
