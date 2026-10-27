"""run_debate_eval.py

Evaluate the panel debate on the Abalone dataset and compare it with a tools-only baseline and with published Kaggle
and literature results.

For each selected row and task (sex, age) it runs one debate (the hidden column is kept as the truth), appends the
result to <output>/results.jsonl at once, and keeps every discussion in <output>/runs/<run>/discussion.md (indexed in
<output>/runs/discussions_index.jsonl). The run is resumable: rows already in results.jsonl are skipped.

Rows: --split test (default) uses the 835 held-out rows of the frozen split, which no tool model saw in training; this
is the split that is fair to compare with published results. --split train or all adds rows the ML tools were trained
on, so their scores are optimistic and are reported apart.

Metrics
- sex: 3-class accuracy and macro-F1 of the arbiter's forced_choice (the class used for benchmark scoring), infant vs
  adult accuracy, and how often the debate answered adult_undetermined.
- age: MAE, RMSE, R², RMSLE of the ring estimate, share within 1 and 2 rings, and coverage of the 80% range.
The baseline calls the tools directly: sex = soft vote of the three probabilistic tools; age = median of the four
estimates with the widest 80% range.

Run from the repository root (Ollama on the GPU, article store built):
    .venv/bin/python evals/abalone_debate/run_debate_eval.py --limit 4                 # quick check
    .venv/bin/python evals/abalone_debate/run_debate_eval.py --concurrency 2           # the whole test split
    .venv/bin/python evals/abalone_debate/run_debate_eval.py --report-only             # rebuild the report
    # Spread the debates over two Ollama servers (each one has a single coordinating CPU thread)
    .venv/bin/python evals/abalone_debate/run_debate_eval.py --concurrency 8 --ollama-hosts http://127.0.0.1:11434 http://127.0.0.1:11435
    # gemma4:31b with the forced-class sex debate (M, F or I), on a server started with OLLAMA_NUM_PARALLEL=4
    .venv/bin/python evals/abalone_debate/run_debate_eval.py --model gemma4:31b --forced-sex --num-ctx 20480 \\
        --concurrency 4 --ollama-hosts http://127.0.0.1:11435 --output artifacts/runs/evals/debate_31b_forced
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from runtime.flow_runner import FlowRun, RunOptions  # noqa: E402
from runtime.ollama_chat import OllamaChat  # noqa: E402
from runtime.problem_statement import problems_from_csv  # noqa: E402

DATASET = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "train" / "abalone_dataset.csv"
SPLIT = REPO_ROOT / "artifacts" / "datasets" / "abalone_dataset" / "splits.json"
FLOWS = {task: REPO_ROOT / "declarations" / "flows" / f"abalone_{task}_debate_flow.json" for task in ("sex", "age")}
# Sex debate in which every tool agent and the arbiter must answer M, F or I (no adult_undetermined).
FORCED_SEX_FLOW = REPO_ROOT / "declarations" / "flows" / "abalone_sex_forced_debate_flow.json"
SEX_PROBABILISTIC = ("abalone_sex_random_forest", "abalone_sex_logistic_regression", "abalone_sex_bivariate_profile")
AGE_TOOLS = ("abalone_age_gradient_boosting", "abalone_age_ridge_regression", "abalone_age_allometric_formula", "abalone_age_shell_weight_lookup")

def runtime_version() -> str:
    """Fingerprint of the code and declarations that decide a debate, stored with every result."""
    import hashlib

    digest = hashlib.sha256()
    files = sorted([*(CODE_DIR / "runtime").glob("*.py"), *(CODE_DIR / "tools" / "article_rag").glob("*.py"),
                    *(REPO_ROOT / "declarations" / "agents").glob("*.json"), *(REPO_ROOT / "declarations" / "flows").glob("abalone_*_debate_flow.json"),
                    *(REPO_ROOT / "declarations" / "tools").glob("article_*.json")])
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


RUNTIME_VERSION = runtime_version()

# Published results to compare with. Different splits and data, so they are a reference, not a leaderboard.
REFERENCES = [
    {"task": "sex", "metric": "accuracy (3 classes)", "value": 0.53, "source": "SVM, cross-validated, original UCI data",
     "url": "https://rstudio-pubs-static.s3.amazonaws.com/863621_0ed8a487ee6c4b7ea07eaacd308e257c.html"},
    {"task": "age", "metric": "RMSE (rings)", "value": 1.957, "source": "BGNLM PRL, best of the paper, 1,000 test rows of the original UCI data",
     "url": "https://arxiv.org/abs/2003.02929"},
    {"task": "age", "metric": "MAE (rings)", "value": 1.447, "source": "BGNLM PRL, same paper and test rows",
     "url": "https://arxiv.org/abs/2003.02929"},
    {"task": "age", "metric": "RMSE (rings)", "value": 2.035, "source": "Random forest, same paper and test rows",
     "url": "https://arxiv.org/abs/2003.02929"},
    {"task": "age", "metric": "RMSE (rings)", "value": 2.088, "source": "XGBoost, same paper and test rows",
     "url": "https://arxiv.org/abs/2003.02929"},
    {"task": "age", "metric": "RMSLE", "value": 0.14379, "source": "Kaggle Playground S4E4, 1st place private score (synthetic data derived from UCI)",
     "url": "https://www.kaggle.com/competitions/playground-series-s4e4/writeups/johannes-heller-1st-place-solution-for-the-regress"},
]


# ---------------------------------------------------------------------------
# Baseline and scoring
# ---------------------------------------------------------------------------

def baseline(problem, tools: dict) -> dict:
    specimen = problem.specimen()
    if problem.task == "sex":
        totals = Counter()
        for name in SEX_PROBABILISTIC:
            prediction = tool_registry.invoke(tools[name], {"specimens": [specimen]})["result"]["predictions"][0]
            totals.update(prediction["probabilities"])
        choice = max(("M", "F", "I"), key=lambda cls: totals[cls])
        return {"forced_choice": choice, "maturity": "infant" if choice == "I" else "adult"}
    estimates, lows, highs = [], [], []
    for name in AGE_TOOLS:
        prediction = tool_registry.invoke(tools[name], {"specimens": [specimen]})["result"]["predictions"][0]
        estimates.append(prediction["predicted_rings"])
        lows.append(prediction["interval_80"]["low"])
        highs.append(prediction["interval_80"]["high"])
    return {"rings": statistics.median(estimates), "rings_low": min(lows), "rings_high": max(highs)}


def debate_answer(task: str, final: dict | None) -> dict:
    result = (final or {}).get("result") or {}
    if task == "sex":
        call = result.get("sex_call") or str(result.get("answer") or "").strip()
        forced = result.get("forced_choice") or (call if call in ("M", "F", "I") else None)
        maturity = result.get("maturity") or {"I": "infant", "M": "adult", "F": "adult", "adult_undetermined": "adult"}.get(call, "uncertain")
        return {"sex_call": call, "forced_choice": forced, "maturity": maturity}
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(result.get("answer") or ""))]
    return {
        "rings": result.get("rings", numbers[0] if numbers else None),
        "rings_low": result.get("rings_low", numbers[1] if len(numbers) >= 3 else None),
        "rings_high": result.get("rings_high", numbers[2] if len(numbers) >= 3 else None),
    }


def sex_metrics(rows: list[dict], key: str) -> dict:
    if not rows:
        return {}
    truth = [r["truth"] for r in rows]
    choice = [r[key].get("forced_choice") for r in rows]
    accuracy = sum(c == t for c, t in zip(choice, truth)) / len(rows)
    f1s = []
    for cls in ("M", "F", "I"):
        tp = sum(c == cls and t == cls for c, t in zip(choice, truth))
        fp = sum(c == cls and t != cls for c, t in zip(choice, truth))
        fn = sum(c != cls and t == cls for c, t in zip(choice, truth))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    maturity_truth = ["infant" if t == "I" else "adult" for t in truth]
    maturity = sum(r[key].get("maturity") == m for r, m in zip(rows, maturity_truth)) / len(rows)
    metrics = {"n": len(rows), "accuracy_3_class": round(accuracy, 4), "macro_f1": round(sum(f1s) / 3, 4), "infant_vs_adult_accuracy": round(maturity, 4)}
    if key == "debate":
        metrics["adult_undetermined_rate"] = round(sum(r["debate"].get("sex_call") == "adult_undetermined" for r in rows) / len(rows), 4)
        metrics["missing_forced_choice"] = sum(r["debate"].get("forced_choice") is None for r in rows)
    return metrics


def age_metrics(rows: list[dict], key: str) -> dict:
    scored = [r for r in rows if r[key].get("rings") is not None]
    if not scored:
        return {}
    truth = [float(r["truth"]) for r in scored]
    pred = [float(r[key]["rings"]) for r in scored]
    errors = [p - t for p, t in zip(pred, truth)]
    mean_truth = sum(truth) / len(truth)
    ss_tot = sum((t - mean_truth) ** 2 for t in truth) or 1.0
    covered = [r for r in scored if r[key].get("rings_low") is not None and r[key].get("rings_high") is not None]
    return {
        "n": len(scored),
        "missing_estimate": len(rows) - len(scored),
        "mae": round(sum(abs(e) for e in errors) / len(errors), 4),
        "rmse": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 4),
        "r2": round(1 - sum(e * e for e in errors) / ss_tot, 4),
        "rmsle": round(math.sqrt(sum((math.log1p(max(p, 0)) - math.log1p(t)) ** 2 for p, t in zip(pred, truth)) / len(pred)), 4),
        "within_1_ring": round(sum(abs(e) <= 1 for e in errors) / len(errors), 4),
        "within_2_rings": round(sum(abs(e) <= 2 for e in errors) / len(errors), 4),
        "range_coverage": round(sum(r[key]["rings_low"] <= float(r["truth"]) <= r[key]["rings_high"] for r in covered) / len(covered), 4) if covered else None,
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def selected_rows(split_name: str, limit: int | None, seed: int) -> list[tuple[int, str]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = {"test": [(i, "test") for i in split["test_indices"]],
            "train": [(i, "train") for i in split["train_indices"]]}
    chosen = rows["test"] + rows["train"] if split_name == "all" else rows[split_name]
    chosen = sorted(chosen)
    if limit is not None:
        chosen = sorted(random.Random(seed).sample(chosen, min(limit, len(chosen))))
    return chosen


def done_keys(results_path: Path) -> set[tuple[int, str]]:
    if not results_path.is_file():
        return set()
    return {(r["row"], r["task"]) for r in map(json.loads, results_path.read_text(encoding="utf-8").splitlines()) if r.get("completed")}


async def run_one(row: int, part: str, task: str, flow: Path, options: RunOptions, tools: dict, results_path: Path, lock: asyncio.Lock, progress: dict) -> None:
    problem = problems_from_csv(DATASET, task, [row])[0]
    truth = problem.truth["sex" if task == "sex" else "number_of_rings"]
    base = baseline(problem, tools)
    started = time.perf_counter()
    try:
        run = await FlowRun(flow, problem, options).run()
        final, stats, run_dir, debate = run.final, run.stats, str(run.run_dir.relative_to(REPO_ROOT)), run.debate
        error = None
    except Exception as exc:  # noqa: BLE001 - one failed debate must not stop the evaluation
        final, stats, run_dir, debate, error = None, {}, None, {}, repr(exc)
    result = (final or {}).get("result") or {}
    record = {
        "row": row, "split": part, "task": task, "truth": truth,
        "debate": debate_answer(task, final), "baseline": base,
        "reliability": result.get("reliability"), "debate_outcome": result.get("debate_outcome"),
        "arbiter_fallback": bool(result.get("arbiter_fallback")),
        "converged": debate.get("converged"), "rounds": debate.get("round"),
        "challenges": dict(Counter(c["status"] for r in debate.get("ledger", []) for c in r["challenges"])),
        "abstained": debate.get("abstained", []),
        "turns": stats.get("turns"), "elapsed_s": round(time.perf_counter() - started, 1),
        "run": run_dir, "completed": final is not None, "error": error, "runtime_version": RUNTIME_VERSION,
        "flow": flow.name, "model": options.model,
    }
    async with lock:
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress["done"] += 1
        rate = (time.perf_counter() - progress["started"]) / progress["done"]
        left = (progress["total"] - progress["done"]) * rate / 3600
        shown = record["debate"].get("forced_choice") if task == "sex" else record["debate"].get("rings")
        print(f"[{progress['done']}/{progress['total']}] row {row} {task}: debate {shown} | truth {truth} | "
              f"{record['elapsed_s']} s | about {left:.1f} h left{' | ERROR ' + error if error else ''}", flush=True)


def write_report(output: Path) -> dict:
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("completed")]
    report: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "references": REFERENCES, "splits": {}}
    lines = ["# Debate evaluation", "", f"Generated {report['generated']}. Rows the ML tools were trained on are reported apart.", ""]
    for part in ("test", "train"):
        part_rows = [r for r in rows if r["split"] == part]
        if not part_rows:
            continue
        sex = [r for r in part_rows if r["task"] == "sex"]
        age = [r for r in part_rows if r["task"] == "age"]
        block = {
            "sex": {"debate": sex_metrics(sex, "debate"), "baseline": sex_metrics(sex, "baseline")},
            "age": {"debate": age_metrics(age, "debate"), "baseline": age_metrics(age, "baseline")},
            "debate": {
                "converged": round(sum(bool(r["converged"]) for r in part_rows) / len(part_rows), 4),
                "mean_rounds": round(sum(r["rounds"] or 0 for r in part_rows) / len(part_rows), 2),
                "challenges": dict(sum((Counter(r["challenges"]) for r in part_rows), Counter())),
                "abstentions": dict(Counter(code for r in part_rows for code in r["abstained"])),
                "mean_seconds": round(sum(r["elapsed_s"] for r in part_rows) / len(part_rows), 1),
                "arbiter_fallbacks": sum(bool(r.get("arbiter_fallback")) for r in part_rows),
            },
        }
        report["splits"][part] = block
        title = "Held-out test split (fair comparison)" if part == "test" else "Training rows (optimistic: the ML tools saw them)"
        lines += [f"## {title}", "", "| Task | Metric | Debate | Tools-only baseline |", "| --- | --- | --- | --- |"]
        for task in ("sex", "age"):
            debate_m, base_m = block[task]["debate"], block[task]["baseline"]
            for metric, value in debate_m.items():
                if metric in ("n",):
                    continue
                lines.append(f"| {task} | {metric} | {value} | {base_m.get(metric, '')} |")
            if debate_m:
                lines.append(f"| {task} | rows | {debate_m.get('n')} | {base_m.get('n')} |")
        lines += ["", f"Debate: {json.dumps(block['debate'], ensure_ascii=False)}", ""]
    lines += ["## Published references", "", "| Task | Metric | Value | Source |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['task']} | {r['metric']} | {r['value']} | [{r['source']}]({r['url']}) |" for r in REFERENCES]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


async def main_async(args: argparse.Namespace) -> int:
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    if not args.report_only:
        # The process id, so a detached run can be stopped with: kill -TERM -$(cat <output>/eval.pid)
        (output / "eval.pid").write_text(str(os.getpid()), encoding="utf-8")
    if args.report_only:
        print((output / "report.md").read_text() if write_report(output) else "no results")
        return 0
    tasks = args.tasks
    flows = {**FLOWS, "sex": FORCED_SEX_FLOW} if args.forced_sex else FLOWS
    done = done_keys(results_path)
    pending = [(row, part, task) for row, part in selected_rows(args.split, args.limit, args.seed) for task in tasks if (row, task) not in done]
    print(f"{len(pending)} debates to run ({args.split} split, tasks {tasks}, concurrency {args.concurrency}, model {args.model}, "
          f"hosts {args.ollama_hosts or ['declared']}, num_ctx {args.num_ctx or 'declared'}, "
          f"flows {[flows[t].name for t in tasks]}, runtime {RUNTIME_VERSION}); "
          f"results in {args.output}", flush=True)
    tools = tool_registry.load_tool_declarations(server="abalone_tools")
    hosts = args.ollama_hosts or [None]
    per_host = max(1, math.ceil(args.concurrency / len(hosts)))
    lanes = []
    for host in hosts:
        def factory(agent, host=host):
            if host:
                agent = {**agent, "model": {**agent["model"], "connection": {**agent["model"]["connection"], "base_url": host}}}
            return OllamaChat(agent, args.model, num_ctx_override=args.num_ctx)
        options = RunOptions(model=args.model, num_ctx=args.num_ctx, runs_dir=output / "runs", echo=lambda _: None, llm_factory=factory)
        lanes.append((options, asyncio.Semaphore(per_host)))
    lock = asyncio.Lock()
    progress = {"done": 0, "total": len(pending), "started": time.perf_counter()}

    async def guarded(index, item):
        options, semaphore = lanes[index % len(lanes)]
        async with semaphore:
            row, part, task = item
            await run_one(row, part, task, flows[task], options, tools, results_path, lock, progress)

    await asyncio.gather(*(guarded(i, item) for i, item in enumerate(pending)))
    write_report(output)
    print((output / "report.md").read_text(), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the panel debate against a tools-only baseline and published results.")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--tasks", nargs="+", choices=["sex", "age"], default=["sex", "age"])
    parser.add_argument("--limit", type=int, help="Run only this many rows (random, seeded) of the split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1, help="Debates at the same time (raise OLLAMA_NUM_PARALLEL to match).")
    parser.add_argument("--model", default="gemma4:e4b")
    parser.add_argument("--ollama-hosts", nargs="+", help="Chat servers to spread the debates over (embeddings stay on the tools' configured server).")
    parser.add_argument("--num-ctx", type=int, help="Context tokens per request for every agent (default: each declaration's num_ctx).")
    parser.add_argument("--forced-sex", action="store_true", help="Sex debates use abalone_sex_forced_debate_flow.json: the answer is always M, F or I.")
    parser.add_argument("--output", default="artifacts/runs/evals/debate_full")
    parser.add_argument("--report-only", action="store_true")
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
