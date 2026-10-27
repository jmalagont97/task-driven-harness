"""run_debate.py

Temporary entry point of the debate runtime. It does the job of infer.py for the Abalone debate flows (problem
ingestion, blackboard initialisation, graph orchestration) without touching infer.py, which belongs to someone else.

Run from the repository root (Ollama must be running):
    # Show the English problem statements without calling any model
    .venv/bin/python code/runtime/run_debate.py --input evals/abalone_debate/smoke_sex.csv --render-only
    # One debate per CSV row; the task comes from the missing column (no sex -> sex, no number_of_rings -> age)
    .venv/bin/python code/runtime/run_debate.py --input evals/abalone_debate/smoke_sex.csv --model gemma4:e4b
    # Rows of the full dataset: choose the task, the other target column is hidden and kept as the truth
    .venv/bin/python code/runtime/run_debate.py --input artifacts/datasets/abalone_dataset/train/abalone_dataset.csv --task age --rows 0 5
    # One standalone question to one article expert
    .venv/bin/python code/runtime/run_debate.py --only article_mcshane_1992_agent --question "Are growth rings formed once a year?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime.flow_runner import REPO_ROOT, RUNS_DIR, FlowRun, RunOptions, run_question  # noqa: E402
from runtime.problem_statement import RowError, problems_from_csv  # noqa: E402

FLOWS = {
    "sex": REPO_ROOT / "declarations" / "flows" / "abalone_sex_debate_flow.json",
    "age": REPO_ROOT / "declarations" / "flows" / "abalone_age_debate_flow.json",
}


def summarize(final: dict | None) -> str:
    if not final:
        return "no final decision"
    result = final.get("result") or {}
    answer = result.get("answer", "?")
    return f"{answer} (reliability {result.get('reliability')}, debate {result.get('debate_outcome')})"


async def main_async(args: argparse.Namespace) -> int:
    options = RunOptions(model=args.model, think=args.think, num_ctx=args.num_ctx, runs_dir=Path(args.runs_dir))
    if args.only:
        if not args.question:
            print("--only needs --question.", file=sys.stderr)
            return 2
        flow = Path(args.flow) if args.flow else FLOWS["sex"]
        result = await run_question(flow, args.only, args.question, options)
        output = result.final or {}
        print(json.dumps(output.get("result", {}).get("answer_markdown", output), ensure_ascii=False, indent=2))
        print(f"coverage: {output.get('result', {}).get('coverage')} | run folder: {result.run_dir.relative_to(REPO_ROOT)}")
        return 0

    try:
        problems = problems_from_csv(Path(args.input), args.task, args.rows)
    except RowError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        if args.render_only:
            print(f"===== row {problem.row_number} ({problem.task}) =====\n{problem.statement}")
            continue
        flow = Path(args.flow) if args.flow else FLOWS[problem.task]
        print(f"Row {problem.row_number}: {problem.task} debate with {args.model or 'the declared models'}")
        result = await FlowRun(flow, problem, options).run()
        truth = f" | truth: {problem.truth}" if problem.truth else ""
        print(f"Row {problem.row_number}: {summarize(result.final)}{truth}")
        print(f"  {result.stats['turns']} turns, {result.stats['skipped']} skipped, {result.stats.get('elapsed_s')} s | run folder: {result.run_dir.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Abalone debate for CSV rows, or ask one article expert a question.")
    parser.add_argument("--input", help="CSV with the dataset's columns (dataset units).")
    parser.add_argument("--task", choices=["sex", "age"], help="Needed only when the CSV has both sex and number_of_rings.")
    parser.add_argument("--rows", type=int, nargs="*", help="0-based data rows to run (default: all).")
    parser.add_argument("--model", help="Ollama model for every agent, e.g. gemma4:e4b or gemma4:31b (default: each declaration's).")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None, help="Override runtime.adapters.ollama.think.")
    parser.add_argument("--num-ctx", type=int, help="Override runtime.adapters.ollama.num_ctx (context tokens per request).")
    parser.add_argument("--render-only", action="store_true", help="Print the problem statements and stop.")
    parser.add_argument("--only", help="Node id of one article expert, to ask it --question.")
    parser.add_argument("--question", help="Standalone question for --only.")
    parser.add_argument("--flow", help="Override the flow file.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="Where run folders are written.")
    args = parser.parse_args(argv)
    if not args.only and not args.input:
        parser.error("give --input (a CSV) or --only with --question")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
