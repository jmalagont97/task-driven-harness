"""run_retrieval_eval.py

Retrieval evaluation of search_article for each article, with no language model involved, and calibration of the
per-article min_score.

For every question in evals/article_agents/questions/<document_id>.jsonl it calls the search_article executor bound
to that article (real index, real embeddings) and measures:
- hit@1 and recall@5: one of the `must_contain` snippets appears in the first result / in the top 5;
- MRR over the in-scope questions;
- isolation: every returned chunk_id belongs to the bound article (must be 100%);
- the top score of in-scope and out-of-scope questions, from which --calibrate picks the min_score that best separates
  them (largest balanced accuracy, ties broken towards the midpoint of the gap) and writes calibration.json.

Run from the repository root after building the index (Ollama running):
    .venv/bin/python evals/article_agents/run_retrieval_eval.py
    .venv/bin/python evals/article_agents/run_retrieval_eval.py --calibrate
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from tools.article_rag.search import search_article  # noqa: E402

QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
DECLARATION = tool_registry.read_declaration(tool_registry.TOOLS_DIR / "article_search_article.json")


def load_questions(document_id: str) -> list[dict]:
    path = QUESTIONS_DIR / f"{document_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hit_rank(results: list[dict], snippets: list[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        text = result["text"].lower()
        if any(snippet.lower() in text for snippet in snippets):
            return rank
    return None


def best_threshold(inside: list[float], outside: list[float]) -> float:
    """min_score with the best balanced accuracy (in-scope top score >= t, out-of-scope < t)."""
    candidates = sorted(set(inside + outside))
    best, best_value = 0.0, -1.0
    for low, high in zip([0.0] + candidates, candidates + [1.0]):
        t = round((low + high) / 2, 4)
        accepted = sum(score >= t for score in inside) / max(len(inside), 1)
        rejected = sum(score < t for score in outside) / max(len(outside), 1)
        value = (accepted + rejected) / 2
        if value > best_value + 1e-9:
            best, best_value = t, value
    return best


def evaluate_document(document_id: str, top_k: int) -> dict:
    config = copy.deepcopy(DECLARATION["config"])
    config.update(document_id=document_id, calibration_path=None)
    rows = []
    for question in load_questions(document_id):
        started = time.perf_counter()
        envelope = search_article({"query": question["question"], "top_k": top_k}, config)
        if not envelope["ok"]:
            raise SystemExit(f"{document_id} {question['id']}: {envelope['error']}")
        result = envelope["result"]
        results = result["results"]
        rows.append({
            "id": question["id"],
            "lang": question["lang"],
            "expect": question["expect"],
            "top_score": results[0]["score"] if results else 0.0,
            "rank": hit_rank(results, question["must_contain"]) if question["must_contain"] else None,
            "isolated": all(r["chunk_id"].startswith(document_id + ":") for r in results),
            "mode": result["retrieval_mode"],
            "ms": round((time.perf_counter() - started) * 1000),
        })
    inside = [r for r in rows if r["expect"] == "covered"]
    outside = [r for r in rows if r["expect"] == "not_covered"]
    threshold = best_threshold([r["top_score"] for r in inside], [r["top_score"] for r in outside])
    return {
        "document_id": document_id,
        "questions": len(rows),
        "hit@1": round(sum(r["rank"] == 1 for r in inside) / len(inside), 3),
        f"recall@{top_k}": round(sum(r["rank"] is not None for r in inside) / len(inside), 3),
        "mrr": round(sum(1 / r["rank"] for r in inside if r["rank"]) / len(inside), 3),
        "recall_es": round(sum(r["rank"] is not None for r in inside if r["lang"] == "es") / max(1, sum(r["lang"] == "es" for r in inside)), 3),
        "isolation": round(sum(r["isolated"] for r in rows) / len(rows), 3),
        "in_scope_top_scores": sorted(r["top_score"] for r in inside),
        "out_of_scope_top_scores": sorted(r["top_score"] for r in outside),
        "suggested_min_score": threshold,
        "out_of_scope_rejected_at_threshold": round(sum(r["top_score"] < threshold for r in outside) / max(1, len(outside)), 3),
        "in_scope_accepted_at_threshold": round(sum(r["top_score"] >= threshold for r in inside) / max(1, len(inside)), 3),
        "misses": [r["id"] for r in inside if r["rank"] is None],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval and min_score calibration for the article servers.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--calibrate", action="store_true", help="Write the suggested min_score per article to calibration.json.")
    parser.add_argument("--output", default="artifacts/runs/evals/retrieval_eval.json")
    args = parser.parse_args(argv)

    documents = sorted(path.stem for path in QUESTIONS_DIR.glob("*.jsonl"))
    report = {"embedding": DECLARATION["config"]["embedding"]["model"], "top_k": args.top_k, "documents": []}
    print(f"{'document':28s} {'hit@1':>6s} {'R@k':>6s} {'MRR':>6s} {'R@k es':>7s} {'isol':>5s} {'min_score':>9s} {'out rej':>8s} {'in acc':>7s}")
    for document_id in documents:
        summary = evaluate_document(document_id, args.top_k)
        report["documents"].append(summary)
        print(
            f"{document_id:28s} {summary['hit@1']:6.2f} {summary[f'recall@{args.top_k}']:6.2f} {summary['mrr']:6.2f} "
            f"{summary['recall_es']:7.2f} {summary['isolation']:5.2f} {summary['suggested_min_score']:9.3f} "
            f"{summary['out_of_scope_rejected_at_threshold']:8.2f} {summary['in_scope_accepted_at_threshold']:7.2f}"
        )
        if summary["misses"]:
            print(f"  misses: {', '.join(summary['misses'])}")

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {args.output}")
    if args.calibrate:
        path = REPO_ROOT / DECLARATION["config"]["calibration_path"]
        calibration = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        for summary in report["documents"]:
            calibration[summary["document_id"]] = {
                "hybrid": summary["suggested_min_score"],
                "embedding_model": report["embedding"],
                "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "questions": summary["questions"],
            }
        path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"calibration written to {DECLARATION['config']['calibration_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
