"""Sweep score threshold to find the optimal RETRIEVAL_SCORE_THRESHOLD.

Runs all scoreable queries from data/eval_queries.json at each threshold in the
sweep range and prints a metrics table. Useful for setting RETRIEVAL_SCORE_THRESHOLD
in config rather than leaving it at the 0.3 default.

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --k 10 --min 0.1 --max 0.7 --step 0.05
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service")
# Disable reranking during calibration — we're tuning the retrieval threshold, not
# the reranker output. Including reranking would conflate two distinct sources of quality.
os.environ.setdefault("ENABLE_RERANKER", "false")

from app.rag.retrieval import search_flights_hybrid  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_queries.json"


async def _eval_at_threshold(entries: list[dict], k: int, threshold: float) -> dict:
    """Run all scoreable queries at a single threshold and aggregate metrics.

    Args:
        entries: Loaded eval_queries.json entries with expected_flight_ids.
        k: Top-k cutoff.
        threshold: Score threshold to apply for this run.

    Returns:
        Dict with keys: threshold, n_scored, hit_rate, mean_recall, mean_precision,
        mean_mrr, mean_returned — mean number of results returned per query.
    """
    scored = [e for e in entries if e.get("expected_flight_ids")]
    if not scored:
        return {}

    hits, recalls, precisions, mrrs, n_returned = [], [], [], [], []

    for entry in scored:
        expected = set(entry["expected_flight_ids"])
        result = await search_flights_hybrid(
            query=entry["query"], limit=k, score_threshold=threshold
        )
        returned = [r["flight_id"] for r in result["results"]]
        hit_ids = [fid for fid in returned if fid in expected]

        hits.append(1 if hit_ids else 0)
        recalls.append(len(hit_ids) / len(expected))
        precisions.append(len(hit_ids) / len(returned) if returned else 0.0)
        n_returned.append(len(returned))

        mrr = 0.0
        for rank, fid in enumerate(returned, start=1):
            if fid in expected:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

    n = len(scored)
    return {
        "threshold": threshold,
        "n_scored": n,
        "hit_rate": sum(hits) / n,
        "mean_recall": sum(recalls) / n,
        "mean_precision": sum(precisions) / n,
        "mean_mrr": sum(mrrs) / n,
        "mean_returned": sum(n_returned) / n,
    }


async def main(k: int, t_min: float, t_max: float, step: float) -> None:
    """Sweep thresholds and print a ranked comparison table.

    Args:
        k: Top-k cutoff for retrieval.
        t_min: Minimum threshold to test (inclusive).
        t_max: Maximum threshold to test (inclusive).
        step: Increment between thresholds.
    """
    if not EVAL_PATH.exists():
        print(f"No eval set found at {EVAL_PATH}. Run scripts/build_eval_set.py first.")
        sys.exit(1)

    entries = json.loads(EVAL_PATH.read_text())
    scoreable = [e for e in entries if e.get("expected_flight_ids")]
    if not scoreable:
        print("No entries with expected_flight_ids — annotate with build_eval_set.py first.")
        sys.exit(1)

    print(f"Sweeping threshold {t_min:.2f}–{t_max:.2f} step={step} | k={k} | {len(scoreable)} scoreable queries\n")
    print(f"{'Thresh':>7}  {'Hit%':>6}  {'Recall':>7}  {'Prec':>7}  {'MRR':>7}  {'Avg_N':>7}")
    print("─" * 52)

    thresholds = []
    t = t_min
    while t <= t_max + 1e-9:
        thresholds.append(round(t, 4))
        t += step

    rows = []
    for threshold in thresholds:
        row = await _eval_at_threshold(entries, k, threshold)
        rows.append(row)
        print(
            f"{row['threshold']:>7.3f}  "
            f"{row['hit_rate']*100:>5.1f}%  "
            f"{row['mean_recall']:>7.3f}  "
            f"{row['mean_precision']:>7.3f}  "
            f"{row['mean_mrr']:>7.3f}  "
            f"{row['mean_returned']:>7.1f}"
        )

    # Recommend the threshold with the best F1 of recall and precision.
    def _f1(r: dict) -> float:
        rec, prec = r["mean_recall"], r["mean_precision"]
        return 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0.0

    best = max(rows, key=_f1)
    print(f"\nRecommended threshold (best Recall/Precision F1): {best['threshold']:.3f}")
    print(f"  Hit rate:  {best['hit_rate']*100:.1f}%")
    print(f"  MRR:       {best['mean_mrr']:.3f}")
    print(f"\nSet RETRIEVAL_SCORE_THRESHOLD={best['threshold']:.3f} in your .env or docker-compose.yml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate RETRIEVAL_SCORE_THRESHOLD")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff (default 10)")
    parser.add_argument("--min", dest="t_min", type=float, default=0.1, help="Min threshold (default 0.1)")
    parser.add_argument("--max", dest="t_max", type=float, default=0.7, help="Max threshold (default 0.7)")
    parser.add_argument("--step", type=float, default=0.05, help="Threshold step (default 0.05)")
    args = parser.parse_args()
    asyncio.run(main(args.k, args.t_min, args.t_max, args.step))
