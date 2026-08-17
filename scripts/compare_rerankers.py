"""Cross-encoder reranker comparison script.

Benchmarks a set of candidate cross-encoder models against the annotated eval
set (data/eval_queries.json) and prints a ranked comparison table of retrieval
quality (Hit@k, MRR) and inference latency (p50 / p95 per rerank call).

Design
------
For each model, the script:

1. Loads the eval queries.
2. For each query: retrieves RERANK_FETCH_K candidates from Qdrant via HyDE,
   runs the cross-encoder on those candidates, and records reranked MRR and
   per-rerank wall-clock time.
3. Aggregates across queries into a summary row.

Because step 2 re-uses the same Qdrant candidates for every model (retrieved
once per query, cached in memory for the model loop), the comparison is fair:
only the cross-encoder step differs between models.

Model ranking uses a composite score:
    score = α × MRR + (1 − α) × (1 − norm_latency)
where norm_latency = mean_rerank_ms / max_mean_rerank_ms across models.  The
default α=0.8 weights quality over speed; adjust with --quality-weight.

Models evaluated by default
----------------------------
- BAAI/bge-reranker-base       current default; BEIR NDCG@10 ≈ 0.49
- cross-encoder/ms-marco-MiniLM-L-6-v2   former default; BEIR NDCG@10 ≈ 0.42
- cross-encoder/ms-marco-MiniLM-L-12-v2  larger ms-marco; BEIR NDCG@10 ≈ 0.45
- cross-encoder/nli-deberta-v3-small      NLI-based; robust on structured queries

All models are downloaded from HuggingFace Hub on first run and cached in
~/.cache/huggingface/hub.  Expect 1–5 minutes for initial downloads.

Usage:
    python scripts/compare_rerankers.py
    python scripts/compare_rerankers.py --k 10 --fetch-k 30
    python scripts/compare_rerankers.py --models BAAI/bge-reranker-base cross-encoder/ms-marco-MiniLM-L-6-v2
    python scripts/compare_rerankers.py --output results/reranker_comparison.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service")
os.environ.setdefault("ENABLE_RERANKER", "false")

from app.rag.retrieval import search_flights_hybrid  # noqa: E402
from app.rag.flight_doc import flight_doc  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_queries.json"

DEFAULT_MODELS = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/nli-deberta-v3-small",
]


def _rerank_with_model(
    model_instance,
    query: str,
    docs: list[str],
    top_n: int,
) -> tuple[list[int], float]:
    """Score all (query, doc) pairs with the given cross-encoder.

    Args:
        model_instance: Loaded sentence_transformers.CrossEncoder.
        query: User query string.
        docs: Document strings to rank.
        top_n: Number of top indices to return.

    Returns:
        Tuple of (sorted_indices, elapsed_ms).
    """
    t0 = time.perf_counter()
    pairs = [(query, d) for d in docs]
    scores = model_instance.predict(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_n], elapsed_ms


def _compute_mrr(returned_ids: list[str], expected_set: set[str]) -> float:
    for rank, fid in enumerate(returned_ids, start=1):
        if fid in expected_set:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (pct / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


async def _fetch_candidates(
    entries: list[dict],
    k: int,
    fetch_k: int,
) -> list[dict | None]:
    """Retrieve Qdrant candidates for every eval query.

    Fetching once and reusing across models ensures the reranker comparison
    is controlled: only the cross-encoder differs between models.

    Args:
        entries: Eval query dicts with ``query`` and ``expected_flight_ids``.
        k: Final top-k (unused here, fetch_k is what matters).
        fetch_k: Number of candidates to retrieve per query.

    Returns:
        List of candidate dicts (one per entry): {candidates, expected_set} or
        None when the entry has no expected_flight_ids (unannotated).
    """
    print(f"Fetching {fetch_k} candidates per query from Qdrant …")
    candidate_sets: list[dict | None] = []
    for entry in entries:
        if not entry.get("expected_flight_ids"):
            candidate_sets.append(None)
            continue
        result = await search_flights_hybrid(query=entry["query"], limit=fetch_k)
        candidate_sets.append({
            "flights": result["results"],
            "docs": [flight_doc(f) for f in result["results"]],
            "expected_set": set(entry["expected_flight_ids"]),
            "query": entry["query"],
        })
    scoreable = sum(1 for c in candidate_sets if c is not None)
    print(f"  {scoreable} annotated queries ready.\n")
    return candidate_sets


def _evaluate_model(
    model_name: str,
    candidate_sets: list[dict | None],
    k: int,
) -> dict:
    """Load a cross-encoder and score all candidate sets.

    Args:
        model_name: HuggingFace model ID to load and evaluate.
        candidate_sets: Output of _fetch_candidates — one entry per eval query.
        k: Top-k cutoff applied after reranking.

    Returns:
        Summary dict with keys: model, n_scored, hit_rate, mean_mrr,
        mean_rerank_ms, p50_rerank_ms, p95_rerank_ms.
    """
    from sentence_transformers import CrossEncoder

    print(f"  Loading {model_name} …", flush=True)
    t_load = time.perf_counter()
    model = CrossEncoder(model_name)
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"  Loaded in {load_ms:.0f} ms", flush=True)

    hits, mrrs, rerank_times = [], [], []

    for cset in candidate_sets:
        if cset is None:
            continue

        flights = cset["flights"]
        docs = cset["docs"]
        expected_set = cset["expected_set"]
        query = cset["query"]

        if not flights:
            hits.append(0)
            mrrs.append(0.0)
            rerank_times.append(0.0)
            continue

        indices, elapsed_ms = _rerank_with_model(model, query, docs, top_n=k)
        rerank_times.append(elapsed_ms)

        reranked_ids = [flights[i]["flight_id"] for i in indices]
        hits.append(1 if any(fid in expected_set for fid in reranked_ids) else 0)
        mrrs.append(_compute_mrr(reranked_ids, expected_set))

    n = len(hits)
    return {
        "model": model_name,
        "n_scored": n,
        "hit_rate": sum(hits) / n if n else 0.0,
        "mean_mrr": statistics.mean(mrrs) if mrrs else 0.0,
        "mean_rerank_ms": statistics.mean(rerank_times) if rerank_times else 0.0,
        "p50_rerank_ms": _percentile(rerank_times, 50),
        "p95_rerank_ms": _percentile(rerank_times, 95),
        "load_ms": load_ms,
    }


def _composite_score(row: dict, max_mean_ms: float, quality_weight: float) -> float:
    """Compute a composite quality-latency score for ranking models.

    Args:
        row: Model evaluation dict from _evaluate_model.
        max_mean_ms: Maximum mean rerank latency across all evaluated models,
            used to normalise latency to [0, 1].
        quality_weight: Weight α applied to MRR; speed gets weight (1 − α).

    Returns:
        Composite score in [0, 1]; higher is better.
    """
    norm_latency = row["mean_rerank_ms"] / max_mean_ms if max_mean_ms > 0 else 0.0
    speed_score = 1.0 - norm_latency
    return quality_weight * row["mean_mrr"] + (1.0 - quality_weight) * speed_score


async def main(
    models: list[str],
    k: int,
    fetch_k: int,
    quality_weight: float,
    output: str | None,
) -> int:
    """Run the full reranker comparison and print a ranked table.

    Args:
        models: List of HuggingFace cross-encoder model IDs to evaluate.
        k: Top-k cutoff applied after reranking.
        fetch_k: Number of Qdrant candidates to retrieve per query.
        quality_weight: Weight α for MRR in the composite ranking score.
        output: Optional path to write results JSON.

    Returns:
        Exit code (0 = success, 1 = eval set missing or empty).
    """
    if not EVAL_PATH.exists():
        print(f"No eval set found at {EVAL_PATH}. Run scripts/build_eval_set.py first.")
        return 1

    entries = json.loads(EVAL_PATH.read_text())
    scoreable = [e for e in entries if e.get("expected_flight_ids")]
    if not scoreable:
        print("eval_queries.json has no annotated entries — run build_eval_set.py first.")
        return 1

    print(f"Reranker comparison | {len(scoreable)} annotated queries | k={k} fetch_k={fetch_k}\n")

    candidate_sets = await _fetch_candidates(entries, k, fetch_k)

    rows: list[dict] = []
    for model_name in models:
        print(f"Evaluating: {model_name}")
        try:
            row = _evaluate_model(model_name, candidate_sets, k)
            rows.append(row)
            print(
                f"  Hit@{k}={row['hit_rate']*100:.1f}%  MRR={row['mean_mrr']:.4f}  "
                f"rerank p50={row['p50_rerank_ms']:.0f}ms  p95={row['p95_rerank_ms']:.0f}ms\n"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}\n")

    if not rows:
        print("No models evaluated successfully.")
        return 1

    max_mean_ms = max(r["mean_rerank_ms"] for r in rows) or 1.0
    for row in rows:
        row["composite"] = _composite_score(row, max_mean_ms, quality_weight)

    rows.sort(key=lambda r: r["composite"], reverse=True)

    col_w = max(len(r["model"]) for r in rows) + 2
    header = f"{'Model':<{col_w}} {'Hit%':>6}  {'MRR':>7}  {'p50ms':>7}  {'p95ms':>7}  {'Score':>7}"
    print(f"\n{'═' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")
    for row in rows:
        print(
            f"{row['model']:<{col_w}} "
            f"{row['hit_rate']*100:>5.1f}%  "
            f"{row['mean_mrr']:>7.4f}  "
            f"{row['p50_rerank_ms']:>6.0f}ms  "
            f"{row['p95_rerank_ms']:>6.0f}ms  "
            f"{row['composite']:>7.4f}"
        )
    print(f"{'═' * len(header)}")

    best = rows[0]
    print(f"\nRecommended model (quality_weight={quality_weight}): {best['model']}")
    print(f"  MRR={best['mean_mrr']:.4f}  Hit@{k}={best['hit_rate']*100:.1f}%  "
          f"p95_rerank={best['p95_rerank_ms']:.0f}ms")
    print(f"\nSet RERANKER_MODEL={best['model']} in your .env or docker-compose.yml")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "k": k,
            "fetch_k": fetch_k,
            "quality_weight": quality_weight,
            "n_queries": len(scoreable),
            "models": rows,
        }, indent=2))
        print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare cross-encoder reranker models")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace cross-encoder model IDs to compare",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff after reranking (default 5)")
    parser.add_argument(
        "--fetch-k", type=int, default=20,
        help="Qdrant candidates to retrieve per query before reranking (default 20)",
    )
    parser.add_argument(
        "--quality-weight", type=float, default=0.8,
        help="Weight α for MRR in composite ranking score (default 0.8; speed gets 1-α)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write JSON results (e.g. results/reranker_comparison.json)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(
        models=args.models,
        k=args.k,
        fetch_k=args.fetch_k,
        quality_weight=args.quality_weight,
        output=args.output,
    ))
    sys.exit(exit_code)
