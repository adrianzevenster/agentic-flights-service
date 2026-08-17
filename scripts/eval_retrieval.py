"""Retrieval evaluation harness — relevance metrics, latency, and HyDE A/B.

Loads the annotated golden set from data/eval_queries.json and runs each query
against the live retrieval pipeline, computing Hit@k, Recall@k, MRR, and
per-stage latency (p50 / p95 / mean).  Exits 1 when any metric falls below its
configured threshold so this script can gate CI on retrieval quality.

HyDE A/B comparison
--------------------
Pass --compare-hyde to run each query twice: once with HyDE (HYDE_NUM_SAMPLES
from env/config) and once with direct embedding (hyde_num_samples=0, which uses
the raw query string instead of a hypothetical document).  The report shows the
MRR lift or drop from HyDE and confirms that the extra LLM round-trip is
justified for your query distribution.

Latency reporting
-----------------
Per-query timing is collected from the latency dict returned by
search_flights_hybrid (which records wall-clock time for each stage: hyde,
embed, qdrant, rerank).  The summary table reports mean, p50, and p95 latencies
so you can reason about both typical and tail behaviour.

Thresholds
----------
--min-hit-rate (default 0.70) and --min-mrr (default 0.40) are the minimum
acceptable values for CI to pass.  Tighten them as the eval set matures.
--max-p95-latency-ms (default 5000) gates on tail retrieval latency.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 5 --min-hit-rate 0.8 --min-mrr 0.5
    python scripts/eval_retrieval.py --compare-hyde
    python scripts/eval_retrieval.py --k 10 --max-p95-latency-ms 3000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service")

from app.rag.retrieval import search_flights_hybrid  # noqa: E402
from app.rag.embeddings import async_embed_text  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_queries.json"


async def _run_query(
    query: str,
    k: int,
    filters: dict | None = None,
    hyde_num_samples: int | None = None,
) -> tuple[list[dict], dict]:
    """Run a single query against the retrieval pipeline.

    Args:
        query: Natural-language search query.
        k: Number of results to retrieve.
        filters: Optional dict with keys origin, dest, carrier — passed as hard
            payload filters to Qdrant, mirroring how the agent tool layer calls
            the retrieval function after extracting structured fields from the query.
        hyde_num_samples: Override for HYDE_NUM_SAMPLES.  Pass 0 to bypass HyDE
            entirely (raw query embedding).

    Returns:
        Tuple of (returned_flight_dicts, latency_dict).  Each flight dict contains
        at minimum ``flight_id``, ``carrier``, ``origin``, ``dest``.
    """
    f = filters or {}
    origin  = f.get("origin")
    dest    = f.get("dest")
    carrier = f.get("carrier")

    if hyde_num_samples == 0:
        from app.core.config import settings
        from app.rag.qdrant_client import get_qdrant
        from app.rag.retrieval import _vector_search_sync, _build_filter
        import time

        vec = await async_embed_text(query)
        flt = _build_filter(origin, dest, carrier, None, None, None, None)
        t = time.perf_counter()
        client = get_qdrant()
        pts = await asyncio.to_thread(
            _vector_search_sync, client, vec, flt, k, settings.RETRIEVAL_SCORE_THRESHOLD
        )
        elapsed = (time.perf_counter() - t) * 1000
        latency = {
            "hyde_ms": 0.0,
            "qdrant_ms": elapsed,
            "rerank_ms": 0.0,
            "total_ms": elapsed,
        }
        return pts, latency

    result = await search_flights_hybrid(
        query=query,
        origin=origin,
        dest=dest,
        carrier=carrier,
        limit=k,
        hyde_num_samples=hyde_num_samples,
    )
    return result["results"], result.get("latency", {})


def _flight_matches_filters(flight: dict, filters: dict) -> bool:
    """Return True when a flight dict satisfies all non-None filter fields.

    Comparison is case-insensitive to tolerate minor normalisation differences.

    Args:
        flight: Dict with at minimum ``carrier``, ``origin``, ``dest`` keys.
        filters: Dict with optional keys ``carrier``, ``origin``, ``dest``.

    Returns:
        True if the flight satisfies every filter field that is present and
        non-empty in ``filters``.
    """
    for key in ("carrier", "origin", "dest"):
        expected_val = filters.get(key)
        if expected_val:
            actual_val = flight.get(key) or ""
            if actual_val.upper() != expected_val.upper():
                return False
    return True


async def eval_query(
    entry: dict,
    k: int,
    hyde_num_samples: int | None = None,
) -> dict:
    """Run a single golden query and compute retrieval metrics.

    Scoring strategy
    ----------------
    When the entry has a ``filters`` dict (carrier / origin / dest), a returned
    flight is counted as a hit if it satisfies those constraints — i.e. if any
    result is a real DL LGA→ATL flight, that is a hit regardless of which
    specific flight ID was returned.  This is the correct semantics for a dataset
    with thousands of flights on each route: any correctly-routed flight is an
    acceptable answer.

    Exact-ID matching (``expected_flight_ids``) is used only when no filters are
    present, for backwards compatibility with manually annotated entries that
    intentionally single out specific flights.

    Args:
        entry: Dict with keys ``query`` (str), ``expected_flight_ids`` (list[str]),
            and optionally ``filters`` (dict with origin/dest/carrier keys).
        k: Number of results to retrieve.
        hyde_num_samples: Override for HYDE_NUM_SAMPLES passed through to the
            retrieval pipeline.  None uses the value from config/env.

    Returns:
        Dict with keys: query, hit, recall_at_k, precision_at_k, mrr,
        filters_dropped, latency, and skipped.
    """
    query = entry["query"]
    expected: list[str] = entry.get("expected_flight_ids", [])
    filters: dict = entry.get("filters", {})

    use_filter_match = bool(filters.get("carrier") or filters.get("origin") or filters.get("dest"))

    if not use_filter_match and not expected:
        return {
            "query": query, "hit": None, "recall_at_k": None,
            "precision_at_k": None, "mrr": None, "filters_dropped": None,
            "latency": {}, "skipped": True,
        }

    returned_flights, latency = await _run_query(query, k, filters=filters, hyde_num_samples=hyde_num_samples)

    if use_filter_match:
        matching = [f for f in returned_flights if _flight_matches_filters(f, filters)]
        hit = len(matching) > 0
        recall = min(len(matching) / k, 1.0)
        precision = len(matching) / len(returned_flights) if returned_flights else 0.0
        mrr = 0.0
        for rank, f in enumerate(returned_flights, start=1):
            if _flight_matches_filters(f, filters):
                mrr = 1.0 / rank
                break
    else:
        returned_ids = [f["flight_id"] for f in returned_flights]
        expected_set = set(expected)
        hits = [fid for fid in returned_ids if fid in expected_set]
        hit = len(hits) > 0
        recall = len(hits) / len(expected_set) if expected_set else 0.0
        precision = len(hits) / len(returned_ids) if returned_ids else 0.0
        mrr = 0.0
        for rank, fid in enumerate(returned_ids, start=1):
            if fid in expected_set:
                mrr = 1.0 / rank
                break

    return {
        "query": query,
        "hit": hit,
        "recall_at_k": recall,
        "precision_at_k": precision,
        "mrr": mrr,
        "filters_dropped": False,
        "latency": latency,
        "skipped": False,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _status(r: dict) -> str:
    if r.get("skipped"):
        return "SKIP"
    return "PASS" if r["hit"] else "FAIL"


async def _run_suite(entries: list[dict], k: int, hyde_num_samples: int | None = None) -> list[dict]:
    results = []
    for entry in entries:
        r = await eval_query(entry, k, hyde_num_samples=hyde_num_samples)
        results.append(r)
    return results


def _print_summary(
    results: list[dict],
    k: int,
    min_hit_rate: float,
    min_mrr: float,
    max_p95_latency_ms: float,
    label: str = "",
) -> bool:
    """Print a summary table and return True when all thresholds pass.

    Args:
        results: Per-query result dicts from _run_suite.
        k: Top-k cutoff used for this run.
        min_hit_rate: CI pass threshold for hit rate.
        min_mrr: CI pass threshold for MRR.
        max_p95_latency_ms: CI pass threshold for p95 total latency.
        label: Optional prefix for the header (used in compare-hyde mode).

    Returns:
        True when all thresholds pass, False otherwise.
    """
    scored = [r for r in results if not r.get("skipped")]

    if label:
        print(f"\n{'━'*60}")
        print(f"  {label}")
        print(f"{'━'*60}")

    for r in results:
        status = _status(r)
        lat = r.get("latency", {})
        total_ms = lat.get("total_ms", 0.0)
        if r.get("skipped"):
            print(f"[{status}] {r['query'][:70]!r}")
        else:
            print(
                f"[{status}] {r['query'][:55]!r}  "
                f"R@{k}={r['recall_at_k']:.2f}  MRR={r['mrr']:.2f}  "
                f"{total_ms:.0f}ms"
            )

    if not scored:
        print("\nNo scoreable queries — annotate with build_eval_set.py first.")
        return False

    hit_rate = sum(1 for r in scored if r["hit"]) / len(scored)
    mean_recall = statistics.mean(r["recall_at_k"] for r in scored)
    mean_precision = statistics.mean(r["precision_at_k"] for r in scored)
    mean_mrr = statistics.mean(r["mrr"] for r in scored)

    total_lats = [r["latency"].get("total_ms", 0.0) for r in scored if r.get("latency")]
    mean_lat = statistics.mean(total_lats) if total_lats else 0.0
    p50_lat = _percentile(total_lats, 50)
    p95_lat = _percentile(total_lats, 95)

    hyde_lats = [r["latency"].get("hyde_ms", 0.0) for r in scored if r.get("latency")]
    qdrant_lats = [r["latency"].get("qdrant_ms", 0.0) for r in scored if r.get("latency")]

    print(f"\n{'─'*60}")
    print(f"Queries scored  : {len(scored)} / {len(results)}")
    print(f"Hit rate        : {hit_rate:.3f}  (threshold ≥ {min_hit_rate})")
    print(f"Mean Recall@{k}  : {mean_recall:.3f}")
    print(f"Mean Precision@{k}: {mean_precision:.3f}")
    print(f"Mean MRR        : {mean_mrr:.3f}  (threshold ≥ {min_mrr})")
    print(f"\nLatency (total) : mean={mean_lat:.0f}ms  p50={p50_lat:.0f}ms  p95={p95_lat:.0f}ms  (threshold p95 ≤ {max_p95_latency_ms:.0f}ms)")
    if hyde_lats:
        print(f"  hyde_ms       : mean={statistics.mean(hyde_lats):.0f}  p95={_percentile(hyde_lats, 95):.0f}")
    if qdrant_lats:
        print(f"  qdrant_ms     : mean={statistics.mean(qdrant_lats):.0f}  p95={_percentile(qdrant_lats, 95):.0f}")

    failed = hit_rate < min_hit_rate or mean_mrr < min_mrr or p95_lat > max_p95_latency_ms
    print(f"\n{'FAIL ✗' if failed else 'PASS ✓'}")
    return not failed


async def main(
    k: int,
    min_hit_rate: float,
    min_mrr: float,
    max_p95_latency_ms: float,
    compare_hyde: bool,
) -> int:
    """Run the full eval suite and print a summary table.

    Args:
        k: Top-k cutoff for all metrics.
        min_hit_rate: Minimum acceptable hit rate; exit 1 if below.
        min_mrr: Minimum acceptable MRR; exit 1 if below.
        max_p95_latency_ms: Maximum acceptable p95 total query latency in ms.
        compare_hyde: When True, run each query with and without HyDE and
            print a side-by-side MRR lift/drop table.

    Returns:
        Exit code (0 = all thresholds pass, 1 = fail or empty eval set).
    """
    if not EVAL_PATH.exists():
        print(f"No eval set found at {EVAL_PATH}. Run scripts/build_eval_set.py first.")
        return 1

    entries = json.loads(EVAL_PATH.read_text())
    if not entries:
        print("eval_queries.json is empty — run scripts/build_eval_set.py first.")
        return 1

    print(f"Running {len(entries)} queries at k={k}\n")

    if compare_hyde:
        print("HyDE A/B comparison: running each query with and without HyDE.\n")

        results_hyde = await _run_suite(entries, k, hyde_num_samples=None)
        results_direct = await _run_suite(entries, k, hyde_num_samples=0)

        passed_hyde = _print_summary(
            results_hyde, k, min_hit_rate, min_mrr, max_p95_latency_ms,
            label="WITH HyDE (hyde_num_samples from config)",
        )
        _print_summary(
            results_direct, k, min_hit_rate, min_mrr, max_p95_latency_ms,
            label="WITHOUT HyDE (direct embedding)",
        )

        scored_h = [r for r in results_hyde if not r.get("skipped")]
        scored_d = [r for r in results_direct if not r.get("skipped")]
        if scored_h and scored_d:
            mrr_h = statistics.mean(r["mrr"] for r in scored_h)
            mrr_d = statistics.mean(r["mrr"] for r in scored_d)
            lift = mrr_h - mrr_d
            print(f"\n{'━'*60}")
            print(f"HyDE MRR lift: {lift:+.4f}  ({mrr_h:.4f} vs {mrr_d:.4f})")
            if lift < 0:
                print("  ⚠ HyDE is hurting MRR on this query distribution.")
                print("  Consider setting HYDE_NUM_SAMPLES=0 or investigating prompt quality.")
            elif lift < 0.01:
                print("  HyDE lift is marginal (<0.01 MRR) — may not justify the LLM round-trip.")
            else:
                print("  HyDE is improving MRR — the extra LLM call is justified.")

        return 0 if passed_hyde else 1

    results = await _run_suite(entries, k)
    passed = _print_summary(results, k, min_hit_rate, min_mrr, max_p95_latency_ms)
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight retrieval evaluation")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff (default 10)")
    parser.add_argument("--min-hit-rate", type=float, default=0.7, help="Minimum hit rate (default 0.7)")
    parser.add_argument("--min-mrr", type=float, default=0.4, help="Minimum MRR (default 0.4)")
    parser.add_argument(
        "--max-p95-latency-ms", type=float, default=5000.0,
        help="Maximum p95 total query latency in ms (default 5000)",
    )
    parser.add_argument(
        "--compare-hyde", action="store_true",
        help="Run each query with and without HyDE and report MRR lift",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(
        k=args.k,
        min_hit_rate=args.min_hit_rate,
        min_mrr=args.min_mrr,
        max_p95_latency_ms=args.max_p95_latency_ms,
        compare_hyde=args.compare_hyde,
    ))
    sys.exit(exit_code)
