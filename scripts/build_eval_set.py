"""Interactive tool for building the retrieval golden set.

Runs each seed query against the live retrieval system, displays the top results,
and prompts you to mark which ones are correct.  Writes selections back to
data/eval_queries.json so eval_retrieval.py can score against them.

Structured filters
------------------
Each seed entry may specify explicit origin, dest, and carrier filters in
addition to the natural-language query.  When present, these are passed to
search_flights_hybrid exactly as the agentic tool dispatch would pass them
(the LLM extracts structured fields from the user's query before calling the
tool).  Displaying filter-qualified results during annotation produces a golden
set that reflects real agent behaviour rather than unfiltered semantic search.

The filters are also persisted in eval_queries.json under "filters" so that
eval_retrieval.py can replay the same call signature when scoring.

Usage:
    python scripts/build_eval_set.py
    python scripts/build_eval_set.py --k 10 --append
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

from app.rag.retrieval import search_flights_hybrid  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_queries.json"

_SEED_QUERIES: list[dict] = [
    {"query": "morning United flight from Newark to Chicago",
     "filters": {"carrier": "UA", "origin": "EWR", "dest": "ORD"}},
    {"query": "Delta flight from LaGuardia to Atlanta",
     "filters": {"carrier": "DL", "origin": "LGA", "dest": "ATL"}},
    {"query": "JetBlue morning flight from JFK to Fort Lauderdale",
     "filters": {"carrier": "B6", "origin": "JFK", "dest": "FLL"}},
    {"query": "American Airlines afternoon flight from LaGuardia to Dallas",
     "filters": {"carrier": "AA", "origin": "LGA", "dest": "DFW"}},
    {"query": "afternoon American Airlines flight from LaGuardia to Miami",
     "filters": {"carrier": "AA", "origin": "LGA", "dest": "MIA"}},
    {"query": "Southwest Airlines flight from Newark to Chicago Midway",
     "filters": {"carrier": "WN", "origin": "EWR", "dest": "MDW"}},
    {"query": "US Airways morning flight from Newark to Charlotte",
     "filters": {"carrier": "US", "origin": "EWR", "dest": "CLT"}},
    {"query": "evening Envoy Air flight from LaGuardia to Atlanta",
     "filters": {"carrier": "MQ", "origin": "LGA", "dest": "ATL"}},
    {"query": "United early morning departure from LaGuardia to Houston",
     "filters": {"carrier": "UA", "origin": "LGA", "dest": "IAH"}},
    {"query": "JetBlue afternoon flight from JFK to Orlando",
     "filters": {"carrier": "B6", "origin": "JFK", "dest": "MCO"}},
]


def _fmt_time(hhmm: int | None) -> str:
    if hhmm is None:
        return "    "
    h, m = divmod(int(hhmm), 100)
    return f"{h:02d}:{m:02d}"


async def annotate_query(entry: dict, k: int) -> dict:
    """Run a query with its structured filters, display results, and collect annotations.

    Args:
        entry: Dict with keys ``query`` (str) and optional ``filters`` (dict with
               origin, dest, carrier keys).  Filters are passed to search_flights_hybrid
               to replicate the call the agent tool layer would make.
        k: Number of results to display.

    Returns:
        Dict with keys ``query``, ``filters``, and ``expected_flight_ids`` (may be
        empty if the user marks none as correct or skips).
    """
    query = entry["query"]
    filters = entry.get("filters", {})

    print(f"\n{'═'*70}")
    print(f"Query: {query!r}")
    if filters:
        print(f"Filters: {filters}")
    print(f"{'═'*70}")

    result = await search_flights_hybrid(
        query=query,
        origin=filters.get("origin"),
        dest=filters.get("dest"),
        carrier=filters.get("carrier"),
        limit=k,
    )
    flights = result["results"]

    if not flights:
        print("  No results returned.")
        return {"query": query, "filters": filters, "expected_flight_ids": [], "notes": "no results"}

    for i, f in enumerate(flights):
        dep = _fmt_time(f.get("sched_dep_time"))
        arr = _fmt_time(f.get("sched_arr_time"))
        score = f.get("score", 0)
        print(
            f"  [{i+1:2d}] {f.get('carrier','?'):2s} {str(f.get('flight','?')):5s} "
            f"{f.get('origin','?')}->{f.get('dest','?')}  "
            f"{dep}->{arr}  "
            f"score={score:.3f}  id={f['flight_id'][:12]}…"
        )

    raw = input("\nEnter correct result numbers (e.g. '1 3'), 's' to skip, or Enter for none: ").strip()

    if raw.lower() == "s":
        return {"query": query, "filters": filters, "expected_flight_ids": [], "notes": "skipped"}

    selected_ids: list[str] = []
    for token in raw.split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(flights):
                selected_ids.append(flights[idx]["flight_id"])
        except ValueError:
            pass

    print(f"  Marked {len(selected_ids)} correct result(s).")
    return {"query": query, "filters": filters, "expected_flight_ids": selected_ids}


async def main(k: int, append: bool) -> None:
    """Annotate seed queries interactively and write to the eval file.

    Args:
        k: Number of results to show per query.
        append: When True, merge into an existing eval file; when False, replace it.
    """
    existing: list[dict] = []
    if append and EVAL_PATH.exists():
        existing = json.loads(EVAL_PATH.read_text())
        existing_queries = {e["query"] for e in existing}
        print(f"Loaded {len(existing)} existing entries.")
    else:
        existing_queries: set[str] = set()

    to_run = [e for e in _SEED_QUERIES if e["query"] not in existing_queries]
    if not to_run:
        print("All seed queries already annotated.")
        return

    print(f"\nAnnotating {len(to_run)} queries at k={k}.")
    print("Filters shown next to each query reflect what the agent would extract.")
    print("Mark results that are genuinely relevant for the query intent.\n")

    new_entries: list[dict] = []
    for entry in to_run:
        result = await annotate_query(entry, k)
        new_entries.append(result)

    all_entries = existing + new_entries
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(all_entries, indent=2))
    print(f"\nSaved {len(all_entries)} entries to {EVAL_PATH}")
    scoreable = sum(1 for e in all_entries if e.get("expected_flight_ids"))
    print(f"{scoreable} entries have expected IDs and will be scored by eval_retrieval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build retrieval golden set interactively")
    parser.add_argument("--k", type=int, default=10, help="Results to display per query")
    parser.add_argument("--append", action="store_true", help="Merge into existing eval file")
    args = parser.parse_args()
    asyncio.run(main(args.k, args.append))
