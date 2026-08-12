"""Quick smoke-test for HyDE query generation.

Checks that _hyde_query produces a plausible flight-record document for
several natural-language queries, and that the output structurally
resembles the indexed document format.

Usage:
    OLLAMA_CHAT_MODEL=llama3.1:latest python scripts/test_hyde.py
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_CHAT_MODEL", "llama3.1:latest")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service")

from app.rag.retrieval import _hyde_query  # noqa: E402

QUERIES = [
    "morning flight from New York to Los Angeles",
    "Delta airlines flight to Chicago in January",
    "short domestic hop under 500 miles",
    "late night transatlantic route",
]

def _looks_like_flight_record(doc: str) -> bool:
    iata = r"(?:\([A-Z]{3}\)|[A-Z]{3})"
    has_route = bool(re.search(rf"{iata}\s*-+>?\s*.{{0,20}}{iata}", doc))
    has_number = bool(re.search(r"\d{3,5}", doc))
    return has_route and has_number


def check(query: str) -> bool:
    t0 = time.monotonic()
    doc = _hyde_query(query)
    elapsed = time.monotonic() - t0

    is_fallback = doc == query
    looks_like_record = _looks_like_flight_record(doc)

    status = "PASS" if (not is_fallback and looks_like_record) else ("FALLBACK" if is_fallback else "WEAK")
    print(f"\n[{status}] {elapsed:.1f}s")
    print(f"  query : {query}")
    print(f"  hyde  : {doc}")
    return status == "PASS"


if __name__ == "__main__":
    results = [check(q) for q in QUERIES]
    passed = sum(results)
    print(f"\n{passed}/{len(QUERIES)} passed")
    sys.exit(0 if passed == len(QUERIES) else 1)
