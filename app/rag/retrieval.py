from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from qdrant_client.http import models as qm

from app.core.config import settings
from app.rag.qdrant_client import get_qdrant
from app.rag.embeddings import async_embed_text
from app.rag.flight_doc import HYDE_PROMPT

log = logging.getLogger(__name__)

_FLIGHT_FIELDS = [
    "flight_id", "carrier", "flight", "origin", "dest",
    "sched_dep_time", "sched_arr_time", "dep_time", "arr_time",
    "distance", "air_time", "year", "month", "day",
]

# Simple LRU-style cache: HyDE calls are expensive (full LLM inference per query).
# 256 slots covers typical session traffic without unbounded growth.
_hyde_cache: dict[str, str] = {}
_HYDE_CACHE_MAX = 256


async def _hyde_query(query: str) -> str:
    """Generate a hypothetical flight document for HyDE retrieval, with caching.

    Args:
        query: Natural-language flight search query.

    Returns:
        Synthetic flight-record string, or the original query on LLM failure.
    """
    if query in _hyde_cache:
        return _hyde_cache[query]

    result = await _hyde_query_uncached(query)

    if len(_hyde_cache) >= _HYDE_CACHE_MAX:
        _hyde_cache.pop(next(iter(_hyde_cache)))
    _hyde_cache[query] = result
    return result


async def _hyde_query_uncached(query: str) -> str:
    from app.core.http_client import get_http_client
    try:
        r = await get_http_client().post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": settings.OLLAMA_CHAT_MODEL,
                "messages": [{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
                "stream": False,
            },
        )
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "").strip()
        if not raw:
            return query
        # Extract the first line that contains a route arrow — the model sometimes
        # prepends explanation text before the record.
        for line in raw.splitlines():
            line = line.strip()
            if "->" in line and len(line) > 20:
                return line
        return raw.splitlines()[-1].strip() or query
    except Exception:
        log.warning("HyDE query failed for %r, falling back to raw query", query)
        return query


def _build_filter(
    origin: str | None,
    dest: str | None,
    carrier: str | None,
    year: int | None,
    month: int | None,
    day: int | None,
    flight: int | None,
) -> qm.Filter | None:
    """Build a Qdrant payload filter from the provided optional fields.

    Args:
        origin: IATA origin code.
        dest: IATA destination code.
        carrier: Carrier code.
        year: Year.
        month: Month.
        day: Day of month.
        flight: Flight number.

    Returns:
        A Qdrant Filter, or None if no constraints are provided.
    """
    must = []
    if origin:
        must.append(qm.FieldCondition(key="origin", match=qm.MatchValue(value=origin.upper())))
    if dest:
        must.append(qm.FieldCondition(key="dest", match=qm.MatchValue(value=dest.upper())))
    if carrier:
        must.append(qm.FieldCondition(key="carrier", match=qm.MatchValue(value=carrier.upper())))
    if year is not None:
        must.append(qm.FieldCondition(key="year", match=qm.MatchValue(value=int(year))))
    if month is not None:
        must.append(qm.FieldCondition(key="month", match=qm.MatchValue(value=int(month))))
    if day is not None:
        must.append(qm.FieldCondition(key="day", match=qm.MatchValue(value=int(day))))
    if flight is not None:
        must.append(qm.FieldCondition(key="flight", match=qm.MatchValue(value=int(flight))))
    return qm.Filter(must=must) if must else None


def _point_to_dict(p: Any) -> dict:
    payload = getattr(p, "payload", None) or {}
    return {
        "flight_id": str(getattr(p, "id", "")),
        "score": float(p.score) if getattr(p, "score", None) is not None else None,
        **{k: payload.get(k) for k in _FLIGHT_FIELDS if k != "flight_id"},
    }


def _vector_search_sync(client: Any, vec: list[float], flt: qm.Filter | None, limit: int) -> list[dict]:
    # Runs inside asyncio.to_thread — keep it purely sync.
    results = client.query_points(
        collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
        query=vec,
        limit=limit,
        query_filter=flt,
        score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
        with_payload=True,
        with_vectors=False,
    )
    return [_point_to_dict(p) for p in results.points]


async def search_flights_hybrid(
    query: str,
    origin: Optional[str] = None,
    dest: Optional[str] = None,
    carrier: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    day: Optional[int] = None,
    flight: Optional[int] = None,
    limit: int = 10,
) -> dict:
    """Search flights using HyDE-expanded embeddings with hard-filter fallback.

    Args:
        query: Natural-language description of the desired flight.
        origin: IATA origin code filter.
        dest: IATA destination code filter.
        carrier: Carrier code filter.
        year: Year filter.
        month: Month filter.
        day: Day-of-month filter.
        flight: Flight number filter.
        limit: Maximum results to return.

    Returns:
        Dict with keys:
            results (list[dict]): Matched flight records, scored and sorted.
            filters_dropped (bool): True when the filtered query returned no
                results and was retried without filters.
    """
    client = get_qdrant()
    flt = _build_filter(origin, dest, carrier, year, month, day, flight)

    hyde_doc = await _hyde_query(query)
    vec = await async_embed_text(hyde_doc)

    points = await asyncio.to_thread(_vector_search_sync, client, vec, flt, limit)

    if not points and flt is not None:
        points = await asyncio.to_thread(_vector_search_sync, client, vec, None, limit)
        return {"results": points, "filters_dropped": True}

    return {"results": points, "filters_dropped": False}


async def get_flight_by_id(flight_id: str) -> dict | None:
    """Retrieve a single flight record by its UUID.

    Args:
        flight_id: UUID string of the flight point in Qdrant.

    Returns:
        Dict of flight fields with `flight_id` injected, or None if not found.
    """
    client = get_qdrant()
    pts = await asyncio.to_thread(
        lambda: client.retrieve(
            collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
            ids=[flight_id],
            with_payload=True,
            with_vectors=False,
        )
    )
    if not pts:
        return None
    p = pts[0]
    payload = p.payload or {}
    payload["flight_id"] = str(p.id)
    return payload
