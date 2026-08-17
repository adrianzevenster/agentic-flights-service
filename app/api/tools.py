"""Tool handler functions called by the agentic loop in chat.py.

Each function validates its kwargs with the corresponding Pydantic schema,
executes the underlying retrieval or booking operation, and returns a dict the
model can interpret.  Errors are returned as structured dicts rather than raised
so the model always receives a response it can relay to the user.

Idempotency
-----------
tool_create_booking derives an idempotency_key from (flight_id, normalised
passenger_name) using UUID5.  This key is passed to store.create_booking, which
will return the existing CONFIRMED booking if the same key is submitted twice
rather than creating a duplicate.  The key is deterministic across processes and
restarts; the UUID5 namespace is the standard URL namespace, which has no special
semantic meaning here — it is just a fixed salt for collision resistance.
"""
from __future__ import annotations

import asyncio
import uuid

from app.api.schemas import SearchFlightsIn, CreateBookingIn, CancelBookingIn, GetBookingIn
from app.rag.retrieval import search_flights_hybrid, get_flight_by_id
from app.booking import store
from app.core.config import settings

_SNAPSHOT_KEYS = [
    "flight_id", "carrier", "flight", "origin", "dest",
    "sched_dep_time", "sched_arr_time", "dep_time", "arr_time",
    "year", "month", "day", "distance", "air_time",
]

_IDEMPOTENCY_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _booking_idempotency_key(flight_id: str, passenger_name: str) -> str:
    """Derive a stable UUID5 key from (flight_id, normalised passenger_name).

    Args:
        flight_id: UUID of the Qdrant flight point.
        passenger_name: Raw passenger name as provided by the user.

    Returns:
        UUID5 string suitable for use as an idempotency key.
    """
    canonical = f"{flight_id}|{passenger_name.strip().lower()}"
    return str(uuid.uuid5(_IDEMPOTENCY_NAMESPACE, canonical))


async def tool_search_flights(**kwargs) -> dict:
    """Execute a semantic flight search and return scored results.

    Args:
        **kwargs: Validated by SearchFlightsIn (query, origin, dest, carrier, limit,
                  month_min, month_max, time_of_day).

    Returns:
        Dict with keys ``results`` (list of flight dicts) and ``filters_dropped`` (bool).
    """
    params = SearchFlightsIn.model_validate(kwargs)
    out = await search_flights_hybrid(
        query=params.query,
        origin=params.origin,
        dest=params.dest,
        carrier=params.carrier,
        month_min=params.month_min,
        month_max=params.month_max,
        time_of_day=params.time_of_day,
        limit=min(params.limit, settings.MAX_TOOL_RESULTS),
    )
    return {"results": out["results"], "filters_dropped": out["filters_dropped"]}


async def tool_get_flight_details(**kwargs) -> dict:
    """Look up full payload for a single flight by UUID.

    Args:
        **kwargs: Must include ``flight_id`` (str UUID).

    Returns:
        Dict with key ``flight`` containing all payload fields, or an error dict.
    """
    flight_id = kwargs.get("flight_id")
    if not flight_id:
        return {"error": "MISSING_FLIGHT_ID"}
    flight = await get_flight_by_id(str(flight_id))
    if not flight:
        return {"error": "FLIGHT_NOT_FOUND", "flight_id": str(flight_id)}
    return {"flight": flight}


async def tool_create_booking(**kwargs) -> dict:
    """Create a booking after verifying the flight exists, with idempotency protection.

    Args:
        **kwargs: Validated by CreateBookingIn (flight_id, passenger_name).

    Returns:
        Dict with key ``booking`` on success, or an error dict with a user-facing
        ``message`` if the flight_id is not found in Qdrant.
    """
    params = CreateBookingIn.model_validate(kwargs)
    flight = await get_flight_by_id(params.flight_id)
    if not flight:
        return {
            "error": "FLIGHT_NOT_FOUND",
            "flight_id": params.flight_id,
            "message": "That flight_id does not exist. Please search again and pick a flight_id from the results.",
        }
    snapshot = {k: flight.get(k) for k in _SNAPSHOT_KEYS if k in flight}
    idem_key = _booking_idempotency_key(params.flight_id, params.passenger_name)
    booking = await asyncio.to_thread(
        store.create_booking,
        flight_id=params.flight_id,
        passenger_name=params.passenger_name,
        flight_snapshot=snapshot,
        idempotency_key=idem_key,
    )
    return {"booking": booking}


async def tool_cancel_booking(**kwargs) -> dict:
    """Cancel a booking by ID.

    Args:
        **kwargs: Validated by CancelBookingIn (booking_id).

    Returns:
        Dict with cancellation result, or an error dict.
    """
    params = CancelBookingIn.model_validate(kwargs)
    out = await asyncio.to_thread(store.cancel_booking, booking_id=params.booking_id)
    if out.get("error"):
        return out
    return {"result": out}


async def tool_get_booking(**kwargs) -> dict:
    """Retrieve a booking's current status and snapshot by ID.

    Args:
        **kwargs: Validated by GetBookingIn (booking_id).

    Returns:
        Dict with key ``booking``, or an error dict if not found.
    """
    params = GetBookingIn.model_validate(kwargs)
    b = await asyncio.to_thread(store.get_booking, params.booking_id)
    if not b:
        return {"error": "BOOKING_NOT_FOUND", "booking_id": params.booking_id}
    return {"booking": b}
