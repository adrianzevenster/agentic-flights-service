"""Synchronous data-access layer for flight bookings.

Idempotency protocol
--------------------
create_booking accepts an optional idempotency_key.  When provided:

1. A SELECT is issued first.  If a CONFIRMED booking with that key exists, it is
   returned immediately without a new INSERT — the operation is idempotent.
2. If no matching booking is found, an INSERT is attempted.
3. A concurrent request that wins the INSERT race will cause our attempt to raise
   sqlalchemy.exc.IntegrityError (UNIQUE violation on idempotency_key).  The
   session is rolled back and we fall through to a final SELECT, which must now
   find the winning row.

This three-step pattern (read → insert → on-conflict read) is the standard
approach for idempotent inserts in Postgres without advisory locks.  It is safe
under any transaction isolation level because the UNIQUE constraint is enforced
by the database, not by application-level locking.

All functions run synchronously and are called via asyncio.to_thread from the
async tool dispatch layer, keeping the SQLAlchemy Session off the event loop.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.exc import IntegrityError

from app.booking.db import SessionLocal
from app.booking.models import Booking


def _safe_json_loads(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _booking_to_dict(b: Booking, snapshot: dict | None = None) -> dict:
    if snapshot is None:
        snapshot = _safe_json_loads(b.flight_snapshot_json)
    return {
        "booking_id": b.booking_id,
        "flight_id": b.flight_id,
        "passenger_name": b.passenger_name,
        "status": b.status,
        "created_at": b.created_at.isoformat(),
        "updated_at": b.updated_at.isoformat(),
        "flight_snapshot": snapshot,
    }


def create_booking(
    flight_id: str,
    passenger_name: str,
    flight_snapshot: dict,
    idempotency_key: str | None = None,
) -> dict:
    """Insert a new booking or return the existing one for the given idempotency key.

    Args:
        flight_id: UUID of the flight in Qdrant.
        passenger_name: Full name of the passenger as provided by the user.
        flight_snapshot: Subset of flight payload fields snapshotted at booking time.
        idempotency_key: Caller-supplied deduplication key (UUID5 recommended).
            When None the INSERT is unconditional; duplicate-prevention is the
            caller's responsibility.

    Returns:
        Booking dict with booking_id, flight_id, passenger_name, status,
        created_at, updated_at, and flight_snapshot.
    """
    with SessionLocal() as s:
        if idempotency_key:
            existing = (
                s.query(Booking)
                .filter_by(idempotency_key=idempotency_key, status="CONFIRMED")
                .first()
            )
            if existing:
                return _booking_to_dict(existing, snapshot=flight_snapshot)

        b = Booking(
            flight_id=flight_id,
            passenger_name=passenger_name,
            idempotency_key=idempotency_key,
            flight_snapshot_json=json.dumps(flight_snapshot or {}, ensure_ascii=False),
        )
        try:
            s.add(b)
            s.commit()
            s.refresh(b)
            return _booking_to_dict(b, snapshot=flight_snapshot)
        except IntegrityError:
            s.rollback()
            winner = (
                s.query(Booking)
                .filter_by(idempotency_key=idempotency_key)
                .first()
            )
            return _booking_to_dict(winner, snapshot=flight_snapshot)


def cancel_booking(booking_id: str) -> dict:
    """Cancel a booking by ID.

    Args:
        booking_id: Primary key of the booking row.

    Returns:
        Dict with booking_id, status, and updated_at; or an error dict if not found.
    """
    with SessionLocal() as s:
        b = s.get(Booking, booking_id)
        if not b:
            return {"error": "BOOKING_NOT_FOUND", "booking_id": booking_id}
        b.status = "CANCELLED"
        s.commit()
        s.refresh(b)
        return {
            "booking_id": b.booking_id,
            "status": b.status,
            "updated_at": b.updated_at.isoformat(),
        }


def get_booking(booking_id: str) -> dict | None:
    """Retrieve a booking's current state by ID.

    Args:
        booking_id: Primary key of the booking row.

    Returns:
        Full booking dict, or None if the ID does not exist.
    """
    with SessionLocal() as s:
        b = s.get(Booking, booking_id)
        if not b:
            return None
        return _booking_to_dict(b)
