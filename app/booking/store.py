from __future__ import annotations

import json
from typing import Any

from app.booking.db import SessionLocal
from app.booking.models import Booking


def _safe_json_loads(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s or "{}")
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def create_booking(flight_id: str, passenger_name: str, flight_snapshot: dict) -> dict:
    with SessionLocal() as s:
        b = Booking(
            flight_id=flight_id,
            passenger_name=passenger_name,
            flight_snapshot_json=json.dumps(flight_snapshot or {}, ensure_ascii=False),
        )
        s.add(b)
        s.commit()
        s.refresh(b)
        return {
            "booking_id": b.booking_id,
            "flight_id": b.flight_id,
            "passenger_name": b.passenger_name,
            "status": b.status,
            "created_at": b.created_at.isoformat(),
            "updated_at": b.updated_at.isoformat(),
            "flight_snapshot": flight_snapshot or {},
        }


def cancel_booking(booking_id: str) -> dict:
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
    with SessionLocal() as s:
        b = s.get(Booking, booking_id)
        if not b:
            return None
        snap = _safe_json_loads(b.flight_snapshot_json)
        return {
            "booking_id": b.booking_id,
            "flight_id": b.flight_id,
            "passenger_name": b.passenger_name,
            "status": b.status,
            "created_at": b.created_at.isoformat(),
            "updated_at": b.updated_at.isoformat(),
            "flight_snapshot": snap,
        }
