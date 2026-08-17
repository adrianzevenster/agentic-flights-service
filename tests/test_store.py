"""Unit tests for booking store idempotency logic.

Tests use SQLite in-memory via a patched DATABASE_URL so no Postgres instance
is required.  The idempotency column and unique constraint are exercised against
a real (in-process) database to validate the three-step insert protocol:
read → insert → on-IntegrityError read.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


@pytest.fixture(scope="module")
def sqlite_session_factory():
    """In-memory SQLite engine with the full bookings schema applied."""
    from app.booking.models import Base
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def patch_session(monkeypatch, sqlite_session_factory):
    """Redirect store.SessionLocal to the in-memory SQLite factory."""
    import app.booking.store as store_mod
    monkeypatch.setattr(store_mod, "SessionLocal", sqlite_session_factory)


def _flight_snapshot() -> dict:
    return {"carrier": "DL", "origin": "ATL", "dest": "JFK", "flight": 100}


def _random_flight_id() -> str:
    return str(uuid.uuid4())


# ─── create_booking — baseline ────────────────────────────────────────────────

def test_create_booking_returns_booking_id():
    from app.booking.store import create_booking
    result = create_booking(
        flight_id=_random_flight_id(),
        passenger_name="Alice Smith",
        flight_snapshot=_flight_snapshot(),
    )
    assert "booking_id" in result
    assert result["status"] == "CONFIRMED"


def test_create_booking_stores_passenger_name():
    from app.booking.store import create_booking
    result = create_booking(
        flight_id=_random_flight_id(),
        passenger_name="Bob Jones",
        flight_snapshot=_flight_snapshot(),
    )
    assert result["passenger_name"] == "Bob Jones"


def test_create_booking_snapshot_preserved():
    from app.booking.store import create_booking
    snap = {"carrier": "UA", "origin": "SFO", "dest": "DEN"}
    result = create_booking(
        flight_id=_random_flight_id(),
        passenger_name="Carol White",
        flight_snapshot=snap,
    )
    assert result["flight_snapshot"]["carrier"] == "UA"


# ─── idempotency ──────────────────────────────────────────────────────────────

def test_idempotent_create_returns_same_booking_id():
    from app.booking.store import create_booking
    fid = _random_flight_id()
    key = str(uuid.uuid4())
    snap = _flight_snapshot()

    first = create_booking(fid, "Dave Brown", snap, idempotency_key=key)
    second = create_booking(fid, "Dave Brown", snap, idempotency_key=key)

    assert first["booking_id"] == second["booking_id"]


def test_idempotent_create_does_not_duplicate_rows():
    from app.booking import store as store_mod
    from app.booking.store import create_booking
    from app.booking.models import Booking

    fid = _random_flight_id()
    key = str(uuid.uuid4())
    snap = _flight_snapshot()

    create_booking(fid, "Eve Green", snap, idempotency_key=key)
    create_booking(fid, "Eve Green", snap, idempotency_key=key)

    with store_mod.SessionLocal() as s:
        count = s.query(Booking).filter_by(idempotency_key=key).count()
    assert count == 1


def test_different_keys_create_distinct_bookings():
    from app.booking.store import create_booking
    fid = _random_flight_id()
    snap = _flight_snapshot()

    b1 = create_booking(fid, "Frank Lee", snap, idempotency_key=str(uuid.uuid4()))
    b2 = create_booking(fid, "Frank Lee", snap, idempotency_key=str(uuid.uuid4()))

    assert b1["booking_id"] != b2["booking_id"]


def test_no_idempotency_key_allows_duplicates():
    from app.booking.store import create_booking
    fid = _random_flight_id()
    snap = _flight_snapshot()

    b1 = create_booking(fid, "Grace Kim", snap, idempotency_key=None)
    b2 = create_booking(fid, "Grace Kim", snap, idempotency_key=None)

    assert b1["booking_id"] != b2["booking_id"]


# ─── cancel / get ─────────────────────────────────────────────────────────────

def test_cancel_booking_changes_status():
    from app.booking.store import create_booking, cancel_booking
    b = create_booking(_random_flight_id(), "Henry Park", _flight_snapshot())
    result = cancel_booking(b["booking_id"])
    assert result["status"] == "CANCELLED"


def test_cancel_booking_not_found_returns_error():
    from app.booking.store import cancel_booking
    result = cancel_booking("nonexistent-id")
    assert result.get("error") == "BOOKING_NOT_FOUND"


def test_get_booking_returns_full_record():
    from app.booking.store import create_booking, get_booking
    b = create_booking(_random_flight_id(), "Iris Wang", _flight_snapshot())
    fetched = get_booking(b["booking_id"])
    assert fetched is not None
    assert fetched["passenger_name"] == "Iris Wang"
    assert fetched["status"] == "CONFIRMED"


def test_get_booking_missing_returns_none():
    from app.booking.store import get_booking
    assert get_booking("does-not-exist") is None


# ─── tool-layer idempotency key derivation ────────────────────────────────────

def test_booking_idempotency_key_is_deterministic():
    from app.api.tools import _booking_idempotency_key
    k1 = _booking_idempotency_key("flight-abc", "Alice Smith")
    k2 = _booking_idempotency_key("flight-abc", "Alice Smith")
    assert k1 == k2


def test_booking_idempotency_key_normalises_case_and_whitespace():
    from app.api.tools import _booking_idempotency_key
    k1 = _booking_idempotency_key("flight-abc", "Alice Smith")
    k2 = _booking_idempotency_key("flight-abc", "  ALICE SMITH  ")
    assert k1 == k2


def test_booking_idempotency_key_differs_on_different_passenger():
    from app.api.tools import _booking_idempotency_key
    k1 = _booking_idempotency_key("flight-abc", "Alice Smith")
    k2 = _booking_idempotency_key("flight-abc", "Bob Jones")
    assert k1 != k2


def test_booking_idempotency_key_differs_on_different_flight():
    from app.api.tools import _booking_idempotency_key
    k1 = _booking_idempotency_key("flight-abc", "Alice Smith")
    k2 = _booking_idempotency_key("flight-xyz", "Alice Smith")
    assert k1 != k2
