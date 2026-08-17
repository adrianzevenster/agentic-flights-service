"""SQLAlchemy ORM models for the booking subsystem.

Idempotency
-----------
The idempotency_key column carries a UUID5 derived server-side from
(flight_id, normalised passenger_name).  A UNIQUE constraint on this column
turns duplicate INSERT attempts — which the agentic loop can produce when it
retries a tool call after a transient error — into a read-then-return of the
existing booking rather than a double-charge.  Postgres treats NULL as not
equal to NULL in UNIQUE constraints, so rows created without an idempotency key
(e.g., direct REST calls during testing) accumulate freely.

flight_snapshot_json
--------------------
The flight payload is snapshotted at booking time so that cancellations and
status lookups remain coherent even after a Qdrant re-index or payload schema
migration.  The snapshot is intentionally denormalised: query correctness is
more important than storage efficiency for booking records.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Booking(Base):
    __tablename__ = "bookings"

    booking_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    flight_id: Mapped[str] = mapped_column(String, nullable=False)
    passenger_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="CONFIRMED")
    idempotency_key: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )

    flight_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now,
    )
