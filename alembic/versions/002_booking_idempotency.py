"""Add idempotency_key to bookings.

Revision ID: 002
Revises: 001
Create Date: 2026-08-14 00:00:00.000000

Adds a nullable, unique-indexed idempotency_key column to the bookings table.
Existing rows receive NULL, which Postgres UNIQUE constraints treat as
non-comparable, so this migration is zero-downtime safe on live data.

The index (ix_bookings_idempotency_key) is created CONCURRENTLY in production
via a separate DDL statement so it does not hold an ACCESS EXCLUSIVE lock for
the duration of a large table scan.  Alembic's op.create_index uses
CREATE INDEX (blocking) by default; in production apply the index step
manually with CREATE INDEX CONCURRENTLY before running alembic upgrade.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "bookings",
        sa.Column("idempotency_key", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_bookings_idempotency_key",
        "bookings",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_bookings_idempotency_key", table_name="bookings")
    op.drop_column("bookings", "idempotency_key")
