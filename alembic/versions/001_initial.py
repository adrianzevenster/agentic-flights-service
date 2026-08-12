"""initial schema

Revision ID: 001
Revises:
Create Date: 2025-08-12 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bookings",
        sa.Column("booking_id", sa.String(), nullable=False),
        sa.Column("flight_id", sa.String(), nullable=False),
        sa.Column("passenger_name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("flight_snapshot_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("booking_id"),
    )


def downgrade() -> None:
    op.drop_table("bookings")
