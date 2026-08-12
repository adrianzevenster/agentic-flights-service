from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

log = logging.getLogger(__name__)

# alembic.ini lives at the project root (alongside requirements.txt).
_ALEMBIC_CFG_PATH = Path(__file__).parent.parent.parent / "alembic.ini"


def init_db() -> None:
    """Run Alembic migrations to bring the schema to head.

    If the bookings table already exists but alembic_version does not
    (i.e. the DB was created by an older create_all call), we stamp the
    current revision as head rather than attempting a re-create.
    """
    from app.booking.db import engine

    cfg = Config(str(_ALEMBIC_CFG_PATH))

    with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
        if "bookings" in tables and "alembic_version" not in tables:
            log.info("Existing schema without migration tracking — stamping head.")
            command.stamp(cfg, "head")
            return

    log.info("Running Alembic migrations.")
    command.upgrade(cfg, "head")
