from __future__ import annotations

import logging
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from qdrant_client.http import models as qm

from app.core.config import settings
from app.rag.qdrant_client import get_qdrant
from app.rag.embeddings import embed_text, embed_texts
from app.rag.flight_doc import flight_doc

log = logging.getLogger(__name__)


def _stable_flight_id(row: dict) -> str:
    key = "|".join([
        str(row.get("year", "")),
        str(row.get("month", "")),
        str(row.get("day", "")),
        str(row.get("carrier", "")),
        str(row.get("flight", "")),
        str(row.get("origin", "")),
        str(row.get("dest", "")),
        str(row.get("sched_dep_time", "")),
    ])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _safe_int(v) -> int | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except Exception:
        return None


def ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection and payload indexes if they don't exist.

    Args:
        vector_size: Dimensionality of the embedding model's output.
    """
    q = get_qdrant()
    existing = {c.name for c in q.get_collections().collections}
    if settings.QDRANT_FLIGHTS_COLLECTION in existing:
        return

    q.create_collection(
        collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )

    for field, schema in [
        ("origin", qm.PayloadSchemaType.KEYWORD),
        ("dest", qm.PayloadSchemaType.KEYWORD),
        ("carrier", qm.PayloadSchemaType.KEYWORD),
        ("year", qm.PayloadSchemaType.INTEGER),
        ("month", qm.PayloadSchemaType.INTEGER),
        ("day", qm.PayloadSchemaType.INTEGER),
        ("flight", qm.PayloadSchemaType.INTEGER),
        ("sched_dep_time", qm.PayloadSchemaType.INTEGER),
    ]:
        try:
            q.create_payload_index(
                collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass

    log.info("Created collection '%s'.", settings.QDRANT_FLIGHTS_COLLECTION)


def ingest_csv(csv_path: Path, limit: int | None = None) -> int:
    """Embed and upsert flight records from a CSV into Qdrant.

    Args:
        csv_path: Path to the CSV file.
        limit: Row cap; None means all rows.

    Returns:
        Number of records upserted.
    """
    df = pd.read_csv(csv_path)
    for col in ("origin", "dest", "carrier"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    if limit:
        df = df.head(limit)

    records = df.to_dict(orient="records")
    if not records:
        return 0

    embed_batch = max(1, settings.INGEST_EMBED_BATCH_SIZE)
    upsert_batch = max(1, settings.INGEST_UPSERT_BATCH_SIZE)
    total_records = len(records)

    first_vec = embed_text(flight_doc(records[0]))
    ensure_collection(vector_size=len(first_vec))

    q = get_qdrant()
    pending: list[qm.PointStruct] = []
    total_upserted = 0

    for batch_start in range(0, total_records, embed_batch):
        batch = records[batch_start: batch_start + embed_batch]
        docs = [flight_doc(r) for r in batch]
        vecs = embed_texts(docs)

        for r, vec in zip(batch, vecs):
            payload = {
                "year": _safe_int(r.get("year")),
                "month": _safe_int(r.get("month")),
                "day": _safe_int(r.get("day")),
                "origin": r.get("origin"),
                "dest": r.get("dest"),
                "carrier": r.get("carrier"),
                "flight": _safe_int(r.get("flight")),
                "sched_dep_time": _safe_int(r.get("sched_dep_time")),
                "sched_arr_time": _safe_int(r.get("sched_arr_time")),
                "dep_time": _safe_int(r.get("dep_time")),
                "arr_time": _safe_int(r.get("arr_time")),
                "distance": _safe_int(r.get("distance")),
                "air_time": _safe_int(r.get("air_time")),
                "tailnum": r.get("tailnum"),
            }
            pending.append(qm.PointStruct(id=_stable_flight_id(r), vector=vec, payload=payload))

            if len(pending) >= upsert_batch:
                q.upsert(collection_name=settings.QDRANT_FLIGHTS_COLLECTION, points=pending)
                total_upserted += len(pending)
                pending = []

        pct = min(100, int((batch_start + len(batch)) / total_records * 100))
        log.info("Embedded %d/%d records (%d%%).", batch_start + len(batch), total_records, pct)

    if pending:
        q.upsert(collection_name=settings.QDRANT_FLIGHTS_COLLECTION, points=pending)
        total_upserted += len(pending)

    log.info("Ingest complete: %d records upserted.", total_upserted)
    return total_upserted


def ingest_archive(zip_path: Path, limit: int | None = None) -> int:
    """Extract the first CSV from a zip archive and ingest it.

    Uses a temporary directory that is cleaned up regardless of outcome.

    Args:
        zip_path: Path to the zip archive.
        limit: Row cap passed through to ingest_csv; None means all rows.

    Returns:
        Number of records upserted.

    Raises:
        ValueError: If the archive contains no CSV files.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in archive.")
        name = csv_names[0]
        log.info("Extracting '%s' from archive.", name)
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract(name, path=tmpdir)
            return ingest_csv(Path(tmpdir) / name, limit=limit)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=str, required=True)
    p.add_argument("--limit", type=int, default=0, help="Row cap (0 = all)")
    args = p.parse_args()

    n = ingest_archive(Path(args.archive), limit=args.limit or None)
    print(f"Ingested {n} records into '{settings.QDRANT_FLIGHTS_COLLECTION}'.")
