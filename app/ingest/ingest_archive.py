from __future__ import annotations

import zipfile
from pathlib import Path
import uuid

import pandas as pd
from qdrant_client.http import models as qm

from app.core.config import settings
from app.rag.qdrant_client import get_qdrant
from app.rag.embeddings import embed_text, embed_texts


def _stable_flight_id(row: dict) -> str:
    key = "|".join(
        [
            str(row.get("year", "")),
            str(row.get("month", "")),
            str(row.get("day", "")),
            str(row.get("carrier", "")),
            str(row.get("flight", "")),
            str(row.get("origin", "")),
            str(row.get("dest", "")),
            str(row.get("sched_dep_time", "")),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _flight_doc(row: dict) -> str:
    return (
        f"{row.get('carrier')} flight {row.get('flight')} "
        f"{row.get('origin')}->{row.get('dest')} "
        f"date {row.get('year')}-{row.get('month')}-{row.get('day')} "
        f"sched {row.get('sched_dep_time')}->{row.get('sched_arr_time')} "
        f"actual {row.get('dep_time')}->{row.get('arr_time')} "
        f"dist {row.get('distance')} air {row.get('air_time')} "
        f"tail {row.get('tailnum')}"
    )


def ensure_collection(vector_size: int) -> None:
    q = get_qdrant()
    existing = [c.name for c in q.get_collections().collections]
    if settings.QDRANT_FLIGHTS_COLLECTION in existing:
        return

    q.create_collection(
        collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )

    index_specs = [
        ("origin", qm.PayloadSchemaType.KEYWORD),
        ("dest", qm.PayloadSchemaType.KEYWORD),
        ("carrier", qm.PayloadSchemaType.KEYWORD),
        ("year", qm.PayloadSchemaType.INTEGER),
        ("month", qm.PayloadSchemaType.INTEGER),
        ("day", qm.PayloadSchemaType.INTEGER),
        ("flight", qm.PayloadSchemaType.INTEGER),
        ("sched_dep_time", qm.PayloadSchemaType.INTEGER),
    ]

    for field, schema in index_specs:
        try:
            q.create_payload_index(
                collection_name=settings.QDRANT_FLIGHTS_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass


def _safe_int(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def ingest_csv(csv_path: Path, limit: int | None = None) -> int:
    df = pd.read_csv(csv_path)

    for c in ["origin", "dest", "carrier"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper()

    if limit:
        df = df.head(limit)

    records = df.to_dict(orient="records")
    if not records:
        return 0

    EMBED_BATCH_SIZE = max(1, int(settings.INGEST_EMBED_BATCH_SIZE))
    UPSERT_BATCH_SIZE = max(1, int(settings.INGEST_UPSERT_BATCH_SIZE))

    first_vec = embed_text(_flight_doc(records[0]))
    ensure_collection(vector_size=len(first_vec))

    q = get_qdrant()

    total = 0
    pending_points: list[qm.PointStruct] = []

    for i in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[i : i + EMBED_BATCH_SIZE]
        docs = [_flight_doc(r) for r in batch]
        vecs = embed_texts(docs)

        for r, vec in zip(batch, vecs):
            fid = _stable_flight_id(r)
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
            pending_points.append(qm.PointStruct(id=fid, vector=vec, payload=payload))

            if len(pending_points) >= UPSERT_BATCH_SIZE:
                q.upsert(collection_name=settings.QDRANT_FLIGHTS_COLLECTION, points=pending_points)
                total += len(pending_points)
                pending_points = []

    if pending_points:
        q.upsert(collection_name=settings.QDRANT_FLIGHTS_COLLECTION, points=pending_points)
        total += len(pending_points)

    return total


def ingest_archive(zip_path: Path, limit: int | None = None) -> int:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in archive.")
        name = csv_names[0]

        z.extract(name, path="/tmp")
        extracted = Path("/tmp") / name

    return ingest_csv(extracted, limit=limit)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=str, required=True, help="Path to archive.zip")
    p.add_argument("--limit", type=int, default=0, help="Optional cap for demo speed")
    args = p.parse_args()

    n = ingest_archive(Path(args.archive), limit=args.limit or None)
    print(f"Ingested {n} flight records into Qdrant collection '{settings.QDRANT_FLIGHTS_COLLECTION}'.")
