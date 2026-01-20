from pathlib import Path

from app.core.config import settings
from app.ingest.ingest_archive import ingest_archive, ingest_csv


def main(path: str) -> None:
    p = Path(path)
    limit = settings.INGEST_LIMIT_DEFAULT or None

    if p.suffix.lower() == ".zip":
        n = ingest_archive(p, limit=limit)
    else:
        n = ingest_csv(p, limit=limit)

    print(f"Ingested {n} flight records into Qdrant collection '{settings.QDRANT_FLIGHTS_COLLECTION}'.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.ingest.ingest_from_archive <path_to_zip_or_csv>")
        raise SystemExit(2)

    main(sys.argv[1])
