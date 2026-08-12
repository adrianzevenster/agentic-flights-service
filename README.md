# Flight Service (Ollama + Qdrant + Postgres + Streamlit)

This repo is a minimal, production-lean flight enquiry + booking system.

- **Ollama**: local LLM + embeddings
- **Qdrant**: semantic flight search index
- **Postgres**: transactional store for bookings
- **FastAPI**: backend (tool-executing orchestrator)
- **Streamlit**: chat UI

## Quick start

### 1) Start services

```bash
docker compose up -d --build
```

### 2) Pull Ollama models

Inside the `ollama` container:

```bash
docker exec -it $(docker ps -qf name=ollama) ollama pull llama3.1:8b
docker exec -it $(docker ps -qf name=ollama) ollama pull nomic-embed-text
```

### 3) Ingest flights into Qdrant

You provided `archive.zip` (contains `flights.csv`). Mount/copy it into the API container or run ingestion locally.

**Local ingestion (recommended while developing):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export OLLAMA_BASE_URL=http://localhost:11434
export QDRANT_URL=http://localhost:6333

python -m app.ingest.ingest_archive --archive /mnt/data/archive.zip --limit 0
```

> The ingestion embeds each record using Ollama. For a first run you can set `--limit 5000` to speed it up.

### 4) Open the UI

- Streamlit: http://localhost:8501
- API health: http://localhost:8000/readyz

## How booking works

Bookings are **tool-driven**:
- The LLM only proposes one of a small set of tools (search, details, create booking, cancel booking, get booking).
- The backend validates the tool schema (Pydantic) and executes it.
- Bookings are stored in Postgres.

## Notes

- Qdrant holds flight records for retrieval. Postgres is used for transactional data.
- For production: add auth, rate limiting, structured logging, Alembic migrations, and a proper evaluation harness.
