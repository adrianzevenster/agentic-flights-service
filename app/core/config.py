"""Application settings loaded from environment variables and .env.

All tuneable knobs live here so they can be overridden per-environment without
touching source.  Defaults are chosen for a single-node local setup; production
docker-compose.yml overrides the service-discovery URLs.

Context window
--------------
CONTEXT_TOKEN_LIMIT controls how many tokens of conversation history are sent to
the LLM.  Token counting uses tiktoken cl100k_base, which closely approximates
LLaMA3's tokenizer (both are tiktoken BPE; LLaMA3 uses a 128k-vocab superset).
The known over-/under-count for JSON-serialised tool calls is ±10-15 %.
20 000 tokens is conservative relative to llama3.1:8b's 128k context window,
leaving headroom for the system prompt, the current user turn, and tool responses.

Retrieval
---------
RETRIEVAL_SCORE_THRESHOLD (default 0.3) is a minimum cosine similarity cut-off.
Use scripts/calibrate_threshold.py to find the Recall/Precision F1-optimal value
for your specific index and query distribution before tightening it.

HYDE_NUM_SAMPLES controls how many hypothetical flight documents are generated per
query.  With n=1, the single HyDE doc is cached; with n>1, n independent LLM
calls are made and their embeddings averaged (centroid HyDE).  Centroid HyDE
measurably improves recall for ambiguous queries at the cost of n×LLM latency
(which is serialised on a single-GPU Ollama instance regardless of concurrency).

Reranking
---------
ENABLE_RERANKER=true fetches RERANK_FETCH_K candidates from Qdrant and rescores
them with a cross-encoder before returning RERANK_RETURN_N results.  The default
model BAAI/bge-reranker-base is trained on heterogeneous retrieval tasks (BEIR
benchmark) and handles the structured flight-record format significantly better
than the ms-marco family, which was pre-trained on web Q&A logs.

bge-reranker-base: ~278 MB download, no GPU required, BEIR average NDCG@10 ≈ 0.49.
bge-reranker-large: ~1.3 GB, GPU recommended, NDCG@10 ≈ 0.52.
ms-marco-MiniLM-L-6-v2: ~90 MB, fast, but -7–10 % NDCG on non-web corpora.

Run scripts/compare_rerankers.py to benchmark candidate models against your
annotated eval set before changing RERANKER_MODEL in production.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.1:latest"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_FLIGHTS_COLLECTION: str = "flights"
    QDRANT_TIMEOUT_SECONDS: int = 30

    DATABASE_URL: str = "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service"

    REDIS_URL: str = ""
    API_KEY: str = ""

    OTEL_EXPORTER_ENDPOINT: str = ""
    OTEL_SERVICE_NAME: str = "flight-service"

    MAX_TOOL_RESULTS: int = 20
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3

    CONTEXT_TOKEN_LIMIT: int = 20_000
    CONTEXT_MIN_MESSAGES: int = 10

    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANK_FETCH_K: int = 20
    RERANK_RETURN_N: int = 5

    HYDE_NUM_SAMPLES: int = 1
    HYDE_CACHE_TTL: int = 3600

    ENABLE_SUMMARIZATION: bool = True

    INGEST_EMBED_BATCH_SIZE: int = 64
    INGEST_UPSERT_BATCH_SIZE: int = 200
    INGEST_LIMIT_DEFAULT: int = 0


settings = Settings()
