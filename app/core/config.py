from pydantic import BaseModel
import os


class Settings(BaseModel):
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_FLIGHTS_COLLECTION: str = os.getenv("QDRANT_FLIGHTS_COLLECTION", "flights")

    QDRANT_TIMEOUT_SECONDS: int = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "30"))

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service",
    )

    MAX_TOOL_RESULTS: int = int(os.getenv("MAX_TOOL_RESULTS", "20"))

    INGEST_EMBED_BATCH_SIZE: int = int(os.getenv("INGEST_EMBED_BATCH_SIZE", "64"))
    INGEST_UPSERT_BATCH_SIZE: int = int(os.getenv("INGEST_UPSERT_BATCH_SIZE", "200"))
    INGEST_LIMIT_DEFAULT: int = int(os.getenv("INGEST_LIMIT_DEFAULT", "0"))


settings = Settings()
