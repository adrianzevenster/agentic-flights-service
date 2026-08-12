from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.1:8b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_FLIGHTS_COLLECTION: str = "flights"
    QDRANT_TIMEOUT_SECONDS: int = 30

    DATABASE_URL: str = "postgresql+psycopg://flight_user:flight_pass@localhost:5432/flight_service"

    MAX_TOOL_RESULTS: int = 20
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3

    INGEST_EMBED_BATCH_SIZE: int = 64
    INGEST_UPSERT_BATCH_SIZE: int = 200
    INGEST_LIMIT_DEFAULT: int = 0


settings = Settings()
