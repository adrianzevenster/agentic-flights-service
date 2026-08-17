"""Set env vars before any app module is imported.

pydantic-settings reads env at class definition time so these must come first.
Values point to non-existent services — tests that need real infrastructure are
marked integration and excluded from the default CI run.

CONTEXT_TOKEN_LIMIT is set to 500 tokens so that trim-history unit tests can
trigger truncation with small synthetic histories rather than building histories
large enough to exceed the 20 000-token production limit.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://x:x@localhost/x")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("OTEL_EXPORTER_ENDPOINT", "")
os.environ.setdefault("API_KEY", "")
os.environ.setdefault("ENABLE_RERANKER", "false")
os.environ.setdefault("ENABLE_SUMMARIZATION", "false")
os.environ.setdefault("CONTEXT_TOKEN_LIMIT", "500")
