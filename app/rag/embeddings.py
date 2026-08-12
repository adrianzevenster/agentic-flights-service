from __future__ import annotations

import requests

from app.core.config import settings


class EmbeddingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Sync — ingest runs outside the event loop (pandas batch loop, no uvicorn).
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings via Ollama (synchronous).

    Tries the batch /api/embed endpoint first; falls back to per-text
    /api/embeddings if the model or version doesn't support it.

    Args:
        texts: Strings to embed.

    Returns:
        One float vector per input, in the same order.

    Raises:
        EmbeddingError: If all Ollama endpoints fail for any input.
    """
    base = settings.OLLAMA_BASE_URL.rstrip("/")

    try:
        data = _post_json(f"{base}/api/embed", {"model": settings.OLLAMA_EMBED_MODEL, "input": texts})
        embeddings = data.get("embeddings") or []
        if len(embeddings) == len(texts):
            return embeddings
    except Exception:
        pass

    results = []
    for text in texts:
        try:
            data = _post_json(f"{base}/api/embeddings", {"model": settings.OLLAMA_EMBED_MODEL, "prompt": text})
            if "embedding" in data:
                results.append(data["embedding"])
                continue
        except Exception:
            pass
        raise EmbeddingError(f"Ollama embedding failed for: {text[:60]!r}")

    return results


def embed_text(text: str) -> list[float]:
    """Embed a single string (synchronous).

    Args:
        text: String to embed.

    Returns:
        Float vector.
    """
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Async — API request path; shares the process-wide httpx.AsyncClient.
# ---------------------------------------------------------------------------

async def async_embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings via Ollama (asynchronous).

    Args:
        texts: Strings to embed.

    Returns:
        One float vector per input, in the same order.

    Raises:
        EmbeddingError: If all Ollama endpoints fail for any input.
    """
    from app.core.http_client import get_http_client

    client = get_http_client()
    base = settings.OLLAMA_BASE_URL.rstrip("/")

    try:
        r = await client.post(f"{base}/api/embed", json={"model": settings.OLLAMA_EMBED_MODEL, "input": texts})
        r.raise_for_status()
        data = r.json()
        embeddings = data.get("embeddings") or []
        if len(embeddings) == len(texts):
            return embeddings
    except Exception:
        pass

    results = []
    for text in texts:
        try:
            r = await client.post(f"{base}/api/embeddings", json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text})
            r.raise_for_status()
            data = r.json()
            if "embedding" in data:
                results.append(data["embedding"])
                continue
        except Exception:
            pass
        raise EmbeddingError(f"Ollama embedding failed for: {text[:60]!r}")

    return results


async def async_embed_text(text: str) -> list[float]:
    """Embed a single string asynchronously.

    Args:
        text: String to embed.

    Returns:
        Float vector.
    """
    return (await async_embed_texts([text]))[0]
