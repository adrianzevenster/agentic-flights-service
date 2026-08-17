"""Shared async Redis client.

A single aioredis.Redis instance is reused across the app. Session store and
HyDE cache both import from here so they share the same connection pool instead
of opening redundant connections.
"""
from __future__ import annotations

import logging

from app.core.config import settings

log = logging.getLogger(__name__)
_client = None


async def get_redis():
    """Return the shared Redis client, or None when REDIS_URL is not configured.

    Returns:
        aioredis.Redis connected to settings.REDIS_URL, or None.
    """
    global _client
    if not settings.REDIS_URL:
        return None
    if _client is None:
        import redis.asyncio as aioredis
        _client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        log.info("Redis client initialised: %s", settings.REDIS_URL)
    return _client


async def close_redis() -> None:
    """Close the shared connection and reset the singleton.

    Should be called in the FastAPI lifespan shutdown handler.
    """
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        log.info("Redis client closed.")
