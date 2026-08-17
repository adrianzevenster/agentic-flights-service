from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_header)) -> None:
    """FastAPI dependency that enforces X-API-Key when API_KEY is configured.

    When API_KEY is blank (local dev default) this is a no-op, so routes
    can always declare this dependency without breaking the dev workflow.

    Args:
        key: Value of the X-API-Key header, injected by FastAPI.

    Raises:
        HTTPException 401: When API_KEY is set and the header is missing or wrong.
    """
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
