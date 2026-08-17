from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.schemas import ChatIn, SearchFlightsIn, CreateBookingIn, CancelBookingIn
from app.api.chat import handle_chat, stream_chat, get_metrics
from app.api import tools
from app.booking.db import engine
from app.booking.init_db import init_db
from app.core.auth import require_api_key
from app.core.config import settings
from app.core.http_client import get_http_client, close_http_client
from app.core.redis_client import close_redis
from app.core.logging_config import configure_logging, request_id_var
from app.core.tracing import configure_tracing


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to every request for log correlation.

    Reads X-Request-ID from the incoming headers; generates a UUID if absent.
    Echoes the ID back in the response header.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    configure_tracing(_app)
    init_db()
    yield
    await close_http_client()
    await close_redis()


app = FastAPI(title="Flight Service", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(_RequestIDMiddleware)


# ─── Chat endpoints ───────────────────────────────────────────────────────────

@app.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatIn):
    return await handle_chat(body.message, session_id=body.session_id)


@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def chat_stream(request: Request, body: ChatIn):
    """SSE endpoint for streaming chat responses.

    Yields Server-Sent Events with ``data: <json>\\n\\n`` frames.
    Each frame is either ``{"chunk": str}`` or the terminal
    ``{"done": true, "session_id": str, "tool_results": list}``.
    """
    async def generate():
        async for event in stream_chat(body.message, session_id=body.session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─── Direct tool endpoints ────────────────────────────────────────────────────

@app.post("/searchFlights", dependencies=[Depends(require_api_key)])
async def search_flights(body: SearchFlightsIn):
    return await tools.tool_search_flights(**body.model_dump())


@app.get("/flight/{flight_id}", dependencies=[Depends(require_api_key)])
async def flight_details(flight_id: str):
    out = await tools.tool_get_flight_details(flight_id=flight_id)
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


@app.post("/bookFlight", dependencies=[Depends(require_api_key)])
async def book_flight(body: CreateBookingIn):
    out = await tools.tool_create_booking(**body.model_dump())
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out)
    return out


@app.post("/cancelBooking", dependencies=[Depends(require_api_key)])
async def cancel_booking(body: CancelBookingIn):
    out = await tools.tool_cancel_booking(**body.model_dump())
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


@app.get("/booking/{booking_id}", dependencies=[Depends(require_api_key)])
async def get_booking(booking_id: str):
    out = await tools.tool_get_booking(booking_id=booking_id)
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


# ─── Observability ────────────────────────────────────────────────────────────

@app.get("/metrics")
async def agent_metrics():
    """Return agent-level operational counters.

    Returns:
        Dict with ``tool_cap_hits``: number of agentic turns that exhausted
        the iteration cap without producing a text response.  A rising rate
        indicates prompt drift or a data distribution shift.
    """
    return get_metrics()


# ─── Health ───────────────────────────────────────────────────────────────────

def _check_postgres() -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


@app.get("/readyz")
async def readyz():
    checks: dict[str, str] = {}
    healthy = True
    client = get_http_client()

    try:
        r = await client.get(f"{settings.QDRANT_URL}/collections", timeout=3.0)
        r.raise_for_status()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = str(exc)
        healthy = False

    try:
        r = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3.0)
        r.raise_for_status()
        checks["ollama"] = "ok"
    except Exception as exc:
        checks["ollama"] = str(exc)
        healthy = False

    try:
        await asyncio.to_thread(_check_postgres)
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = str(exc)
        healthy = False

    if not healthy:
        raise HTTPException(status_code=503, detail={"ok": False, "checks": checks})

    return {"ok": True, "checks": checks}
