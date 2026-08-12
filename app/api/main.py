from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.schemas import ChatIn, SearchFlightsIn, CreateBookingIn, CancelBookingIn
from app.api.chat import handle_chat, stream_chat
from app.api import tools
from app.booking.db import engine
from app.booking.init_db import init_db
from app.core.config import settings
from app.core.http_client import get_http_client, close_http_client
from app.core.logging_config import configure_logging, request_id_var


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    init_db()
    yield
    await close_http_client()


app = FastAPI(title="Flight Service", lifespan=lifespan)
app.add_middleware(_RequestIDMiddleware)


@app.post("/chat")
async def chat(body: ChatIn):
    return await handle_chat(body.message, session_id=body.session_id)


@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """SSE endpoint for streaming chat responses.

    Yields Server-Sent Events with `data: <json>\\n\\n` frames.
    Each frame is either `{"chunk": str}` or the terminal
    `{"done": true, "session_id": str, "tool_results": list}`.
    """
    async def generate():
        async for event in stream_chat(body.message, session_id=body.session_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/searchFlights")
async def search_flights(body: SearchFlightsIn):
    return await tools.tool_search_flights(**body.model_dump())


@app.get("/flight/{flight_id}")
async def flight_details(flight_id: str):
    out = await tools.tool_get_flight_details(flight_id=flight_id)
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


@app.post("/bookFlight")
async def book_flight(body: CreateBookingIn):
    out = await tools.tool_create_booking(**body.model_dump())
    if out.get("error"):
        raise HTTPException(status_code=400, detail=out)
    return out


@app.post("/cancelBooking")
async def cancel_booking(body: CancelBookingIn):
    out = await tools.tool_cancel_booking(**body.model_dump())
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


@app.get("/booking/{booking_id}")
async def get_booking(booking_id: str):
    out = await tools.tool_get_booking(booking_id=booking_id)
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out)
    return out


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
