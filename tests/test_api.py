"""Integration tests for the FastAPI layer.

Covers auth, routing, session continuity, the booking flow, health probes, and
the Prometheus metrics endpoint.  All external I/O (Ollama, Qdrant, Postgres) is
replaced with lightweight doubles; the booking store runs on SQLite in-memory so
the full create→retrieve→cancel path exercises real SQLAlchemy code.

Mocking boundary
----------------
- ``app.core.http_client.get_http_client`` — controls every httpx call: Ollama
  chat/embed, and the Qdrant + Ollama liveness GETs in /readyz.
- ``app.rag.qdrant_client.get_qdrant`` — controls Qdrant vector search and point
  retrieval.
- ``app.booking.store.SessionLocal`` — swapped for an in-memory SQLite session.
- ``app.api.main.init_db`` — replaced with a no-op so the lifespan does not
  attempt a real Alembic migration.
- ``app.api.main.engine`` — replaced with the SQLite engine so _check_postgres
  in /readyz connects to SQLite instead of Postgres.

The Ollama post mock dispatches on URL and message content:
- /api/embed → returns a 768-dim zero vector.
- /api/chat with a HyDE prompt → returns a minimal flight record string.
- /api/chat with an agentic payload (tools present) → returns a text response
  with no tool_calls, terminating the agentic loop after one iteration.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.booking.models import Base

_FAKE_VEC = [0.0] * 768
_FLIGHT_ID = "abc00000-0000-0000-0000-000000000001"


# ─── Mock factories ───────────────────────────────────────────────────────────

def _ollama_text_response(content: str) -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"message": {"content": content, "tool_calls": []}}
    return r


def _embed_response() -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"embeddings": [_FAKE_VEC]}
    return r


def _health_ok_response() -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {}
    return r


def _qdrant_point(flight_id: str = _FLIGHT_ID) -> MagicMock:
    p = MagicMock()
    p.id = flight_id
    p.score = 0.85
    p.payload = {
        "carrier": "DL", "flight": 100, "origin": "ATL", "dest": "JFK",
        "year": 2013, "month": 1, "day": 1,
        "sched_dep_time": 800, "sched_arr_time": 1030,
        "dep_time": 810, "arr_time": 1045,
        "distance": 760, "air_time": 115,
    }
    return p


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def sqlite_session():
    """In-memory SQLite engine + session factory with the booking schema created.

    StaticPool is required so that threads spawned by asyncio.to_thread share the
    same single connection — without it each thread opens a new :memory: database
    and the tables created by create_all are invisible to the store's session.
    """
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    yield eng, Session
    eng.dispose()


@pytest.fixture()
def mock_http():
    """Async httpx client double that handles Ollama and health-check calls.

    Returns:
        AsyncMock with post and get side-effects pre-configured.
    """
    async def _aiter_lines():
        yield json.dumps({"message": {"content": "Here are your options!"}, "done": False})
        yield json.dumps({"message": {"content": ""}, "done": True})

    @asynccontextmanager
    async def _fake_stream(*args, **kwargs):
        resp = AsyncMock()
        resp.raise_for_status = MagicMock()
        resp.aiter_lines = _aiter_lines
        yield resp

    async def _post(url, *, json=None, **kwargs):
        if "/api/embed" in url or "/api/embeddings" in url:
            return _embed_response()
        msgs = (json or {}).get("messages", [])
        last = msgs[-1].get("content", "") if msgs else ""
        if "Generate a realistic" in last:
            return _ollama_text_response(
                "DL Delta Air Lines flight 100 ATL Atlanta to JFK "
                "New York JFK 2013-1-1 sched 800->1030 actual 810->1045 dist 760 air 115"
            )
        return _ollama_text_response("I found some great options for you!")

    async def _get(url, **kwargs):
        return _health_ok_response()

    client = AsyncMock()
    client.post.side_effect = _post
    client.get.side_effect = _get
    client.stream = _fake_stream
    return client


@pytest.fixture()
def mock_qdrant():
    """Qdrant client double returning one deterministic flight point.

    Returns:
        MagicMock whose query_points and retrieve methods return _qdrant_point().
    """
    q = MagicMock()
    result = MagicMock()
    result.points = [_qdrant_point()]
    q.query_points.return_value = result
    q.retrieve.return_value = [_qdrant_point()]
    return q


@pytest.fixture()
def client(sqlite_session, mock_http, mock_qdrant):
    """TestClient with all external deps replaced.

    Patches applied:
        - init_db → no-op (avoids Alembic migration attempt)
        - SessionLocal → SQLite session factory
        - engine (main.py scope) → SQLite engine so /readyz _check_postgres works
        - get_qdrant → mock_qdrant
        - get_http_client → mock_http (covers both inline imports and main.py import)

    Args:
        sqlite_session: Tuple of (engine, sessionmaker) from sqlite_session fixture.
        mock_http: AsyncMock http client from mock_http fixture.
        mock_qdrant: MagicMock Qdrant client from mock_qdrant fixture.

    Yields:
        Configured fastapi.testclient.TestClient instance.
    """
    eng, Session = sqlite_session
    with (
        patch("app.api.main.init_db"),
        patch("app.booking.store.SessionLocal", Session),
        patch("app.api.main.engine", eng),
        patch("app.rag.retrieval.get_qdrant", return_value=mock_qdrant),
        patch("app.core.http_client.get_http_client", return_value=mock_http),
        patch("app.api.main.get_http_client", return_value=mock_http),
    ):
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ─── Probes ───────────────────────────────────────────────────────────────────

def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_readyz_healthy(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["checks"]["qdrant"] == "ok"
    assert body["checks"]["ollama"] == "ok"
    assert body["checks"]["postgres"] == "ok"


def test_metrics_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "flight_agent_tool_cap_hits_total" in r.text


# ─── Auth ─────────────────────────────────────────────────────────────────────

def test_auth_rejected_when_api_key_set(sqlite_session, mock_http, mock_qdrant):
    """When API_KEY is configured, requests without the header must be rejected."""
    eng, Session = sqlite_session
    with (
        patch("app.api.main.init_db"),
        patch("app.booking.store.SessionLocal", Session),
        patch("app.api.main.engine", eng),
        patch("app.rag.retrieval.get_qdrant", return_value=mock_qdrant),
        patch("app.core.http_client.get_http_client", return_value=mock_http),
        patch("app.api.main.get_http_client", return_value=mock_http),
        patch("app.core.auth.settings") as mock_settings,
    ):
        mock_settings.API_KEY = "secret"
        from app.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            r = c.post("/chat", json={"message": "hello"})
            assert r.status_code == 401

            r = c.post("/chat", json={"message": "hello"},
                       headers={"X-API-Key": "secret"})
            assert r.status_code == 200


# ─── Chat ─────────────────────────────────────────────────────────────────────

def test_chat_returns_assistant_response(client):
    r = client.post("/chat", json={"message": "Find me a flight to Atlanta"})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "assistant"
    assert isinstance(body["content"], str)
    assert len(body["content"]) > 0
    assert "session_id" in body


def test_chat_session_continuity(client):
    """Session ID returned on first call must produce a non-empty second response."""
    r1 = client.post("/chat", json={"message": "Find flights to Atlanta"})
    session_id = r1.json()["session_id"]

    r2 = client.post("/chat", json={"message": "Book the first one", "session_id": session_id})
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_chat_stream_sse_format(client):
    """SSE stream must end with a done frame containing session_id and tool_results."""
    r = client.post("/chat/stream", json={"message": "Show me Delta flights"})
    assert r.status_code == 200

    events = [
        json.loads(line[len("data: "):])
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) >= 1

    done_events = [e for e in events if e.get("done")]
    assert len(done_events) == 1
    done = done_events[0]
    assert "session_id" in done
    assert "tool_results" in done


def test_chat_with_search_tool_call(client, mock_http):
    """Agentic loop dispatches search_flights when the model returns a tool call."""
    search_tool_call = [{
        "function": {
            "name": "search_flights",
            "arguments": {"query": "morning Delta ATL to JFK"},
        }
    }]

    call_count = 0

    async def _post_with_tool_first_call(url, *, json=None, **kwargs):
        nonlocal call_count
        if "/api/embed" in url or "/api/embeddings" in url:
            return _embed_response()
        msgs = (json or {}).get("messages", [])
        last = msgs[-1].get("content", "") if msgs else ""
        if "Generate a realistic" in last:
            return _ollama_text_response(
                "DL Delta Air Lines flight 100 ATL Atlanta to JFK 2013-1-1 "
                "sched 800->1030 actual 810->1045 dist 760 air 115"
            )
        call_count += 1
        if call_count == 1:
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {
                "message": {"content": "", "tool_calls": search_tool_call}
            }
            return r
        return _ollama_text_response("I found 1 Delta flight from ATL to JFK.")

    mock_http.post.side_effect = _post_with_tool_first_call

    r = client.post("/chat", json={"message": "Find morning Delta flights ATL to JFK"})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "assistant"
    assert len(body["tool_results"]) > 0
    assert body["tool_results"][0]["type"] == "search_results"


# ─── Booking flow ─────────────────────────────────────────────────────────────

def test_book_flight_and_retrieve(client):
    r = client.post("/bookFlight", json={
        "flight_id": _FLIGHT_ID,
        "passenger_name": "Jane Smith",
    })
    assert r.status_code == 200
    booking = r.json()["booking"]
    assert booking["passenger_name"] == "Jane Smith"
    assert booking["status"] == "CONFIRMED"
    booking_id = booking["booking_id"]

    r2 = client.get(f"/booking/{booking_id}")
    assert r2.status_code == 200
    assert r2.json()["booking"]["booking_id"] == booking_id


def test_cancel_booking(client):
    r = client.post("/bookFlight", json={
        "flight_id": _FLIGHT_ID,
        "passenger_name": "John Doe",
    })
    booking_id = r.json()["booking"]["booking_id"]

    rc = client.post("/cancelBooking", json={"booking_id": booking_id})
    assert rc.status_code == 200
    assert rc.json()["result"]["status"] == "CANCELLED"


def test_cancel_unknown_booking_returns_404(client):
    r = client.post("/cancelBooking", json={"booking_id": "does-not-exist"})
    assert r.status_code == 404


def test_get_unknown_booking_returns_404(client):
    r = client.get("/booking/does-not-exist")
    assert r.status_code == 404


def test_book_flight_unknown_id_returns_400(client, mock_qdrant):
    """Booking an unknown flight_id must fail with 400, not 500."""
    mock_qdrant.retrieve.return_value = []
    r = client.post("/bookFlight", json={
        "flight_id": "00000000-0000-0000-0000-000000000000",
        "passenger_name": "Ghost Rider",
    })
    assert r.status_code == 400


def test_book_flight_idempotency(client):
    """Submitting the same booking twice must return the same booking_id."""
    payload = {"flight_id": _FLIGHT_ID, "passenger_name": "Alice Wonder"}
    r1 = client.post("/bookFlight", json=payload)
    r2 = client.post("/bookFlight", json=payload)
    assert r1.json()["booking"]["booking_id"] == r2.json()["booking"]["booking_id"]


# ─── Direct tool endpoints ────────────────────────────────────────────────────

def test_search_flights_endpoint(client):
    r = client.post("/searchFlights", json={"query": "morning Delta flights"})
    assert r.status_code == 200
    assert "results" in r.json()


def test_get_flight_details_found(client):
    r = client.get(f"/flight/{_FLIGHT_ID}")
    assert r.status_code == 200
    assert r.json()["flight"]["flight_id"] == _FLIGHT_ID


def test_get_flight_details_not_found(client, mock_qdrant):
    mock_qdrant.retrieve.return_value = []
    r = client.get("/flight/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
