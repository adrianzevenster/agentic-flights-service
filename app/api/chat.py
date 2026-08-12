from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import AsyncIterator

from pydantic import ValidationError

from app.api import tools as _tools
from app.core.config import settings
from app.core.logging_config import request_id_var

log = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights using semantic search plus optional keyword filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "origin": {"type": "string", "description": "IATA airport code, e.g. JFK"},
                    "dest": {"type": "string", "description": "IATA airport code, e.g. LAX"},
                    "carrier": {"type": "string", "description": "Carrier code, e.g. DL"},
                    "limit": {"type": "integer", "description": "1–50, default 10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_details",
            "description": "Retrieve full details for a flight by its UUID flight_id.",
            "parameters": {
                "type": "object",
                "properties": {"flight_id": {"type": "string"}},
                "required": ["flight_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": (
                "Book a flight for a named passenger. "
                "Only call this when the user has explicitly confirmed both flight_id and passenger_name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "passenger_name": {"type": "string"},
                },
                "required": ["flight_id", "passenger_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel a booking by booking_id.",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Look up a booking's status and details by booking_id.",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string"}},
                "required": ["booking_id"],
            },
        },
    },
]

_SYSTEM = (
    "You are FlightService, a flight search and booking assistant.\n"
    "- Use search_flights to find flights.\n"
    "- Use create_booking only when the user has confirmed a specific flight_id and provided their passenger name.\n"
    "- If the user refers to a result by position, surface the flight_id from search results and ask them to confirm before booking.\n"
    "- Ask for missing required fields before calling any tool."
)

_MAX_SESSIONS = 500
_SESSION_TTL = 3600.0
# Guard against pathological tool loops (e.g. model repeatedly calling search).
_MAX_TOOL_ITERATIONS = 10


class _SessionStore:
    """In-process session store backed by an LRU OrderedDict.

    Thread-safety is via asyncio.Lock — safe for the single event-loop model
    uvicorn runs. Not suitable for multi-process deployments without Redis.
    """

    def __init__(self) -> None:
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[dict]:
        """Fetch message history for a session, refreshing its TTL.

        Args:
            session_id: Opaque session token.

        Returns:
            Copy of the message list, or an empty list if the session is unknown.
        """
        async with self._lock:
            entry = self._store.get(session_id)
            if not entry:
                return []
            entry["ts"] = time.monotonic()
            self._store.move_to_end(session_id)
            return list(entry["messages"])

    async def save(self, session_id: str, messages: list[dict]) -> None:
        """Persist message history and evict stale/overflow sessions.

        Args:
            session_id: Opaque session token.
            messages: Full message list to store.
        """
        async with self._lock:
            self._store[session_id] = {"messages": list(messages), "ts": time.monotonic()}
            self._store.move_to_end(session_id)
            self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        stale = [k for k, v in self._store.items() if now - v["ts"] > _SESSION_TTL]
        for k in stale:
            del self._store[k]
        while len(self._store) > _MAX_SESSIONS:
            self._store.popitem(last=False)


_sessions = _SessionStore()


async def _call_ollama(messages: list[dict]) -> dict:
    from app.core.http_client import get_http_client
    r = await get_http_client().post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={"model": settings.OLLAMA_CHAT_MODEL, "messages": messages, "tools": TOOLS, "stream": False},
    )
    r.raise_for_status()
    return r.json()


async def _stream_final_response(history: list[dict]) -> AsyncIterator[str]:
    """Stream the assistant's final text response without exposing tools.

    Omitting `tools` from the request prevents the model from calling tools
    in the streaming pass, which would produce unparseable partial JSON chunks.

    Args:
        history: Full conversation history up to (but not including) the final
                 assistant turn.

    Yields:
        Text chunks as they arrive from Ollama.
    """
    from app.core.http_client import get_http_client
    async with get_http_client().stream(
        "POST",
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={"model": settings.OLLAMA_CHAT_MODEL, "messages": history, "stream": True},
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def _parse_args(raw: str | dict) -> dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}


async def _dispatch(tool_name: str, args: dict) -> tuple[str, dict]:
    """Route a tool call to the appropriate handler.

    Catches ValidationError and unexpected exceptions so the model always
    receives a structured error it can relay to the user instead of a 500.

    Args:
        tool_name: Name of the tool as declared in TOOLS.
        args: Parsed keyword arguments for the tool.

    Returns:
        Tuple of (response_type, result_dict).
    """
    try:
        if tool_name == "search_flights":
            return "search_results", await _tools.tool_search_flights(**args)
        if tool_name == "get_flight_details":
            return "flight_details", await _tools.tool_get_flight_details(**args)
        if tool_name == "create_booking":
            out = await _tools.tool_create_booking(**args)
            return ("booking_confirmation" if not out.get("error") else "error"), out
        if tool_name == "cancel_booking":
            return "cancel_result", await _tools.tool_cancel_booking(**args)
        if tool_name == "get_booking":
            return "booking", await _tools.tool_get_booking(**args)
        return "error", {"error": "UNKNOWN_TOOL", "tool_name": tool_name}
    except ValidationError as exc:
        log.warning("Tool %r validation error: %s", tool_name, exc)
        return "error", {"error": "INVALID_ARGS", "tool": tool_name, "detail": str(exc)}
    except Exception:
        log.exception("Tool %r raised an unexpected error", tool_name)
        return "error", {"error": "TOOL_ERROR", "tool": tool_name}


async def _run_tool_loop(
    history: list[dict],
    session_id: str,
) -> tuple[list[dict], list[dict], str | None]:
    """Execute the tool-calling phase of an agentic turn.

    Loops until the model produces a text response (no tool_calls) or the
    iteration cap is hit. Tool results are appended to history so the model
    sees them in each subsequent call.

    Args:
        history: Conversation history including the latest user message.
        session_id: Used only for structured log fields.

    Returns:
        Tuple of:
            history: Updated message list (tool turns appended).
            tool_results: Ordered list of {type, data} dicts for every tool call made.
            content: The model's final text response, or None if the cap was hit.
    """
    tool_results: list[dict] = []

    for iteration in range(_MAX_TOOL_ITERATIONS):
        response = await _call_ollama(history)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            content = msg.get("content") or "I'm not sure how to help with that. Try asking about a flight route."
            log.info("chat.complete session=%s iterations=%d", session_id, iteration + 1)
            return history, tool_results, content

        log.info("chat.tool_round session=%s iteration=%d tools=%d", session_id, iteration, len(tool_calls))
        history.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            raw_args = _parse_args(fn.get("arguments", {}))
            log.info("chat.dispatch session=%s tool=%s", session_id, tool_name)
            resp_type, tool_result = await _dispatch(tool_name, raw_args)
            tool_results.append({"type": resp_type, "data": tool_result})
            history.append({"role": "tool", "content": json.dumps(tool_result), "name": tool_name})

        await _sessions.save(session_id, history)

    log.warning("chat.max_iterations session=%s", session_id)
    return history, tool_results, None


async def handle_chat(user_message: str, session_id: str | None = None) -> dict:
    """Process one user turn and return the model's synthesized text response.

    The agentic loop runs to completion before returning, so the response
    always contains the model's natural-language interpretation of any tool
    results — not raw tool JSON.

    Args:
        user_message: Raw message from the user.
        session_id: Opaque token from a prior call. Pass None to start a new session.

    Returns:
        Dict with keys:
            type (str): Always "assistant".
            content (str): The model's final text response.
            session_id (str): Token to pass on the next call.
            tool_results (list[dict]): Ordered tool calls made this turn, for
                the UI to optionally render structured data alongside the text.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    log.info("chat.start session=%s msg_len=%d", session_id, len(user_message))

    history = await _sessions.get(session_id)
    if not history:
        history = [{"role": "system", "content": _SYSTEM}]
    history.append({"role": "user", "content": user_message})

    history, tool_results, content = await _run_tool_loop(history, session_id)

    if content is None:
        content = "I've been working on your request but couldn't complete it. Please try rephrasing."

    history.append({"role": "assistant", "content": content})
    await _sessions.save(session_id, history)

    return {
        "type": "assistant",
        "content": content,
        "session_id": session_id,
        "tool_results": tool_results,
    }


async def stream_chat(user_message: str, session_id: str | None = None) -> AsyncIterator[dict]:
    """Process one user turn and stream the model's final response as SSE chunks.

    Tool-calling rounds run non-streaming (they complete before the first chunk
    is yielded). The final text response is streamed via a second Ollama call
    using stream=True without tools, which prevents partial tool-call JSON from
    appearing in the stream.

    Args:
        user_message: Raw message from the user.
        session_id: Opaque token from a prior call. Pass None to start a new session.

    Yields:
        Dicts of the form {"chunk": str} during generation, then a final
        {"done": True, "session_id": str, "tool_results": list[dict]}.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    history = await _sessions.get(session_id)
    if not history:
        history = [{"role": "system", "content": _SYSTEM}]
    history.append({"role": "user", "content": user_message})

    history, tool_results, _ = await _run_tool_loop(history, session_id)

    collected: list[str] = []
    async for chunk in _stream_final_response(history):
        collected.append(chunk)
        yield {"chunk": chunk}

    content = "".join(collected) or "Done."
    history.append({"role": "assistant", "content": content})
    await _sessions.save(session_id, history)

    yield {"done": True, "session_id": session_id, "tool_results": tool_results}
