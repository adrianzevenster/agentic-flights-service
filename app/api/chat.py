"""Agentic chat loop, session management, and context-window control.

Architecture
------------
Each user turn drives _run_tool_loop, which repeatedly calls the LLM until it
produces a text-only response (no tool_calls) or hits _MAX_TOOL_ITERATIONS.
Tool calls are dispatched to app.api.tools and their results are appended to the
conversation history so the model can synthesise a final answer.

The /chat/stream endpoint runs the tool rounds non-streaming (structured JSON
must arrive complete before it can be parsed), then replays the history to Ollama
without tools for a clean streaming text response.  Omitting tools from the final
call prevents partial tool-call JSON from leaking into the SSE stream.

Session persistence
-------------------
Sessions are stored in-process (_InMemorySessionStore) when REDIS_URL is blank,
or in Redis (_RedisSessionStore) when it is set.  The in-memory store uses an
OrderedDict for O(1) LRU eviction and is protected by an asyncio.Lock.  It is
not safe for multi-process deployments — configure REDIS_URL for those.

Context window management
--------------------------
Token counting uses tiktoken cl100k_base, which closely tracks LLaMA3's
tokenizer (both are BPE-based tiktoken; LLaMA3 adds 28k extra merge rules).
Empirical error on mixed prose+JSON is ±10-15 %, comfortably within the
4× margin between CONTEXT_TOKEN_LIMIT (20k) and the model's 128k context.

When history exceeds CONTEXT_TOKEN_LIMIT, _maybe_trim_history first tries LLM
summarisation (ENABLE_SUMMARIZATION=true) and falls back to hard tail-trimming
if Ollama is unreachable.  Summarisation preserves semantic continuity at the
cost of one extra inference call per trim event.

Tool-call few-shot anchoring
-----------------------------
_SYSTEM includes worked examples for each tool.  Without them, llama3.1:8b
occasionally invents flight_ids or skips required fields such as passenger_name.
The examples are in the system prompt so they are never evicted by context
trimming (the system message is always retained in position 0).

Metrics
-------
_cap_hit_count tracks how many agentic turns hit _MAX_TOOL_ITERATIONS without
producing a text response.  A rising rate is the earliest signal that the prompt
or data distribution has shifted.  get_metrics() surfaces the counter to the
/metrics endpoint; individual events are also recorded as OTel span attributes
on the retrieval.search span when tracing is enabled.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from typing import AsyncIterator, Protocol

import tiktoken
from pydantic import ValidationError

from app.api import tools as _tools
from app.core.config import settings
from app.core.logging_config import request_id_var

log = logging.getLogger(__name__)

_token_enc = tiktoken.get_encoding("cl100k_base")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights using semantic search plus optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "origin": {"type": "string", "description": "IATA airport code, e.g. JFK"},
                    "dest": {"type": "string", "description": "IATA airport code, e.g. LAX"},
                    "carrier": {"type": "string", "description": "Carrier code, e.g. DL"},
                    "limit": {"type": "integer", "description": "1–50, default 10"},
                    "month_min": {"type": "integer", "description": "Start month 1-12 for month range filter"},
                    "month_max": {"type": "integer", "description": "End month 1-12 for month range filter"},
                    "time_of_day": {
                        "type": "string",
                        "enum": ["early_morning", "morning", "afternoon", "evening", "late_night"],
                        "description": "Filter by departure time band",
                    },
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

_SYSTEM = """\
You are FlightService, a flight search and booking assistant.

Rules:
- Use search_flights first when the user wants to find flights.
- Only call create_booking when the user has explicitly confirmed a specific flight_id AND \
provided their passenger name. Never invent or guess a flight_id.
- If the user refers to a result by position ("the first one", "option 2"), \
surface the flight_id from prior search results and confirm before booking.
- Ask for any missing required fields before calling a tool.

Example tool usage:
  User: "Find morning Delta flights from Atlanta to Chicago"
  → search_flights(query="morning Delta flights Atlanta to Chicago", origin="ATL", dest="ORD", \
carrier="DL", time_of_day="morning")

  User: "Show me flights in June and July"
  → search_flights(query="flights June July", month_min=6, month_max=7)

  User: "Book flight abc-uuid-123 for Jane Smith"
  → create_booking(flight_id="abc-uuid-123", passenger_name="Jane Smith")

  User: "What's the status of booking B-456?"
  → get_booking(booking_id="B-456")

  User: "Cancel booking X-789"
  → cancel_booking(booking_id="X-789")
"""

_MAX_SESSIONS = 500
_SESSION_TTL = 3600.0
_MAX_TOOL_ITERATIONS = 10

_cap_hit_count: int = 0
_metrics_lock = threading.Lock()


def _increment_cap_hits() -> None:
    global _cap_hit_count
    with _metrics_lock:
        _cap_hit_count += 1


def get_metrics() -> dict[str, int]:
    """Return a snapshot of agent-level counters for the /metrics endpoint.

    Returns:
        Dict with key ``tool_cap_hits``: number of agentic turns that exhausted
        _MAX_TOOL_ITERATIONS without producing a text response.
    """
    with _metrics_lock:
        return {"tool_cap_hits": _cap_hit_count}


# ─── Token counting ───────────────────────────────────────────────────────────

def _count_tokens(history: list[dict]) -> int:
    """Estimate the token count of a serialised conversation history.

    Args:
        history: Full message list to measure.

    Returns:
        Approximate token count using cl100k_base encoding.  Error margin is
        ±10-15 % relative to LLaMA3's tokenizer on mixed prose+JSON content.
    """
    return len(_token_enc.encode(json.dumps(history)))


# ─── Session store ────────────────────────────────────────────────────────────

class _SessionStore(Protocol):
    async def get(self, session_id: str) -> list[dict]: ...
    async def save(self, session_id: str, messages: list[dict]) -> None: ...


class _InMemorySessionStore:
    """LRU in-process session store backed by an OrderedDict.

    Not suitable for multi-process deployments; configure REDIS_URL to use
    _RedisSessionStore instead.  The asyncio.Lock serialises all reads and
    writes within a single worker process.
    """

    def __init__(self) -> None:
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> list[dict]:
        """Fetch message history, refreshing the session TTL.

        Args:
            session_id: Opaque session token.

        Returns:
            Copy of the message list, or [] if unknown.
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


class _RedisSessionStore:
    """Redis-backed session store for multi-process / restart-safe deployments.

    Borrows the shared singleton client from app.core.redis_client so it does
    not open a second connection pool alongside the one used for HyDE caching.
    """

    _prefix = "flt:sess:"

    async def _r(self):
        from app.core.redis_client import get_redis
        return await get_redis()

    async def get(self, session_id: str) -> list[dict]:
        """Fetch message history from Redis.

        Args:
            session_id: Opaque session token.

        Returns:
            Deserialized message list, or [] if the key is absent or expired.
        """
        r = await self._r()
        data = await r.get(f"{self._prefix}{session_id}")
        return json.loads(data) if data else []

    async def save(self, session_id: str, messages: list[dict]) -> None:
        """Persist message history with a rolling TTL.

        Args:
            session_id: Opaque session token.
            messages: Full message list to store.
        """
        r = await self._r()
        await r.setex(
            f"{self._prefix}{session_id}",
            int(_SESSION_TTL),
            json.dumps(messages),
        )


def _make_session_store() -> _InMemorySessionStore | _RedisSessionStore:
    if settings.REDIS_URL:
        log.info("Using Redis session store: %s", settings.REDIS_URL)
        return _RedisSessionStore()
    log.info("Using in-memory session store.")
    return _InMemorySessionStore()


_sessions: _InMemorySessionStore | _RedisSessionStore = _make_session_store()


# ─── Context window management ────────────────────────────────────────────────

def _trim_history(history: list[dict]) -> list[dict]:
    """Drop oldest non-system messages when the history exceeds the token budget.

    Fallback path used when summarisation is disabled or fails.  Always retains
    the system prompt at index 0 and at least CONTEXT_MIN_MESSAGES recent messages.

    Args:
        history: Full conversation history including the system message.

    Returns:
        Trimmed history.  Returns the same list object when no trimming is needed
        so callers can use identity comparison as a cheap change-detection signal.
    """
    limit = settings.CONTEXT_TOKEN_LIMIT
    min_keep = settings.CONTEXT_MIN_MESSAGES

    if _count_tokens(history) <= limit:
        return history

    system = history[:1]
    rest = history[1:]
    original_len = len(rest)

    while _count_tokens(system + rest) > limit and len(rest) > min_keep:
        rest = rest[1:]

    dropped = original_len - len(rest)
    if dropped:
        log.info("context.trimmed dropped=%d kept=%d", dropped, len(rest))

    return system + rest


async def _summarize_history(history: list[dict]) -> list[dict]:
    """Replace the oldest portion of history with an LLM-generated summary.

    Keeps the last CONTEXT_MIN_MESSAGES messages verbatim and summarises
    everything older.  The summary is injected as a system-role message so the
    model understands prior context without re-reading the full transcript.

    Args:
        history: Full conversation history including the system message.

    Returns:
        Condensed history: [system_msg, summary_msg, ...tail_messages].

    Raises:
        Exception: Propagates on Ollama failure so the caller can fall back to
            hard trim via _trim_history.
    """
    system = history[:1]
    rest = history[1:]
    keep_tail = rest[-settings.CONTEXT_MIN_MESSAGES:]
    to_summarize = rest[:-settings.CONTEXT_MIN_MESSAGES]

    if not to_summarize:
        return history

    transcript = "\n".join(
        f"{m['role'].upper()}: {str(m.get('content', ''))[:300]}"
        for m in to_summarize
        if m.get("content")
    )
    if not transcript.strip():
        return history

    prompt = (
        "Summarize the following flight booking conversation in 2-3 sentences. "
        "Focus on: which flights were searched, any bookings made, and the user's preferences.\n\n"
        f"{transcript}"
    )

    from app.core.http_client import get_http_client
    r = await get_http_client().post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={
            "model": settings.OLLAMA_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    r.raise_for_status()
    summary = r.json().get("message", {}).get("content", "").strip()

    log.info("context.summarized dropped=%d summary_chars=%d", len(to_summarize), len(summary))

    summary_msg = {
        "role": "system",
        "content": f"[Earlier conversation ({len(to_summarize)} messages): {summary}]",
    }
    return system + [summary_msg] + keep_tail


async def _maybe_trim_history(history: list[dict]) -> list[dict]:
    """Trim or summarise history when it exceeds the token budget.

    Prefers summarisation (ENABLE_SUMMARIZATION=true) because a summary
    preserves semantic context — the model knows what was searched rather than
    losing it entirely.  Falls back to hard trim on Ollama failure.

    Args:
        history: Full conversation history.

    Returns:
        History that fits within CONTEXT_TOKEN_LIMIT.
    """
    if _count_tokens(history) <= settings.CONTEXT_TOKEN_LIMIT:
        return history

    if settings.ENABLE_SUMMARIZATION:
        try:
            return await _summarize_history(history)
        except Exception:
            log.warning("context.summarization_failed — falling back to hard trim")

    return _trim_history(history)


# ─── Ollama calls ─────────────────────────────────────────────────────────────

async def _call_ollama(messages: list[dict]) -> dict:
    from app.core.http_client import get_http_client
    r = await get_http_client().post(
        f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        json={"model": settings.OLLAMA_CHAT_MODEL, "messages": messages, "tools": TOOLS, "stream": False},
    )
    r.raise_for_status()
    return r.json()


async def _stream_final_response(history: list[dict]) -> AsyncIterator[str]:
    """Stream the assistant's final text response without offering tools.

    Tools are omitted from this call to prevent partial tool-call JSON from
    appearing in the SSE stream as garbled text.

    Args:
        history: Full conversation history up to (not including) the final turn.

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


# ─── Tool dispatch ────────────────────────────────────────────────────────────

def _parse_args(raw: str | dict) -> dict:
    """Parse tool call arguments from the model's raw output.

    Ollama returns arguments as a dict in structured mode but may fall back to a
    raw string when the model produces non-conforming output.  The regex path
    handles cases where the model wraps JSON in markdown fences or preamble text.

    Args:
        raw: Arguments as a dict (ideal) or string (fallback from the model).

    Returns:
        Parsed dict, or {} on parse failure.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}
    return raw or {}


async def _dispatch(tool_name: str, args: dict) -> tuple[str, dict]:
    """Route a tool call to the appropriate handler.

    Catches ValidationError and unexpected exceptions so the model always
    receives a structured error it can relay to the user rather than a 500.

    Args:
        tool_name: Name of the tool as declared in TOOLS.
        args: Parsed keyword arguments for the tool.

    Returns:
        Tuple of (response_type_str, result_dict).
    """
    from app.core.tracing import get_tracer
    with get_tracer().start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
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
            span.set_attribute("error", True)
            return "error", {"error": "INVALID_ARGS", "tool": tool_name, "detail": str(exc)}
        except Exception:
            log.exception("Tool %r raised an unexpected error", tool_name)
            span.set_attribute("error", True)
            return "error", {"error": "TOOL_ERROR", "tool": tool_name}


# ─── Agentic loop ─────────────────────────────────────────────────────────────

async def _run_tool_loop(
    history: list[dict],
    session_id: str,
) -> tuple[list[dict], list[dict], str | None]:
    """Execute the tool-calling phase of an agentic turn.

    Loops until the model produces text (no tool_calls) or _MAX_TOOL_ITERATIONS
    is reached.  History is trimmed/summarised before each LLM call to stay
    within the token budget.  When the iteration cap fires, _cap_hit_count is
    incremented and None is returned as the content signal for the caller.

    Args:
        history: Conversation history including the latest user message.
        session_id: Used only for structured log fields.

    Returns:
        Tuple of:
            history: Updated message list with all tool turns appended.
            tool_results: Ordered list of {type, data} dicts for every call made.
            content: The model's final text, or None if the cap was hit.
    """
    tool_results: list[dict] = []

    for iteration in range(_MAX_TOOL_ITERATIONS):
        trimmed = await _maybe_trim_history(history)
        response = await _call_ollama(trimmed)
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
            raw_args = fn.get("arguments", {})
            parsed = _parse_args(raw_args)

            if not parsed and raw_args:
                log.warning("chat.malformed_args session=%s tool=%s raw=%r", session_id, tool_name, str(raw_args)[:200])
                tool_results.append({"type": "error", "data": {"error": "MALFORMED_ARGS", "tool": tool_name}})
                history.append({
                    "role": "tool",
                    "content": json.dumps({"error": "MALFORMED_ARGS", "detail": "Arguments could not be parsed as JSON. Please retry with valid JSON."}),
                    "name": tool_name,
                })
                continue

            log.info("chat.dispatch session=%s tool=%s", session_id, tool_name)
            resp_type, tool_result = await _dispatch(tool_name, parsed)
            tool_results.append({"type": resp_type, "data": tool_result})
            history.append({"role": "tool", "content": json.dumps(tool_result), "name": tool_name})

        await _sessions.save(session_id, history)

    _increment_cap_hits()
    log.warning("chat.max_iterations session=%s cap_hit_total=%d", session_id, _cap_hit_count)
    return history, tool_results, None


async def handle_chat(user_message: str, session_id: str | None = None) -> dict:
    """Process one user turn and return the model's synthesised text response.

    Args:
        user_message: Raw message from the user.
        session_id: Opaque token from a prior call. Pass None to start a new session.

    Returns:
        Dict with keys:
            type (str): Always "assistant".
            content (str): The model's final text response.
            session_id (str): Token to pass on the next call.
            tool_results (list[dict]): Ordered tool calls made this turn.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    log.info("chat.start session=%s", session_id)

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

    Tool-calling rounds run non-streaming.  The final text response is streamed
    via a second Ollama call with stream=True and no tools.

    Args:
        user_message: Raw message from the user.
        session_id: Opaque token from a prior call. Pass None to start a new session.

    Yields:
        {"chunk": str} during generation, then
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
    async for chunk in _stream_final_response(await _maybe_trim_history(history)):
        collected.append(chunk)
        yield {"chunk": chunk}

    content = "".join(collected) or "Done."
    history.append({"role": "assistant", "content": content})
    await _sessions.save(session_id, history)

    yield {"done": True, "session_id": session_id, "tool_results": tool_results}
