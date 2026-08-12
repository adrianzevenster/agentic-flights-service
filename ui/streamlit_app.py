from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
STREAM_URL = f"{API_BASE}/chat/stream"

st.set_page_config(page_title="Flight Service", layout="wide")


# ─── Formatters ───────────────────────────────────────────────────────────────

def _fmt_time(hhmm: int | None) -> str:
    """Convert an integer HHMM value to HH:MM, e.g. 1430 → '14:30'."""
    if hhmm is None:
        return "—"
    h, m = divmod(int(hhmm), 100)
    return f"{h:02d}:{m:02d}"


def _fmt_dist(miles: int | None) -> str:
    if miles is None:
        return "—"
    return f"{miles:,} mi"


def _short_id(uid: str) -> str:
    return uid[:8] + "…"


# ─── Booking dialog ───────────────────────────────────────────────────────────

@st.dialog("Confirm Booking")
def _booking_dialog(flight: dict) -> None:
    """Modal that collects the passenger name before injecting a booking message.

    Args:
        flight: Flight dict from a search_results tool result.
    """
    carrier = flight.get("carrier", "")
    fn = flight.get("flight", "")
    origin = flight.get("origin", "")
    dest = flight.get("dest", "")
    dep = _fmt_time(flight.get("sched_dep_time"))
    arr = _fmt_time(flight.get("sched_arr_time"))
    date = f"{flight.get('year')}-{flight.get('month', 0):02d}-{flight.get('day', 0):02d}"

    st.markdown(f"**{carrier} {fn}** &nbsp; {origin} → {dest}")
    st.caption(f"{date} &nbsp;&nbsp; {dep} → {arr} &nbsp;&nbsp; {_fmt_dist(flight.get('distance'))}")
    st.divider()

    name = st.text_input("Passenger name")
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("Confirm", type="primary", disabled=not name.strip()):
            st.session_state.pending_booking_msg = (
                f"Book flight {flight['flight_id']} for {name.strip()}"
            )
            st.session_state.flight_to_book = None
            st.rerun()
    with col_cancel:
        if st.button("Cancel"):
            st.session_state.flight_to_book = None
            st.rerun()


# ─── Flight card + tool result renderers ──────────────────────────────────────

def _render_flight_card(flight: dict, key_prefix: str, show_book: bool) -> None:
    """Render a single flight as a bordered card with formatted fields.

    Args:
        flight: Flight dict with carrier, flight, origin, dest, time, score, etc.
        key_prefix: Unique prefix for Streamlit widget keys to prevent collisions
                    across multiple render passes (e.g. history vs. current turn).
        show_book: When True, render a Book button that opens the booking dialog.
    """
    carrier = flight.get("carrier", "?")
    fn = flight.get("flight", "?")
    origin = flight.get("origin", "?")
    dest = flight.get("dest", "?")
    dep = _fmt_time(flight.get("sched_dep_time"))
    arr = _fmt_time(flight.get("sched_arr_time"))
    date = f"{flight.get('year')}-{flight.get('month', 0):02d}-{flight.get('day', 0):02d}"
    score = flight.get("score")

    with st.container(border=True):
        col_info, col_time, col_action = st.columns([4, 3, 2])
        with col_info:
            st.markdown(f"**{carrier} {fn}** &nbsp; `{_short_id(flight['flight_id'])}`")
            st.caption(f"{origin} → {dest} &nbsp; · &nbsp; {_fmt_dist(flight.get('distance'))}")
        with col_time:
            st.markdown(f"{dep} → {arr}")
            st.caption(date)
        with col_action:
            if score is not None:
                st.metric("Match", f"{score:.0%}")
            if show_book:
                if st.button("Book ✈", key=f"{key_prefix}_{flight['flight_id']}"):
                    st.session_state.flight_to_book = flight
                    st.rerun()


def _render_tool_result(tr: dict, key_prefix: str = "", show_actions: bool = True) -> None:
    """Render structured tool output beneath an assistant message.

    Args:
        tr: Dict with keys `type` (str) and `data` (dict).
        key_prefix: Forwarded to _render_flight_card for unique widget keys.
        show_actions: When False, suppresses Book buttons (used for history replay).
    """
    tr_type = tr.get("type")
    data = tr.get("data") or {}

    if tr_type == "search_results":
        results = data.get("results") or []
        if results:
            if data.get("filters_dropped"):
                st.caption("No results matched all filters — showing unfiltered results.")
            for flight in results:
                _render_flight_card(flight, key_prefix=key_prefix, show_book=show_actions)

    elif tr_type == "booking_confirmation":
        booking = data.get("booking", data)
        st.success(f"Booked! ID: `{booking.get('booking_id', '—')}`")
        # Persist the ID so the sidebar booking tracker can display it.
        bid = booking.get("booking_id")
        if bid and bid not in st.session_state.booking_ids:
            st.session_state.booking_ids.append(bid)

    elif tr_type == "cancel_result":
        result = data.get("result", data)
        st.info(f"Booking `{result.get('booking_id', '—')}` cancelled.")

    elif tr_type == "booking":
        st.json(data.get("booking", data))

    elif tr_type == "error":
        st.error(data.get("message") or data.get("error") or "An error occurred.")


# ─── SSE streaming with retry ─────────────────────────────────────────────────

def _stream_events(url: str, payload: dict, timeout: int = 180):
    """Yield SSE event dicts from a streaming POST, retrying once on early drop.

    The retry only fires if no events have been received yet — mid-stream
    reconnect would replay the whole turn from a duplicate message, which
    is worse than surfacing the error.

    Args:
        url: SSE endpoint URL.
        payload: JSON body for the POST request.
        timeout: Per-request timeout in seconds.

    Yields:
        Parsed event dicts ({"chunk": str} or {"done": True, ...}).

    Raises:
        requests.RequestException: On the second failure or any mid-stream drop.
    """
    received_any = False
    for attempt in range(2):
        try:
            with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    received_any = True
                    yield event
                    if event.get("done"):
                        return
            return
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 0 and not received_any:
                continue
            raise


# ─── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "booking_ids" not in st.session_state:
    st.session_state.booking_ids = []
if "pending_booking_msg" not in st.session_state:
    st.session_state.pending_booking_msg = None
if "flight_to_book" not in st.session_state:
    st.session_state.flight_to_book = None

st.title("Flight Service")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Session")
    if st.session_state.session_id:
        st.caption(f"ID: `{st.session_state.session_id}`")
    else:
        st.caption("No active session")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()
    st.subheader("Quick Filters")
    st.caption("Appended to your next message")
    fc1, fc2 = st.columns(2)
    origin_filter = fc1.text_input("From", placeholder="JFK", max_chars=3).strip().upper()
    dest_filter = fc2.text_input("To", placeholder="LAX", max_chars=3).strip().upper()

    st.divider()
    st.subheader("Your Bookings")
    if not st.session_state.booking_ids:
        st.caption("No bookings yet.")
    else:
        for bid in list(st.session_state.booking_ids):
            try:
                r = requests.get(f"{API_BASE}/booking/{bid}", timeout=5)
                b = r.json().get("booking", {}) if r.ok else {}
                status = b.get("status", "?")
                snap = b.get("flight_snapshot") or {}
                colour = "🟢" if status == "CONFIRMED" else "🔴"
                with st.container(border=True):
                    st.caption(f"{colour} `{_short_id(bid)}`")
                    st.write(b.get("passenger_name", "Unknown"))
                    if snap.get("carrier"):
                        st.caption(
                            f"{snap['carrier']} {snap.get('origin', '')}→{snap.get('dest', '')}"
                        )
                    if status == "CONFIRMED":
                        if st.button("Cancel", key=f"sb_cancel_{bid}"):
                            try:
                                requests.post(
                                    f"{API_BASE}/cancelBooking",
                                    json={"booking_id": bid},
                                    timeout=5,
                                )
                            except Exception:
                                pass
                            st.rerun()
            except Exception:
                st.caption(f"Could not load `{_short_id(bid)}`")

# ─── Chat history ─────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # show_actions=False: history cards have no Book buttons — the search
        # results are stale and clicking Book would confuse the session context.
        for tr in msg.get("tool_results", []):
            _render_tool_result(tr, key_prefix=f"hist_{i}", show_actions=False)

# ─── Booking dialog trigger ───────────────────────────────────────────────────

# The dialog is triggered on the rerun that follows a Book button click.
if st.session_state.flight_to_book:
    _booking_dialog(st.session_state.flight_to_book)

# ─── Input and response ───────────────────────────────────────────────────────

user_msg = st.chat_input("Ask about flights or make a booking…")

# Pending message is set by the booking dialog confirming; process it exactly
# like a user-typed message so the agent loop handles it normally.
pending = st.session_state.pending_booking_msg
if pending:
    st.session_state.pending_booking_msg = None

msg_to_send = pending or user_msg

if msg_to_send:
    # Inject quick-filter context so the model can extract it as tool args.
    api_msg = msg_to_send
    if origin_filter or dest_filter:
        parts = []
        if origin_filter:
            parts.append(f"from {origin_filter}")
        if dest_filter:
            parts.append(f"to {dest_filter}")
        api_msg = f"{msg_to_send} ({' '.join(parts)})"

    st.session_state.messages.append({"role": "user", "content": msg_to_send, "tool_results": []})
    st.chat_message("user").write(msg_to_send)

    content = ""
    tool_results: list[dict] = []
    session_id = st.session_state.session_id
    error_text: str | None = None

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for event in _stream_events(STREAM_URL, {"message": api_msg, "session_id": session_id}):
                if "chunk" in event:
                    content += event["chunk"]
                    placeholder.write(content + "▌")
                if event.get("done"):
                    session_id = event.get("session_id", session_id)
                    tool_results = event.get("tool_results", [])

            placeholder.write(content)

        except requests.HTTPError as exc:
            error_text = f"API error {exc.response.status_code}: {exc.response.text}"
            placeholder.error(error_text)
        except Exception as exc:
            error_text = f"Request failed: {exc}"
            placeholder.error(error_text)

        for tr in tool_results:
            _render_tool_result(tr, key_prefix="cur", show_actions=True)

    st.session_state.session_id = session_id
    st.session_state.messages.append({
        "role": "assistant",
        "content": error_text or content,
        "tool_results": tool_results,
    })
