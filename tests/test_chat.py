"""Unit tests for chat.py pure functions.

No network or session store calls — all tests operate on local state only.
Token-based context tests rely on CONTEXT_TOKEN_LIMIT=500 set in conftest.py;
synthetic histories are sized to straddle that threshold.
"""
import json

from app.api.chat import _parse_args, _trim_history, _count_tokens
from app.core.config import settings


def _make_history(n_user_msgs: int, content_size: int = 10) -> list[dict]:
    history = [{"role": "system", "content": "sys"}]
    for i in range(n_user_msgs):
        history.append({"role": "user", "content": f"u{i}" * content_size})
        history.append({"role": "assistant", "content": f"a{i}" * content_size})
    return history


# ─── _count_tokens ────────────────────────────────────────────────────────────

def test_count_tokens_nonempty():
    history = _make_history(1)
    assert _count_tokens(history) > 0


def test_count_tokens_grows_with_history():
    small = _make_history(1)
    large = _make_history(5)
    assert _count_tokens(large) > _count_tokens(small)


def test_count_tokens_returns_int():
    assert isinstance(_count_tokens(_make_history(1)), int)


# ─── _trim_history ────────────────────────────────────────────────────────────

def test_trim_history_no_op_when_under_limit():
    history = _make_history(1)
    assert _count_tokens(history) < settings.CONTEXT_TOKEN_LIMIT
    assert _trim_history(history) is history


def test_trim_history_drops_oldest_non_system():
    history = _make_history(200)
    assert _count_tokens(history) > settings.CONTEXT_TOKEN_LIMIT
    trimmed = _trim_history(history)
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) < len(history)


def test_trim_history_preserves_system_always():
    history = _make_history(200)
    trimmed = _trim_history(history)
    assert trimmed[0]["content"] == "sys"


def test_trim_history_keeps_at_least_min_messages():
    history = _make_history(200)
    trimmed = _trim_history(history)
    assert len(trimmed) >= 1 + settings.CONTEXT_MIN_MESSAGES


def test_trim_history_result_fits_token_limit():
    history = _make_history(200)
    trimmed = _trim_history(history)
    assert _count_tokens(trimmed) <= settings.CONTEXT_TOKEN_LIMIT


def test_trim_history_no_mutation_when_no_drop_needed():
    history = _make_history(1)
    result = _trim_history(history)
    assert result is history


# ─── _parse_args ──────────────────────────────────────────────────────────────

def test_parse_args_passthrough_dict():
    d = {"query": "JFK to LAX", "limit": 5}
    assert _parse_args(d) == d


def test_parse_args_valid_json_string():
    s = json.dumps({"flight_id": "abc-123"})
    assert _parse_args(s) == {"flight_id": "abc-123"}


def test_parse_args_malformed_string_returns_empty():
    assert _parse_args("not json at all") == {}


def test_parse_args_json_embedded_in_text():
    s = 'Here are the arguments: {"query": "morning flights", "origin": "JFK"}'
    result = _parse_args(s)
    assert result.get("query") == "morning flights"
    assert result.get("origin") == "JFK"


def test_parse_args_json_in_markdown_fence():
    s = "```json\n{\"passenger_name\": \"Alice\"}\n```"
    result = _parse_args(s)
    assert result.get("passenger_name") == "Alice"


def test_parse_args_empty_string_returns_empty():
    assert _parse_args("") == {}


def test_parse_args_none_dict_returns_empty():
    assert _parse_args(None) == {}  # type: ignore[arg-type]
