"""Unit tests for retrieval.py pure functions.

No Qdrant or Ollama connections — tests target filter construction and
result conversion logic only.
"""
import pytest
from qdrant_client.http import models as qm

from app.rag.retrieval import _build_filter, _point_to_dict, _TIME_BANDS


# ─── _build_filter ────────────────────────────────────────────────────────────

def test_build_filter_none_when_no_args():
    assert _build_filter(None, None, None, None, None, None, None) is None


def test_build_filter_origin_only():
    f = _build_filter(origin="jfk", dest=None, carrier=None, year=None, month=None, day=None, flight=None)
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "origin"
    assert f.must[0].match.value == "JFK"


def test_build_filter_uppercases_string_fields():
    f = _build_filter(origin="jfk", dest="lax", carrier="dl", year=None, month=None, day=None, flight=None)
    values = {c.key: c.match.value for c in f.must}
    assert values["origin"] == "JFK"
    assert values["dest"] == "LAX"
    assert values["carrier"] == "DL"


def test_build_filter_all_exact_fields():
    f = _build_filter(
        origin="ATL", dest="ORD", carrier="UA",
        year=2023, month=6, day=15, flight=200,
    )
    assert len(f.must) == 7


def test_build_filter_month_range():
    f = _build_filter(
        origin=None, dest=None, carrier=None,
        year=None, month=None, day=None, flight=None,
        month_min=6, month_max=8,
    )
    assert f is not None
    range_conds = [c for c in f.must if hasattr(c, "range") and c.range is not None]
    assert len(range_conds) == 1
    assert range_conds[0].key == "month"
    assert range_conds[0].range.gte == 6
    assert range_conds[0].range.lte == 8


def test_build_filter_time_of_day_morning():
    f = _build_filter(
        origin=None, dest=None, carrier=None,
        year=None, month=None, day=None, flight=None,
        time_of_day="morning",
    )
    assert f is not None
    range_conds = [c for c in f.must if hasattr(c, "range") and c.range is not None]
    assert len(range_conds) == 1
    lo, hi = _TIME_BANDS["morning"]
    assert range_conds[0].range.gte == lo
    assert range_conds[0].range.lte == hi


def test_build_filter_unknown_time_of_day_ignored():
    # Unknown band names must not add a filter — don't corrupt the query.
    f = _build_filter(
        origin=None, dest=None, carrier=None,
        year=None, month=None, day=None, flight=None,
        time_of_day="whenever",
    )
    assert f is None


def test_build_filter_combined_carrier_and_morning():
    f = _build_filter(
        origin=None, dest=None, carrier="AA",
        year=None, month=None, day=None, flight=None,
        time_of_day="morning",
    )
    assert len(f.must) == 2


# ─── _point_to_dict ───────────────────────────────────────────────────────────

class _FakePoint:
    def __init__(self, id, payload, score):
        self.id = id
        self.payload = payload
        self.score = score


def test_point_to_dict_basic():
    p = _FakePoint(
        id="abc-123",
        payload={"carrier": "DL", "origin": "ATL", "dest": "JFK", "flight": 100},
        score=0.87,
    )
    d = _point_to_dict(p)
    assert d["flight_id"] == "abc-123"
    assert d["score"] == pytest.approx(0.87)
    assert d["carrier"] == "DL"


def test_point_to_dict_none_score():
    p = _FakePoint(id="x", payload={}, score=None)
    d = _point_to_dict(p)
    assert d["score"] is None


def test_point_to_dict_none_payload():
    p = _FakePoint(id="y", payload=None, score=0.5)
    d = _point_to_dict(p)
    assert d["flight_id"] == "y"
    assert d["carrier"] is None
