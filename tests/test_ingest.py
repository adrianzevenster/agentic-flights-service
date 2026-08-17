"""Unit tests for ingest utilities and flight_doc.

All tests are pure — no Qdrant, Ollama, or database connections.
"""
import uuid

import pytest
from pydantic import ValidationError

from app.ingest.ingest_archive import _FlightRow, _stable_flight_id
from app.rag.flight_doc import flight_doc


# ─── _stable_flight_id ────────────────────────────────────────────────────────

_BASE_ROW = {
    "year": 2023, "month": 6, "day": 15,
    "carrier": "DL", "flight": 100,
    "origin": "ATL", "dest": "JFK",
    "sched_dep_time": 800,
}


def test_stable_flight_id_deterministic():
    assert _stable_flight_id(_BASE_ROW) == _stable_flight_id(_BASE_ROW)


def test_stable_flight_id_is_valid_uuid():
    fid = _stable_flight_id(_BASE_ROW)
    uuid.UUID(fid)  # raises ValueError if not a valid UUID


def test_stable_flight_id_differs_on_distinct_inputs():
    other = {**_BASE_ROW, "flight": 101}
    assert _stable_flight_id(_BASE_ROW) != _stable_flight_id(other)


def test_stable_flight_id_differs_on_carrier_change():
    other = {**_BASE_ROW, "carrier": "UA"}
    assert _stable_flight_id(_BASE_ROW) != _stable_flight_id(other)


# ─── _FlightRow validation ────────────────────────────────────────────────────

_VALID_ROW = {
    "year": 2023, "month": 6, "day": 15,
    "carrier": "dl",  # lowercase — validator should uppercase
    "flight": 100,
    "origin": "atl",
    "dest": "jfk",
}


def test_flight_row_valid():
    row = _FlightRow.model_validate(_VALID_ROW)
    assert row.carrier == "DL"
    assert row.origin == "ATL"
    assert row.dest == "JFK"


def test_flight_row_uppercases_carrier_origin_dest():
    row = _FlightRow.model_validate(_VALID_ROW)
    assert row.carrier == row.carrier.upper()
    assert row.origin == row.origin.upper()
    assert row.dest == row.dest.upper()


def test_flight_row_optional_fields_default_none():
    row = _FlightRow.model_validate(_VALID_ROW)
    assert row.sched_dep_time is None
    assert row.distance is None


def test_flight_row_missing_required_raises():
    bad = {k: v for k, v in _VALID_ROW.items() if k != "carrier"}
    with pytest.raises(ValidationError):
        _FlightRow.model_validate(bad)


def test_flight_row_missing_year_raises():
    bad = {k: v for k, v in _VALID_ROW.items() if k != "year"}
    with pytest.raises(ValidationError):
        _FlightRow.model_validate(bad)


# ─── flight_doc ───────────────────────────────────────────────────────────────

def test_flight_doc_contains_all_key_fields():
    row = {
        "carrier": "DL", "flight": 100, "origin": "ATL", "dest": "JFK",
        "year": 2023, "month": 6, "day": 15,
        "sched_dep_time": 800, "sched_arr_time": 1030,
        "dep_time": 810, "arr_time": 1045,
        "distance": 760, "air_time": 115,
        "tailnum": "N123DL",
    }
    doc = flight_doc(row)
    assert "DL" in doc
    assert "ATL" in doc
    assert "JFK" in doc
    assert "N123DL" in doc
    assert "->" in doc


def test_flight_doc_handles_missing_keys_gracefully():
    # Missing keys should not raise — they render as None/empty string.
    doc = flight_doc({})
    assert isinstance(doc, str)
    assert "->" in doc  # format string is always present
