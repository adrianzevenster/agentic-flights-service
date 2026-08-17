from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class SearchFlightsIn(BaseModel):
    query: str = Field(..., description="Natural language query")
    origin: Optional[str] = Field(None, description="IATA origin, e.g. JFK")
    dest: Optional[str] = Field(None, description="IATA destination, e.g. LAX")
    carrier: Optional[str] = Field(None, description="Carrier code, e.g. DL")
    limit: int = Field(10, ge=1, le=50)
    month_min: Optional[int] = Field(None, ge=1, le=12, description="Start month (1-12) for a month range filter")
    month_max: Optional[int] = Field(None, ge=1, le=12, description="End month (1-12) for a month range filter")
    time_of_day: Optional[str] = Field(
        None,
        description="Departure time band: early_morning | morning | afternoon | evening | late_night",
    )


class CreateBookingIn(BaseModel):
    flight_id: str
    passenger_name: str


class CancelBookingIn(BaseModel):
    booking_id: str


class GetBookingIn(BaseModel):
    booking_id: str
