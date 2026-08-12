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


class CreateBookingIn(BaseModel):
    flight_id: str
    passenger_name: str


class CancelBookingIn(BaseModel):
    booking_id: str


class GetBookingIn(BaseModel):
    booking_id: str
