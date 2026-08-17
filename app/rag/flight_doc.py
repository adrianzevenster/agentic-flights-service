"""Flight document format shared by the ingest pipeline and HyDE retrieval.

Both ingest (app/ingest/ingest_archive.py) and HyDE query generation
(app/rag/retrieval.py) must produce identical surface forms so that their
embeddings land in the same region of the vector space.  Both call flight_doc()
or refer to HYDE_PROMPT defined here, so any format change must be made once
and re-ingested.

Document format
---------------
The indexed string includes both the IATA code and the English name for each
carrier and airport.  For example:

    DL Delta flight 1047 LGA LaGuardia to ATL Atlanta 2013-1-1 sched 1759->2042 actual 1757->2027 dist 762 air 125

Including both representations means:
- Queries using full names ("Delta", "LaGuardia") land close to indexed docs.
- Queries using IATA codes ("DL", "LGA") also land close to indexed docs.
- The embedding model does not need to know that "DL" means "Delta" out of
  context — the mapping is explicit in every document.

The year-month-day and time fields use the compact integer representation that
matches the Qdrant payload (sched_dep_time=1759 means 17:59) so that retrieval
results can be mapped directly to filter conditions.

HyDE prompt
-----------
The prompt instructs the LLM to produce this exact format, including both IATA
code and name for each field.  It also provides a worked example so the model
does not abbreviate or embellish the output.  The example year (2013) matches
the indexed dataset so that generated distances and flight times are plausible.
"""
from __future__ import annotations

_CARRIER_NAMES: dict[str, str] = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "EV": "ExpressJet Airlines",
    "F9": "Frontier Airlines",
    "FL": "AirTran Airways",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "WN": "Southwest Airlines",
    "YV": "Mesa Airlines",
}

_AIRPORT_CITIES: dict[str, str] = {
    "ATL": "Atlanta",
    "AUS": "Austin",
    "BDL": "Hartford",
    "BNA": "Nashville",
    "BOS": "Boston",
    "BTV": "Burlington",
    "BUF": "Buffalo",
    "BWI": "Baltimore",
    "CAK": "Akron Canton",
    "CHS": "Charleston",
    "CLE": "Cleveland",
    "CLT": "Charlotte",
    "CMH": "Columbus",
    "CVG": "Cincinnati",
    "DAY": "Dayton",
    "DCA": "Washington Reagan",
    "DEN": "Denver",
    "DFW": "Dallas Fort Worth",
    "DSM": "Des Moines",
    "DTW": "Detroit",
    "EWR": "Newark",
    "FLL": "Fort Lauderdale",
    "GRR": "Grand Rapids",
    "GSO": "Greensboro",
    "GSP": "Greenville",
    "HNL": "Honolulu",
    "HOU": "Houston Hobby",
    "IAD": "Washington Dulles",
    "IAH": "Houston",
    "IND": "Indianapolis",
    "JAX": "Jacksonville",
    "JFK": "New York JFK",
    "LAS": "Las Vegas",
    "LAX": "Los Angeles",
    "LGA": "LaGuardia",
    "LGB": "Long Beach",
    "MCI": "Kansas City",
    "MCO": "Orlando",
    "MDW": "Chicago Midway",
    "MEM": "Memphis",
    "MHT": "Manchester",
    "MIA": "Miami",
    "MKE": "Milwaukee",
    "MSN": "Madison",
    "MSP": "Minneapolis",
    "MSY": "New Orleans",
    "MYR": "Myrtle Beach",
    "OAK": "Oakland",
    "ORD": "Chicago O'Hare",
    "ORF": "Norfolk",
    "PBI": "West Palm Beach",
    "PDX": "Portland",
    "PHL": "Philadelphia",
    "PHX": "Phoenix",
    "PIT": "Pittsburgh",
    "PSE": "Ponce",
    "PVD": "Providence",
    "PWM": "Portland ME",
    "RDU": "Raleigh Durham",
    "RIC": "Richmond",
    "ROC": "Rochester",
    "RSW": "Fort Myers",
    "SAN": "San Diego",
    "SAT": "San Antonio",
    "SAV": "Savannah",
    "SEA": "Seattle",
    "SFO": "San Francisco",
    "SJC": "San Jose",
    "SJU": "San Juan",
    "SLC": "Salt Lake City",
    "SMF": "Sacramento",
    "SNA": "Orange County",
    "SRQ": "Sarasota",
    "STL": "St Louis",
    "STT": "St Thomas",
    "SYR": "Syracuse",
    "TPA": "Tampa",
    "TUL": "Tulsa",
    "TYS": "Knoxville",
    "XNA": "Northwest Arkansas",
}


HYDE_PROMPT = (
    "Generate a realistic flight record matching this search: {query}\n\n"
    "Use this EXACT format (one line):\n"
    "<2-LETTER-CODE> <AIRLINE-NAME> flight <NUMBER> <ORIGIN-IATA> <ORIGIN-CITY> to <DEST-IATA> <DEST-CITY> "
    "<YEAR>-<MONTH>-<DAY> sched <DEP_HHMM>-><ARR_HHMM> actual <DEP_HHMM>-><ARR_HHMM> dist <MILES> air <MINUTES>\n\n"
    "Rules:\n"
    "- Carrier: 2-letter IATA code first, then full airline name (e.g. 'DL Delta Air Lines', 'UA United Airlines', "
    "'B6 JetBlue Airways', 'AA American Airlines', 'WN Southwest Airlines')\n"
    "- Airports: 3-letter IATA code first, then city name (e.g. 'LGA LaGuardia', 'JFK New York JFK', "
    "'EWR Newark', 'ATL Atlanta', 'ORD Chicago O\\'Hare', 'LAX Los Angeles')\n"
    "- Year must be 2013. Times in HHMM integer format (e.g. 0800 = 8am, 1430 = 2:30pm)\n"
    "- Distances in miles, airtime in minutes\n\n"
    "Example: DL Delta Air Lines flight 1047 LGA LaGuardia to ATL Atlanta 2013-1-1 "
    "sched 1759->2042 actual 1757->2027 dist 762 air 125\n\n"
    "Output only the record, nothing else."
)


def flight_doc(row: dict) -> str:
    """Render a flight row as the canonical indexed document string.

    Includes both the IATA code and English name for carrier and airports so
    that natural-language queries ("Delta from LaGuardia to Atlanta") and
    code-based queries ("DL LGA ATL") both embed close to this document.

    Args:
        row: Dict with flight fields (carrier, flight, origin, dest, year, month,
             day, sched_dep_time, sched_arr_time, dep_time, arr_time, distance,
             air_time, tailnum). Missing keys render as None.

    Returns:
        Single-line string in the format consumed by HYDE_PROMPT.
    """
    carrier = row.get("carrier") or ""
    origin  = row.get("origin")  or ""
    dest    = row.get("dest")    or ""

    carrier_name = _CARRIER_NAMES.get(carrier, carrier)
    origin_city  = _AIRPORT_CITIES.get(origin, origin)
    dest_city    = _AIRPORT_CITIES.get(dest, dest)

    return (
        f"{carrier} {carrier_name} flight {row.get('flight')} "
        f"{origin} {origin_city} to {dest} {dest_city} "
        f"{row.get('year')}-{row.get('month')}-{row.get('day')} "
        f"sched {row.get('sched_dep_time')}->{row.get('sched_arr_time')} "
        f"actual {row.get('dep_time')}->{row.get('arr_time')} "
        f"dist {row.get('distance')} air {row.get('air_time')}"
    )
