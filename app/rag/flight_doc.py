# The ingest doc format and the HyDE prompt must produce identical surface forms so
# their embeddings land in the same region of the vector space. Keep both here.
from __future__ import annotations

HYDE_PROMPT = (
    "Generate a realistic flight record matching this search: {query}\n\n"
    "Format exactly (one line): "
    "<CARRIER> flight <NUMBER> <ORIGIN>-><DEST> "
    "date <YEAR>-<MONTH>-<DAY> "
    "sched <DEP_HHMM>-><ARR_HHMM> "
    "actual <DEP_HHMM>-><ARR_HHMM> "
    "dist <MILES> air <MINUTES> tail <TAILNUM>\n\n"
    "Output only the record."
)


def flight_doc(row: dict) -> str:
    """Render a flight row as the canonical indexed document string.

    Args:
        row: Dict with flight fields (carrier, flight, origin, dest, year, month,
             day, sched_dep_time, sched_arr_time, dep_time, arr_time, distance,
             air_time, tailnum). Missing keys render as empty strings.

    Returns:
        Single-line string in the format expected by the embedding model.
    """
    return (
        f"{row.get('carrier')} flight {row.get('flight')} "
        f"{row.get('origin')}->{row.get('dest')} "
        f"date {row.get('year')}-{row.get('month')}-{row.get('day')} "
        f"sched {row.get('sched_dep_time')}->{row.get('sched_arr_time')} "
        f"actual {row.get('dep_time')}->{row.get('arr_time')} "
        f"dist {row.get('distance')} air {row.get('air_time')} "
        f"tail {row.get('tailnum')}"
    )
