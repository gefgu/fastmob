"""Shared column auto-detection for the Staypoints/Triplegs/Trips/Tours levels.

These levels need two datetime columns (``started_at``/``finished_at``),
unlike `TrajDataFrame`'s single ``datetime_col`` model, so they get their own
small detection helper rather than reusing
`fastmob.utils._common._detect_trajectory_columns`.
"""

from __future__ import annotations

from typing import Any

from fastmob.utils._common import UID_CANDIDATES, _pick_existing_column

STARTED_AT_CANDIDATES: list[str] = ["started_at", "datetime", "start_time"]
FINISHED_AT_CANDIDATES: list[str] = ["finished_at", "leaving_datetime", "end_time"]


def _detect_interval_columns(
    columns: Any,
    uid_col: str | None = None,
    started_at_col: str | None = None,
    finished_at_col: str | None = None,
) -> tuple[str | None, str, str]:
    """Resolve ``(uid_col, started_at_col, finished_at_col)`` from ``columns``.

    Accepts explicit overrides for any column; auto-detects the rest.
    """
    if uid_col is None:
        uid_col = _pick_existing_column(columns, UID_CANDIDATES)
    if started_at_col is None:
        started_at_col = _pick_existing_column(columns, STARTED_AT_CANDIDATES)
    if finished_at_col is None:
        finished_at_col = _pick_existing_column(columns, FINISHED_AT_CANDIDATES)

    if started_at_col is None:
        raise ValueError(f"Could not find a 'started_at' column; checked {STARTED_AT_CANDIDATES}")
    if finished_at_col is None:
        raise ValueError(f"Could not find a 'finished_at' column; checked {FINISHED_AT_CANDIDATES}")

    return uid_col, started_at_col, finished_at_col
