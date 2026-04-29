from __future__ import annotations

import math
from typing import Iterable

import narwhals as nw

_ROW_ORDER_COL = "__skmob2_row_order__"
_USER_RANGE_START_COL = "__skmob2_user_range_start__"

# ---------------------------------------------------------------------------
# Authoritative candidate lists for column auto-detection.
# All measure files should import from here rather than defining their own.
# ---------------------------------------------------------------------------

DATETIME_CANDIDATES: list[str] = ["datetime", "check-in_time", "timestamp", "time"]
LAT_CANDIDATES: list[str] = ["lat", "latitude"]
LNG_CANDIDATES: list[str] = ["lng", "lon", "longitude"]
UID_CANDIDATES: list[str] = ["uid", "user", "user_id"]

ACTIVITY_CANDIDATES: list[str] = ["purpose", "activity", "act", "location_type"]
TIMESTAMP_CANDIDATES: list[str] = ["start_timestamp", "timestamp", "datetime"]
DAY_CANDIDATES: list[str] = ["day_of_week", "day", "weekday"]
USER_ID_CANDIDATES: list[str] = ["user_id", "uid", "agent_id", "user"]
LOCATION_CANDIDATES: list[str] = ["location_id", "area", "venueId", "location"]
DURATION_CANDIDATES: list[str] = ["duration_steps", "duration_minutes", "duration"]
PURPOSE_CANDIDATES: list[str] = ["purpose", "activity", "location_type"]
LOCATION_TYPE_CANDIDATES: list[str] = ["location_type", "purpose", "activity"]
ORIGIN_CANDIDATES: list[str] = ["origin_area", "Origin_Area", "area_o", "ORIGIN_AREA"]
DEST_CANDIDATES: list[str] = ["destination_area", "Dest_Area", "area_d", "DESTINATION_AREA"]


def _shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy in bits for a list of event counts.

    Parameters
    ----------
    counts:
        A list of non-negative integer counts (e.g. visit counts per location).
        Zero-count items are ignored (contribute 0 to entropy, matching the
        information-theoretic convention ``0 * log2(0) = 0``).

    Returns
    -------
    float
        Shannon entropy in bits.  Returns 0.0 when the total count is 0 or
        when all probability mass is concentrated on a single item.

    Examples
    --------
    >>> _shannon_entropy([1, 1])   # two equal-probability items -> 1 bit
    1.0
    >>> _shannon_entropy([1, 0])   # one item -> 0 bits
    0.0
    >>> _shannon_entropy([])
    0.0
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _pick_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in ``columns``, or None."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _detect_trajectory_columns(
    nw_df: nw.DataFrame,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> tuple[str, str, str, str | None]:
    """Auto-detect trajectory column names from a Narwhals DataFrame.

    Accepts explicit overrides for any column; auto-detects the rest from the
    authoritative candidate lists defined in this module.

    Parameters
    ----------
    nw_df:
        A Narwhals DataFrame (already wrapped via ``nw.from_native``).
    datetime_col:
        Explicit datetime column name; auto-detected when None.
    lat_col:
        Explicit latitude column name; auto-detected when None.
    lng_col:
        Explicit longitude column name; auto-detected when None.
    uid_col:
        Explicit user-ID column name; auto-detected when None. When no
        user-ID column is found the whole frame is treated as one user.

    Returns
    -------
    tuple[str, str, str, str | None]
        ``(datetime_col, lat_col, lng_col, uid_col)`` where ``uid_col`` may
        be None if no user-ID column exists.

    Raises
    ------
    ValueError
        When a required column (datetime, lat, or lng) cannot be found.

    Examples
    --------
    >>> import pandas as pd, narwhals as nw
    >>> df = pd.DataFrame({"datetime": [], "lat": [], "lng": [], "uid": []})
    >>> nw_df = nw.from_native(df, eager_only=True)
    >>> _detect_trajectory_columns(nw_df)
    ('datetime', 'lat', 'lng', 'uid')
    """
    columns = nw_df.columns

    if datetime_col is None:
        datetime_col = _pick_existing_column(columns, DATETIME_CANDIDATES)
    if lat_col is None:
        lat_col = _pick_existing_column(columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(columns, LNG_CANDIDATES)
    if uid_col is None:
        uid_col = _pick_existing_column(columns, UID_CANDIDATES)

    missing = [
        name
        for name, col in zip(
            ["datetime", "latitude", "longitude"],
            [datetime_col, lat_col, lng_col],
        )
        if col is None
    ]
    if missing:
        hints = {
            "datetime": DATETIME_CANDIDATES,
            "latitude": LAT_CANDIDATES,
            "longitude": LNG_CANDIDATES,
        }
        details = "; ".join(f"'{m}' (looked for: {hints[m]})" for m in missing)
        raise ValueError(
            f"Could not find required column(s): {details}. "
            f"Available columns: {columns}. "
            f"Pass the column name(s) explicitly."
        )

    return datetime_col, lat_col, lng_col, uid_col


def _prepare_trajectory(
    traj,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sort: bool = True,
) -> tuple[nw.DataFrame, str, str, str, str | None]:
    """Wrap, detect columns, optionally sort, and cast a raw trajectory into a clean DataFrame.

    This is the standard preprocessing pipeline shared by all trajectory-based
    measures (jump lengths, radius of gyration, etc.).  It:

    1. Wraps the input in Narwhals (accepting any eager backend).
    2. Auto-detects column names (with optional overrides).
    3. Drops nulls in the required coordinate/datetime columns.
    4. Optionally sorts by ``[uid, datetime]`` (with a stable row-order
       tiebreaker) to ensure chronological order within each user.
    5. Casts lat/lng to ``Float64``.

    Parameters
    ----------
    traj:
        Raw trajectory dataframe (any Narwhals-compatible backend).
    datetime_col, lat_col, lng_col, uid_col:
        Optional explicit column overrides; auto-detected when None.
    sort:
        Whether to sort by user and datetime. When False, rows keep their
        input order after null rows are dropped.

    Returns
    -------
    tuple[nw.DataFrame, str, str, str, str | None]
        ``(df, datetime_col, lat_col, lng_col, uid_col)`` where ``df`` is
        the cleaned Narwhals DataFrame and ``uid_col`` may be None.

    Raises
    ------
    ValueError
        When required columns cannot be found.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ...     "uid": ["u1", "u1", "u1"],
    ... })
    >>> clean_df, dt, lat, lng, uid = _prepare_trajectory(df)
    """
    nw_df = nw.from_native(traj, eager_only=True).with_row_index(_ROW_ORDER_COL)

    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        nw_df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    df = nw_df.drop_nulls(subset=[datetime_col, lat_col, lng_col])
    if sort:
        sort_cols = [uid_col, datetime_col, _ROW_ORDER_COL] if uid_col else [datetime_col, _ROW_ORDER_COL]
        df = df.sort(*sort_cols)

    df = (
        df
        .with_columns(
            nw.col(lat_col).cast(nw.Float64),
            nw.col(lng_col).cast(nw.Float64),
        )
        .drop(_ROW_ORDER_COL)
    )

    return df, datetime_col, lat_col, lng_col, uid_col


def _build_user_ranges(df: nw.DataFrame, uid_col: str | None) -> tuple[list, list[tuple[int, int]]]:
    """Split a uid-sorted DataFrame into per-user (uid_value, index_range) pairs.

    Returns
    -------
    tuple[list, list[tuple[int, int]]]
        ``(uid_values, ranges)`` where ``ranges[i]`` is the half-open row
        interval ``[start, end)`` for ``uid_values[i]``.  When ``uid_col`` is
        None the whole frame is treated as one user and returns
        ``([None], [(0, len(df))])``.
    """
    n = len(df)
    if uid_col is None:
        return [None], [(0, n)]
    if n == 0:
        return [], []

    starts_df = (
        df.select([uid_col])
        .with_row_index(_USER_RANGE_START_COL)
        .filter((nw.col(uid_col) != nw.col(uid_col).shift(1)).fill_null(True))
    )
    starts = starts_df.get_column(_USER_RANGE_START_COL).to_list()
    uid_values = starts_df.get_column(uid_col).to_list()
    ends = starts[1:] + [n]
    ranges = list(zip(starts, ends))

    return uid_values, ranges
