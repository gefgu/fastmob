from __future__ import annotations

import re
from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob.utils._common import _detect_trajectory_columns, _with_datetime_column

_FREQ_RE = re.compile(r"^\s*(?P<count>\d+)?\s*(?P<unit>[A-Za-z]+)\s*$")
_CASE_SENSITIVE_ALIASES = {
    "MS": "mo",  # pandas month-start = Narwhals month truncation
}

_FREQ_UNIT_ALIASES = {
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "t": "m",
    "m": "m",
    "min": "m",
    "mins": "m",
    "minute": "m",
    "minutes": "m",
    "h": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
    "mo": "mo",
    "month": "mo",
    "months": "mo",
    "q": "q",
    "quarter": "q",
    "quarters": "q",
    "y": "y",
    "year": "y",
    "years": "y",
}


def _normalize_frequency_for_narwhals(freq: str) -> str | None:
    """Return a Narwhals duration string for common pandas offset aliases."""
    match = _FREQ_RE.match(freq)
    if match is None:
        return None

    count = match.group("count") or "1"
    if int(count) <= 0:
        return None

    unit = match.group("unit")
    if unit in _CASE_SENSITIVE_ALIASES:
        return f"{count}{_CASE_SENSITIVE_ALIASES[unit]}"
    if any(char.isupper() for char in unit) and unit not in {"D", "H", "S", "T"}:
        return None
    if unit == "M":
        return None
    unit_key = unit.lower()
    normalized_unit = _FREQ_UNIT_ALIASES.get(unit_key)
    if normalized_unit is None:
        return None
    return f"{count}{normalized_unit}"


def visits_per_time_unit(
    traj: Any,
    time_unit: str | None = None,
    *,
    freq: str = "1h",
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the number of trajectory points per time unit across all users.

    Groups all trajectory records by time bins of width ``freq`` and counts
    the number of records (visits) in each bin.  The result covers only bins
    that contain at least one record.

    The ``freq`` parameter follows pandas offset alias syntax (e.g. ``"1h"``,
    ``"1D"``, ``"15min"``).  Results are returned using the same dataframe
    backend as the input.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    time_unit : str or None, optional
        Alias for ``freq`` (skmob compatibility).  When provided, overrides
        ``freq``.
    freq : str, optional
        Pandas-compatible offset alias for the time bin width.  Default: ``"1h"``.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per non-empty time bin with columns
        ``[datetime_col, "n_visits"]``, sorted chronologically.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.data.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from fastmob import visits_per_time_unit
    >>> result = visits_per_time_unit(df)
    >>> print(result.head().to_string(index=False))
                     datetime  n_visits
    2008-06-21 17:00:00+00:00         1
    2008-06-22 01:00:00+00:00         1
    2008-06-22 05:00:00+00:00         1
    2008-06-22 17:00:00+00:00         1
    2008-06-22 18:00:00+00:00         1

    References
    ----------
    - [PRS2016] Pappalardo, L., Rinzivillo, S. & Simini, F. (2016) Human Mobility Modelling: exploration and preferential return meet the gravity model. Procedia Computer Science 83, 934-939, http://dx.doi.org/10.1016/j.procs.2016.04.188
    """
    if time_unit is not None:
        freq = time_unit

    df = nw.from_native(unwrap_native(traj), eager_only=True)

    datetime_col, lat_col, lng_col, _ = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    # Narrow to only the columns needed to reproduce _prepare_trajectory's
    # null-drop criterion (datetime, lat, lng -- matching its exact subset)
    # before dropping nulls, then drop lat/lng immediately: this op never
    # reads their values, only whether a row's location was present, so no
    # Float64 cast is needed either. Avoids materializing/casting the
    # caller's full (possibly much wider, e.g. visit-shaped) trajectory
    # frame for an op that only ever returns datetime/n_visits.
    df = df.select([datetime_col, lat_col, lng_col])
    df = _with_datetime_column(df, datetime_col)
    df = df.drop_nulls(subset=[datetime_col, lat_col, lng_col]).select(datetime_col)

    normalized_freq = _normalize_frequency_for_narwhals(freq)

    if normalized_freq is None:
        # Pandas resample fallback for frequency strings Narwhals cannot truncate
        # (e.g. "1W"). Uses nw.DataFrame.to_pandas() as the bridge; no direct
        # pandas import needed. Preserves the caller's original backend via
        # nw.from_dict(..., backend=backend).
        backend = df.implementation
        pd_df = df.select(datetime_col).to_pandas()
        resampled = pd_df.resample(freq, on=datetime_col).size().rename("n_visits").reset_index()
        resampled = resampled[resampled["n_visits"] > 0]
        return (
            nw.from_dict(
                {
                    datetime_col: resampled[datetime_col].tolist(),
                    "n_visits": resampled["n_visits"].tolist(),
                },
                backend=backend,
            )
            .sort(datetime_col)
            .to_native()
        )

    return (
        df.select(nw.col(datetime_col).dt.truncate(normalized_freq).alias(datetime_col))
        .group_by(datetime_col)
        .agg(nw.len().alias("n_visits"))
        .sort(datetime_col)
        .to_native()
    )
