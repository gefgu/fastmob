from __future__ import annotations

import re
from typing import Any

import narwhals as nw

from .._common import _detect_trajectory_columns, _prepare_trajectory

_FREQ_RE = re.compile(r"^\s*(?P<count>\d+)?\s*(?P<unit>[A-Za-z]+)\s*$")
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
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    time_unit:
        Alias for ``freq`` (skmob compatibility).  When provided, overrides
        ``freq``.
    freq:
        Pandas-compatible offset alias for the time bin width.  Default: ``"1h"``.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    DataFrame
        One row per non-empty time bin with columns
        ``[datetime_col, "n_visits"]``, sorted chronologically.



    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from skmob2 import visits_per_time_unit
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
    import pandas as pd  # noqa: PLC0415 - pandas offset aliases are the public compatibility contract

    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    normalized_freq = _normalize_frequency_for_narwhals(freq)
    if normalized_freq is not None:
        try:
            return (
                df.select(nw.col(datetime_col).dt.truncate(normalized_freq).alias(datetime_col))
                .group_by(datetime_col)
                .agg(nw.len().alias("n_visits"))
                .sort(datetime_col)
                .to_native()
            )
        except Exception:
            pass

    native = df.select([datetime_col]).to_native()
    if isinstance(native, pd.DataFrame):
        pandas_df = native.copy()
    else:
        pandas_df = pd.DataFrame({datetime_col: df.get_column(datetime_col).to_list()})

    pandas_df[datetime_col] = pd.to_datetime(pandas_df[datetime_col])
    counts = pandas_df.set_index(datetime_col).resample(freq).size().rename("n_visits").reset_index()
    counts = counts[counts["n_visits"] > 0].reset_index(drop=True)

    if isinstance(df.to_native(), pd.DataFrame):
        return counts

    return nw.from_dict(
        {
            datetime_col: counts[datetime_col].tolist(),
            "n_visits": counts["n_visits"].astype("int64").tolist(),
        },
        backend=df.implementation,
    ).to_native()
