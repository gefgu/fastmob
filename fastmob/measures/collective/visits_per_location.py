from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.utils._common import (
    DATETIME_CANDIDATES,
    LOCATION_CANDIDATES,
    UID_CANDIDATES,
    _detect_required_column,
    _pick_existing_column,
)


def visits_per_location(
    traj: Any,
    *,
    datetime_col: str | None = None,
    location_id_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total number of visits for each distinct location.

    Counts all trajectory rows (visits) per unique location ID across all
    users. Location IDs must be global-scoped, so the same ID represents the
    same location for every user. This is the population-level analogue of
    per-user ``location_frequency``.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …). Must have datetime and a globally scoped location-ID
        column.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    location_id_col : str or None, optional
        Explicit location-ID column name. Auto-detected when None. IDs must
        be global-scoped rather than user-scoped.
    uid_col : str or None, optional
        Explicit user-ID column name. Auto-detected when None (not used for
        aggregation, but retained for API consistency).

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per distinct location ID with columns
        ``[location_id_col, "n_visits"]``, sorted by descending visit count.
        The returned backend matches the input backend.

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
    >>> df = df.dropna(subset=["uid", "datetime", "location_id"])[
    ...     ["uid", "datetime", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime                       location_id
       0 2010-10-16 06:02:04+00:00 7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 424eb3dd143292f9e013efa00486c907
    >>> from fastmob import visits_per_location
    >>> result = visits_per_location(df)
    >>> print(result.head().to_string(index=False))
                              location_id  n_visits
    7a0f88982aa015062b95e3b4843f9ca2              340
    dd7cd3d264c2d063832db506fba8bf79              297
    9848afcc62e500a01cf6fbf24b797732f8963683       297
    2ef143e12038c870038df53e0478cefc               249
    424eb3dd143292f9e013efa00486c907               232

    References
    ----------
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

    See Also
    --------
    homes_per_location : Number of users whose home is at each location.
    location_frequency : Per-user visit frequency (individual measure).
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col = _detect_required_column(df, datetime_col, DATETIME_CANDIDATES)
    location_id_col = _detect_required_column(df, location_id_col, LOCATION_CANDIDATES)
    uid_col = uid_col or _pick_existing_column(df.columns, UID_CANDIDATES)
    required_cols = [datetime_col, location_id_col]
    if uid_col is not None:
        required_cols.append(uid_col)
    df = df.drop_nulls(subset=required_cols)

    result = (
        df.select(location_id_col)
        .group_by(location_id_col)
        .agg(nw.len().alias("n_visits"))
        .sort("n_visits", descending=True)
    )

    return result.to_native()
