from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob._core import number_of_visits_indexed, number_of_visits_presorted
from fastmob.utils._common import (
    _build_user_ranges_auto,
    _detect_trajectory_columns,
    _to_native,
)


def number_of_visits(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total number of trajectory points (visits) for each user.

    A "visit" is defined as one row in the trajectory dataframe after null
    removal. The result is the row count per user — identical to the skmob
    ``number_of_visits`` individual measure.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per user with columns ``[uid_col, "number_of_visits"]``.
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
    >>> from fastmob import number_of_visits
    >>> result = number_of_visits(df)
    >>> print(result.head().to_string(index=False))
     uid  number_of_visits
       0              2099
       1              1210
       2              1691
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = df.with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    uid_values, indices, ends = _build_user_ranges_auto(
        df,
        uid_col,
        coordinates=(df.get_column(lat_col).to_arrow(), df.get_column(lng_col).to_arrow()),
    )
    if indices is None:
        counts = number_of_visits_presorted(len(df), ends)
        if uid_col is None:
            return _to_native({"number_of_visits": counts}, df)
        return _to_native({uid_col: uid_values, "number_of_visits": counts}, df)

    valid_mask = (~df.get_column(lat_col).is_null() & ~df.get_column(lng_col).is_null()).to_numpy()
    counts = number_of_visits_indexed(len(df), indices, ends, valid_mask)

    if uid_col is None:
        return _to_native({"number_of_visits": counts}, df)
    return _to_native({uid_col: uid_values, "number_of_visits": counts}, df)
