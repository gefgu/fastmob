from __future__ import annotations

import math
from typing import Any

import narwhals as nw

from fastmob.utils._common import _detect_trajectory_columns, _prepare_trajectory


def random_location_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the random entropy for each distinct location across all users.

    Random location entropy is \\(\\log_2(n)\\), where \\(n\\) is the number of
    distinct users who visited the location.  A location is a unique exact
    ``(lat, lng)`` pair — matching the skmob convention.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent all rows are treated as
        coming from a single individual (entropy will be 0 everywhere).
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
        One row per distinct ``(lat, lng)`` location with columns
        ``[lat_col, lng_col, "random_entropy"]``.
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
    >>> from fastmob import random_location_entropy
    >>> result = random_location_entropy(df)
    >>> print(result.round({"random_entropy": 3}).head().to_string(index=False))
          lat        lng  random_entropy
     0.000000   0.000000           1.585
    29.532220 -98.300914           0.000
    29.942673 -90.064455           0.000
    29.948116 -90.063436           0.000
    29.948125 -90.063510           0.000

    See Also
    --------
    uncorrelated_location_entropy : Location entropy weighted by visitor frequency.
    """
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

    if uid_col is None:
        # Single user: each location is visited by exactly 1 user -> entropy = 0.
        locs = df.select([lat_col, lng_col]).unique().sort([lat_col, lng_col])
        result = locs.with_columns(nw.lit(0.0).alias("random_entropy"))
        return result.to_native()

    # Count distinct users per (lat, lng) location.
    grouped = (
        df.select([uid_col, lat_col, lng_col])
        .unique()  # one row per (uid, lat, lng) combination
        .group_by([lat_col, lng_col])
        .agg(nw.col(uid_col).n_unique().alias("__n_users__"))
        .sort([lat_col, lng_col])
    )

    result = grouped.with_columns((nw.col("__n_users__").log() / math.log(2)).alias("random_entropy")).select(
        [lat_col, lng_col, "random_entropy"]
    )

    return result.to_native()
