from __future__ import annotations

from typing import Any

import narwhals as nw

from .._common import _detect_trajectory_columns, _prepare_trajectory


def visits_per_location(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the total number of visits for each distinct location.

    Counts all trajectory rows (visits) per unique ``(lat, lng)`` pair across
    all users.  This is the population-level analogue of per-user
    ``location_frequency``.

    Parameters
    ----------
    traj : DataFrame-like
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    datetime_col : str or None, optional
        Explicit datetime column name.  Auto-detected when None.
    lat_col : str or None, optional
        Explicit latitude column name.  Auto-detected when None.
    lng_col : str or None, optional
        Explicit longitude column name.  Auto-detected when None.
    uid_col : str or None, optional
        Explicit user-ID column name.  Auto-detected when None (not used for
        aggregation, but retained for consistent preprocessing).

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per distinct ``(lat, lng)`` location with columns
        ``[lat_col, lng_col, "n_visits"]``, sorted by descending visit count.
        The returned backend matches the input backend.

    Examples
    --------
    >>> import pandas as pd
    >>> import fkmob
    >>> url = fkmob.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from fkmob import visits_per_location
    >>> result = visits_per_location(df)
    >>> print(result.head().to_string(index=False))
          lat         lng  n_visits
    39.739154 -104.984703       340
    37.630490 -122.411084       297
    37.580304 -122.343679       297
    37.584103 -122.366083       249
    39.762146 -104.982480       232

    References
    ----------
    - [PF2018] Pappalardo, L. & Simini, F. (2018) Data-driven generation of spatio-temporal routines in human mobility. Data Mining and Knowledge Discovery 32, 787-829, https://link.springer.com/article/10.1007/s10618-017-0548-4

    See Also
    --------
    homes_per_location : Number of users whose home is at each location.
    location_frequency : Per-user visit frequency (individual measure).
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

    result = (
        df.select([lat_col, lng_col])
        .group_by([lat_col, lng_col])
        .agg(nw.len().alias("n_visits"))
        .sort("n_visits", descending=True)
    )

    return result.to_native()
