from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob._core import recency_rank_presorted, recency_rank_values_indexed
from fastmob.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_presorted_user_ends,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_ms,
    _take_uid_values,
    _to_native,
    _with_datetime_column,
)

_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _unpack_rank(raw: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        _arrow_result_values(raw[0]),
        _arrow_result_values(raw[1]),
        _arrow_result_values(raw[2]),
        raw[3],
    )


def recency_rank(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Return the recency rank of each distinct location for every user.

    The recency rank ``K_s(r_i)`` of location ``r_i`` is 1 if it is the most
    recently visited location, 2 if it is the second-most recently visited, and
    so on.  Ties (multiple visits to the same ``(lat, lng)`` pair) are resolved
    by keeping only the latest visit for each location before ranking.

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
    presorted : bool, optional
        When True, trust that rows are already grouped by user and ordered by
        datetime within each user, then use the contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per ``(user, location)`` pair with columns
        ``[uid_col, lat_col, lng_col, "recency_rank"]``.
        Rank 1 is the most recently visited location.
        The returned backend matches the input backend.



    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from fastmob import recency_rank
    >>> result = recency_rank(df)
    >>> print(result.head().to_string(index=False))
     uid       lat         lng  recency_rank
       0 39.891383 -105.070814             1
       0 39.891077 -105.068532             2
       0 39.750469 -104.999073             3
       0 39.752713 -104.996337             4
       0 39.752508 -104.996637             5

    References
    ----------
    - [BDEM2015] Barbosa, H., de Lima-Neto, F. B., Evsukoff, A., Menezes, R. (2015) The effect of recency to human mobility, EPJ Data Science 4(21), https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-015-0059-8
    """
    df = nw.from_native(traj, eager_only=True)
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df = _with_datetime_column(df, datetime_col).with_columns(
        nw.col(lat_col).cast(nw.Float64),
        nw.col(lng_col).cast(nw.Float64),
    )

    ops = _EXTRACTOR.get_ops(df)
    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        raw = recency_rank_presorted(lats_data, lngs_data, ends)
    else:
        timestamps = _extract_timestamps_ms(df, datetime_col)
        timestamps_data = ops["extract_data"](timestamps)
        uid_values, indices, ends = _build_time_ordered_user_ranges(df, uid_col, datetime_col, timestamps_data)
        raw = recency_rank_values_indexed(lats_data, lngs_data, indices, ends)
    out_lats, out_lngs, ranks, user_indices = _unpack_rank(raw)

    if uid_col is None:
        return _to_native({lat_col: out_lats, lng_col: out_lngs, "recency_rank": ranks}, df)
    return _to_native(
        {
            uid_col: _take_uid_values(uid_values, user_indices),
            lat_col: out_lats,
            lng_col: out_lngs,
            "recency_rank": ranks,
        },
        df,
    )
