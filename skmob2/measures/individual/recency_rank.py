from __future__ import annotations
import narwhals as nw

from typing import Any

from skmob2._core import recency_rank_indexed_arrow, recency_rank_indexed_numpy
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _detect_trajectory_columns,
    _prepare_trajectory,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "kernel": recency_rank_indexed_arrow,
        "unpack": lambda raw: (
            _arrow_result_values(raw[0]).to_pylist(),
            _arrow_result_values(raw[1]).to_pylist(),
            raw[2],
            raw[3],
        ),
    },
    numpy_ops={
        "kernel": recency_rank_indexed_numpy,
        "unpack": lambda raw: (raw[0].tolist(), raw[1].tolist(), raw[2], raw[3]),
    },
)


def recency_rank(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the recency rank of each distinct location for every user.

    The recency rank ``K_s(r_i)`` of location ``r_i`` is 1 if it is the most
    recently visited location, 2 if it is the second-most recently visited, and
    so on.  Ties (multiple visits to the same ``(lat, lng)`` pair) are resolved
    by keeping only the latest visit for each location before ranking.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
        One row per ``(user, location)`` pair with columns
        ``[uid_col, lat_col, lng_col, "recency_rank"]``.
        Rank 1 is the most recently visited location.
        The returned backend matches the input backend.



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
    >>> from skmob2 import recency_rank
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
    # Kernel requires chronological order within each user; sort but keep nulls
    # for the Rust indexed path to handle natively.
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        drop_nulls=False,
    )

    ops = _DISPATCHER.get_ops(df)
    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"
    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)

    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    out_lats, out_lngs, out_starts, out_ends = ops["unpack"](
        ops["kernel"](lats_data, lngs_data, indices, ends)
    )

    ranks_all: list[int] = []
    uid_vals_all: list = []
    for i, (s, e) in enumerate(zip(out_starts.tolist(), out_ends.tolist())):
        n = e - s
        ranks_all.extend(range(1, n + 1))
        if uid_values is not None:
            uid_vals_all.extend([uid_values[i]] * n)

    if uid_col is None:
        return _to_native({lat_col: out_lats, lng_col: out_lngs, "recency_rank": ranks_all}, df)
    return _to_native(
        {uid_col: uid_vals_all, lat_col: out_lats, lng_col: out_lngs, "recency_rank": ranks_all},
        df,
    )
