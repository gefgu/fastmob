from __future__ import annotations
import narwhals as nw

from typing import Any

from skmob2._core import (
    number_of_visits_arrow,
    number_of_visits_indexed_arrow,
    number_of_visits_indexed_numpy,
    number_of_visits_numpy,
)

from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _dispatch_kernel,
    _detect_trajectory_columns,
    _ranges_from_ends,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def number_of_visits(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
) -> Any:
    """Return the total number of trajectory points (visits) for each user.

    A "visit" is defined as one row in the trajectory dataframe after null
    removal. The result is the row count per user — identical to the skmob
    ``number_of_visits`` individual measure.

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
    sorted:
        When True, trust that rows are already grouped by user and use the
        contiguous fast path.

    Returns
    -------
    DataFrame
        One row per user with columns ``[uid_col, "number_of_visits"]``.
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
    >>> from skmob2 import number_of_visits
    >>> result = number_of_visits(df)
    >>> print(result.head().to_string(index=False))
     uid  number_of_visits
       0              2099
       1              1210
       2              1691
    """
    df = nw.from_native(traj, eager_only=True)
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

    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"
    if sorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        ranges = _ranges_from_ends(ends)
        counts = _dispatch_kernel(
            number_of_visits_numpy,
            number_of_visits_arrow,
            [],
            len(df),
            ranges,
            use_arrow=use_arrow,
        )
        if uid_col is None:
            return _to_native({"number_of_visits": counts}, df)
        return _to_native({uid_col: uid_values, "number_of_visits": counts}, df)

    valid_mask = (
        ~df.get_column(lat_col).is_null() & ~df.get_column(lng_col).is_null()
    ).to_numpy()
    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)
    counts = _dispatch_kernel(
        number_of_visits_indexed_numpy,
        number_of_visits_indexed_arrow,
        [],
        len(df),
        indices,
        ends,
        valid_mask,
        use_arrow=use_arrow,
    )

    if uid_col is None:
        return _to_native({"number_of_visits": counts}, df)
    return _to_native({uid_col: uid_values, "number_of_visits": counts}, df)
