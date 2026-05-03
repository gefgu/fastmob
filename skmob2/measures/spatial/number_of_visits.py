from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import number_of_visits_indexed_arrow, number_of_visits_indexed_numpy

from .._common import _arrow_result_values, _build_indexed_user_ranges_fast, _is_polars_backed, _prepare_trajectory


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

    @usedBy
        skmob2.measures.__init__, skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
    )

    if uid_col is None:
        use_arrow = _is_polars_backed(df)
        _, indices, starts, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)
        if use_arrow:
            count_values = _arrow_result_values(number_of_visits_indexed_arrow(len(df), indices, starts, ends))
        else:
            count_values = number_of_visits_indexed_numpy(len(df), indices, starts, ends)
        return nw.from_dict(
            {"number_of_visits": count_values},
            backend=df.implementation,
        ).to_native()

    use_arrow = _is_polars_backed(df)
    uid_values, indices, starts, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)
    if use_arrow:
        counts = _arrow_result_values(number_of_visits_indexed_arrow(len(df), indices, starts, ends))
    else:
        counts = number_of_visits_indexed_numpy(len(df), indices, starts, ends)

    result = nw.from_dict({uid_col: uid_values, "number_of_visits": counts}, backend=df.implementation)

    return result.to_native()
