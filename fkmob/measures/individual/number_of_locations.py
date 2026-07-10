from __future__ import annotations

from typing import Any

import narwhals as nw

from fkmob._core import (
    number_of_locations_arrow,
    number_of_locations_indexed_arrow,
    number_of_locations_indexed_numpy,
    number_of_locations_numpy,
)
from fkmob.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "presorted": number_of_locations_arrow,
        "indexed": number_of_locations_indexed_arrow,
        "format_values": _arrow_result_values,
    },
    numpy_ops={
        "presorted": number_of_locations_numpy,
        "indexed": number_of_locations_indexed_numpy,
        "format_values": lambda values: values,
    },
)


def number_of_locations(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
) -> Any:
    """Return the number of distinct locations visited by each user.

    A distinct location is a unique exact ``(lat, lng)`` pair — matching the
    skmob convention of float equality without spatial clustering.

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
        When True, trust that rows are already grouped by user and use the
        contiguous fast path.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        One row per user with columns ``[uid_col, "number_of_locations"]``.
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
    >>> from fkmob import number_of_locations
    >>> result = number_of_locations(df)
    >>> print(result.head().to_string(index=False))
     uid  number_of_locations
       0                  542
       1                   97
       2                  427

    References
    ----------
    - [GHB2008] Gonzalez, M. C., Hidalgo, C. A. & Barabasi, A. L. (2008) Understanding individual human mobility patterns. Nature, 453, 779-782, https://www.nature.com/articles/nature06958.
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

    ops = _DISPATCHER.get_ops(df)
    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        n_locs = ops["format_values"](ops["presorted"](lats_data, lngs_data, ends))
        if uid_col is None:
            return _to_native({"number_of_locations": n_locs}, df)
        return _to_native({uid_col: uid_values, "number_of_locations": n_locs}, df)

    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col)

    n_locs = ops["format_values"](ops["indexed"](lats_data, lngs_data, indices, ends))

    if uid_col is None:
        result = _to_native({"number_of_locations": n_locs}, df)
    else:
        result = _to_native({uid_col: uid_values, "number_of_locations": n_locs}, df)

    return result
