from __future__ import annotations
import narwhals as nw
import numpy as np

from typing import Any

from skmob2._core import (
    k_radius_of_gyration_arrow,
    k_radius_of_gyration_indexed_arrow,
    k_radius_of_gyration_indexed_numpy,
    k_radius_of_gyration_numpy,
)

from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _extract_timestamps_ms,
    _detect_trajectory_columns,
    _ranges_from_ends,
    _to_native,
)

_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "sorted": k_radius_of_gyration_arrow,
        "indexed": k_radius_of_gyration_indexed_arrow,
        "format_values": _arrow_result_values,
    },
    numpy_ops={
        "sorted": k_radius_of_gyration_numpy,
        "indexed": k_radius_of_gyration_indexed_numpy,
        "format_values": lambda values: np.asarray(values, dtype=np.float64),
    },
)


def k_radius_of_gyration(
    traj: Any,
    k: int = 2,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
):
    """Compute the k-radius of gyration (km) for each user in the trajectory.

    The k-radius of gyration is the radius of gyration computed using only the
    k most-visited locations for each user. Formally:

    \\[
    r_g^{(k)}(u) =
    \\sqrt{
        \\frac{\\sum_{i \\in \\mathrm{top}\\text{-}k} w_i d_\\mathrm{haversine}(r_i, r_\\mathrm{cm})^2}
             {\\sum_{i \\in \\mathrm{top}\\text{-}k} w_i}
    }
    \\]

    where \\(r_\\mathrm{cm}\\) is the weighted center of mass over the top-k
    locations and \\(w_i\\) is the visit count of location \\(i\\).

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, ...).  Must have columns for datetime, latitude, and longitude.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
    k:
        Number of most-visited locations to consider.  Defaults to 2.
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
        One row per user with columns ``[uid_col, "k_radius_of_gyration"]``.
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
    >>> from skmob2 import k_radius_of_gyration
    >>> result = k_radius_of_gyration(df, k=2)
    >>> print(result.round({"k_radius_of_gyration": 3}).head().to_string(index=False))
     uid  k_radius_of_gyration
       0                 7.859
       1                 4.069
       2                 5.794

    References
    ----------
    - [PSRPGB2015] Pappalardo, L., Simini, F. Rinzivillo, S., Pedreschi, D. Giannotti, F. & Barabasi, A. L. (2015) Returners and Explorers dichotomy in human mobility. Nature Communications 6, https://www.nature.com/articles/ncomms9166


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
    use_arrow = _DISPATCHER.get_backend_key(df) == "arrow"

    lats = ops["extract_data"](df.get_column(lat_col))
    lngs = ops["extract_data"](df.get_column(lng_col))
    timestamps = _extract_timestamps_ms(df, datetime_col)
    timestamps_data = ops["extract_data"](timestamps)

    if sorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        ranges = _ranges_from_ends(ends)
        krg_values = ops["format_values"](ops["sorted"](lats, lngs, timestamps_data, ranges, k))
        if uid_col is None:
            return _to_native({"k_radius_of_gyration": krg_values}, df)
        return _to_native({uid_col: uid_values, "k_radius_of_gyration": krg_values}, df)

    uid_values, indices, ends = _build_indexed_user_ranges_fast(df, uid_col, use_arrow=use_arrow)

    krg_values = ops["format_values"](ops["indexed"](lats, lngs, timestamps_data, indices, ends, k))

    if uid_col is None:
        return _to_native({"k_radius_of_gyration": krg_values}, df)
    return _to_native({uid_col: uid_values, "k_radius_of_gyration": krg_values}, df)
