from __future__ import annotations
import narwhals as nw

from typing import Any

import numpy as np
from skmob2._core import (
    radius_of_gyration_arrow,
    radius_of_gyration_indexed_arrow,
    radius_of_gyration_indexed_numpy,
    radius_of_gyration_numpy,
    radius_of_gyration_valid_user_indices_arrow,
    radius_of_gyration_valid_user_indices_numpy,
)

from .._common import (
    _as_index_array,
    _build_presorted_user_ranges,
    _build_user_ranges,
    _dispatch_kernel,
    _is_polars_backed,
    _detect_trajectory_columns,
    _prepare_trajectory,
    _ranges_to_starts_ends,
    _result_scalar,
    _to_native,
    _uid_values_from_index_ranges,
)

_ROG_ROW_INDEX_COL = "__skmob2_rog_row_index__"


def _build_valid_indexed_user_ranges(
    df, uid_col: str, lat_col: str, lng_col: str
) -> tuple[list, Any, Any, Any]:
    """Build sorted row indices and per-user ranges for rows with valid lat/lng.

    Tries the Rust fast path first; falls back to a Narwhals filter+sort when
    the UID dtype is not supported by the Rust kernel.
    """
    n = len(df)
    if n == 0:
        empty = np.array([], dtype=np.uintp)
        return [], empty, empty, empty

    use_arrow = _is_polars_backed(df)
    uid_series = df.get_column(uid_col)
    try:
        indices, starts, ends = _dispatch_kernel(
            radius_of_gyration_valid_user_indices_numpy,
            radius_of_gyration_valid_user_indices_arrow,
            [uid_series, df.get_column(lat_col), df.get_column(lng_col)],
            use_arrow=use_arrow,
            convert_arrow_result=False,
        )
        uid_values = _uid_values_from_index_ranges(uid_series, indices, starts, use_arrow=use_arrow)
        return uid_values, indices, starts, ends
    except ValueError:
        pass

    index_df = (
        df.with_row_index(_ROG_ROW_INDEX_COL)
        .drop_nulls(subset=[lat_col, lng_col])
        .select([uid_col, _ROG_ROW_INDEX_COL])
        .sort(uid_col, _ROG_ROW_INDEX_COL)
    )
    uid_values, ranges = _build_user_ranges(index_df, uid_col)
    indices = [int(idx) for idx in index_df.get_column(_ROG_ROW_INDEX_COL).to_list()]
    starts, ends = _ranges_to_starts_ends(ranges)
    return uid_values, _as_index_array(indices), starts, ends


def radius_of_gyration(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted: bool = False,
):
    """Compute the radius of gyration (km) for each user in the trajectory.

    The radius of gyration captures how far a user typically roams from their
    center of mass. Formally:

    \\[
    r_g(u) =
    \\sqrt{\\frac{1}{n_u} \\sum_i d_\\mathrm{haversine}(r_i, r_\\mathrm{cm})^2}
    \\]

    where \\(r_\\mathrm{cm}\\) is the arithmetic mean of the user's lat/lng
    coordinates.

    Radius of gyration is order-independent, so chronological sorting is not
    required for correctness; grouping rows by user is what matters.

    Parameters
    ----------
    traj:
        Trajectory data; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must have columns for datetime, latitude, and longitude.
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
        One row per user with columns ``[uid_col, "radius_of_gyration"]``.
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
    >>> from skmob2 import radius_of_gyration
    >>> result = radius_of_gyration(df)
    >>> print(result.round({"radius_of_gyration": 3}).head().to_string(index=False))
     uid  radius_of_gyration
       0            1564.439
       1            2467.777
       2            1600.452

    References
    ----------
    - [GHB2008] Gonzalez, M. C., Hidalgo, C. A. & Barabasi, A. L. (2008) Understanding individual human mobility patterns. Nature, 453, 779-782, https://www.nature.com/articles/nature06958.
    - [PRQPG2013] Pappalardo, L., Rinzivillo, S., Qu, Z., Pedreschi, D. & Giannotti, F. (2013) Understanding the patterns of car travel. European Physics Journal Special Topics 215(1), 61-73, https://link.springer.com/article/10.1140%2Fepjst%2Fe2013-01715-5


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
        drop_nulls=False,
    )

    lats = df.get_column(lat_col)
    lngs = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        rg = _result_scalar(
            _dispatch_kernel(
                radius_of_gyration_numpy,
                radius_of_gyration_arrow,
                [lats, lngs],
                [(0, len(df))],
                use_arrow=use_arrow,
            )
        )
        return _to_native({"radius_of_gyration": [rg]}, df)

    if sorted:
        valid_df = df.drop_nulls(subset=[lat_col, lng_col]).filter(
            (~nw.col(lat_col).is_nan()) & (~nw.col(lng_col).is_nan())
        )
        uid_values, ranges = _build_presorted_user_ranges(valid_df, uid_col)
        rog_values = _dispatch_kernel(
            radius_of_gyration_numpy,
            radius_of_gyration_arrow,
            [valid_df.get_column(lat_col), valid_df.get_column(lng_col)],
            ranges,
            use_arrow=use_arrow,
        )
        return _to_native({uid_col: uid_values, "radius_of_gyration": rog_values}, df)

    uid_values, indices, starts, ends = _build_valid_indexed_user_ranges(df, uid_col, lat_col, lng_col)
    rog_values = _dispatch_kernel(
        radius_of_gyration_indexed_numpy,
        radius_of_gyration_indexed_arrow,
        [lats, lngs],
        indices,
        starts,
        ends,
        use_arrow=use_arrow,
    )
    return _to_native({uid_col: uid_values, "radius_of_gyration": rog_values}, df)
