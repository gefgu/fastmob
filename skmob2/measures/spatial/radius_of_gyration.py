from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    radius_of_gyration_arrow,
    radius_of_gyration_indexed_arrow,
    radius_of_gyration_indexed_numpy,
    radius_of_gyration_valid_user_indices_arrow,
    radius_of_gyration_valid_user_indices_numpy,
    radius_of_gyration_numpy,
    radius_of_gyration_user_indices_arrow,
    radius_of_gyration_user_indices_numpy,
)

from .._common import _build_indexed_user_ranges, _build_user_ranges, _is_polars_backed, _prepare_trajectory

_ROG_ROW_INDEX_COL = "__skmob2_rog_row_index__"


def _route_and_call(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> Any:
    if use_arrow:
        return radius_of_gyration_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    return radius_of_gyration_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def _route_and_call_indexed(
    lats: nw.Series,
    lngs: nw.Series,
    indices: Any,
    starts: Any,
    ends: Any,
    *,
    use_arrow: bool,
) -> Any:
    if use_arrow:
        return radius_of_gyration_indexed_arrow(lats.to_arrow(), lngs.to_arrow(), indices, starts, ends)

    return radius_of_gyration_indexed_numpy(lats.to_numpy(), lngs.to_numpy(), indices, starts, ends)


def _as_index_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.uintp)


def _ranges_to_starts_ends(ranges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.fromiter((start for start, _ in ranges), dtype=np.uintp, count=len(ranges))
    ends = np.fromiter((end for _, end in ranges), dtype=np.uintp, count=len(ranges))
    return starts, ends


def _arrow_result_values(values: Any) -> Any:
    if hasattr(values, "to_pyarrow"):
        return values.to_pyarrow()
    return values


def _result_scalar(values: Any) -> float:
    if hasattr(values, "to_numpy"):
        return float(values.to_numpy()[0])
    return float(np.asarray(values)[0])


def _build_rog_indexed_user_ranges(df: nw.DataFrame, uid_col: str) -> tuple[list, Any, Any, Any]:
    n = len(df)
    if n == 0:
        empty = np.array([], dtype=np.uintp)
        return [], empty, empty, empty

    uid_series = df.get_column(uid_col)
    try:
        if _is_polars_backed(df):
            uid_arrow = uid_series.to_arrow()
            indices, starts, ends = radius_of_gyration_user_indices_arrow(uid_arrow)
            uid_values = [uid_arrow[int(indices[start])].as_py() for start in starts]
        else:
            indices, starts, ends = radius_of_gyration_user_indices_numpy(uid_series.to_numpy())
            start_indices = indices[starts]
            uid_values = uid_series.to_numpy()[start_indices].tolist()
        return uid_values, indices, starts, ends
    except ValueError:
        pass

    uid_values, indices, ranges = _build_indexed_user_ranges(df, uid_col, row_index_col=_ROG_ROW_INDEX_COL)
    starts, ends = _ranges_to_starts_ends(ranges)
    return uid_values, _as_index_array(indices), starts, ends


def _build_rog_valid_indexed_user_ranges(
    df: nw.DataFrame,
    uid_col: str,
    lat_col: str,
    lng_col: str,
) -> tuple[list, Any, Any, Any]:
    n = len(df)
    if n == 0:
        empty = np.array([], dtype=np.uintp)
        return [], empty, empty, empty

    uid_series = df.get_column(uid_col)
    lat_series = df.get_column(lat_col)
    lng_series = df.get_column(lng_col)
    try:
        if _is_polars_backed(df):
            uid_arrow = uid_series.to_arrow()
            indices, starts, ends = radius_of_gyration_valid_user_indices_arrow(
                uid_arrow,
                lat_series.to_arrow(),
                lng_series.to_arrow(),
            )
            uid_values = [uid_arrow[int(indices[start])].as_py() for start in starts]
        else:
            indices, starts, ends = radius_of_gyration_valid_user_indices_numpy(
                uid_series.to_numpy(),
                lat_series.to_numpy(),
                lng_series.to_numpy(),
            )
            start_indices = indices[starts]
            uid_values = uid_series.to_numpy()[start_indices].tolist()
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
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=False,
        drop_nulls=False,
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        rg = _result_scalar(_route_and_call(lats_full, lngs_full, [(0, len(df))], use_arrow=use_arrow))
        return nw.from_dict({"radius_of_gyration": [rg]}, backend=df.implementation).to_native()

    uid_values, indices, starts, ends = _build_rog_valid_indexed_user_ranges(df, uid_col, lat_col, lng_col)
    rog_values = _route_and_call_indexed(lats_full, lngs_full, indices, starts, ends, use_arrow=use_arrow)
    if use_arrow:
        rog_values = _arrow_result_values(rog_values)

    result = nw.from_dict(
        {uid_col: uid_values, "radius_of_gyration": rog_values},
        backend=df.implementation,
    )

    return result.to_native()
