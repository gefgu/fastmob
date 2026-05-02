from __future__ import annotations

from typing import Any

import narwhals as nw
from skmob2._core import (
    radius_of_gyration_arrow,
    radius_of_gyration_indexed_arrow,
    radius_of_gyration_indexed_numpy,
    radius_of_gyration_numpy,
    radius_of_gyration_user_indices_arrow,
    radius_of_gyration_user_indices_numpy,
)

from .._common import _build_indexed_user_ranges, _is_polars_backed, _prepare_trajectory

_ROG_ROW_INDEX_COL = "__skmob2_rog_row_index__"


def _route_and_call(
    lats: nw.Series,
    lngs: nw.Series,
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return radius_of_gyration_arrow(lats.to_arrow(), lngs.to_arrow(), ranges)

    return radius_of_gyration_numpy(lats.to_numpy(), lngs.to_numpy(), ranges)


def _route_and_call_indexed(
    lats: nw.Series,
    lngs: nw.Series,
    indices: list[int],
    ranges: list[tuple[int, int]],
    *,
    use_arrow: bool,
) -> list[float]:
    if use_arrow:
        return radius_of_gyration_indexed_arrow(lats.to_arrow(), lngs.to_arrow(), indices, ranges)

    return radius_of_gyration_indexed_numpy(lats.to_numpy(), lngs.to_numpy(), indices, ranges)


def _build_rog_indexed_user_ranges(df: nw.DataFrame, uid_col: str) -> tuple[list, list[int], list[tuple[int, int]]]:
    n = len(df)
    if n == 0:
        return [], [], []

    uid_series = df.get_column(uid_col)
    try:
        if _is_polars_backed(df):
            uid_arrow = uid_series.to_arrow()
            indices, ranges = radius_of_gyration_user_indices_arrow(uid_arrow)
            uid_values = [uid_arrow[indices[start]].as_py() for start, _ in ranges]
        else:
            indices, ranges = radius_of_gyration_user_indices_numpy(uid_series.to_numpy())
            start_indices = [indices[start] for start, _ in ranges]
            uid_values = uid_series.to_numpy()[start_indices].tolist()
        return uid_values, indices, ranges
    except ValueError:
        pass

    return _build_indexed_user_ranges(df, uid_col, row_index_col=_ROG_ROW_INDEX_COL)


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
    )

    lats_full = df.get_column(lat_col)
    lngs_full = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if uid_col is None:
        (rg,) = _route_and_call(lats_full, lngs_full, [(0, len(df))], use_arrow=use_arrow)
        return nw.from_dict({"radius_of_gyration": [rg]}, backend=df.implementation).to_native()

    uid_values, indices, ranges = _build_rog_indexed_user_ranges(df, uid_col)
    rog_values = _route_and_call_indexed(lats_full, lngs_full, indices, ranges, use_arrow=use_arrow)

    result = nw.from_dict(
        {uid_col: uid_values, "radius_of_gyration": rog_values},
        backend=df.implementation,
    )

    return result.to_native()
