from __future__ import annotations

from itertools import compress
from typing import Any

import narwhals as nw
import numpy as np

from skmob2._core import (
    radius_of_gyration_arrow_with_counts,
    radius_of_gyration_indexed_arrow,
    radius_of_gyration_indexed_numpy,
    radius_of_gyration_numpy_with_counts,
)
from skmob2.core.dispatch import TrajectoryDispatcher

from .._common import (
    _arrow_result_values,
    _build_indexed_user_ranges_fast,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _ranges_from_ends,
    _result_scalar,
    _to_native,
)


def _filter_arrow_values(values: Any, keep: np.ndarray) -> Any:
    import pyarrow as pa
    import pyarrow.compute as pc

    arrow_values = _arrow_result_values(values)
    if hasattr(arrow_values, "__arrow_c_array__"):
        arrow_values = pa.array(arrow_values)
    return pc.filter(arrow_values, pa.array(keep))


def _filter_numpy_values(values: Any, keep: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[keep]


ROG_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "rog_sorted": radius_of_gyration_arrow_with_counts,
        "rog_indexed": radius_of_gyration_indexed_arrow,
        "format_values": _arrow_result_values,
        "filter_values": _filter_arrow_values,
    },
    numpy_ops={
        "rog_sorted": radius_of_gyration_numpy_with_counts,
        "rog_indexed": radius_of_gyration_indexed_numpy,
        "format_values": lambda values: np.asarray(values, dtype=np.float64),
        "filter_values": _filter_numpy_values,
    },
)


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

    ops = ROG_DISPATCHER.get_ops(df)
    use_arrow = ROG_DISPATCHER.get_backend_key(df) == "arrow"

    _, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    schema = df.schema
    if schema[lat_col] != nw.Float64 or schema[lng_col] != nw.Float64:
        df = df.with_columns(
            nw.col(lat_col).cast(nw.Float64),
            nw.col(lng_col).cast(nw.Float64),
        )

    if sorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        ranges = _ranges_from_ends(ends)
        lats_data = ops["extract_data"](df.get_column(lat_col))
        lngs_data = ops["extract_data"](df.get_column(lng_col))
        raw_values, raw_counts = ops["rog_sorted"](lats_data, lngs_data, ranges)
    else:
        lats_data = ops["extract_data"](df.get_column(lat_col))
        lngs_data = ops["extract_data"](df.get_column(lng_col))
        uid_values, indices, ends = _build_indexed_user_ranges_fast(
            df,
            uid_col,
            use_arrow=use_arrow,
        )
        raw_values, raw_counts = ops["rog_indexed"](lats_data, lngs_data, indices, ends)

    rog_values = ops["format_values"](raw_values)
    valid_counts = np.asarray(raw_counts, dtype=np.uintp)
    keep = valid_counts > 0

    if uid_col is None:
        if not keep.any():
            return _to_native({"radius_of_gyration": [0.0]}, df)
        filtered_values = ops["filter_values"](rog_values, keep)
        return _to_native(
            {"radius_of_gyration": [_result_scalar(filtered_values)]},
            df,
        )

    filtered_uid_values = list(compress(uid_values, keep))
    filtered_rog_values = ops["filter_values"](rog_values, keep)
    return _to_native(
        {uid_col: filtered_uid_values, "radius_of_gyration": filtered_rog_values},
        df,
    )
