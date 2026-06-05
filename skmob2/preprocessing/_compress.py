from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
from skmob2._core import (
    compress_trajectory_representatives_arrow as _compress_arrow,
    compress_trajectory_representatives_indexed_arrow as _compress_indexed_arrow,
    compress_trajectory_representatives_indexed_numpy as _compress_indexed_numpy,
    compress_trajectory_representatives_numpy as _compress_numpy,
)

from ..measures._common import (
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _build_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _is_polars_backed,
    _prepare_trajectory,
)


def compress(
    traj: Any,
    spatial_radius_km: float = 0.2,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted=False,
) -> Any:
    """Compress trajectory by collapsing nearby points into single representative points.

    All points within spatial_radius_km from an initial point are collapsed into
    one point with median lat/lng and the initial point's timestamp.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    spatial_radius_km:
        Minimum distance (km) between consecutive output points.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    sorted:
        Whether the trajectory is already sorted by user and time.

    Returns
    -------
    DataFrame
        Compressed trajectory in the same backend as input.


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
    >>> from skmob2.preprocessing import compress
    >>> compressed = compress(df, spatial_radius_km=0.2)
    >>> print(len(compressed))
    3173
    >>> print(compressed.head().to_string(index=False))
     uid                  datetime       lat         lng                      location_id
       0 2009-05-25 20:56:10+00:00 37.774929 -122.419415 ee81ef22a22411ddb5e97f082c799f59
       0 2009-05-25 21:35:28+00:00 37.600747 -122.382376 248b82709e6c11ddbf68003048c0801e
       0 2009-05-25 22:13:23+00:00 37.615223 -122.389979 be2f1e669cc111dd9a50003048c0801e
       0 2009-05-26 02:21:12+00:00 39.878664 -104.682105 e12721ce84e911dd8019003048c0801e
       0 2009-05-26 04:59:44+00:00 39.739154 -104.984703 ee8b1d0ea22411ddb074dbd65f1665cf

    References
    ----------
    - [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), <a href="https://dl.acm.org/citation.cfm?id=2743025">https://dl.acm.org/citation.cfm?id=2743025</a>

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
    )

    timestamps_s = _extract_timestamps_s(df, datetime_col)
    lats = df.get_column(lat_col)
    lngs = df.get_column(lng_col)
    use_arrow = _is_polars_backed(df)

    if sorted:
        _, ranges = _build_user_ranges(df, uid_col)

        if use_arrow:
            _i, _l, _g = _compress_arrow(lats.to_arrow(), lngs.to_arrow(), ranges, spatial_radius_km)
            representative_indices = np.asarray(_arrow_result_values(_i), dtype=np.intp)
            median_lats = np.asarray(_arrow_result_values(_l), dtype=np.float64)
            median_lngs = np.asarray(_arrow_result_values(_g), dtype=np.float64)
        else:
            representative_indices, median_lats, median_lngs = _compress_numpy(
                lats.to_numpy(), lngs.to_numpy(), ranges, spatial_radius_km
            )
    else:
        _, sorted_indices, ends = _build_time_ordered_user_ranges(
            df,
            uid_col,
            datetime_col=datetime_col,
            timestamps=timestamps_s,
            use_arrow=use_arrow,
        )

        if use_arrow:
            _i, _l, _g = _compress_indexed_arrow(
                lats.to_arrow(),
                lngs.to_arrow(),
                sorted_indices,
                ends,
                spatial_radius_km,
            )
            representative_indices = np.asarray(_arrow_result_values(_i), dtype=np.intp)
            median_lats = np.asarray(_arrow_result_values(_l), dtype=np.float64)
            median_lngs = np.asarray(_arrow_result_values(_g), dtype=np.float64)
        else:
            representative_indices, median_lats, median_lngs = _compress_indexed_numpy(
                lats.to_numpy(),
                lngs.to_numpy(),
                sorted_indices,
                ends,
                spatial_radius_km,
            )

    result = df[representative_indices].with_columns(
        nw.new_series(lat_col, median_lats, backend=df.implementation),
        nw.new_series(lng_col, median_lngs, backend=df.implementation),
    )
    return result.to_native()


compress.__module__ = "skmob2.preprocessing"
