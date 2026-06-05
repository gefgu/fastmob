from __future__ import annotations
import time
from typing import Any

import narwhals as nw
from skmob2._core import FilterConfig
from skmob2._core import (
    filter_trajectory_arrow as _filter_trajectory_arrow,
    filter_trajectory_indexed_arrow as _filter_trajectory_indexed_arrow,
    filter_trajectory_indexed_numpy as _filter_trajectory_indexed_numpy,
    filter_trajectory_numpy as _filter_trajectory_numpy,
)
from ..measures._common import (
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _build_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _prepare_trajectory,
)

# Import the new dispatcher from wherever you saved it
from skmob2.core.dispatch import TrajectoryDispatcher

# Instantiate the dispatcher for this specific module
FILTER_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "filter_sorted": _filter_trajectory_arrow,
        "filter_indexed": _filter_trajectory_indexed_arrow,
        "format_mask": _arrow_result_values,
    },
    numpy_ops={
        "filter_sorted": _filter_trajectory_numpy,
        "filter_indexed": _filter_trajectory_indexed_numpy,
        "format_mask": lambda mask: mask,  # No-op for NumPy
    }
)

def filter(
    traj: Any,
    max_speed_kmh: float = 500.0,
    include_loops: bool = False,
    speed_kmh: float = 5.0,
    max_loop: int = 6,
    ratio_max: float = 0.25,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    sorted=False,
) -> Any:
    """Filter trajectory noise by removing high-speed outlier points.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    max_speed_kmh:
        Remove points where speed from the previous point exceeds this threshold.
    include_loops:
        If True, also remove points forming short fast return loops.
    speed_kmh:
        Minimum loop speed threshold (km/h); only used when include_loops=True.
    max_loop:
        Maximum number of points to look ahead for loop detection.
    ratio_max:
        Distance ratio threshold for loop detection.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    sorted: Whether the trajectory is already sorted by user and time;

    Returns
    -------
    DataFrame
        Filtered trajectory in the same backend as input.

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
    >>> from skmob2.preprocessing import filter
    >>> filtered = filter(df, max_speed_kmh=500.0)
    >>> print(len(filtered))
    4898
    >>> print(filtered.head().to_string(index=False))
     uid                  datetime       lat         lng                      location_id
       0 2009-05-25 20:56:10+00:00 37.774929 -122.419415 ee81ef22a22411ddb5e97f082c799f59
       0 2009-05-25 21:35:28+00:00 37.600747 -122.382376 248b82709e6c11ddbf68003048c0801e
       0 2009-05-25 21:37:44+00:00 37.600747 -122.382376 248b82709e6c11ddbf68003048c0801e
       0 2009-05-25 21:42:47+00:00 37.600747 -122.382376 248b82709e6c11ddbf68003048c0801e
       0 2009-05-25 22:13:23+00:00 37.615223 -122.389979 be2f1e669cc111dd9a50003048c0801e

    References
    ----------
    - [Z2015] Zheng, Y. (2015) Trajectory data mining: an overview. ACM Transactions on Intelligent Systems and Technology 6(3), <a href="https://dl.acm.org/citation.cfm?id=2743025">https://dl.acm.org/citation.cfm?id=2743025</a>
    """
    df = nw.from_native(traj, eager_only=True)
    
    # 1. Fetch backend context from the dispatcher
    ops = FILTER_DISPATCHER.get_ops(df)
    use_arrow = FILTER_DISPATCHER.get_backend_key(df) == "arrow"

    start = time.perf_counter_ns()
    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    print(f"Column detection took {(time.perf_counter_ns() - start) / 1e9:.3f} seconds")

    start = time.perf_counter_ns()
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=use_arrow,
    )
    print(f"Trajectory preparation took {(time.perf_counter_ns() - start) / 1e9:.3f} seconds")

    start = time.perf_counter_ns()
    timestamps_s = _extract_timestamps_s(df, datetime_col)
    print(f"Timestamp extraction took {(time.perf_counter_ns() - start) / 1e9:.3f} seconds")
    
    lats = df.get_column(lat_col)
    lngs = df.get_column(lng_col)

    # 2. Extract arrays dynamically using the backend's method
    lats_data = ops["extract_data"](lats)
    lngs_data = ops["extract_data"](lngs)
    times_data = ops["extract_data"](timestamps_s)

    config = FilterConfig(
        max_speed_kmh=max_speed_kmh,
        include_loops=include_loops,
        speed_kmh=speed_kmh,
        max_loop=max_loop,
        ratio_max=ratio_max,
    )
    sorted = use_arrow or sorted

    start = time.perf_counter_ns()
    
    # 3. Build ranges and select the appropriate core function from the dictionary
    if sorted:
        _, ranges = _build_user_ranges(df, uid_col)
        filter_func = ops["filter_sorted"]
        args = (lats_data, lngs_data, times_data, ranges, config)
    else:
        _, sorted_indices, ends = _build_time_ordered_user_ranges(
            df,
            uid_col,
            datetime_col=datetime_col,
            timestamps=timestamps_s,
            use_arrow=use_arrow,
        )
        filter_func = ops["filter_indexed"]
        args = (lats_data, lngs_data, times_data, sorted_indices, ends, config)

    # 4. Execute the math and format the resulting mask dynamically
    raw_mask = filter_func(*args)
    keep_mask = ops["format_mask"](raw_mask)
    
    print(f"Filtering took {(time.perf_counter_ns() - start) / 1e9:.3f} seconds")

    native_df = df.to_native()
    if hasattr(native_df, "iloc") and hasattr(native_df, "dtypes"):
        return native_df[keep_mask]

    return df.filter(
        nw.new_series("__keep__", keep_mask, backend=df.implementation)
    ).to_native()

filter.__module__ = "skmob2.preprocessing"