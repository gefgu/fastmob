from __future__ import annotations

from typing import Any, Callable

import narwhals as nw

from fastmob._core import (
    FilterConfig,
    OutlierConfig,
    filter_trajectory_indexed,
    filter_trajectory_sorted,
    outlier_trajectory_indexed,
    outlier_trajectory_sorted,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _arrow_result_values,
    _build_time_ordered_user_ranges,
    _build_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _narwhals_safe_value,
)

_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _filter_speed(
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
    is_sorted=False,
) -> Any:
    """Filter trajectory noise by removing high-speed outlier points.

    This is a pure extraction of the original ``filter()`` implementation
    (method="speed" is the default and only pre-existing behavior); see
    ``filter()`` for the public entry point and full parameter docs.

    @usedBy `filter()` via `FILTER_METHODS["speed"]` (the default method).
    """
    df = nw.from_native(traj, eager_only=True)

    datetime_col, lat_col, lng_col, uid_col = _detect_trajectory_columns(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    timestamp_s = _extract_timestamps_s(df, datetime_col)
    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    # Also feeds _build_time_ordered_user_ranges below, so its extraction must
    # keep matching whatever backend that helper's own uid-code extraction
    # picks internally (see waiting_times.py/mean_square_displacement.py for
    # why this one can't be forced to Arrow independently).
    times_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamp_s)

    config = FilterConfig(
        max_speed_kmh=max_speed_kmh,
        include_loops=include_loops,
        speed_kmh=speed_kmh,
        max_loop=max_loop,
        ratio_max=ratio_max,
    )

    # 3. Build ranges and select the appropriate core function from the dictionary
    if is_sorted:
        _, ranges = _build_user_ranges(df, uid_col)
        raw_mask = filter_trajectory_sorted(lats_data, lngs_data, times_data, ranges, config)
    else:
        _, sorted_indices, ends = _build_time_ordered_user_ranges(
            df,
            uid_col,
            datetime_col=datetime_col,
            timestamps_data=times_data,
        )
        raw_mask = filter_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config)

    keep_mask = _narwhals_safe_value(_arrow_result_values(raw_mask))
    result = df.filter(nw.new_series("__keep__", keep_mask, backend=df.implementation)).to_native()

    return result


def _prepare_hampel(window_size: int = 5, n_sigma: float = 3.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Hampel outlier detection.

    @usedBy `filter()` via `FILTER_METHODS["hampel"]`.
    """
    return "hampel", {"window_size": window_size, "n_sigma": n_sigma}


def _prepare_greedy(max_speed_kmh: float = 100.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Greedy outlier detection.

    @usedBy `filter()` via `FILTER_METHODS["greedy"]`.
    """
    return "greedy", {"max_speed_kmh": max_speed_kmh}


def _prepare_smart_greedy(max_speed_kmh: float = 100.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for SmartGreedy outlier detection.

    @usedBy `filter()` via `FILTER_METHODS["smart_greedy"]`.
    """
    return "smart_greedy", {"max_speed_kmh": max_speed_kmh}


def _prepare_zheng(max_speed_kmh: float = 100.0, min_seg_size: int = 1, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Zheng outlier detection.

    @usedBy `filter()` via `FILTER_METHODS["zheng"]`.
    """
    return "zheng", {"max_speed_kmh": max_speed_kmh, "min_seg_size": min_seg_size}


FILTER_METHODS: dict[str, Callable] = {
    "speed": _filter_speed,
    "hampel": _prepare_hampel,
    "greedy": _prepare_greedy,
    "smart_greedy": _prepare_smart_greedy,
    "zheng": _prepare_zheng,
}


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
    is_sorted=False,
    method: str = "speed",
    **method_kwargs: Any,
) -> Any:
    """Filter trajectory noise by removing outlier points using a named algorithm.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    max_speed_kmh, include_loops, speed_kmh, max_loop, ratio_max:
        Parameters for the default ``method="speed"`` algorithm; unused by
        every other method. See the ``method="speed"`` description below.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    is_sorted: Whether the trajectory is already sorted by user and time; And Nulls/NaNs have been dropped.
    Setting this to True can speed up processing but may lead to incorrect results if the data is not properly preprocessed.
    method:
        Name of the outlier-detection algorithm to run. One of ``"speed"``
        (default, backward-compatible with every prior ``filter()`` call
        that never passed ``method``), ``"hampel"``, ``"greedy"``,
        ``"smart_greedy"``, or ``"zheng"``.
    **method_kwargs:
        Method-specific parameters for ``method != "speed"``, forwarded to
        the matching ``FILTER_METHODS[method]`` preparer.

        - ``speed``: uses ``max_speed_kmh``, ``include_loops``,
          ``speed_kmh``, ``max_loop``, ``ratio_max`` above (unchanged from
          every prior release).
        - ``hampel``: ``window_size`` (default ``5``) — centered rolling
          window length, in points, over the per-user consecutive-point
          speed (km/h) series; ``n_sigma`` (default ``3.0``) — outlier
          threshold as a multiple of ``1.4826 * MAD(window)``.
        - ``greedy``: ``max_speed_kmh`` (default ``100.0``) — maximum
          physically-consistent speed, in km/h, between each point and the
          immediately preceding point.
        - ``smart_greedy``: ``max_speed_kmh`` (default ``100.0``) — same
          consistency threshold as ``greedy``, but keeps the single longest
          run of mutually consistent points rather than testing fixed
          adjacent pairs.
        - ``zheng``: ``max_speed_kmh`` (default ``100.0``) — consistency
          threshold as above; ``min_seg_size`` (default ``1``) — minimum
          consecutive-consistent-run length required to keep a segment
          (shorter runs are dropped in full).

    Returns
    -------
    DataFrame
        Filtered trajectory in the same backend as input.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.utils.constants.BRIGHTKITE_SAMPLE
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
    >>> from fastmob.preprocessing import filter
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
    - [PT2020] Pedrido, M.O. (2020). Hampel filter. GitHub repository, <a href="https://github.com/MichaelisTrofficus/hampel_filter">https://github.com/MichaelisTrofficus/hampel_filter</a>
    - [CKMSS2019] Custers, B., van de Kerkhof, M., Meulemans, W., Speckmann, B., & Staals, F. (2019). Maximum physically consistent trajectories. SIGSPATIAL 2019.
    """
    if method not in FILTER_METHODS:
        raise ValueError(f"unknown filter method: {method!r}; choose from {sorted(FILTER_METHODS)}")

    if method == "speed":
        return _filter_speed(
            traj,
            max_speed_kmh,
            include_loops,
            speed_kmh,
            max_loop,
            ratio_max,
            datetime_col=datetime_col,
            lat_col=lat_col,
            lng_col=lng_col,
            uid_col=uid_col,
            is_sorted=is_sorted,
        )

    method_name, params = FILTER_METHODS[method](**method_kwargs)

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

    lats_data = df.get_column(lat_col).to_arrow()
    lngs_data = df.get_column(lng_col).to_arrow()
    timestamps_s = _extract_timestamps_s(df, datetime_col)
    times_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps_s)

    config = OutlierConfig(method=method_name, **params)

    if is_sorted:
        _, ranges = _build_user_ranges(df, uid_col)
        raw_mask = outlier_trajectory_sorted(lats_data, lngs_data, times_data, ranges, config)
    else:
        _, sorted_indices, ends = _build_time_ordered_user_ranges(
            df,
            uid_col,
            datetime_col=datetime_col,
            timestamps_data=times_data,
        )
        raw_mask = outlier_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config)

    keep_mask = _narwhals_safe_value(_arrow_result_values(raw_mask))
    result = df.filter(nw.new_series("__keep__", keep_mask, backend=df.implementation)).to_native()

    return result


filter.__module__ = "fastmob.preprocessing"
