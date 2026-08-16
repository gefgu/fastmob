from __future__ import annotations

from typing import Any, Callable

import narwhals as nw

from fastmob._core import (
    SimplifyConfig,
    simplify_trajectory_indexed,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _as_arrow,
    _build_indexed_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamp_arrow,
    _extract_timestamps,
    _narwhals_safe_value,
)

# Bare extractor supplies numeric timestamps to the simplification kernel.
# Ordering receives the original Arrow timestamp array separately. lat/lng go
# to Arrow unconditionally.
_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _prepare_douglas_peucker(epsilon_km: float = 0.001, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Douglas-Peucker simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["douglas_peucker"]`.
    """
    return "douglas_peucker", {"epsilon_km": epsilon_km}


def _prepare_top_down_time_ratio(epsilon_km: float = 0.001, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Top-Down Time Ratio simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["top_down_time_ratio"]`.
    """
    return "top_down_time_ratio", {"epsilon_km": epsilon_km}


def _prepare_min_distance(min_distance_km: float = 0.01, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for MinDistance simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["min_distance"]`.
    """
    return "min_distance", {"min_distance_km": min_distance_km}


def _prepare_min_time_delta(min_time_delta_s: float = 60.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for MinTimeDelta simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["min_time_delta"]`.
    """
    return "min_time_delta", {"min_time_delta_s": min_time_delta_s}


def _prepare_max_distance(epsilon_km: float = 0.001, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for MaxDistance simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["max_distance"]`.
    """
    return "max_distance", {"epsilon_km": epsilon_km}


def _prepare_chan_chin(epsilon_km: float = 0.001, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Chan-Chin simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["chan_chin"]`.
    """
    return "chan_chin", {"epsilon_km": epsilon_km}


def _prepare_imai_iri(epsilon_km: float = 0.001, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for Imai-Iri simplification.

    @usedBy `simplify()` via `SIMPLIFY_METHODS["imai_iri"]`.
    """
    return "imai_iri", {"epsilon_km": epsilon_km}


SIMPLIFY_METHODS: dict[str, Callable[..., tuple[str, dict]]] = {
    "douglas_peucker": _prepare_douglas_peucker,
    "top_down_time_ratio": _prepare_top_down_time_ratio,
    "min_distance": _prepare_min_distance,
    "min_time_delta": _prepare_min_time_delta,
    "max_distance": _prepare_max_distance,
    "chan_chin": _prepare_chan_chin,
    "imai_iri": _prepare_imai_iri,
}


def simplify(
    traj: Any,
    method: str = "douglas_peucker",
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    is_sorted: bool = False,
    **method_kwargs: Any,
) -> Any:
    """Simplify a trajectory by dropping points using a named algorithm.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    method:
        Name of the simplification algorithm to run. One of
        ``"douglas_peucker"``, ``"top_down_time_ratio"``, ``"min_distance"``,
        ``"min_time_delta"``, ``"max_distance"``, ``"chan_chin"``, or
        ``"imai_iri"``.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    is_sorted:
        Whether the trajectory is already sorted by user and time. Setting
        this to True can speed up processing but may lead to incorrect
        results if the data is not properly preprocessed.
    **method_kwargs:
        Method-specific parameters, forwarded to the matching
        ``SIMPLIFY_METHODS[method]`` preparer. See each algorithm below for
        its accepted keyword(s) and default.

        - ``douglas_peucker``: ``epsilon_km`` (default ``0.001``) — maximum
          perpendicular distance, in km, a dropped point may deviate from
          the simplified line.
        - ``top_down_time_ratio``: ``epsilon_km`` (default ``0.001``) —
          maximum spatiotemporal distance, in km, between a point and its
          time-interpolated position on the simplified line.
        - ``min_distance``: ``min_distance_km`` (default ``0.01``) — minimum
          distance, in km, required between consecutive kept points.
        - ``min_time_delta``: ``min_time_delta_s`` (default ``60.0``) —
          minimum time, in seconds, required between consecutive kept
          points.
        - ``max_distance``: ``epsilon_km`` (default ``0.001``) — maximum
          perpendicular distance, in km, allowed in this single-pass
          streaming approximation of Douglas-Peucker.
        - ``chan_chin``: ``epsilon_km`` (default ``0.001``) — maximum
          perpendicular distance, in km, allowed between an original point
          and its nearest retained segment.
        - ``imai_iri``: ``epsilon_km`` (default ``0.001``) — same tolerance
          as ``chan_chin``, but searching for the true minimum number of
          retained segments.

    Returns
    -------
    DataFrame
        Simplified trajectory (row subset) in the same backend as input.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> url = fastmob.data.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng"]
    ... ]
    >>> from fastmob.preprocessing import simplify
    >>> simplified = simplify(df, method="douglas_peucker", epsilon_km=0.1)
    >>> len(simplified) <= len(df)
    True

    References
    ----------
    - [DP1973] Douglas, D., & Peucker, T. (1973). Algorithms for the
      reduction of the number of points required to represent a digitized
      line or its caricature. The Canadian Cartographer 10(2), 112-122.
    - [MdB2004] Meratnia, N., & de By, R.A. (2004). Spatiotemporal
      compression techniques for moving point objects. EDBT 2004.
    - [CC1996] Chan, W. S., & Chin, F. (1996). Approximation of polygonal
      curves with minimum number of line segments or minimum error.
      International Journal of Computational Geometry & Applications, 6(1).
    - [II1988] Imai, H., & Iri, M. (1988). Polygonal approximations of a
      curve — formulations and algorithms. Computational Morphology.
    """
    if method not in SIMPLIFY_METHODS:
        raise ValueError(f"unknown simplify method: {method!r}; choose from {sorted(SIMPLIFY_METHODS)}")
    method_name, params = SIMPLIFY_METHODS[method](**method_kwargs)

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
    timestamps = _extract_timestamps(df, datetime_col)
    times_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps)

    config = SimplifyConfig(method=method_name, **params)

    if is_sorted:
        _, sorted_indices, ends = _build_indexed_user_ranges(df, uid_col)
        raw_mask = simplify_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config)
    else:
        timestamp_arrow = _extract_timestamp_arrow(df, datetime_col)
        _, sorted_indices, ends = _build_indexed_user_ranges(
            df,
            uid_col,
            timestamps=timestamp_arrow,
        )
        raw_mask = simplify_trajectory_indexed(lats_data, lngs_data, times_data, sorted_indices, ends, config)

    keep_mask = _narwhals_safe_value(_as_arrow(raw_mask))
    result = df.filter(nw.new_series("__keep__", keep_mask, backend=df.implementation)).to_native()

    return result


simplify.__module__ = "fastmob.preprocessing"
