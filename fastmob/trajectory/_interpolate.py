from __future__ import annotations

from typing import Any, Callable

import narwhals as nw

from fastmob._core import (
    InterpolationConfig,
    interpolate_trajectory_indexed,
    interpolate_trajectory_presorted,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _arrow_flat_result_values,
    _build_presorted_user_ends,
    _build_time_ordered_user_ranges,
    _detect_trajectory_columns,
    _extract_timestamps_s,
    _take_uid_values,
    _timestamps_s_to_datetime_ns,
    _to_native,
)

# Bare extractor used only for `.get_ops(df)["extract_data"]` (Rule 1: never
# inline backend branching). Which logical Rust function to call is a
# presorted-vs-indexed choice, not a per-backend one; only column extraction
# differs by backend.
_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _prepare_linear(**_: Any) -> tuple[str, dict]:
    """Build Rust config params for linear interpolation.

    @usedBy `interpolate()` via `INTERPOLATE_METHODS["linear"]`.
    """
    return "linear", {}


def _prepare_cubic_spline(**_: Any) -> tuple[str, dict]:
    """Build Rust config params for cubic-spline interpolation.

    @usedBy `interpolate()` via `INTERPOLATE_METHODS["cubic_spline"]`.
    """
    return "cubic_spline", {}


def _prepare_kinematic(max_speed_kmh: float = 300.0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for kinematic interpolation.

    @usedBy `interpolate()` via `INTERPOLATE_METHODS["kinematic"]`.
    """
    return "kinematic", {"max_speed_kmh": max_speed_kmh}


def _prepare_random_walk(step_std_km: float = 0.01, seed: int = 0, **_: Any) -> tuple[str, dict]:
    """Build Rust config params for random-walk interpolation.

    @usedBy `interpolate()` via `INTERPOLATE_METHODS["random_walk"]`.
    """
    return "random_walk", {"step_std_km": step_std_km, "seed": seed}


INTERPOLATE_METHODS: dict[str, Callable[..., tuple[str, dict]]] = {
    "linear": _prepare_linear,
    "cubic_spline": _prepare_cubic_spline,
    "kinematic": _prepare_kinematic,
    "random_walk": _prepare_random_walk,
}


def interpolate(
    traj: Any,
    method: str = "linear",
    sampling_rate_s: float = 3600.0,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
    **method_kwargs: Any,
) -> Any:
    """Fill gaps in a trajectory by inserting interpolated points.

    For every gap between two chronologically consecutive points of the same
    user whose time delta exceeds ``sampling_rate_s``, exactly one new point
    is inserted at ``t[i-1] + sampling_rate_s`` (matching PTRAIL's
    ``Interpolation.interpolate_position`` insertion policy: a large gap is
    not filled iteratively down to ``sampling_rate_s``-sized steps -- only
    one point is added per gap, regardless of how large it is).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    method:
        Name of the interpolation algorithm to run. One of ``"linear"``,
        ``"cubic_spline"``, ``"kinematic"``, or ``"random_walk"``.
    sampling_rate_s:
        Maximum time gap, in seconds, allowed between consecutive points
        before an interpolated point is inserted. Default ``3600.0``.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    presorted:
        Whether the trajectory is already sorted by user and time. Setting
        this to True can speed up processing but may lead to incorrect
        results if the data is not properly preprocessed.
    **method_kwargs:
        Method-specific parameters, forwarded to the matching
        ``INTERPOLATE_METHODS[method]`` preparer.

        - ``linear``, ``cubic_spline``: no extra parameters.
        - ``kinematic``: ``max_speed_kmh`` (default ``300.0``) -- safety
          clamp; a gap falls back to linear interpolation if the fitted
          kinematic position implies a speed above this bound from the
          preceding point.
        - ``random_walk``: ``step_std_km`` (default ``0.01``) -- half-normal
          standard deviation, in km, of the random perturbation applied on
          top of the linear-interpolated base position; ``seed`` (default
          ``0``) -- RNG seed; each user draws an independent, seed-derived
          sequence, so results are reproducible but do not attempt to
          reproduce any other library's own random draws.

    Returns
    -------
    DataFrame
        The expanded trajectory (original points, plus any inserted points)
        in the same backend as input, sorted chronologically per user.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> df = pd.DataFrame({
    ...     "uid": [1, 1, 1],
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ...     "datetime": pd.to_datetime(
    ...         ["2020-01-01 00:00", "2020-01-01 02:00", "2020-01-01 03:00"]
    ...     ),
    ... })
    >>> from fastmob.trajectory import interpolate
    >>> out = interpolate(df, method="linear", sampling_rate_s=3600.0)
    >>> len(out) >= len(df)
    True

    References
    ----------
    - [PTRAIL] Haranwala, Y.J., & Haidri, S. PTRAIL: A Python package for
      parallel trajectory data preprocessing.
    - [Nogueira2016] Nogueira, T.O. "kinematic_interpolation.py" (2016).
    """
    if method not in INTERPOLATE_METHODS:
        raise ValueError(f"unknown interpolate method: {method!r}; choose from {sorted(INTERPOLATE_METHODS)}")
    method_name, params = INTERPOLATE_METHODS[method](**method_kwargs)

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

    ops = _EXTRACTOR.get_ops(df)
    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))
    timestamps_s = _extract_timestamps_s(df, datetime_col)
    times_data = ops["extract_data"](timestamps_s)

    config = InterpolationConfig(method=method_name, sampling_rate_s=sampling_rate_s, **params)

    if presorted:
        uid_values, ends = _build_presorted_user_ends(df, uid_col)
        out_lats, out_lngs, out_times, out_user_idx = interpolate_trajectory_presorted(
            lats_data, lngs_data, times_data, ends, config
        )
    else:
        uid_values, indices, ends = _build_time_ordered_user_ranges(df, uid_col, datetime_col, times_data)
        out_lats, out_lngs, out_times, out_user_idx = interpolate_trajectory_indexed(
            lats_data, lngs_data, times_data, indices, ends, config
        )

    result_dict: dict[str, Any] = {}
    if uid_col is not None:
        result_dict[uid_col] = _take_uid_values(uid_values, out_user_idx)
    result_dict[datetime_col] = _timestamps_s_to_datetime_ns(out_times)
    result_dict[lat_col] = _arrow_flat_result_values(out_lats)
    result_dict[lng_col] = _arrow_flat_result_values(out_lngs)

    return _to_native(result_dict, df)


interpolate.__module__ = "fastmob.trajectory"
