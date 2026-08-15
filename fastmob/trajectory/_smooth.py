"""Named trajectory smoothing algorithms.

Unlike `interpolate` (which inserts new points and grows row count),
`smooth` replaces every existing row's position at its original timestamp:
output cardinality always equals input cardinality, and row order is
preserved exactly. Follows the same "method name -> Rust config" dispatch
shape as `fastmob/trajectory/_interpolate.py`.
"""

from __future__ import annotations

from typing import Any, Callable

import narwhals as nw

from fastmob._core import (
    SmoothConfig,
    smooth_trajectory_indexed,
    smooth_trajectory_presorted,
)
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import (
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _detect_trajectory_columns,
    _extract_timestamp_arrow,
    _extract_timestamps,
)


def _to_new_series_values(values: Any) -> Any:
    """Coerce a NumPy or pyo3-arrow ``arro3`` result into a value narwhals'
    ``new_series`` can safely add to a Polars frame.

    ``arro3.core.Array`` (the type pyo3-arrow's ``PyArray`` unwraps to in
    Python) implements ``__arrow_c_array__`` but not ``to_pyarrow``, so it
    slips past ``_as_arrow``'s pyarrow-conversion check. Passing
    it directly to ``nw.new_series`` triggers a narwhals/Polars bug where two
    such nameless arrays added in the same ``with_columns`` call collide on
    an empty default name (``the name '' passed to LazyFrame.with_columns is
    duplicate``). Converting to NumPy first sidesteps it; lat/lng values are
    already plain floats at this point, so no information is lost.
    """
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return values


# Bare extractor supplies numeric timestamps to the smoothing kernel.
# Ordering receives the original Arrow timestamp array separately. lat/lng go
# to Arrow unconditionally.
_TIMESTAMP_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _prepare_kalman_cv(
    process_noise_std_km: float = 0.05,
    measurement_noise_std_km: float = 0.1,
    **_: Any,
) -> tuple[str, dict]:
    """Build Rust config params for the Kalman constant-velocity smoother.

    @usedBy `smooth()` via `SMOOTH_METHODS["kalman_cv"]`.
    """
    return "kalman_cv", {
        "process_noise_std_km": process_noise_std_km,
        "measurement_noise_std_km": measurement_noise_std_km,
    }


SMOOTH_METHODS: dict[str, Callable[..., tuple[str, dict]]] = {
    "kalman_cv": _prepare_kalman_cv,
}


def smooth(
    traj: Any,
    method: str = "kalman_cv",
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted: bool = False,
    **method_kwargs: Any,
) -> Any:
    """Smooth a trajectory's positions using a named algorithm.

    Every row's ``(lat, lng)`` is replaced with a denoised estimate at its
    original timestamp; row count and row order are unchanged (unlike
    :func:`fastmob.trajectory.interpolate`, which inserts new points).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    method:
        Name of the smoothing algorithm to run. Only ``"kalman_cv"`` is
        shipped currently.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.
    presorted:
        Whether the trajectory is already sorted by user and time. Setting
        this to True can speed up processing but may lead to incorrect
        results if the data is not properly preprocessed.
    **method_kwargs:
        Method-specific parameters, forwarded to the matching
        ``SMOOTH_METHODS[method]`` preparer.

        - ``kalman_cv``: ``process_noise_std_km`` (default ``0.05``) --
          acceleration-noise standard deviation (km); higher values trust the
          constant-velocity assumption less, producing less smoothing.
          ``measurement_noise_std_km`` (default ``0.1``) -- assumed GPS noise
          standard deviation (km); higher values pull the smoothed path
          closer to a constant-velocity line.

    Returns
    -------
    DataFrame
        The trajectory with smoothed ``lat``/``lng`` values, same row count
        and order as input, in the same backend as input. Rows with a null
        latitude/longitude/datetime are left as null (excluded from
        smoothing per Rule 2, not dropped).

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> df = pd.DataFrame({
    ...     "uid": [1, 1, 1],
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ... })
    >>> from fastmob.trajectory import smooth
    >>> out = smooth(df, method="kalman_cv")
    >>> len(out) == len(df)
    True

    References
    ----------
    - [GJH2020] Graser, A., Jankowski, P., & Heistermann, M. (2020).
      MovingPandas ``KalmanSmootherCV``.
      <a href="https://github.com/movingpandas/movingpandas">https://github.com/movingpandas/movingpandas</a>
    """
    if method not in SMOOTH_METHODS:
        raise ValueError(f"unknown smooth method: {method!r}; choose from {sorted(SMOOTH_METHODS)}")
    method_name, params = SMOOTH_METHODS[method](**method_kwargs)

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
    timestamp_arrow = _extract_timestamp_arrow(df, datetime_col)
    times_data = _TIMESTAMP_EXTRACTOR.get_ops(df)["extract_data"](timestamps)

    config = SmoothConfig(method=method_name, **params)

    if presorted:
        _, ends = _build_presorted_user_ends(df, uid_col)
        out_lats, out_lngs = smooth_trajectory_presorted(lats_data, lngs_data, times_data, ends, config)
    else:
        _, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamp_arrow)
        out_lats, out_lngs = smooth_trajectory_indexed(lats_data, lngs_data, times_data, indices, ends, config)

    result = df.with_columns(
        nw.new_series(lat_col, _to_new_series_values(out_lats), backend=df.implementation),
        nw.new_series(lng_col, _to_new_series_values(out_lngs), backend=df.implementation),
    )

    return result.to_native()


smooth.__module__ = "fastmob.trajectory"
