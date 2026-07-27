from __future__ import annotations

from typing import Any, Callable

from fastmob._core import DistanceConfig
from fastmob._core import trajectory_distance as _trajectory_distance_kernel
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.measures.evaluation.spatial import _trajectory_input
from fastmob.utils._common import _prepare_trajectory

# Bare extractor used only for `.get_backend_key(df)` (Rule 1: never inline
# backend branching).
_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _prepare_dtw(**_: Any) -> tuple[str, dict]:
    """@usedBy `trajectory_distance()` via `DISTANCE_METHODS["dtw"]`."""
    return "dtw", {}


def _prepare_frechet(**_: Any) -> tuple[str, dict]:
    """@usedBy `trajectory_distance()` via `DISTANCE_METHODS["frechet"]`."""
    return "frechet", {}


def _prepare_hausdorff(**_: Any) -> tuple[str, dict]:
    """@usedBy `trajectory_distance()` via `DISTANCE_METHODS["hausdorff"]`."""
    return "hausdorff", {}


def _prepare_lcss(epsilon_km: float = 0.1, **_: Any) -> tuple[str, dict]:
    """@usedBy `trajectory_distance()` via `DISTANCE_METHODS["lcss"]`."""
    return "lcss", {"epsilon_km": epsilon_km}


DISTANCE_METHODS: dict[str, Callable[..., tuple[str, dict]]] = {
    "dtw": _prepare_dtw,
    "frechet": _prepare_frechet,
    "hausdorff": _prepare_hausdorff,
    "lcss": _prepare_lcss,
}


def _single_sequence(
    traj: Any,
    *,
    datetime_col: str | None,
    lat_col: str | None,
    lng_col: str | None,
    uid_col: str | None,
) -> tuple[Any, str, str]:
    """Prepare one side of a trajectory-pair comparison: a single,
    chronologically-sorted, null-cleaned point sequence.

    Deliberately deviates from Rule 2's nullable-indexed-path convention:
    this always calls ``_prepare_trajectory(sort=True, drop_nulls=True)``
    rather than passing an Arrow validity mask into the Rust kernel. DTW,
    Fréchet, Hausdorff, and LCSS each compare exactly one trajectory per
    side -- there is no per-user grouping to preserve partial validity
    across, and none of the four define a "skip this point but keep the
    sequence" semantics distinct from "drop the null row and shrink the
    sequence." Cleaning once, up front, in Python is both correct and
    simpler than threading a validity mask through a kernel that has no
    group boundaries to align it with.
    """
    df, datetime_col, lat_col, lng_col, uid_col = _trajectory_input(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    if uid_col is not None:
        n_unique = df.get_column(uid_col).n_unique()
        if n_unique > 1:
            raise ValueError(
                "trajectory_distance compares two single point-sequences; got "
                f"{n_unique} distinct users in column {uid_col!r}. Slice to one user first."
            )
    df = _prepare_trajectory(
        df,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
        sort=True,
        drop_nulls=True,
    )
    return df, lat_col, lng_col


def trajectory_distance(
    traj_a: Any,
    traj_b: Any,
    method: str = "dtw",
    *,
    datetime_col_a: str | None = None,
    lat_col_a: str | None = None,
    lng_col_a: str | None = None,
    uid_col_a: str | None = None,
    datetime_col_b: str | None = None,
    lat_col_b: str | None = None,
    lng_col_b: str | None = None,
    uid_col_b: str | None = None,
    **method_kwargs: Any,
) -> float:
    """Compute a similarity/distance metric between two trajectories.

    Compares exactly two whole point-sequences -- each side must be a single
    user's trajectory (or have no user column at all), not a multi-user
    dataframe. Point-to-point distances are computed via haversine, matching
    every other fastmob measure's geographic-distance convention.

    Parameters
    ----------
    traj_a, traj_b:
        Trajectory dataframes; any Narwhals-compatible eager backend, or a
        ``TrajDataFrame``.
    method:
        ``"dtw"``, ``"frechet"``, or ``"hausdorff"`` return a distance in km
        (``0`` = identical shape); ``"lcss"`` returns a similarity in
        ``[0, 1]`` (``1`` = identical shape).
    datetime_col_a, lat_col_a, lng_col_a, uid_col_a:
        Explicit column overrides for `traj_a`; auto-detected when None.
    datetime_col_b, lat_col_b, lng_col_b, uid_col_b:
        Explicit column overrides for `traj_b`; auto-detected when None.
    **method_kwargs:
        Method-specific parameters, forwarded to the matching
        ``DISTANCE_METHODS[method]`` preparer.

        - ``lcss``: ``epsilon_km`` (default ``0.1``) -- two points are
          considered a match when their haversine distance is within this
          threshold.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If either side's user-ID column contains more than one distinct
        user.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> traj_a = pd.DataFrame({
    ...     "lat": [0.0, 1.0, 2.0], "lng": [0.0, 0.0, 0.0],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ... })
    >>> traj_b = pd.DataFrame({
    ...     "lat": [0.0, 1.0, 2.0], "lng": [0.1, 0.1, 0.1],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ... })
    >>> from fastmob.trajectory import trajectory_distance
    >>> trajectory_distance(traj_a, traj_b, method="dtw") > 0
    True

    References
    ----------
    - [Vlachos2002] Vlachos, M., Kollios, G., & Gunopulos, D. (2002).
      Discovering similar multidimensional trajectories. ICDE 2002.
    """
    if method not in DISTANCE_METHODS:
        raise ValueError(f"unknown distance method: {method!r}; choose from {sorted(DISTANCE_METHODS)}")
    method_name, params = DISTANCE_METHODS[method](**method_kwargs)

    df_a, lat_a, lng_a = _single_sequence(
        traj_a,
        datetime_col=datetime_col_a,
        lat_col=lat_col_a,
        lng_col=lng_col_a,
        uid_col=uid_col_a,
    )
    df_b, lat_b, lng_b = _single_sequence(
        traj_b,
        datetime_col=datetime_col_b,
        lat_col=lat_col_b,
        lng_col=lng_col_b,
        uid_col=uid_col_b,
    )

    use_arrow = _EXTRACTOR.get_backend_key(df_a) == "arrow" and _EXTRACTOR.get_backend_key(df_b) == "arrow"

    if use_arrow:
        lats_a = df_a.get_column(lat_a).to_arrow()
        lngs_a = df_a.get_column(lng_a).to_arrow()
        lats_b = df_b.get_column(lat_b).to_arrow()
        lngs_b = df_b.get_column(lng_b).to_arrow()
    else:
        lats_a = df_a.get_column(lat_a).to_numpy()
        lngs_a = df_a.get_column(lng_a).to_numpy()
        lats_b = df_b.get_column(lat_b).to_numpy()
        lngs_b = df_b.get_column(lng_b).to_numpy()

    config = DistanceConfig(method=method_name, **params)
    return float(_trajectory_distance_kernel(lats_a, lngs_a, lats_b, lngs_b, config))


trajectory_distance.__module__ = "fastmob.trajectory"
