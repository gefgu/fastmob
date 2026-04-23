"""Spatio-temporal Wasserstein distance between two GPS trajectories."""
from __future__ import annotations

from typing import Any

from skmob2._core import wasserstein_emd

from ._common import _prepare_trajectory


def wasserstein_distance(
    traj_a: Any,
    traj_b: Any,
    reg: float = 0.1,
    max_iter: int = 100,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> float:
    """Compute the Wasserstein Earth Mover's Distance between two GPS point clouds.

    Builds a pairwise Haversine cost matrix between the two trajectories and
    solves the optimal transport problem via the Sinkhorn log-domain algorithm.
    Both trajectories are treated as uniform discrete distributions over their
    GPS fixes (equal weight per point).

    Parameters
    ----------
    traj_a:
        First trajectory; any Narwhals-compatible eager dataframe (pandas,
        polars, …).  Must contain lat/lng columns (auto-detected or explicit).
    traj_b:
        Second trajectory; same format as ``traj_a``.
    reg:
        Sinkhorn regularisation parameter (``> 0``).  Smaller values give a
        more exact transport distance but converge more slowly.  Default ``0.1``.
    max_iter:
        Maximum number of Sinkhorn iterations.  Default ``100``.
    datetime_col:
        Explicit datetime column name for both trajectories.  Auto-detected
        when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.  When absent
        the whole frame is treated as a single individual.

    Returns
    -------
    float
        The Wasserstein distance in km between the two spatial distributions.

    Raises
    ------
    ValueError
        When required columns cannot be found, when either trajectory is empty,
        or when ``reg <= 0``.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.wasserstein_distance import wasserstein_distance
    >>> df_a = pd.DataFrame({
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ... })
    >>> df_b = pd.DataFrame({
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ...     "lat": [0.0, 1.0, 2.0],
    ...     "lng": [0.0, 0.0, 0.0],
    ... })
    >>> wasserstein_distance(df_a, df_b)
    0.0
    """
    if reg <= 0:
        raise ValueError(f"reg must be positive, got {reg}")

    df_a, _, lat_col_a, lng_col_a, _ = _prepare_trajectory(
        traj_a,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )
    df_b, _, lat_col_b, lng_col_b, _ = _prepare_trajectory(
        traj_b,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    lats_a = df_a.get_column(lat_col_a).to_list()
    lons_a = df_a.get_column(lng_col_a).to_list()
    lats_b = df_b.get_column(lat_col_b).to_list()
    lons_b = df_b.get_column(lng_col_b).to_list()

    return wasserstein_emd(lats_a, lons_a, lats_b, lons_b, reg, max_iter)
