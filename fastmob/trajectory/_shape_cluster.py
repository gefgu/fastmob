"""Clustering trajectories by shape (as opposed to `fastmob.preprocessing.
cluster`, which clusters *stop-location points* within one user's own
trajectory).

Ported from tracktable's `cluster_trajectories_shape` /
`distance_geometry_by_distance`: each trajectory is reduced to a compact,
rotation/translation-invariant distance-geometry signature (see
`fastmob-core/src/trajectory/shape_signature.rs`), then those signature
vectors are clustered with a Euclidean-eps DBSCAN (deliberately not
tracktable's own box/Chebyshev-metric DBSCAN, since signature values are
already normalized to `[0,1]`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fastmob._core import cluster_trajectory_shape_signatures, trajectory_shape_signatures
from fastmob.core.base import unwrap_native
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import _prepare_trajectory, _trajectory_input

_EXTRACTOR = TrajectoryDispatcher(arrow_ops={}, numpy_ops={})


def _single_trajectory_lat_lng(
    traj: Any,
    *,
    datetime_col: str | None,
    lat_col: str | None,
    lng_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare one trajectory into a chronologically-sorted, null-cleaned
    (lats, lngs) NumPy pair.

    Deliberately deviates from Rule 2's nullable-indexed-path convention,
    the same documented precedent `fastmob/trajectory/_distance.py`'s
    `_single_sequence` already established: each list element here is
    already one independent whole sequence, with no per-user grouping to
    preserve partial validity across.
    """
    df, datetime_col, lat_col, lng_col, uid_col = _trajectory_input(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=None,
    )
    if uid_col is not None and uid_col in df.columns:
        n_unique = df.get_column(uid_col).n_unique()
        if n_unique > 1:
            raise ValueError(
                "cluster_trajectory_shapes requires each list element to be a single "
                f"user's trajectory; got {n_unique} distinct users in column {uid_col!r}."
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
    ops = _EXTRACTOR.get_ops(df)
    lats = np.asarray(ops["extract_data"](df.get_column(lat_col)), dtype=np.float64)
    lngs = np.asarray(ops["extract_data"](df.get_column(lng_col)), dtype=np.float64)
    return lats, lngs


def cluster_trajectory_shapes(
    trajectories: Sequence[Any],
    depth: int = 4,
    epsilon: float = 0.05,
    min_cluster_size: int = 2,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    datetime_col: str | None = None,
) -> np.ndarray:
    """Cluster a list of independent trajectories by their overall shape.

    Parameters
    ----------
    trajectories : Sequence
        Each element is a single trajectory: any Narwhals-compatible eager
        dataframe (or `TrajDataFrame`) representing one user's/one trip's
        point sequence. Must have at least 2 points and no more than one
        distinct user.
    depth : int, optional
        Number of distance-geometry levels (signature length is
        ``depth * (depth + 1) // 2``). Higher values capture finer shape
        detail at the cost of a higher-dimensional signature. Default ``4``.
    epsilon : float, optional
        DBSCAN neighborhood radius over the normalized signature space
        (values are in ``[0, 1]``, so a typical ``epsilon`` is a small
        fraction like ``0.05``). Default ``0.05``.
    min_cluster_size : int, optional
        DBSCAN minimum points per cluster; must be ``>= 2``. Default ``2``.
    lat_col, lng_col, datetime_col : str, optional
        Explicit column name overrides; auto-detected per trajectory when
        None.

    Returns
    -------
    numpy.ndarray
        One cluster label per input trajectory, in input order. ``-1``
        marks DBSCAN noise (an outlier shape), matching
        `fastmob.preprocessing.cluster`'s convention.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> from fastmob.trajectory import cluster_trajectory_shapes
    >>> line_a = pd.DataFrame({
    ...     "lat": [0.0, 1.0, 2.0], "lng": [0.0, 0.0, 0.0],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ... })
    >>> line_b = pd.DataFrame({
    ...     "lat": [10.0, 11.0, 12.0], "lng": [10.0, 10.0, 10.0],
    ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
    ... })
    >>> labels = cluster_trajectory_shapes([line_a, line_b], depth=2, epsilon=0.1)
    >>> labels[0] == labels[1]
    np.True_
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if len(trajectories) == 0:
        return np.empty(0, dtype=np.int64)

    prepared = [
        _single_trajectory_lat_lng(traj, datetime_col=datetime_col, lat_col=lat_col, lng_col=lng_col)
        for traj in trajectories
    ]

    signature_len = depth * (depth + 1) // 2
    flat_signatures = trajectory_shape_signatures(prepared, depth)
    labels = cluster_trajectory_shape_signatures(flat_signatures, signature_len, epsilon, min_cluster_size)
    return np.asarray(labels)


cluster_trajectory_shapes.__module__ = "fastmob.trajectory"


def cluster_trajectory_shapes_from_segments(
    traj: Any,
    segment_col: str = "segment_id",
    uid_col: str | None = None,
    depth: int = 4,
    epsilon: float = 0.05,
    min_cluster_size: int = 2,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    datetime_col: str | None = None,
) -> Any:
    """Cluster trajectory shapes directly from a `segment()`-output (or
    `Triplegs.points`) frame, splitting into per-segment trajectories internally.

    Parameters
    ----------
    traj : DataFrame-like
        A dataframe already carrying a segment/group id column (e.g. the
        output of `fastmob.preprocessing.segment`).
    segment_col : str, optional
        Column identifying which rows belong to the same trajectory.
        Default ``"segment_id"``.
    uid_col : str, optional
        When given, groups by ``(uid_col, segment_col)`` instead of
        ``segment_col`` alone (segment ids commonly restart per user).
    **kwargs
        Forwarded to :func:`cluster_trajectory_shapes`.

    Returns
    -------
    DataFrame
        The distinct ``(uid_col?, segment_col)`` keys with an added
        ``shape_cluster`` column, in the same backend as input.
    """
    import narwhals as nw

    nw_df = nw.from_native(unwrap_native(traj), eager_only=True)
    group_cols = [uid_col, segment_col] if uid_col else [segment_col]
    keys = nw_df.select(group_cols).unique(maintain_order=True)

    trajectories = []
    for row in keys.iter_rows(named=True):
        mask = nw.col(segment_col) == row[segment_col]
        if uid_col:
            mask = mask & (nw.col(uid_col) == row[uid_col])
        trajectories.append(nw_df.filter(mask).to_native())

    labels = cluster_trajectory_shapes(
        trajectories,
        depth=depth,
        epsilon=epsilon,
        min_cluster_size=min_cluster_size,
        lat_col=lat_col,
        lng_col=lng_col,
        datetime_col=datetime_col,
    )
    result = keys.with_columns(nw.new_series("shape_cluster", np.asarray(labels), backend=keys.implementation))
    return result.to_native()


cluster_trajectory_shapes_from_segments.__module__ = "fastmob.trajectory"
