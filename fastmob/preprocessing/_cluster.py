from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from fastmob.utils._common import _build_presorted_user_ends, _detect_trajectory_columns, _prepare_trajectory

_KMS_PER_RADIAN = 6371.0088


def _dbscan_cls():
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError("scikit-learn is required for cluster: pip install fastmob[ai]") from exc
    return DBSCAN


def cluster(
    traj: Any,
    cluster_radius_km: float = 0.1,
    min_samples: int = 1,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
    presorted=False,
    n_jobs: int | None = None,
) -> Any:
    """Cluster stop locations using DBSCAN with Haversine metric."""

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
        sort=not presorted,
    )

    eps_rad = cluster_radius_km / _KMS_PER_RADIAN
    DBSCAN = _dbscan_cls()

    _, group_ends = _build_presorted_user_ends(df, uid_col)
    ends = np.asarray(group_ends, dtype=np.intp)
    starts = np.concatenate(([0], ends[:-1]))

    # 1. PRE-COMPUTE COORDINATES globally
    lats = df.get_column(lat_col).to_numpy()
    lngs = df.get_column(lng_col).to_numpy()
    all_coords = np.radians(np.column_stack([lats, lngs]))

    # 2. PRE-ALLOCATE MEMORY for the output labels
    n_rows = len(df)
    all_labels = np.empty(n_rows, dtype=np.int32)

    # 3. REUSE ESTIMATOR to avoid initialization overhead
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree", metric="haversine", n_jobs=n_jobs)

    for start, end in zip(starts, ends):
        user_coords = all_coords[start:end]
        if len(user_coords) == 0:
            continue

        raw_labels = db.fit(user_coords).labels_

        # 4. VECTORIZE THE REMAPPING
        mask = raw_labels >= 0
        valid_labels = raw_labels[mask]

        if len(valid_labels) > 0:
            # Count occurrences of valid labels
            unique_lbls, counts = np.unique(valid_labels, return_counts=True)

            # Sort labels by count descending
            sorted_idx = np.argsort(-counts)
            sorted_lbls = unique_lbls[sorted_idx]

            # Create a direct array lookup map
            max_lbl = np.max(raw_labels)
            map_arr = np.full(max_lbl + 1, -1, dtype=np.int32)
            map_arr[sorted_lbls] = np.arange(len(sorted_lbls))

            # Apply the mapping only to valid labels
            remapped = raw_labels.copy()
            remapped[mask] = map_arr[valid_labels]
        else:
            remapped = raw_labels

        # Assign directly into the pre-allocated array
        all_labels[start:end] = remapped

    # 5. USE DATAFRAME API safely based on your specific Narwhals version
    try:
        cluster_series = nw.new_series(name="cluster", values=all_labels, backend=df.implementation)
    except Exception:  # noqa: BLE001
        # Fallback to from_dict (which your original code successfully used)
        cluster_series = nw.from_dict({"cluster": all_labels}, backend=df.implementation).get_column("cluster")

    result = df.with_columns(cluster_series)

    return result.to_native()


cluster.__module__ = "fastmob.preprocessing"
