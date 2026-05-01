from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import narwhals as nw
from sklearn.cluster import DBSCAN

from ..measures._common import _build_user_ranges, _prepare_trajectory

_KMS_PER_RADIAN = 6371.0088


def cluster(
    traj: Any,
    cluster_radius_km: float = 0.1,
    min_samples: int = 1,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Cluster stop locations using DBSCAN with Haversine metric.

    Groups stops that are within cluster_radius_km of each other. Cluster labels
    are ranked by visit frequency (most visited = 0). Noise points get label -1.

    Parameters
    ----------
    traj:
        Stop location dataframe (typically the output of stay_locations).
        Any Narwhals-compatible eager backend.
    cluster_radius_km:
        DBSCAN eps parameter in kilometers.
    min_samples:
        Minimum number of stops to form a cluster.
    datetime_col, lat_col, lng_col, uid_col:
        Explicit column name overrides; auto-detected when None.

    Returns
    -------
    DataFrame
        Input dataframe with an added 'cluster' integer column.

    References
    ----------
    - [DBSCAN] DBSCAN implementation, scikit-learn, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    - [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf

    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    eps_rad = cluster_radius_km / _KMS_PER_RADIAN

    _, ranges = _build_user_ranges(df, uid_col)

    lats = df.get_column(lat_col).to_numpy()
    lngs = df.get_column(lng_col).to_numpy()

    all_labels: list[int] = []

    for start, end in ranges:
        coords = np.radians(np.column_stack([lats[start:end], lngs[start:end]]))
        if len(coords) == 0:
            continue

        db = DBSCAN(
            eps=eps_rad,
            min_samples=min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )
        raw_labels = db.fit(coords).labels_

        # Remap labels by visit frequency (most visited → 0)
        counts = Counter(lbl for lbl in raw_labels if lbl >= 0)
        sorted_clusters = sorted(counts.keys(), key=lambda lbl: -counts[lbl])
        remap = {old: new for new, old in enumerate(sorted_clusters)}

        remapped = [remap[lbl] if lbl >= 0 else -1 for lbl in raw_labels]
        all_labels.extend(remapped)

    col_dict = {col: df.get_column(col).to_list() for col in df.columns}
    col_dict["cluster"] = all_labels
    result = nw.from_dict(col_dict, backend=df.implementation)
    return result.to_native()
