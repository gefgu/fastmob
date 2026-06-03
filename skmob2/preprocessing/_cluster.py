from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import narwhals as nw

from ..measures._common import _build_user_ranges, _detect_trajectory_columns, _prepare_trajectory

_KMS_PER_RADIAN = 6371.0088


def _dbscan_cls():
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError("scikit-learn is required for cluster: pip install skmob2[ai]") from exc
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
    sorted=False,
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
    sorted:
        Whether the trajectory is already sorted by user and time.

    Returns
    -------
    DataFrame
        Input dataframe with an added 'cluster' integer column.


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
    >>> from skmob2.preprocessing import cluster, stay_locations
    >>> stops = stay_locations(df, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    >>> clustered = cluster(stops, cluster_radius_km=0.1)
    >>> print(clustered.head().to_string(index=False))
          lat         lng            datetime  uid    leaving_datetime  cluster
    37.774929 -122.419415 2009-05-25 20:56:10    0 2009-05-25 21:35:28      119
    37.600747 -122.382376 2009-05-25 21:35:28    0 2009-05-25 22:13:23       81
    37.615223 -122.389979 2009-05-25 22:13:23    0 2009-05-26 02:21:12       60
    39.878664 -104.682105 2009-05-26 02:21:12    0 2009-05-26 04:59:44       40
    39.739154 -104.984703 2009-05-26 04:59:44    0 2009-05-26 16:43:59        2

    References
    ----------
    - [DBSCAN] DBSCAN implementation, scikit-learn, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html">https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html</a>
    - [RT2004] Ramaswamy, H. & Toyama, K. (2004) Project Lachesis: parsing and modeling location histories. In International Conference on Geographic Information Science, 106-124, <a href="http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf">http://kentarotoyama.com/papers/Hariharan_2004_Project_Lachesis.pdf</a>

    """
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
        sort=not sorted,
    )

    eps_rad = cluster_radius_km / _KMS_PER_RADIAN
    DBSCAN = _dbscan_cls()

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

        counts = Counter(lbl for lbl in raw_labels if lbl >= 0)
        sorted_clusters = list(counts.keys())
        sorted_clusters.sort(key=lambda lbl: -counts[lbl])
        remap = {old: new for new, old in enumerate(sorted_clusters)}

        remapped = [remap[lbl] if lbl >= 0 else -1 for lbl in raw_labels]
        all_labels.extend(remapped)

    col_dict = {col: df.get_column(col).to_list() for col in df.columns}
    col_dict["cluster"] = all_labels
    result = nw.from_dict(col_dict, backend=df.implementation)
    return result.to_native()


cluster.__module__ = "skmob2.preprocessing"
