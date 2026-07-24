from __future__ import annotations

import math

import numpy as np

from ._util import haversine_m_batch


def nearest_candidate(
    query_lat: np.ndarray,
    query_lng: np.ndarray,
    ref_lat: np.ndarray,
    ref_lng: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find each query point's nearest reference point by exact Haversine distance.

    Builds a longitude-scaled (by ``cos(mean reference latitude)``) KD-tree
    for an approximate nearest-neighbor candidate, then re-scores that single
    candidate with an exact Haversine distance -- the approximate-then-exact
    pattern shared by :func:`fastmob.network.snap.snap_locations_to_graph`
    and the PyMove-style POI/event join functions.

    Parameters
    ----------
    query_lat, query_lng: np.ndarray
        Points to look up.
    ref_lat, ref_lng: np.ndarray
        Reference points to search (e.g. graph nodes, POIs, events).

    Returns
    -------
    (indices, distances_m)
        ``indices`` is each query point's nearest reference row index (into
        `ref_lat`/`ref_lng`), ``-1`` if `ref_lat` is empty; ``distances_m``
        is its exact Haversine distance, ``inf`` in that same case.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(query_lat)
    if len(ref_lat) == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf, dtype=np.float64)

    mean_lat = float(np.mean(ref_lat))
    scale = math.cos(math.radians(mean_lat))

    ref_xy = np.column_stack([ref_lat, ref_lng * scale])
    tree = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    tree.fit(ref_xy)

    query_xy = np.column_stack([query_lat, query_lng * scale])
    _, indices = tree.kneighbors(query_xy)
    nearest_idx = indices[:, 0]

    dist_m = haversine_m_batch(query_lat, query_lng, ref_lat[nearest_idx], ref_lng[nearest_idx])
    return nearest_idx.astype(np.int64), dist_m
