from __future__ import annotations

import numpy as np


def nearest_candidate(
    query_lat: np.ndarray,
    query_lng: np.ndarray,
    ref_lat: np.ndarray,
    ref_lng: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find each query point's nearest reference point by exact Haversine distance.

    The spatial index and queries execute in Rust on Arrow arrays. The public
    NumPy-shaped helper remains for the integration module's existing adapter
    contract.

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
    import pyarrow as pa
    from fastmob._core import nearest_nodes_arrow

    n = len(query_lat)
    if len(ref_lat) == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf, dtype=np.float64)

    indices, distances_m = nearest_nodes_arrow(
        pa.array(query_lat, type=pa.float64()),
        pa.array(query_lng, type=pa.float64()),
        pa.array(ref_lat, type=pa.float64()),
        pa.array(ref_lng, type=pa.float64()),
        float("inf"),
    )
    return np.asarray(indices, dtype=np.int64), np.asarray(distances_m, dtype=np.float64)
