from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._util import haversine_m_batch


def snap_locations_to_graph(
    tessellation_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    max_distance_m: float,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> np.ndarray:
    """Snap each tessellation row to its nearest road/rail graph node.

    Parameters
    ----------
    tessellation_df:
        Rows with lat/lng columns to snap.
    nodes_df:
        Graph nodes with columns ``node_idx``, ``lat``, ``lng`` (as returned
        by :func:`fastmob.network.builder.fetch_road_network` / `fetch_rail_network`).
    max_distance_m:
        Maximum snap distance; farther rows are reported unsnapped.
    lat_col, lng_col:
        Column names on ``tessellation_df``.

    Returns
    -------
    numpy.ndarray
        int64 array aligned 1:1 with ``tessellation_df`` rows; ``-1`` when
        the nearest node is farther than ``max_distance_m`` (unsnapped).
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(tessellation_df)
    if len(nodes_df) == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64)

    mean_lat = float(nodes_df["lat"].mean())
    scale = math.cos(math.radians(mean_lat))

    node_lat = nodes_df["lat"].to_numpy(dtype=float)
    node_lng = nodes_df["lng"].to_numpy(dtype=float)
    node_xy = np.column_stack([node_lat, node_lng * scale])

    tree = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    tree.fit(node_xy)

    loc_lat = tessellation_df[lat_col].to_numpy(dtype=float)
    loc_lng = tessellation_df[lng_col].to_numpy(dtype=float)
    loc_xy = np.column_stack([loc_lat, loc_lng * scale])
    _, indices = tree.kneighbors(loc_xy)
    nearest_idx = indices[:, 0]

    dist_m = haversine_m_batch(loc_lat, loc_lng, node_lat[nearest_idx], node_lng[nearest_idx])
    node_idx = nodes_df["node_idx"].to_numpy(dtype=np.int64)[nearest_idx]
    return np.where(dist_m <= max_distance_m, node_idx, -1).astype(np.int64)
