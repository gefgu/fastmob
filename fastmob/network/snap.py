from __future__ import annotations

import narwhals as nw
import numpy as np

from ._nearest import nearest_candidate


def snap_locations_to_graph(
    tessellation_df,
    nodes_df,
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
    n = len(tessellation_df)
    nodes = nw.from_native(nodes_df, eager_only=True)
    if len(nodes) == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64)

    tess = nw.from_native(tessellation_df, eager_only=True)
    node_lat = nodes.get_column("lat").to_numpy().astype(np.float64, copy=False)
    node_lng = nodes.get_column("lng").to_numpy().astype(np.float64, copy=False)
    loc_lat = tess.get_column(lat_col).to_numpy().astype(np.float64, copy=False)
    loc_lng = tess.get_column(lng_col).to_numpy().astype(np.float64, copy=False)

    nearest_idx, dist_m = nearest_candidate(loc_lat, loc_lng, node_lat, node_lng)
    node_idx = nodes.get_column("node_idx").to_numpy().astype(np.int64, copy=False)[nearest_idx]
    return np.where(dist_m <= max_distance_m, node_idx, -1).astype(np.int64)
