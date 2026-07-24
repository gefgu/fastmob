from __future__ import annotations

import numpy as np
import pandas as pd


def od_desire_lines(
    road_network,
    from_nodes: np.ndarray,
    to_nodes: np.ndarray,
    flows: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Aggregate OD-pair flows onto the road/rail graph's edges (desire lines).

    The Overture-native analogue of stplanr's ``overline``/``overline2``:
    for each ``(from_node, to_node, flow)`` triple, walks the time-optimal
    route between the two nodes and adds ``flow`` to every edge it crosses,
    so many overlapping OD pairs accumulate onto shared road segments
    instead of remaining one separate desire line per pair.

    Parameters
    ----------
    road_network: :class:`fastmob.network.road_graph.RoadNetwork`
        A prepared network (see ``RoadNetwork.build``); ``from_nodes``/
        ``to_nodes`` are node ids from that network's ``nodes_df`` (e.g.
        via :func:`fastmob.network.snap.snap_locations_to_graph`).
    from_nodes, to_nodes: np.ndarray
        Node ids per OD pair; a negative value marks an unsnapped origin
        or destination.
    flows: np.ndarray
        Flow volume (e.g. trip count) per OD pair.

    Returns
    -------
    (edges_df, dropped_flow)
        ``edges_df`` has columns ``edge_from``, ``edge_to``, ``from_lat``,
        ``from_lng``, ``to_lat``, ``to_lng``, ``total_flow``, sorted by
        descending ``total_flow``, one row per edge touched by at least one
        route. ``dropped_flow`` is the summed flow of OD pairs that were
        unsnapped or whose endpoints sit in disconnected graph components.
    """
    edge_from, edge_to, total_flow, dropped_flow = road_network._handle.route_edge_flows(
        from_nodes.astype(np.int64), to_nodes.astype(np.int64), flows.astype(np.float64)
    )
    edge_from = np.asarray(edge_from, dtype=np.int64)
    edge_to = np.asarray(edge_to, dtype=np.int64)
    total_flow = np.asarray(total_flow, dtype=np.float64)

    # `node_idx` is dense/0-based (see `RoadNetwork.build`), so a
    # sort-then-position lookup maps node id -> (lat, lng) directly.
    nodes_sorted = road_network.nodes_df.sort_values("node_idx")
    lat_by_node = nodes_sorted["lat"].to_numpy()
    lng_by_node = nodes_sorted["lng"].to_numpy()

    edges_df = pd.DataFrame(
        {
            "edge_from": edge_from,
            "edge_to": edge_to,
            "from_lat": lat_by_node[edge_from],
            "from_lng": lng_by_node[edge_from],
            "to_lat": lat_by_node[edge_to],
            "to_lng": lng_by_node[edge_to],
            "total_flow": total_flow,
        }
    )
    return edges_df, float(dropped_flow)
