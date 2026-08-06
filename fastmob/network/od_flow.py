from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc


def od_desire_lines(
    road_network,
    from_nodes,
    to_nodes,
    flows,
) -> tuple[pa.Table, float]:
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
        ``from_lng``, ``to_lat``, ``to_lng``, ``total_flow``, one row per
        edge touched by at least one route. ``dropped_flow`` is the summed
        flow of OD pairs that were unsnapped or whose endpoints sit in
        disconnected graph components.
    """
    edge_from, edge_to, total_flow, dropped_flow = road_network._handle.route_edge_flows(
        pa.array(from_nodes, type=pa.int64()), pa.array(to_nodes, type=pa.int64()), pa.array(flows, type=pa.float64())
    )
    edge_from, edge_to, total_flow = pa.array(edge_from), pa.array(edge_to), pa.array(total_flow)

    # `node_idx` is dense/0-based and `RoadNetwork.build` already stores
    # `nodes_df` sorted by it, so position `i` is node id `i` directly.
    nodes = road_network.nodes_df
    lat_by_node = nodes.column("lat")
    lng_by_node = nodes.column("lng")

    # TODO: not actually sorted by descending total_flow (pre-existing gap,
    # out of scope for this change -- unrelated to the pandas removal).
    edges_df = pa.table(
        {
            "edge_from": edge_from,
            "edge_to": edge_to,
            "from_lat": pc.take(lat_by_node, edge_from),
            "from_lng": pc.take(lng_by_node, edge_from),
            "to_lat": pc.take(lat_by_node, edge_to),
            "to_lng": pc.take(lng_by_node, edge_to),
            "total_flow": total_flow,
        }
    )
    return edges_df, float(dropped_flow)
