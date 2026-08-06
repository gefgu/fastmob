from __future__ import annotations

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc


class RoadNetwork:
    """A road (or rail) network prepared once (contraction hierarchy) and
    reused for many point-to-point physical-distance queries.

    Bundles the snap-target nodes together with the routing handle (unlike
    a split ``handle`` + separate ``nodes_df`` pair), since a caller always
    needs both to go from raw lat/lng to a routed distance.
    """

    def __init__(self, nodes_df: pa.Table, edges_df: pa.Table, handle: object) -> None:
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self._handle = handle

    @classmethod
    def build(cls, edges_df, nodes_df) -> RoadNetwork:
        """Prepare a contraction hierarchy from a road/rail graph's edges.

        Parameters
        ----------
        edges_df:
            Columns ``from_node``, ``to_node``, ``weight_ds``, ``length_m``
            (as returned by :func:`fastmob.network.builder.fetch_road_network`
            / `fetch_rail_network`); any Narwhals-compatible backend.
        nodes_df:
            Columns ``node_idx``, ``lat``, ``lng`` -- used later for snapping
            via :func:`fastmob.network.snap.snap_locations_to_graph` (stored
            as a pyarrow.Table here regardless of input backend, sorted by
            ``node_idx``); any Narwhals-compatible backend.
        """
        from fastmob._core import RoadNetworkHandle

        # `node_idx` is the dense 0-based id `from_node`/`to_node` reference;
        # sort defensively so `node_lat[i]`/`node_lng[i]` line up with node id
        # `i` regardless of the input row order.
        nodes_sorted = nw.from_native(nodes_df, eager_only=True).sort("node_idx")
        edges = nw.from_native(edges_df, eager_only=True)
        handle = RoadNetworkHandle(
            edges.get_column("from_node").to_arrow(),
            edges.get_column("to_node").to_arrow(),
            edges.get_column("weight_ds").to_arrow(),
            edges.get_column("length_m").to_arrow(),
            nodes_sorted.get_column("lat").to_arrow(),
            nodes_sorted.get_column("lng").to_arrow(),
        )
        return cls(nodes_sorted.to_arrow(), edges.to_arrow(), handle)

    def batch_distances(self, from_nodes, to_nodes) -> tuple[pa.Array, pa.Array]:
        """Batch physical-distance (metres) query for `(from_node, to_node)` pairs.

        Returns ``(distances_m, connected)``, ``connected`` as a bool array;
        `False` for negative/unsnapped node ids or a disconnected graph
        component (fall back to straight-line Haversine in that case).
        """
        distances, connected = self._handle.batch_distances(
            pa.array(from_nodes, type=pa.int64()), pa.array(to_nodes, type=pa.int64())
        )
        return pa.array(distances), pa.array(connected)

    def batch_routes(self, from_nodes, to_nodes, max_waypoints: int = 50) -> pa.Table:
        """Batch route-geometry query for `(from_node, to_node)` pairs.

        Returns a flat pyarrow.Table with one row per waypoint: columns
        ``query_id`` (0-based index into `from_nodes`/`to_nodes`), ``lat``,
        ``lng``, ``cum_weight_ds`` (cumulative travel-time weight from the
        route's start), following the same flat-output + boundary convention
        used by other variable-rows-per-group measures in this library
        (rather than one Python object per query). A query with `connected
        == False` (unsnapped/disconnected) contributes zero rows; join back
        on ``query_id`` against a `connected` array from `batch_distances`
        if you need to distinguish "no route" from "route with no
        waypoints" (the latter cannot happen: every connected route has at
        least its two endpoints).

        Waypoints are decimated to at most `max_waypoints` per query
        (always keeping the first and last), so a caller wanting the full,
        undecimated node path should pass a large `max_waypoints`.
        """
        lats, lngs, cum_weight_ds, _connected, starts, ends = self._handle.batch_routes(
            pa.array(from_nodes, type=pa.int64()), pa.array(to_nodes, type=pa.int64()), max_waypoints
        )
        lats, lngs, cum_weight_ds = pa.array(lats), pa.array(lngs), pa.array(cum_weight_ds)
        starts, ends = pa.array(starts), pa.array(ends)
        counts = pc.subtract(ends, starts).to_pylist()
        query_id = pa.array([query for query, count in enumerate(counts) for _ in range(count)], type=pa.int64())
        return pa.table(
            {
                "query_id": query_id,
                "lat": lats,
                "lng": lngs,
                "cum_weight_ds": cum_weight_ds,
            }
        )
