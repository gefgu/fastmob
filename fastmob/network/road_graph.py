from __future__ import annotations

import narwhals as nw
import numpy as np
import pyarrow as pa


class RoadNetwork:
    """A road (or rail) network prepared once (contraction hierarchy) and
    reused for many point-to-point physical-distance queries.

    Bundles the snap-target nodes together with the routing handle (unlike
    a split ``handle`` + separate ``nodes_df`` pair), since a caller always
    needs both to go from raw lat/lng to a routed distance.
    """

    def __init__(self, nodes_df: pa.Table, handle: object) -> None:
        self.nodes_df = nodes_df
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
            edges.get_column("from_node").to_numpy().astype(np.int64, copy=False),
            edges.get_column("to_node").to_numpy().astype(np.int64, copy=False),
            edges.get_column("weight_ds").to_numpy().astype(np.int64, copy=False),
            edges.get_column("length_m").to_numpy().astype(np.float64, copy=False),
            nodes_sorted.get_column("lat").to_numpy().astype(np.float64, copy=False),
            nodes_sorted.get_column("lng").to_numpy().astype(np.float64, copy=False),
        )
        return cls(nodes_sorted.to_arrow(), handle)

    def batch_distances(self, from_nodes: np.ndarray, to_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch physical-distance (metres) query for `(from_node, to_node)` pairs.

        Returns ``(distances_m, connected)``, ``connected`` as a bool array;
        `False` for negative/unsnapped node ids or a disconnected graph
        component (fall back to straight-line Haversine in that case).
        """
        distances_m, connected = self._handle.batch_distances(from_nodes.astype(np.int64), to_nodes.astype(np.int64))
        return np.asarray(distances_m, dtype=np.float64), np.asarray(connected, dtype=bool)

    def batch_routes(self, from_nodes: np.ndarray, to_nodes: np.ndarray, max_waypoints: int = 50) -> pa.Table:
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
            from_nodes.astype(np.int64), to_nodes.astype(np.int64), max_waypoints
        )
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        counts = ends - starts
        query_id = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        return pa.table(
            {
                "query_id": query_id,
                "lat": np.asarray(lats, dtype=np.float64),
                "lng": np.asarray(lngs, dtype=np.float64),
                "cum_weight_ds": np.asarray(cum_weight_ds, dtype=np.int64),
            }
        )
