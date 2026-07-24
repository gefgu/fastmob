from __future__ import annotations

import numpy as np
import pandas as pd


class RoadNetwork:
    """A road (or rail) network prepared once (contraction hierarchy) and
    reused for many point-to-point physical-distance queries.

    Bundles the snap-target nodes together with the routing handle (unlike
    a split ``handle`` + separate ``nodes_df`` pair), since a caller always
    needs both to go from raw lat/lng to a routed distance.
    """

    def __init__(self, nodes_df: pd.DataFrame, handle: object) -> None:
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
            / `fetch_rail_network`); pandas or Polars.
        nodes_df:
            Columns ``node_idx``, ``lat``, ``lng`` -- used later for snapping
            via :func:`fastmob.network.snap.snap_locations_to_graph` (pandas-typed,
            so normalized to pandas here regardless of input backend); pandas
            or Polars.
        """
        from fastmob._core import RoadNetworkHandle

        nodes_pd = nodes_df if isinstance(nodes_df, pd.DataFrame) else nodes_df.to_pandas()
        # `node_idx` is the dense 0-based id `from_node`/`to_node` reference;
        # sort defensively so `node_lat[i]`/`node_lng[i]` line up with node id
        # `i` regardless of the input row order.
        nodes_sorted = nodes_pd.sort_values("node_idx")
        handle = RoadNetworkHandle(
            np.asarray(edges_df["from_node"]).astype(np.int64),
            np.asarray(edges_df["to_node"]).astype(np.int64),
            np.asarray(edges_df["weight_ds"]).astype(np.int64),
            np.asarray(edges_df["length_m"]).astype(np.float64),
            np.asarray(nodes_sorted["lat"]).astype(np.float64),
            np.asarray(nodes_sorted["lng"]).astype(np.float64),
        )
        return cls(nodes_pd, handle)

    def batch_distances(self, from_nodes: np.ndarray, to_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch physical-distance (metres) query for `(from_node, to_node)` pairs.

        Returns ``(distances_m, connected)``, ``connected`` as a bool array;
        `False` for negative/unsnapped node ids or a disconnected graph
        component (fall back to straight-line Haversine in that case).
        """
        distances_m, connected = self._handle.batch_distances(from_nodes.astype(np.int64), to_nodes.astype(np.int64))
        return np.asarray(distances_m, dtype=np.float64), np.asarray(connected, dtype=bool)

    def batch_routes(self, from_nodes: np.ndarray, to_nodes: np.ndarray, max_waypoints: int = 50) -> pd.DataFrame:
        """Batch route-geometry query for `(from_node, to_node)` pairs.

        Returns a flat pandas DataFrame with one row per waypoint: columns
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
        return pd.DataFrame(
            {
                "query_id": query_id,
                "lat": np.asarray(lats, dtype=np.float64),
                "lng": np.asarray(lngs, dtype=np.float64),
                "cum_weight_ds": np.asarray(cum_weight_ds, dtype=np.int64),
            }
        )
