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
    def build(cls, edges_df, nodes_df) -> "RoadNetwork":
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

        handle = RoadNetworkHandle(
            np.asarray(edges_df["from_node"]).astype(np.int64),
            np.asarray(edges_df["to_node"]).astype(np.int64),
            np.asarray(edges_df["weight_ds"]).astype(np.int64),
            np.asarray(edges_df["length_m"]).astype(np.float64),
        )
        nodes_pd = nodes_df if isinstance(nodes_df, pd.DataFrame) else nodes_df.to_pandas()
        return cls(nodes_pd, handle)

    def batch_distances(self, from_nodes: np.ndarray, to_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch physical-distance (metres) query for `(from_node, to_node)` pairs.

        Returns ``(distances_m, connected)``, ``connected`` as a bool array;
        `False` for negative/unsnapped node ids or a disconnected graph
        component (fall back to straight-line Haversine in that case).
        """
        distances_m, connected = self._handle.batch_distances(
            from_nodes.astype(np.int64), to_nodes.astype(np.int64)
        )
        return np.asarray(distances_m, dtype=np.float64), np.asarray(connected, dtype=bool)
