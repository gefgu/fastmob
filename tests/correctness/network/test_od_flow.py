"""Correctness tests for fastmob.network.od_flow.od_desire_lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.network import RoadNetwork
from fastmob.network.od_flow import od_desire_lines

# Same 4-node chain fixture as test_road_graph.py.
_NODES = pd.DataFrame(
    {
        "node_idx": [0, 1, 2, 3],
        "lat": [0.0, 0.0009, 0.0018, 0.0027],
        "lng": [0.0, 0.0, 0.0, 0.0],
    }
)
_EDGES = pd.DataFrame(
    {
        "from_node": [0, 1, 2],
        "to_node": [1, 2, 3],
        "length_m": [100.0, 100.0, 100.0],
        "weight_ds": [10, 10, 10],
    }
)


@pytest.fixture()
def network() -> RoadNetwork:
    return RoadNetwork.build(_EDGES, _NODES)


def test_od_desire_lines_accumulates_shared_edges(network):
    # (0,3) and (1,3) share the (1,2) and (2,3) edges.
    edges_df, dropped = od_desire_lines(
        network,
        np.array([0, 1]),
        np.array([3, 3]),
        np.array([5.0, 3.0]),
    )
    assert dropped == 0.0
    rows = edges_df.to_pylist()
    by_pair = {(row["edge_from"], row["edge_to"]): row["total_flow"] for row in rows}
    assert by_pair[(0, 1)] == pytest.approx(5.0)
    assert by_pair[(1, 2)] == pytest.approx(8.0)
    assert by_pair[(2, 3)] == pytest.approx(8.0)


def test_od_desire_lines_includes_endpoint_coordinates(network):
    edges_df, _ = od_desire_lines(network, np.array([0]), np.array([1]), np.array([1.0]))
    row = edges_df.to_pylist()[0]
    assert row["from_lat"] == pytest.approx(0.0)
    assert row["from_lng"] == pytest.approx(0.0)
    assert row["to_lat"] == pytest.approx(0.0009)
    assert row["to_lng"] == pytest.approx(0.0)


def test_od_desire_lines_unsnapped_and_disconnected_flow_is_dropped():
    nodes = pd.DataFrame({"node_idx": [0, 1, 2, 3], "lat": [0.0, 0.1, 10.0, 10.1], "lng": [0.0, 0.0, 0.0, 0.0]})
    edges = pd.DataFrame({"from_node": [0, 2], "to_node": [1, 3], "length_m": [100.0, 100.0], "weight_ds": [10, 10]})
    disconnected_network = RoadNetwork.build(edges, nodes)

    edges_df, dropped = od_desire_lines(
        disconnected_network,
        np.array([0, -1]),
        np.array([3, 2]),
        np.array([2.0, 4.0]),
    )
    assert edges_df.num_rows == 0
    assert dropped == pytest.approx(6.0)


def test_od_desire_lines_output_is_pyarrow_native(network):
    """od_desire_lines must not depend on pandas: its return type is a
    pyarrow.Table regardless of the RoadNetwork's own input backend."""
    import pyarrow as pa

    edges_df, _ = od_desire_lines(network, np.array([0]), np.array([1]), np.array([1.0]))
    assert isinstance(edges_df, pa.Table)
