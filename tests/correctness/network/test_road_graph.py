"""Correctness tests for fastmob.network.RoadNetwork and fastmob.measures.individual.network_distance."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fastmob.measures.individual.network_distance import jump_lengths_km, radius_of_gyration_km
from fastmob.network import RoadNetwork

# 4-node chain, ~100m apart along a line of longitude (bidirectional edges).
_NODES = pd.DataFrame(
    {
        "node_idx": [0, 1, 2, 3],
        "lat": [0.0, 0.0009, 0.0018, 0.0027],
        "lng": [0.0, 0.0, 0.0, 0.0],
    }
)
_EDGES = pd.DataFrame(
    {
        "from_node": [0, 1, 2, 1, 2, 3],
        "to_node": [1, 2, 3, 0, 1, 2],
        "length_m": [100.0] * 6,
        "weight_ds": [10] * 6,
    }
)


@pytest.fixture()
def network() -> RoadNetwork:
    return RoadNetwork.build(_EDGES, _NODES)


def _chain_traj() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]),
            "lat": [0.0, 0.0009, 0.0018],
            "lng": [0.0, 0.0, 0.0],
        }
    )


def test_jump_lengths_km_matches_chain_edge_length(network):
    result = jump_lengths_km(_chain_traj(), network=network)
    assert result == pytest.approx([0.1, 0.1])


def test_jump_lengths_km_polars_matches_pandas(network):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    traj = _chain_traj()
    pandas_result = jump_lengths_km(traj, network=network)
    polars_result = jump_lengths_km(pl.from_pandas(traj), network=network)
    assert list(pandas_result) == pytest.approx(list(polars_result))


def test_jump_lengths_km_single_point_returns_empty(network):
    traj = _chain_traj().iloc[:1]
    result = jump_lengths_km(traj, network=network)
    assert len(result) == 0


def test_jump_lengths_km_falls_back_to_haversine_when_unsnapped(network):
    # Middle point is 5 degrees away -- unsnappable at a tight max_distance.
    traj = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]),
            "lat": [0.0, 5.0, 0.0018],
            "lng": [0.0, 5.0, 0.0],
        }
    )
    result = jump_lengths_km(traj, network=network, snap_max_distance_m=50.0)
    # ~5-degree jump is hundreds of km via Haversine, not ~0.1km via network.
    assert result[0] > 100.0
    assert result[1] > 100.0


def test_radius_of_gyration_km_matches_hand_computed(network):
    result = radius_of_gyration_km(_chain_traj(), network=network)
    # centroid = middle stop (node 1); distances 100m, 0m, 100m -> RMS = sqrt(20000/3) m
    expected_km = math.sqrt((100.0**2 + 0.0**2 + 100.0**2) / 3.0) / 1000.0
    assert result["radius_of_gyration"].iloc[0] == pytest.approx(expected_km, rel=1e-6)


def test_radius_of_gyration_km_polars_matches_pandas(network):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    traj = _chain_traj()
    pandas_result = radius_of_gyration_km(traj, network=network)
    polars_result = radius_of_gyration_km(pl.from_pandas(traj), network=network).to_pandas()
    pd.testing.assert_frame_equal(
        pandas_result.reset_index(drop=True), polars_result.reset_index(drop=True), check_dtype=False
    )


def test_radius_of_gyration_km_empty_input_returns_empty(network):
    empty = pd.DataFrame({"uid": [], "datetime": pd.to_datetime([]), "lat": [], "lng": []})
    result = radius_of_gyration_km(empty, network=network)
    assert len(result) == 0
    assert list(result.columns) == ["uid", "radius_of_gyration"]


def test_radius_of_gyration_km_works_without_a_datetime_column(network):
    # radius_of_gyration doesn't need a temporal ordering (unlike jump_lengths) --
    # a frame with only uid/lat/lng must not be rejected for lacking one.
    df = pd.DataFrame({"uid": [1, 1], "lat": [0.0, 0.0018], "lng": [0.0, 0.0]})
    result = radius_of_gyration_km(df, network=network)
    assert len(result) == 1
    assert result["radius_of_gyration"].iloc[0] > 0.0


def test_road_network_build_accepts_polars_nodes_and_edges():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    nodes = pl.DataFrame({"node_idx": [0, 1], "lat": [0.0, 0.0009], "lng": [0.0, 0.0]})
    edges = pl.DataFrame({"from_node": [0, 1], "to_node": [1, 0], "length_m": [100.0, 100.0], "weight_ds": [10, 10]})
    polars_network = RoadNetwork.build(edges, nodes)

    df = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "lat": [0.0, 0.0009],
            "lng": [0.0, 0.0],
        }
    )
    result = jump_lengths_km(df, network=polars_network)
    assert result == pytest.approx([0.1])


def test_road_network_batch_distances_disconnected_reports_not_connected():
    nodes = pd.DataFrame({"node_idx": [0, 1, 2, 3], "lat": [0.0, 0.1, 10.0, 10.1], "lng": [0.0, 0.0, 0.0, 0.0]})
    edges = pd.DataFrame(
        {"from_node": [0, 2], "to_node": [1, 3], "length_m": [100.0, 100.0], "weight_ds": [10, 10]}
    )
    network = RoadNetwork.build(edges, nodes)
    import numpy as np

    distances, connected = network.batch_distances(np.array([0]), np.array([3]))
    assert not connected[0]
    assert distances[0] == 0.0
