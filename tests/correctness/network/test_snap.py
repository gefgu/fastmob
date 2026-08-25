"""Correctness tests for fastmob.network.snap_locations_to_graph."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fastmob.network import snap_locations_to_graph

_NODES = pd.DataFrame({"node_idx": [0, 1, 2], "lat": [0.0, 1.0, 2.0], "lng": [0.0, 0.0, 0.0]})


def test_snap_finds_nearest_node():
    points = pd.DataFrame({"lat": [0.05, 0.95, 2.02], "lng": [0.0, 0.0, 0.0]})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=50_000.0)
    assert result.to_pylist() == [0, 1, 2]


def test_snap_reports_unsnapped_beyond_max_distance():
    points = pd.DataFrame({"lat": [0.001], "lng": [0.0]})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=1.0)
    assert result[0].as_py() == -1


def test_snap_empty_nodes_returns_all_unsnapped():
    points = pd.DataFrame({"lat": [0.0, 1.0], "lng": [0.0, 0.0]})
    empty_nodes = pd.DataFrame({"node_idx": [], "lat": [], "lng": []})
    result = snap_locations_to_graph(points, empty_nodes, max_distance_m=100.0)
    np.testing.assert_array_equal(result, [-1, -1])


def test_snap_empty_points_returns_empty():
    points = pd.DataFrame({"lat": [], "lng": []})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=100.0)
    assert len(result) == 0


def test_snap_custom_column_names():
    points = pd.DataFrame({"latitude": [0.05], "longitude": [0.0]})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=50_000.0, lat_col="latitude", lng_col="longitude")
    assert result[0].as_py() == 0


def test_snap_uses_great_circle_distance_across_antimeridian():
    nodes = pd.DataFrame({"node_idx": [10, 20], "lat": [0.0, 0.0], "lng": [179.9, -170.0]})
    points = pd.DataFrame({"lat": [0.0], "lng": [-179.95]})
    result = snap_locations_to_graph(points, nodes, max_distance_m=50_000.0)
    assert result[0].as_py() == 10
