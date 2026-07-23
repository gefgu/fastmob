"""Correctness tests for fastmob.network.snap_locations_to_graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed (fastmob[ai])")

from fastmob.network import snap_locations_to_graph  # noqa: E402

_NODES = pd.DataFrame({"node_idx": [0, 1, 2], "lat": [0.0, 1.0, 2.0], "lng": [0.0, 0.0, 0.0]})


def test_snap_finds_nearest_node():
    points = pd.DataFrame({"lat": [0.05, 0.95, 2.02], "lng": [0.0, 0.0, 0.0]})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=50_000.0)
    assert list(result) == [0, 1, 2]


def test_snap_reports_unsnapped_beyond_max_distance():
    points = pd.DataFrame({"lat": [0.001], "lng": [0.0]})
    result = snap_locations_to_graph(points, _NODES, max_distance_m=1.0)
    assert result[0] == -1


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
    assert result[0] == 0
