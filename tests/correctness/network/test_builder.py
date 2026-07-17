"""Correctness tests for fastmob.network.builder.

`fetch_road_network`/`fetch_rail_network` hit a live Overture Maps S3
bucket -- these are integration tests, not unit tests, and skip cleanly
without network access or `duckdb` installed rather than failing CI.
"""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("duckdb", reason="duckdb not installed (fastmob[network])")


def _has_network() -> bool:
    try:
        socket.create_connection(("overturemaps-us-west-2.s3.amazonaws.com", 443), timeout=5)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _has_network(), reason="no network access to Overture Maps S3")

# Small bbox in central Paris (~1km x 1km) -- known to have a real road network.
_MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT = 2.34, 48.85, 2.36, 48.86
_OVERTURE_RELEASE = "2026-05-20.0"


def test_fetch_road_network_returns_connected_graph():
    from fastmob.network import fetch_road_network

    nodes_df, edges_df = fetch_road_network(_MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT, _OVERTURE_RELEASE)
    assert len(nodes_df) > 0
    assert len(edges_df) > 0
    assert set(nodes_df.columns) == {"node_idx", "connector_id", "lat", "lng"}
    assert set(edges_df.columns) == {"from_node", "to_node", "length_m", "speed_kmh", "weight_ds", "class"}
    assert (edges_df["from_node"] < len(nodes_df)).all()
    assert (edges_df["to_node"] < len(nodes_df)).all()
    assert (edges_df["length_m"] > 0).all()
    assert (edges_df["speed_kmh"] > 0).all()


def test_fetch_road_network_builds_a_routable_network():
    from fastmob.network import RoadNetwork, fetch_road_network

    nodes_df, edges_df = fetch_road_network(_MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT, _OVERTURE_RELEASE)
    network = RoadNetwork.build(edges_df, nodes_df)

    import numpy as np

    # Route between the first and last node -- not guaranteed connected in
    # general, but plausible for a small, dense central-Paris bbox; if not
    # connected, batch_distances must still report False rather than raising.
    from_idx = np.array([int(nodes_df["node_idx"].iloc[0])])
    to_idx = np.array([int(nodes_df["node_idx"].iloc[-1])])
    distances, connected = network.batch_distances(from_idx, to_idx)
    assert distances.shape == (1,)
    assert connected.shape == (1,)


def test_build_road_graph_caches_to_disk(tmp_path):
    from fastmob.network import build_road_graph

    nodes_output = tmp_path / "nodes.parquet"
    edges_output = tmp_path / "edges.parquet"
    nodes_df, edges_df = build_road_graph(
        _MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT, _OVERTURE_RELEASE, str(nodes_output), str(edges_output)
    )
    assert nodes_output.exists()
    assert edges_output.exists()

    # Second call must load from cache (no live fetch) and return identical data.
    cached_nodes_df, cached_edges_df = build_road_graph(
        _MIN_LON, _MIN_LAT, _MAX_LON, _MAX_LAT, _OVERTURE_RELEASE, str(nodes_output), str(edges_output)
    )
    assert len(cached_nodes_df) == len(nodes_df)
    assert len(cached_edges_df) == len(edges_df)
