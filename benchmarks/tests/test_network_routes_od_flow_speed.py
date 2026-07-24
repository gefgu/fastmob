from __future__ import annotations

from benchmarks import network_routes_od_flow_speed as suite


def test_build_grid_network_produces_expected_node_and_edge_counts():
    network, nodes_df, edges_df = suite._build_grid_network(3)
    assert len(nodes_df) == 9
    # 3x3 grid: 2 horizontal + 2 vertical connections per row/col, both
    # directions -- 12 undirected connections * 2 = 24 directed edges.
    assert len(edges_df) == 24
    assert network is not None


def test_main_runs_at_tiny_scale(tmp_path):
    output_path = tmp_path / "result.json"
    exit_code = suite.main(
        [
            "--grid-size",
            "3",
            "--query-counts",
            "2",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    assert output_path.exists()

    import json

    payload = json.loads(output_path.read_text())
    assert payload["metadata"]["graph_nodes"] == 9
    by_count = payload["results"]["by_query_count"]
    assert len(by_count) == 1
    assert by_count[0]["n_queries"] == 2
