from __future__ import annotations

import pyarrow as pa

from fastmob.network import RoadNetwork, match_trajectory


def _network() -> RoadNetwork:
    nodes = pa.table({"node_idx": [0, 1, 2], "lat": [0.0, 0.0, 0.0], "lng": [0.0, 0.001, 0.002]})
    edges = pa.table(
        {
            "from_node": [0, 1, 1, 2],
            "to_node": [1, 0, 2, 1],
            "weight_ds": [10, 10, 10, 10],
            "length_m": [111.0, 111.0, 111.0, 111.0],
        }
    )
    return RoadNetwork.build(edges, nodes)


def test_match_trajectory_returns_arrow_rows_in_input_order():
    matched = match_trajectory(
        pa.table(
            {
                "uid": ["a", "b", "a"],
                "datetime": [0, 0, 60],
                "lat": [0.0, 0.0, 0.0],
                "lng": [0.0001, 0.001, 0.0019],
            }
        ),
        network=_network(),
    )

    assert isinstance(matched, pa.Table)
    assert matched.num_rows == 3
    assert matched.column("match_status").to_pylist() == ["gap_start", "gap_start", "matched"]
    assert all(value is not None for value in matched.column("matched_edge_from").to_pylist())
