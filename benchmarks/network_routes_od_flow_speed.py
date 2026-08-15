"""Synthetic-network speed benchmark for route geometry and OD desire lines.

Exercises the two new `fastmob.network` capabilities that have no direct
skmob equivalent -- `RoadNetwork.batch_routes` (route geometry) and
`fastmob.network.od_desire_lines` (OD-pair flow aggregation onto graph
edges) -- so their scaling isn't otherwise covered by the skmob-comparison
benchmark suites.

Builds a synthetic grid road network (no external data/network access
needed, unlike `network_yjmob_large_scale.py`'s real Overture fetch) and
times both functions over increasing query-count batches against one
prepared contraction hierarchy, amortizing CH-prep cost the same way real
usage would.

Usage:
    python benchmarks/network_routes_od_flow_speed.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_env import detect_cpu_info, get_default_output_dir
from benchmarks.utils import write_json

GRID_SIZE = 100  # 100x100 nodes = 10,000 nodes, ~19,800 directed edges
QUERY_COUNTS = [1_000, 10_000, 100_000]
MAX_WAYPOINTS = 50


def _build_grid_network(grid_size: int):
    import numpy as np
    import pandas as pd
    from fastmob.network import RoadNetwork

    node_idx = np.arange(grid_size * grid_size, dtype=np.int64)
    row = node_idx // grid_size
    col = node_idx % grid_size
    # ~100m spacing (0.0009 deg latitude, longitude scaled by the same
    # spacing at the equator for a roughly-square grid).
    lat = row.astype(np.float64) * 0.0009
    lng = col.astype(np.float64) * 0.0009
    nodes_df = pd.DataFrame({"node_idx": node_idx, "lat": lat, "lng": lng})

    def node_id(r, c):
        return r * grid_size + c

    from_nodes: list[int] = []
    to_nodes: list[int] = []
    for r in range(grid_size):
        for c in range(grid_size):
            if c + 1 < grid_size:
                from_nodes += [node_id(r, c), node_id(r, c + 1)]
                to_nodes += [node_id(r, c + 1), node_id(r, c)]
            if r + 1 < grid_size:
                from_nodes += [node_id(r, c), node_id(r + 1, c)]
                to_nodes += [node_id(r + 1, c), node_id(r, c)]

    n_edges = len(from_nodes)
    edges_df = pd.DataFrame(
        {
            "from_node": np.asarray(from_nodes, dtype=np.int64),
            "to_node": np.asarray(to_nodes, dtype=np.int64),
            "length_m": np.full(n_edges, 100.0),
            "weight_ds": np.full(n_edges, 10, dtype=np.int64),
        }
    )
    return RoadNetwork.build(edges_df, nodes_df), nodes_df, edges_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--query-counts", type=int, nargs="+", default=QUERY_COUNTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import numpy as np
    from fastmob.network import od_desire_lines

    print(f"Building synthetic {args.grid_size}x{args.grid_size} grid road network ...")
    prep_start = time.perf_counter()
    network, nodes_df, edges_df = _build_grid_network(args.grid_size)
    prep_seconds = time.perf_counter() - prep_start
    n_nodes = len(nodes_df)
    print(f"  nodes={n_nodes} edges={len(edges_df)} ch_prep_seconds={prep_seconds:.4f}")

    rng = np.random.default_rng(0)
    per_count_results = []
    for n_queries in args.query_counts:
        from_nodes = rng.integers(0, n_nodes, size=n_queries, dtype=np.int64)
        to_nodes = rng.integers(0, n_nodes, size=n_queries, dtype=np.int64)
        flows = rng.uniform(1.0, 10.0, size=n_queries)

        t0 = time.perf_counter()
        _distances, connected = network.batch_distances(from_nodes, to_nodes)
        distances_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        routes = network.batch_routes(from_nodes, to_nodes, max_waypoints=MAX_WAYPOINTS)
        routes_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        edges_with_flow, dropped_flow = od_desire_lines(network, from_nodes, to_nodes, flows)
        od_flow_seconds = time.perf_counter() - t0

        result = {
            "n_queries": n_queries,
            "connected_fraction": float(connected.to_numpy(zero_copy_only=False).mean()),
            "batch_distances_seconds": distances_seconds,
            "batch_routes_seconds": routes_seconds,
            "batch_routes_rows": len(routes),
            "od_desire_lines_seconds": od_flow_seconds,
            "od_desire_lines_edges_touched": len(edges_with_flow),
            "od_desire_lines_dropped_flow": dropped_flow,
        }
        per_count_results.append(result)
        print(
            f"  n_queries={n_queries}: "
            f"batch_distances={distances_seconds:.4f}s "
            f"batch_routes={routes_seconds:.4f}s ({len(routes)} waypoint rows) "
            f"od_desire_lines={od_flow_seconds:.4f}s ({len(edges_with_flow)} edges touched)"
        )

    payload = {
        "metadata": {
            "benchmark": "network.batch_routes+od_desire_lines",
            "dataset": "synthetic_grid",
            "grid_size": args.grid_size,
            "graph_nodes": n_nodes,
            "graph_edges": len(edges_df),
            "max_waypoints": MAX_WAYPOINTS,
            "cpu_info": detect_cpu_info(),
        },
        "results": {
            "ch_prep_seconds": prep_seconds,
            "by_query_count": per_count_results,
        },
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_network_routes_od_flow_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
