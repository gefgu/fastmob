"""Populate the TransBigData reference cache used by cached comparison tests.

Runs the real ``transbigdata.traj_mapmatch`` once against a small synthetic
road network built as an osmnx-style ``networkx.MultiDiGraph`` -- the exact
input shape ``traj_mapmatch`` requires ("a networkx multidigraph, created by
osmnx"). ``osmnx``/``networkx`` are only used here, inside this comparison
harness; fastmob's own code never imports them (see CLAUDE.md).

Run inside .venv-transbigdata (a dedicated venv: installing transbigdata's
extras into an already-populated .venv can silently downgrade unrelated
packages like numpy via a full dependency re-resolution -- see
pyproject.toml's dev-transbigdata comment):
    bash scripts/setup_env.sh --venv .venv-transbigdata --transbigdata
    .venv-transbigdata/bin/python tests/populate_transbigdata_cache.py

Or via the shell wrapper:
    bash scripts/populate_transbigdata_cache.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import osmnx as ox
import pandas as pd
import transbigdata as tbd
from fastmob.network._util import haversine_m_batch
from shapely.geometry import LineString

from tests.shared.transbigdata_cache import _REFERENCE_DIR


def _build_toy_grid_graph() -> tuple[nx.MultiDiGraph, dict[int, tuple[float, float]]]:
    """A small 4-node chain-with-a-branch network, hand-built in osmnx's
    MultiDiGraph shape (node `x`/`y` = lon/lat, edge `geometry` = LineString)
    -- small enough to hand-verify, but with enough branching that map
    matching isn't trivially "nearest of 2 nodes"."""
    nodes = {
        0: (0.0, 0.0),
        1: (0.001, 0.0),
        2: (0.002, 0.0),
        3: (0.002, 0.001),
    }
    edge_pairs = [(0, 1), (1, 2), (2, 3)]

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"
    for node_id, (lng, lat) in nodes.items():
        graph.add_node(node_id, x=lng, y=lat)
    for u, v in edge_pairs:
        lng1, lat1 = nodes[u]
        lng2, lat2 = nodes[v]
        graph.add_edge(u, v, geometry=LineString([(lng1, lat1), (lng2, lat2)]), length=1.0, osmid=0)
        graph.add_edge(v, u, geometry=LineString([(lng2, lat2), (lng1, lat1)]), length=1.0, osmid=0)
    return graph, nodes


def _populate_toy_grid() -> None:
    out_dir = _REFERENCE_DIR / "toy_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, nodes = _build_toy_grid_graph()

    node_ids = sorted(nodes)
    nodes_df = pd.DataFrame(
        {
            "node_idx": node_ids,
            "lat": [nodes[n][1] for n in node_ids],
            "lng": [nodes[n][0] for n in node_ids],
        }
    )

    edge_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    from_lat = [nodes[u][1] for u, _ in edge_pairs]
    from_lng = [nodes[u][0] for u, _ in edge_pairs]
    to_lat = [nodes[v][1] for _, v in edge_pairs]
    to_lng = [nodes[v][0] for _, v in edge_pairs]
    length_m = haversine_m_batch(
        pd.Series(from_lat).to_numpy(),
        pd.Series(from_lng).to_numpy(),
        pd.Series(to_lat).to_numpy(),
        pd.Series(to_lng).to_numpy(),
    )
    edges_df = pd.DataFrame(
        {
            "from_node": [u for u, _ in edge_pairs],
            "to_node": [v for _, v in edge_pairs],
            "length_m": length_m,
            # arbitrary but proportional travel-time weight (deciseconds);
            # map matching doesn't depend on routing, only on node distance.
            "weight_ds": (length_m / 10.0).round().astype(int).clip(1, None),
        }
    )

    # Points near the network, but not exactly on any node -- one near the
    # 0-1 edge, one near the branch at node 3.
    traj_df = pd.DataFrame({"lat": [0.00005, 0.0007], "lng": [0.0005, 0.0021]})

    print("    nodes.parquet / edges.parquet / input.parquet")
    nodes_df.to_parquet(out_dir / "nodes.parquet", index=False)
    edges_df.to_parquet(out_dir / "edges.parquet", index=False)
    traj_df.to_parquet(out_dir / "input.parquet", index=False)

    print("    running transbigdata.traj_mapmatch ...")
    tbd_input = traj_df.rename(columns={"lng": "lon"})[["lon", "lat"]]
    matched = tbd.traj_mapmatch(tbd_input, graph, col=["lon", "lat"])
    matched_df = pd.DataFrame(
        {
            "lat": matched["lat"].to_numpy(dtype=float),
            "lng": matched["lon"].to_numpy(dtype=float),
            "dist_m": matched["dist"].to_numpy(dtype=float),
        }
    )
    print("    matched.parquet")
    matched_df.to_parquet(out_dir / "matched.parquet", index=False)

    meta = {"transbigdata_version": tbd.__version__, "osmnx_version": ox.__version__}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"    -> {out_dir}")


def main() -> None:
    print("\n==> toy_grid")
    _populate_toy_grid()
    print("\nAll done. Commit tests/shared/transbigdata_reference/ to git to track the snapshot.")


if __name__ == "__main__":
    main()
