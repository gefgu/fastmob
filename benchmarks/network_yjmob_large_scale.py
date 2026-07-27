"""Big-scale road-network distance benchmark against real Overture + YJMob100K data.

Fetches a real Overture road network for a bounding box inside YJMob100K's
extent (a full-dataset-covering bbox would mean scanning and preparing a CH
for a road network spanning the whole Chita/Nagoya region -- see the module
docstring caveat below), then snaps every YJMob100K ping that falls inside
that bbox and computes `jump_lengths_km`/`radius_of_gyration_km` over the
real road network, comparing against fastmob's existing straight-line
`jump_lengths`/`radius_of_gyration` on the same points. Reports CH-preparation
time separately from batch query time, since amortizing prep cost across
many queries is the whole design point.

**Why a sub-region, not the full YJMob100K bbox**: YJMob100K's full extent
is roughly 1 degree x 1 degree (~100km x 100km). Fetching and preparing a
contraction hierarchy for a road network at that scale is a legitimately
large one-time operation (tens of minutes to hours depending on road
density) -- exactly the kind of cost this benchmark exists to amortize away
across many queries, but not one to eat on every benchmark run. A dense
~5km x 5km sub-region (Nagoya-area, well inside the dataset's bbox) already
has 8,000+ real road segments and thousands of real pings to route, enough
to prove correctness and report honest timing at meaningful (if not
maximal) scale.

Requires FASTMOB_YJMOB_DATA_PATH and network access to Overture Maps S3;
skips cleanly (exit 0) when the dataset path is unset. The road-network
fetch itself is cached to disk after the first run.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/network_yjmob_large_scale.py
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
from benchmarks.shared.yjmob import load_yjmob, skip_reason, yjmob_data_path
from benchmarks.utils import write_json

# A dense ~5km x 5km sub-region well inside YJMob100K's bbox
# (lat 34.67-35.66, lon 136.61-137.61) -- see module docstring.
BBOX = (137.0, 35.1, 137.05, 35.15)  # (min_lon, min_lat, max_lon, max_lat)
OVERTURE_RELEASE = "2026-05-20.0"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "network_yjmob"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    data_path = args.data_path or yjmob_data_path()
    if data_path is None:
        print(skip_reason())
        return 0

    from fastmob.measures.individual import jump_lengths, radius_of_gyration
    from fastmob.measures.individual.network_distance import jump_lengths_km, radius_of_gyration_km
    from fastmob.network import RoadNetwork, build_road_graph

    min_lon, min_lat, max_lon, max_lat = BBOX
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.cache_dir / "nodes.parquet"
    edges_path = args.cache_dir / "edges.parquet"

    print(f"Fetching/loading Overture road network for bbox {BBOX} ...")
    fetch_start = time.perf_counter()
    nodes_df, edges_df = build_road_graph(
        min_lon, min_lat, max_lon, max_lat, OVERTURE_RELEASE, str(nodes_path), str(edges_path)
    )
    fetch_seconds = time.perf_counter() - fetch_start
    print(f"  nodes={len(nodes_df)} edges={len(edges_df)} fetch/load_seconds={fetch_seconds:.2f}")

    prep_start = time.perf_counter()
    network = RoadNetwork.build(edges_df, nodes_df)
    prep_seconds = time.perf_counter() - prep_start
    print(f"  contraction-hierarchy prep_seconds={prep_seconds:.4f}")

    print("Loading YJMob100K and restricting to the fetched bbox ...")
    df = load_yjmob(data_path).rename({"lon": "lng"})
    df = df.filter((df["lat"] >= min_lat) & (df["lat"] <= max_lat) & (df["lng"] >= min_lon) & (df["lng"] <= max_lon))
    print(f"  pings inside bbox: {len(df)} ({df['uid'].n_unique()} distinct users)")

    traj = df.rename({"timestamp": "datetime"}).to_pandas()

    t0 = time.perf_counter()
    straight_jumps = jump_lengths(traj, merge=True)
    straight_jump_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    network_jumps = jump_lengths_km(traj, network=network)
    network_jump_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    straight_rg = radius_of_gyration(traj)
    straight_rg_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    network_rg = radius_of_gyration_km(traj, network=network)
    network_rg_seconds = time.perf_counter() - t0

    results = {
        "n_rows": len(traj),
        "n_users": int(df["uid"].n_unique()),
        "graph_nodes": len(nodes_df),
        "graph_edges": len(edges_df),
        "road_fetch_or_load_seconds": fetch_seconds,
        "ch_prep_seconds": prep_seconds,
        "straight_line_jump_lengths_seconds": straight_jump_seconds,
        "network_jump_lengths_seconds": network_jump_seconds,
        "straight_line_radius_of_gyration_seconds": straight_rg_seconds,
        "network_radius_of_gyration_seconds": network_rg_seconds,
        "mean_straight_line_jump_km": float(straight_jumps.mean()) if len(straight_jumps) else None,
        "mean_network_jump_km": float(network_jumps.mean()) if len(network_jumps) else None,
        "mean_straight_line_radius_of_gyration_km": float(straight_rg["radius_of_gyration"].mean()),
        "mean_network_radius_of_gyration_km": float(network_rg["radius_of_gyration"].mean()),
    }

    payload = {
        "metadata": {
            "benchmark": "network.jump_lengths_km+radius_of_gyration_km",
            "dataset": "yjmob100k",
            "bbox": BBOX,
            "overture_release": OVERTURE_RELEASE,
            "cpu_info": detect_cpu_info(),
            "note": (
                "Bbox is a ~5km x 5km sub-region of YJMob100K's full ~100km x 100km "
                "extent, not the full dataset area -- see module docstring."
            ),
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_network_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for key, value in results.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
