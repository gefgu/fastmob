"""Big-scale raw contact-network construction benchmark on YJMob100K.

Pings are converted to stay intervals and globally assigned H3 locations.
The timed operation builds raw contacts from RECAST's shared daily event
graphs; it does not estimate random thresholds or classify relationship ties.

Requires FASTMOB_YJMOB_DATA_PATH; skips cleanly (exit 0) when unset. See
`benchmarks/shared/yjmob.py`.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/collective_yjmob_large_scale.py --n-users 1000 4000 8000
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
from benchmarks.utils import size_label, write_json

DEFAULT_N_USERS = [1_000, 4_000, 8_000]
H3_RESOLUTION = 9


def benchmark_contact_network(data_path: Path, n_users: int) -> dict:
    import polars as pl
    from fastmob.core import Locations, Staypoints
    from fastmob.preprocessing import latlng_to_h3
    from fastmob.social import co_presence_graph_from_staypoints

    df = load_yjmob(data_path, n_users=n_users)
    df = latlng_to_h3(df, resolution=H3_RESOLUTION, output_col="location_id")
    df = df.sort(["uid", "timestamp"]).with_columns(
        pl.col("timestamp").shift(-1).over("uid")
        .fill_null(pl.col("timestamp") + pl.duration(minutes=30)).alias("finished_at")
    ).rename({"timestamp": "started_at"})
    locations = df.group_by("location_id").agg(
        pl.col("lat").mean().alias("center_lat"), pl.col("lon").mean().alias("center_lng")
    )
    staypoints = Staypoints(df.select("uid", "lat", "lon", "started_at", "finished_at", "location_id"))
    locations = Locations(locations, scope="global")

    build_start = time.perf_counter()
    result = co_presence_graph_from_staypoints(staypoints, locations)
    build_seconds = time.perf_counter() - build_start
    return {
        "n_users": n_users,
        "n_rows": len(df),
        "size_label": size_label(len(df)),
        "graph_node_count": result.graph.node_count,
        "graph_edge_count": result.graph.edge_count,
        "time_steps": result.time_steps,
        "build_seconds": build_seconds,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-users", type=int, nargs="+", default=DEFAULT_N_USERS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = args.data_path or yjmob_data_path()
    if data_path is None:
        print(skip_reason())
        return 0

    results = []
    for n_users in sorted(args.n_users):
        print(f"Benchmarking raw contact construction: n_users={n_users}...")
        results.append(benchmark_contact_network(data_path, n_users))

    payload = {
        "metadata": {
            "benchmark": "social.co_presence_graph_from_staypoints",
            "dataset": "yjmob100k",
            "h3_resolution": H3_RESOLUTION,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
            "location_semantics": "resolution 9 H3 cells, global Locations",
            "interval_semantics": "each ping lasts until the next ping by the same user, final ping lasts 30 minutes",
            "minimum_encounter_minutes": 5,
            "group_size_cap": None,
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_contact_network_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        print(
            f"  n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
            f"graph_edges={r['graph_edge_count']:>10} days={r['time_steps']:>4} "
            f"build_s={r['build_seconds']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
