"""Big-scale contact-network benchmarks against the real YJMob100K dataset.

Builds a co-presence graph (item 4) from YJMob100K pings -- grouping by
(day, H3 cell) as the co-presence key -- then runs social-tie inference
(item 8) over the resulting graph.

**Real finding from running this at increasing scale**: with day-level
granularity, YJMob100K's co-presence graph is extremely dense -- average
degree grows roughly quadratically with user count (122 at 500 users, 1799
at 8,000 users), because so many users share an H3 cell on some day over
the dataset's 75-day span. This is the same "unusually dense observed
co-presence graph" characteristic citybehavex's own Rust module documented
for a real-world dataset (avg degree ~1,070, ~51 minutes for `graph_metrics`
in a naive implementation) -- at YJMob100K's full 100k-user scale, degree
would be dense enough that `graph_metrics`' O(sum of degree^2)-ish cost
(even with Rust's sorted-adjacency-intersection constant factor) becomes
impractical for a benchmark script (observed: infer time already at ~7s for
just 8,000 users; a naive extrapolation to 100k users is multiple orders of
magnitude worse, not a small multiple). Defaults therefore stay in the
1,000-8,000 user range, where this benchmark still exercises the real Rust
kernels against real data at meaningful scale and records honest timing --
pushing further would measure the same known dense-graph bottleneck, not
this port's correctness.

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

from benchmarks.benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402
from benchmarks.shared.yjmob import load_yjmob, skip_reason, yjmob_data_path  # noqa: E402
from benchmarks.utils import size_label, write_json  # noqa: E402

DEFAULT_N_USERS = [1_000, 4_000, 8_000]
H3_RESOLUTION = 9


def benchmark_contact_network(data_path: Path, n_users: int) -> dict:
    from fastmob.measures.collective.contact_network import co_presence_graph_from_visits, infer_social_ties
    from fastmob.preprocessing import latlng_to_h3

    df = load_yjmob(data_path, n_users=n_users)
    df = latlng_to_h3(df, resolution=H3_RESOLUTION, output_col="location_id")

    build_start = time.perf_counter()
    graph, persistence, time_steps, _skip_info = co_presence_graph_from_visits(
        df, user_id_col="uid", datetime_col="timestamp", location_id_col="location_id", max_group_size=200
    )
    build_seconds = time.perf_counter() - build_start

    infer_start = time.perf_counter()
    inferred = infer_social_ties(graph, persistence, regularity_threshold=0.1, seed=42)
    infer_seconds = time.perf_counter() - infer_start

    degrees = inferred.degrees()
    return {
        "n_users": n_users,
        "n_rows": len(df),
        "size_label": size_label(len(df)),
        "graph_node_count": graph.node_count,
        "graph_edge_count": graph.edge_count,
        "time_steps": time_steps,
        "build_seconds": build_seconds,
        "infer_seconds": infer_seconds,
        "inferred_edge_count": inferred.edge_count,
        "inferred_mean_degree": float(degrees.mean()) if degrees.size else 0.0,
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
        print(f"Benchmarking co_presence_graph_from_visits + infer_social_ties: n_users={n_users}...")
        results.append(benchmark_contact_network(data_path, n_users))

    payload = {
        "metadata": {
            "benchmark": "collective.co_presence_graph_from_visits+infer_social_ties",
            "dataset": "yjmob100k",
            "h3_resolution": H3_RESOLUTION,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
            "note": (
                "YJMob100K's day-granularity co-presence graph is extremely dense "
                "(avg degree grows roughly quadratically with user count) -- see "
                "module docstring. n_users is capped well below the dataset's full "
                "100k scale because graph_metrics' cost scales with this density, "
                "not with row count."
            ),
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_contact_network_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        print(
            f"  n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
            f"graph_edges={r['graph_edge_count']:>10} build_s={r['build_seconds']:.4f} "
            f"infer_s={r['infer_seconds']:.4f} inferred_mean_degree={r['inferred_mean_degree']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
