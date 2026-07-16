"""Big-scale fitting benchmarks against the real YJMob100K dataset.

Times `daily_location_lognormal_fit`'s per-user-per-day aggregation step (the
only part of the fit with real data-scale cost -- the lognormal parameters
themselves are computed from the aggregated per-day counts, a small array
regardless of input size). Locations are derived via
`fastmob.preprocessing.latlng_to_h3` (item 1) since YJMob100K ships raw
pings, not pre-tessellated location IDs.

Requires FASTMOB_YJMOB_DATA_PATH; skips cleanly (exit 0) when unset. See
`benchmarks/shared/yjmob.py`.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/fitting_yjmob_large_scale.py --n-users 1000 10000 100000
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
from benchmarks.utils import size_label, summarize_times, write_json  # noqa: E402

DEFAULT_N_USERS = [1_000, 10_000, 100_000]
H3_RESOLUTION = 9


def benchmark_daily_lognormal(data_path: Path, n_users: int, iterations: int) -> dict:
    from fastmob.measures.fitting import daily_location_lognormal_fit
    from fastmob.preprocessing import latlng_to_h3

    df = load_yjmob(data_path, n_users=None if n_users >= 100_000 else n_users)
    visits = latlng_to_h3(df, resolution=H3_RESOLUTION, output_col="location_id")
    n_rows = len(visits)

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        x_points, y_points, mu, sigma = daily_location_lognormal_fit(
            visits, user_id_col="uid", location_id_col="location_id", timestamp_col="timestamp"
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    stats = summarize_times(times)
    rows_per_second = n_rows / stats["minimum_seconds"] if stats["minimum_seconds"] else None
    return {
        "n_users": n_users,
        "n_rows": n_rows,
        "size_label": size_label(n_rows),
        **stats,
        "rows_per_second": rows_per_second,
        "n_distinct_daily_counts": len(x_points),
        "mu": mu,
        "sigma": sigma,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-users", type=int, nargs="+", default=DEFAULT_N_USERS)
    parser.add_argument("--iterations", type=int, default=3)
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
        print(f"Benchmarking daily_location_lognormal_fit: n_users={n_users}...")
        results.append(benchmark_daily_lognormal(data_path, n_users, args.iterations))

    payload = {
        "metadata": {
            "benchmark": "fitting.daily_location_lognormal_fit",
            "dataset": "yjmob100k",
            "h3_resolution": H3_RESOLUTION,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_daily_lognormal_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        print(
            f"  n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
            f"min_seconds={r['minimum_seconds']:.4f} mu={r['mu']:.4f} sigma={r['sigma']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
