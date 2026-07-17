"""Big-scale fitting benchmarks against the real YJMob100K dataset.

Two benchmarks, both against real YJMob100K-derived data:

- `daily_lognormal`: times `daily_location_lognormal_fit`'s per-user-per-day
  aggregation step (the only part of the fit with real data-scale cost --
  the lognormal parameters themselves are computed from the aggregated
  per-day counts, a small array regardless of input size). Locations are
  derived via `fastmob.preprocessing.latlng_to_h3` (item 1) since YJMob100K
  ships raw pings, not pre-tessellated location IDs.
- `truncated_powerlaw`: times `fit_values_to_truncated_powerlaw`'s
  `method="scipy"` vs `method="grid"` (item 5) on the histogram derived from
  YJMob100K's full jump-length distribution. Both methods consume the same
  small aggregated histogram regardless of how many raw rows built it, so
  the interesting number here is fit time, not data-loading time.

Requires FASTMOB_YJMOB_DATA_PATH; skips cleanly (exit 0) when unset. See
`benchmarks/shared/yjmob.py`.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/fitting_yjmob_large_scale.py --benchmark daily_lognormal truncated_powerlaw
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
        "benchmark": "daily_lognormal",
        "n_users": n_users,
        "n_rows": n_rows,
        "size_label": size_label(n_rows),
        **stats,
        "rows_per_second": rows_per_second,
        "n_distinct_daily_counts": len(x_points),
        "mu": mu,
        "sigma": sigma,
    }


def benchmark_truncated_powerlaw(data_path: Path, n_users: int, iterations: int) -> dict:
    from fastmob.measures.fitting import fit_values_to_truncated_powerlaw
    from fastmob.measures.individual import jump_lengths

    df = load_yjmob(data_path, n_users=None if n_users >= 100_000 else n_users).to_pandas()
    jumps = jump_lengths(df, merge=True, uid_col="uid", datetime_col="timestamp", lat_col="lat", lng_col="lon")
    jumps = jumps[jumps > 0]
    n_rows = len(df)

    result = {"benchmark": "truncated_powerlaw", "n_users": n_users, "n_rows": n_rows, "size_label": size_label(n_rows)}
    for method in ("scipy", "grid"):
        times: list[float] = []
        popt = None
        for _ in range(iterations):
            start = time.perf_counter()
            popt, _x_data, _y_data = fit_values_to_truncated_powerlaw(jumps, bins=100, method=method)
            times.append(time.perf_counter() - start)
        stats = summarize_times(times)
        result[f"{method}_minimum_seconds"] = stats["minimum_seconds"]
        result[f"{method}_average_seconds"] = stats["average_seconds"]
        result[f"{method}_popt"] = [float(v) for v in popt]
    return result


BENCHMARKS = {
    "daily_lognormal": benchmark_daily_lognormal,
    "truncated_powerlaw": benchmark_truncated_powerlaw,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-users", type=int, nargs="+", default=DEFAULT_N_USERS)
    parser.add_argument("--benchmark", choices=list(BENCHMARKS) + ["all"], nargs="+", default=["all"])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_path = args.data_path or yjmob_data_path()
    if data_path is None:
        print(skip_reason())
        return 0

    benchmark_names = list(BENCHMARKS) if "all" in args.benchmark else args.benchmark
    results = []
    for name in benchmark_names:
        func = BENCHMARKS[name]
        for n_users in sorted(args.n_users):
            print(f"Benchmarking {name}: n_users={n_users}...")
            results.append(func(data_path, n_users, args.iterations))

    payload = {
        "metadata": {
            "benchmarks": benchmark_names,
            "dataset": "yjmob100k",
            "h3_resolution": H3_RESOLUTION,
            "data_path": str(data_path),
            "cpu_info": detect_cpu_info(),
        },
        "results": results,
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_fitting_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        if r["benchmark"] == "daily_lognormal":
            print(
                f"  [daily_lognormal] n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
                f"min_seconds={r['minimum_seconds']:.4f} mu={r['mu']:.4f} sigma={r['sigma']:.4f}"
            )
        else:
            print(
                f"  [truncated_powerlaw] n_users={r['n_users']:>7} rows={r['n_rows']:>10} "
                f"scipy={r['scipy_minimum_seconds']:.4f}s grid={r['grid_minimum_seconds']:.4f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
