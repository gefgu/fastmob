"""Big-scale preprocessing benchmarks against the real YJMob100K dataset.

Unlike `benchmarks/preprocessing/speed_suite.py` (synthetic Brightkite-derived
data, swept across libraries and sizes), this targets one real, large
trajectory dataset -- 100,000 users, ~111.5M rows -- to prove new Rust-backed
preprocessing kernels are both correct and fast at real scale, not just on a
toy fixture. See `benchmarks/shared/yjmob.py` for how the dataset is sourced.

Requires FASTMOB_YJMOB_DATA_PATH to point at a local YJMob100K parquet; skips
cleanly (exit 0) when unset, so CI (which won't have this file) doesn't fail.

Usage:
    export FASTMOB_YJMOB_DATA_PATH=/path/to/yjmob_wgs84_simple.parquet
    python benchmarks/preprocessing_yjmob_large_scale.py --benchmark h3 od --n-users 1000 10000 100000
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


def benchmark_h3(data_path: Path, n_users: int, backend: str, iterations: int) -> dict:
    from fastmob.preprocessing import latlng_to_h3

    df = load_yjmob(data_path, n_users=None if n_users >= 100_000 else n_users)
    if backend == "pandas":
        df = df.to_pandas()

    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = latlng_to_h3(df, resolution=H3_RESOLUTION)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    n_rows = len(df)
    stats = summarize_times(times)
    rows_per_second = (
        n_rows / stats["minimum_seconds"] if stats["minimum_seconds"] else None
    )
    return {
        "benchmark": "latlng_to_h3",
        "backend": backend,
        "n_users": n_users,
        "n_rows": n_rows,
        "size_label": size_label(n_rows),
        **stats,
        "rows_per_second": rows_per_second,
        "sample_output_rows": len(result),
    }


def benchmark_od(data_path: Path, n_users: int, backend: str, iterations: int) -> dict:
    from fastmob.preprocessing import trajectory_to_od

    df = load_yjmob(data_path, n_users=None if n_users >= 100_000 else n_users)
    if backend == "pandas":
        df = df.to_pandas()

    times: list[float] = []
    n_trips = 0
    for _ in range(iterations):
        start = time.perf_counter()
        result = trajectory_to_od(df, resolution=H3_RESOLUTION, uid_col="uid", datetime_col="timestamp")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        n_trips = len(result)

    n_rows = len(df)
    stats = summarize_times(times)
    rows_per_second = (
        n_rows / stats["minimum_seconds"] if stats["minimum_seconds"] else None
    )
    return {
        "benchmark": "trajectory_to_od",
        "backend": backend,
        "n_users": n_users,
        "n_rows": n_rows,
        "size_label": size_label(n_rows),
        **stats,
        "rows_per_second": rows_per_second,
        "distinct_od_pairs": n_trips,
    }


BENCHMARKS = {"h3": benchmark_h3, "od": benchmark_od}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-users", type=int, nargs="+", default=DEFAULT_N_USERS)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
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
    backends = ["pandas", "polars"] if args.backend == "both" else [args.backend]
    results = []
    for name in benchmark_names:
        func = BENCHMARKS[name]
        for n_users in sorted(args.n_users):
            for backend in backends:
                print(f"Benchmarking {name}: n_users={n_users}, backend={backend}...")
                results.append(func(data_path, n_users, backend, args.iterations))

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

    output_path = args.output or (get_default_output_dir() / "fastmob_preprocessing_yjmob_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    for r in results:
        extra = r.get("rows_per_second")
        extra_str = f"rows/sec={extra:.0f}" if extra else ""
        print(
            f"  [{r['benchmark']}] n_users={r['n_users']:>7} backend={r['backend']:<7} "
            f"rows={r['n_rows']:>10} min_seconds={r['minimum_seconds']:.4f} {extra_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
