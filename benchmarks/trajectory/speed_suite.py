"""Standalone fastmob.trajectory speed benchmark suite.

No comparable cross-library API exists for trajectory interpolation or
trajectory-pair distance metrics (see `FASTMOB_API_COMPARISION.md`'s gap
analysis in the sibling `fastmob_benchmarks` repo), so unlike
`benchmarks/individual|preprocessing|privacy/speed_suite.py`, this suite
benchmarks fastmob only -- there is no `--library skmob`/`movingpandas`/
`ptrail` mode.

Run from the repository root, for example:

    python benchmarks/trajectory/speed_suite.py --backend both --sizes 1000,10000
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.utils import nonnegative_float, positive_int, run_timed_call, size_label, summarize_times, write_json

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402

DEFAULT_SIZES = [1_000, 10_000, 100_000]
INTERPOLATE_METHODS = ("linear", "cubic_spline", "kinematic", "random_walk")
DISTANCE_METHODS = ("dtw", "frechet", "hausdorff", "lcss")


def make_synthetic_trajectory_pandas(rows: int, num_users: int = 20) -> Any:
    """Synthetic multi-user trajectory: `num_users` users, each with a slow
    drift plus small noise sampled every 10 minutes -- some consecutive gaps
    exceed the default 1h `sampling_rate_s` used below, so `interpolate()`
    has real gap-filling work to do at every size.
    """
    import numpy as np
    import pandas as pd

    per_user = max(rows // num_users, 2)
    total = per_user * num_users
    rng = np.random.default_rng(0)
    uid = np.repeat(np.arange(num_users), per_user)
    step_minutes = np.tile(np.arange(per_user), num_users) * 10
    datetime_col = pd.Timestamp("2020-01-01") + pd.to_timedelta(step_minutes, unit="m")
    lat = 40.0 + 0.001 * np.tile(np.arange(per_user), num_users) + rng.normal(scale=0.0005, size=total)
    lng = -74.0 + 0.001 * np.tile(np.arange(per_user), num_users) + rng.normal(scale=0.0005, size=total)
    return pd.DataFrame({"uid": uid, "datetime": datetime_col, "lat": lat, "lng": lng})


def make_synthetic_trajectory_polars(rows: int, num_users: int = 20) -> Any:
    import polars as pl

    return pl.from_pandas(make_synthetic_trajectory_pandas(rows, num_users))


def make_distance_sequence_pandas(rows: int) -> Any:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    lat = 40.0 + 0.001 * np.arange(rows) + rng.normal(scale=0.0005, size=rows)
    lng = -74.0 + 0.001 * np.arange(rows) + rng.normal(scale=0.0005, size=rows)
    datetime_col = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(rows) * 10, unit="m")
    return pd.DataFrame({"datetime": datetime_col, "lat": lat, "lng": lng})


def benchmark_interpolate(df: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import interpolate

    results = {}
    for method in INTERPOLATE_METHODS:
        print(f"  interpolate(method={method})")
        results[method] = run_timed_call(
            interpolate,
            lambda df=df: df,
            {"method": method, "sampling_rate_s": 3600.0},
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    return results


def benchmark_interpolate_at(df: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import interpolate_at

    print("  interpolate_at")
    return run_timed_call(
        interpolate_at,
        lambda df=df: df,
        {"at": "2020-01-05 00:00:00", "method": "linear"},
        iterations=iterations,
        sleep_seconds=sleep_seconds,
    )


def benchmark_distance(seq_a: Any, seq_b: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    """`trajectory_distance` takes two positional trajectories, so it can't
    go through `run_timed_call`'s single-input-argument shape -- this is a
    minimal from-scratch timing loop reusing `summarize_times` for the same
    result shape the other benchmarks return.
    """
    from fastmob.trajectory import trajectory_distance

    results = {}
    for method in DISTANCE_METHODS:
        print(f"  trajectory_distance(method={method})")
        print("    Warming up...")
        trajectory_distance(seq_a, seq_b, method=method)

        times: list[float] = []
        for i in range(iterations):
            if sleep_seconds:
                time.sleep(sleep_seconds)
            start = time.perf_counter()
            trajectory_distance(seq_a, seq_b, method=method)
            duration = time.perf_counter() - start
            times.append(duration)
            print(f"    Round {i + 1}: {duration:.4f} seconds")

        summary = summarize_times(times)
        print(f"    Average Time: {summary['average_seconds']:.4f} s")
        results[method] = {
            "status": "ok",
            "times_seconds": times,
            "iterations_completed": len(times),
            **summary,
        }
    return results


def run_size(size: int, backend: str, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    if backend == "polars":
        df = make_synthetic_trajectory_polars(size)
    else:
        df = make_synthetic_trajectory_pandas(size)

    # trajectory_distance compares two single-sequence trajectories, not a
    # multi-user batch, so its own size scales independently (proportional
    # to `size`, not equal to it).
    distance_size = max(size // 20, 10)
    seq_a = make_distance_sequence_pandas(distance_size)
    seq_b = make_distance_sequence_pandas(distance_size)
    if backend == "polars":
        import polars as pl

        seq_a = pl.from_pandas(seq_a)
        seq_b = pl.from_pandas(seq_b)

    print(f"\n{size_label(size)} rows, backend={backend}")
    return {
        "rows": size,
        "backend": backend,
        "interpolate": benchmark_interpolate(df, iterations, sleep_seconds),
        "interpolate_at": benchmark_interpolate_at(df, iterations, sleep_seconds),
        "distance_sequence_rows": distance_size,
        "trajectory_distance": benchmark_distance(seq_a, seq_b, iterations, sleep_seconds),
    }


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "trajectory",
        "library": "fastmob",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sizes": args.sizes,
        "backend": args.backend,
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
    }


def concrete_backends(backend: str) -> tuple[str, ...]:
    return ("pandas", "polars") if backend == "both" else (backend,)


def parse_sizes(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone fastmob.trajectory speed benchmarks.")
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    args.sizes = parse_sizes(args.sizes)
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metadata": build_metadata(args),
        "results": [
            run_size(size, backend, args.iterations, args.sleep_seconds)
            for backend in concrete_backends(args.backend)
            for size in args.sizes
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    output_path = Path(args.output_dir) / "fastmob_trajectory_speed.json"
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
