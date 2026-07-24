"""Standalone fastmob.trajectory speed benchmark suite.

Uses the real Brightkite check-in dataset at the same 5 standard sizes
(1k/10k/100k/1M/4M rows) as `benchmarks/individual|preprocessing/speed_suite.py`.

Comparison libraries (see `FASTMOB_API_COMPARISION.md`'s gap analysis for why
these are the only two with any real analogue):
  - PTRAIL (`--library ptrail`): real `Interpolation.interpolate_position`
    for `interpolate`'s linear/cubic/kinematic methods (its own public,
    multiprocessing-based entry point -- the intended way to call it on a
    multi-user dataframe, not the bespoke per-user `Helpers.*_help` calls
    `tests/populate_ptrail_cache.py` uses for exact-value correctness
    testing). PTRAIL has no random-walk analogue benchmarked here (its own
    `ip_type='random-walk'` draws one random step per *trajectory*, not per
    gap -- not a fair apples-to-apples timing comparison).
  - MovingPandas (`--library movingpandas`): real `Trajectory.get_position_at`
    for `interpolate_at`, and `Trajectory.hausdorff_distance` for
    `trajectory_distance` (MovingPandas has no DTW/Fréchet/LCSS analogue).
    `smooth` has no MovingPandas comparison path here: its `KalmanSmootherCV`
    depends on the optional `stonesoup` package, which was not available in
    the local `.venv-movingpandas` when this suite was written -- fastmob's
    `smooth` is only benchmarked standalone (`--library fastmob`).

`trajectory_distance` compares exactly 2 single-trajectory sequences -- an
O(n*m) (or worse) operation for DTW/Fréchet/LCSS -- so it uses its own,
deliberately much smaller `--distance-sizes` axis instead of `--sizes`.

Run from the repository root, for example:

    python benchmarks/trajectory/speed_suite.py --library fastmob --backend both
    python benchmarks/trajectory/speed_suite.py --library ptrail
    python benchmarks/trajectory/speed_suite.py --library movingpandas
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.utils import (
    error_result,
    nonnegative_float,
    positive_int,
    run_timed_call,
    size_label,
    skipped_result,
    summarize_times,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402

BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
DEFAULT_DISTANCE_SIZES = [100, 500, 2_000]
SAMPLING_RATE_S = 3600.0
INTERPOLATE_METHODS = ("linear", "cubic_spline", "kinematic", "random_walk")
PTRAIL_INTERPOLATE_METHODS = ("linear", "cubic", "kinematic")
INTERPOLATE_AT_METHODS = ("linear", "nearest")
DISTANCE_METHODS = ("dtw", "frechet", "hausdorff", "lcss")
SMOOTH_METHODS = ("kalman_cv",)


class SkippedMetric(Exception):
    """Raised when a metric cannot be benchmarked in the selected library."""


# ---------------------------------------------------------------------------
# Data loading (all column names -- `user`/`check-in_time`/`latitude`/
# `longitude` -- are already in fastmob's auto-detection candidate lists, so
# fastmob calls below pass the raw Brightkite frame directly, unrenamed).
# ---------------------------------------------------------------------------


def load_brightkite_pandas(data_path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(data_path, sep="\t", header=None, names=BRIGHTKITE_COLUMNS)
    df["check-in_time"] = pd.to_datetime(df["check-in_time"], errors="coerce")
    return df


def load_brightkite_polars(data_path: Path) -> Any:
    import polars as pl

    return pl.read_csv(
        data_path,
        separator="\t",
        has_header=False,
        new_columns=BRIGHTKITE_COLUMNS,
        try_parse_dates=True,
    )


def make_distance_sequences_pandas(df: Any, size: int) -> tuple[Any, Any]:
    """The 2 users with the most check-ins, each truncated to `size` rows --
    a real (not synthetic) two-single-trajectory pair for `trajectory_distance`.
    """
    top_users = df["user"].value_counts().index[:2]
    seq_a = df[df["user"] == top_users[0]].sort_values("check-in_time", kind="mergesort").head(size)
    seq_b = df[df["user"] == top_users[1]].sort_values("check-in_time", kind="mergesort").head(size)
    return seq_a.reset_index(drop=True), seq_b.reset_index(drop=True)


def load_brightkite_ptrail(data_path: Path, size: int) -> Any:
    from ptrail.core.TrajectoryDF import PTRAILDataFrame

    df = load_brightkite_pandas(data_path).head(size).copy()
    df = df.dropna(subset=["user", "check-in_time", "latitude", "longitude"])
    if df["check-in_time"].dt.tz is not None:
        df["check-in_time"] = df["check-in_time"].dt.tz_localize(None)
    return PTRAILDataFrame(df, latitude="latitude", longitude="longitude", datetime="check-in_time", traj_id="user")


def load_brightkite_movingpandas_collection(data_path: Path, size: int) -> Any:
    import geopandas as gpd
    import movingpandas as mpd
    import pandas as pd

    df = load_brightkite_pandas(data_path).head(size).copy()
    df = df.dropna(subset=["user", "check-in_time", "latitude", "longitude"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf["check-in_time"] = pd.to_datetime(gdf["check-in_time"])
    gdf = gdf.set_index("check-in_time")
    return mpd.TrajectoryCollection(gdf, traj_id_col="user", min_length=0)


def load_brightkite_movingpandas_pair(data_path: Path, size: int) -> tuple[Any, Any]:
    import geopandas as gpd
    import movingpandas as mpd

    df = load_brightkite_pandas(data_path)
    df = df.dropna(subset=["user", "check-in_time", "latitude", "longitude"])
    top_users = df["user"].value_counts().index[:2]
    trajs = []
    for uid in top_users:
        user_df = df[df["user"] == uid].sort_values("check-in_time", kind="mergesort").head(size).copy()
        gdf = gpd.GeoDataFrame(
            user_df, geometry=gpd.points_from_xy(user_df["longitude"], user_df["latitude"]), crs="EPSG:4326"
        )
        gdf = gdf.set_index("check-in_time")
        trajs.append(mpd.Trajectory(gdf, traj_id=uid))
    return trajs[0], trajs[1]


# ---------------------------------------------------------------------------
# fastmob
# ---------------------------------------------------------------------------


def benchmark_fastmob_interpolate(df: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import interpolate

    results = {}
    for method in INTERPOLATE_METHODS:
        print(f"  interpolate(method={method})")
        try:
            results[method] = run_timed_call(
                interpolate,
                lambda df=df: df,
                {"method": method, "sampling_rate_s": SAMPLING_RATE_S},
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[method] = error_result(str(exc))
    return results


def benchmark_fastmob_interpolate_at(df: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import interpolate_at

    results = {}
    for method in INTERPOLATE_AT_METHODS:
        print(f"  interpolate_at(method={method})")
        try:
            results[method] = run_timed_call(
                interpolate_at,
                lambda df=df: df,
                {"at": "2010-01-01 00:00:00", "method": method},
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[method] = error_result(str(exc))
    return results


def benchmark_fastmob_smooth(df: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import smooth

    results = {}
    for method in SMOOTH_METHODS:
        print(f"  smooth(method={method})")
        try:
            results[method] = run_timed_call(
                smooth,
                lambda df=df: df,
                {"method": method},
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[method] = error_result(str(exc))
    return results


def benchmark_fastmob_distance(seq_a: Any, seq_b: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from fastmob.trajectory import trajectory_distance

    results = {}
    for method in DISTANCE_METHODS:
        print(f"  trajectory_distance(method={method})")
        try:
            results[method] = _run_two_input_timed_call(
                lambda a, b, method=method: trajectory_distance(a, b, method=method),
                seq_a,
                seq_b,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[method] = error_result(str(exc))
    return results


def _run_two_input_timed_call(
    func: Callable[[Any, Any], Any], a: Any, b: Any, *, iterations: int, sleep_seconds: float
) -> dict[str, Any]:
    """`run_timed_call` assumes a single-input callable; `trajectory_distance`
    and MovingPandas' `hausdorff_distance` both take 2 positional trajectory
    arguments, so this is a minimal from-scratch timing loop reusing
    `summarize_times` for the same result shape the rest of this file uses.
    """
    print("    Warming up...")
    func(a, b)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        start = time.perf_counter()
        func(a, b)
        duration = time.perf_counter() - start
        times.append(duration)
        print(f"    Round {i + 1}: {duration:.4f} seconds")

    summary = summarize_times(times)
    print(f"    Average Time: {summary['average_seconds']:.4f} s")
    return {"status": "ok", "times_seconds": times, "iterations_completed": len(times), **summary}


# ---------------------------------------------------------------------------
# PTRAIL
# ---------------------------------------------------------------------------


def benchmark_ptrail_interpolate(tdf: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    from ptrail.preprocessing.interpolation import Interpolation

    results = {}
    for ip_type in PTRAIL_INTERPOLATE_METHODS:
        print(f"  interpolate_position(ip_type={ip_type})")
        try:
            results[ip_type] = run_timed_call(
                lambda tdf, **kw: Interpolation.interpolate_position(tdf, SAMPLING_RATE_S, **kw),
                lambda tdf=tdf: tdf,
                {"ip_type": ip_type},
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[ip_type] = error_result(str(exc))
    return results


# ---------------------------------------------------------------------------
# MovingPandas
# ---------------------------------------------------------------------------


def movingpandas_interpolate_at_all(tc: Any, method: str) -> list[Any]:
    positions = []
    for traj in tc.trajectories:
        start, end = traj.get_start_time(), traj.get_end_time()
        if start == end:
            continue
        query_time = start + (end - start) / 2
        positions.append(traj.get_position_at(query_time, method=method))
    return positions


def benchmark_movingpandas_interpolate_at(tc: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    results = {}
    for method in ("interpolated", "nearest"):
        print(f"  get_position_at(method={method})")
        try:
            results[method] = run_timed_call(
                movingpandas_interpolate_at_all,
                lambda tc=tc: tc,
                {"method": method},
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            print(f"    error: {exc}")
            results[method] = error_result(str(exc))
    return results


def benchmark_movingpandas_distance(traj_a: Any, traj_b: Any, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    print("  hausdorff_distance")
    try:
        result = _run_two_input_timed_call(
            lambda a, b: a.hausdorff_distance(b),
            traj_a,
            traj_b,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:
        print(f"    error: {exc}")
        result = error_result(str(exc))
    return {"hausdorff": result}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_fastmob_size(
    data_path: Path, size: int, backend: str, distance_sizes: list[int], iterations: int, sleep_seconds: float
) -> dict[str, Any]:
    df = (
        load_brightkite_polars(data_path).head(size)
        if backend == "polars"
        else load_brightkite_pandas(data_path).head(size)
    )
    print(f"\nSize {size_label(size)} ({len(df)} rows), backend={backend}")

    # Distance sequences are built from the 2 globally most-active users'
    # own full check-in histories, not from this size's row slice -- a small
    # `size` slice of Brightkite's user-grouped row order can easily contain
    # only 1 (or 0) distinct users, since rows are not interleaved by user.
    full_pandas_df = load_brightkite_pandas(data_path)
    distance_results = {}
    for distance_size in distance_sizes:
        seq_a, seq_b = make_distance_sequences_pandas(full_pandas_df, distance_size)
        if backend == "polars":
            import polars as pl

            seq_a, seq_b = pl.from_pandas(seq_a), pl.from_pandas(seq_b)
        print(f"  -- trajectory_distance sequence size {size_label(distance_size)} --")
        distance_results[distance_size] = benchmark_fastmob_distance(seq_a, seq_b, iterations, sleep_seconds)

    return {
        "size": size,
        "label": size_label(size),
        "rows": len(df),
        "backend": backend,
        "interpolate": benchmark_fastmob_interpolate(df, iterations, sleep_seconds),
        "interpolate_at": benchmark_fastmob_interpolate_at(df, iterations, sleep_seconds),
        "smooth": benchmark_fastmob_smooth(df, iterations, sleep_seconds),
        "trajectory_distance": distance_results,
    }


def run_ptrail_size(data_path: Path, size: int, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    print(f"\nSize {size_label(size)}")
    try:
        tdf = load_brightkite_ptrail(data_path, size)
    except Exception as exc:
        reason = f"ptrail input setup failed: {exc}"
        print(f"  skipped: {reason}")
        return {
            "size": size,
            "label": size_label(size),
            "interpolate": {m: skipped_result(reason) for m in PTRAIL_INTERPOLATE_METHODS},
        }
    return {
        "size": size,
        "label": size_label(size),
        "rows": len(tdf),
        "interpolate": benchmark_ptrail_interpolate(tdf, iterations, sleep_seconds),
    }


def run_movingpandas_size(
    data_path: Path, size: int, distance_sizes: list[int], iterations: int, sleep_seconds: float
) -> dict[str, Any]:
    print(f"\nSize {size_label(size)}")
    try:
        tc = load_brightkite_movingpandas_collection(data_path, size)
    except Exception as exc:
        reason = f"movingpandas input setup failed: {exc}"
        print(f"  skipped: {reason}")
        interpolate_at_result = {m: skipped_result(reason) for m in ("interpolated", "nearest")}
    else:
        interpolate_at_result = benchmark_movingpandas_interpolate_at(tc, iterations, sleep_seconds)

    distance_results = {}
    for distance_size in distance_sizes:
        print(f"  -- hausdorff_distance sequence size {size_label(distance_size)} --")
        try:
            traj_a, traj_b = load_brightkite_movingpandas_pair(data_path, distance_size)
            distance_results[distance_size] = benchmark_movingpandas_distance(traj_a, traj_b, iterations, sleep_seconds)
        except Exception as exc:
            reason = f"movingpandas distance input setup failed: {exc}"
            print(f"    skipped: {reason}")
            distance_results[distance_size] = {"hausdorff": skipped_result(reason)}

    return {
        "size": size,
        "label": size_label(size),
        "interpolate_at": interpolate_at_result,
        "trajectory_distance": distance_results,
    }


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "trajectory",
        "library": args.library,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sizes": args.sizes,
        "distance_sizes": args.distance_sizes,
        "backend": getattr(args, "backend", None),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "dataset_path": str(args.data_path),
    }


def concrete_backends(args: argparse.Namespace) -> tuple[str, ...]:
    if args.library != "fastmob":
        return (None,)
    return ("pandas", "polars") if args.backend == "both" else (args.backend,)


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fastmob.trajectory speed benchmarks against real Brightkite data."
    )
    parser.add_argument("--library", choices=["fastmob", "ptrail", "movingpandas"], default="fastmob")
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--distance-sizes", type=str, default=",".join(str(s) for s in DEFAULT_DISTANCE_SIZES))
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)
    args.sizes = parse_int_list(args.sizes)
    args.distance_sizes = parse_int_list(args.distance_sizes)
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.data_path).exists():
        raise SystemExit(f"Dataset not found at {args.data_path}.")

    metadata = build_metadata(args)
    results = []
    if args.library == "fastmob":
        for backend in concrete_backends(args):
            for size in args.sizes:
                results.append(
                    run_fastmob_size(
                        args.data_path, size, backend, args.distance_sizes, args.iterations, args.sleep_seconds
                    )
                )
    elif args.library == "ptrail":
        for size in args.sizes:
            results.append(run_ptrail_size(args.data_path, size, args.iterations, args.sleep_seconds))
    else:
        for size in args.sizes:
            results.append(
                run_movingpandas_size(args.data_path, size, args.distance_sizes, args.iterations, args.sleep_seconds)
            )
    return {"metadata": metadata, "results": results}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    suffix = f"_{args.backend}" if args.library == "fastmob" else ""
    output_path = Path(args.output_dir) / f"{args.library}_trajectory_speed{suffix}.json"
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
