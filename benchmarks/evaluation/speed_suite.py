"""Standalone evaluation speed benchmark suite.

Run from the repository root, for example:

    python benchmarks/evaluation/speed_suite.py --library fastmob --sizes 1000 10000
    python benchmarks/evaluation/speed_suite.py --library skmob --sizes 1000 10000 100000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

_BENCH_PROC = psutil.Process(os.getpid())

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
DEFAULT_SEED = 20260608
INPUT_SOURCE = "synthetic"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fastmob_module_path: str
    skmob_module_path: str
    func_name: str
    input_kind: str = "array_pair"
    kwargs: dict[str, Any] = field(default_factory=dict)


LEGACY_EVALUATION_METRICS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "common_part_of_commuters",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "common_part_of_commuters",
        input_kind="flow_pair",
    ),
    BenchmarkSpec(
        "common_part_of_links",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "common_part_of_links",
        input_kind="flow_pair",
    ),
    BenchmarkSpec(
        "common_part_of_commuters_distance",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "common_part_of_commuters_distance",
        input_kind="trips_pair",
    ),
    BenchmarkSpec(
        "r_squared",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "r_squared",
    ),
    BenchmarkSpec(
        "rmse",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "rmse",
    ),
    BenchmarkSpec(
        "nrmse",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "nrmse",
    ),
    BenchmarkSpec(
        "information_gain",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "information_gain",
    ),
    BenchmarkSpec(
        "kullback_leibler_divergence",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "kullback_leibler_divergence",
    ),
    BenchmarkSpec(
        "max_error",
        "fastmob.measures.evaluation",
        "skmob.measures.evaluation",
        "max_error",
    ),
)
EVALUATION_METRICS = LEGACY_EVALUATION_METRICS


class SkippedMetric(Exception):
    """Raised when a metric cannot be benchmarked."""


def size_label(size: int) -> str:
    if size >= 1_000_000 and size % 1_000_000 == 0:
        return f"{size // 1_000_000}M"
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def summarize_times(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"average_seconds": None, "minimum_seconds": None}
    return {"average_seconds": sum(times) / len(times), "minimum_seconds": min(times)}


def summarize_memory(peak_memory_mb: list[float]) -> dict[str, float | None]:
    if not peak_memory_mb:
        return {
            "average_peak_memory_mb": None,
            "minimum_peak_memory_mb": None,
            "maximum_peak_memory_mb": None,
        }
    return {
        "average_peak_memory_mb": sum(peak_memory_mb) / len(peak_memory_mb),
        "minimum_peak_memory_mb": min(peak_memory_mb),
        "maximum_peak_memory_mb": max(peak_memory_mb),
    }


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def build_output_path(output_dir: Path, library: str, profile: str = "speed", backend: str = INPUT_SOURCE) -> Path:
    return output_dir / f"{library}_evaluation_{profile}_{backend}.json"


def module_path_for_library(spec: BenchmarkSpec, library: str) -> str:
    if library == "fastmob":
        return spec.fastmob_module_path
    if library == "skmob":
        return spec.skmob_module_path
    raise ValueError(f"unknown library: {library}")


def import_metric(spec: BenchmarkSpec, library: str) -> Callable[..., Any]:
    import importlib

    module_path = module_path_for_library(spec, library)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def synthetic_seed(size: int, seed: int, input_kind: str) -> int:
    kind_offset = 17 if input_kind == "distance_pair" else 0
    return seed + size + kind_offset


def make_synthetic_pair(size: int, *, seed: int, input_kind: str = "array_pair") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(synthetic_seed(size, seed, input_kind))
    if input_kind == "distance_pair":
        observed = rng.gamma(shape=2.0, scale=8.0, size=size) + 0.001
        predicted = observed * rng.lognormal(mean=0.0, sigma=0.15, size=size)
        return observed.astype(float, copy=False), predicted.astype(float, copy=False)

    observed = rng.lognormal(mean=2.0, sigma=0.8, size=size) + 0.001
    noise = rng.normal(loc=1.0, scale=0.08, size=size)
    predicted = np.maximum(observed * noise, 0.001)
    return observed.astype(float, copy=False), predicted.astype(float, copy=False)


def make_inputs_for_size(size: int, *, seed: int) -> dict[str, Any]:
    array_pair = make_synthetic_pair(size, seed=seed, input_kind="array_pair")
    distance_pair = make_synthetic_pair(size, seed=seed, input_kind="distance_pair")
    import pandas as pd

    locations = np.arange(size, dtype=np.int64)
    flow = {
        "origin": locations,
        "destination": np.roll(locations, -1),
        "flow": np.maximum(array_pair[0], 0.001),
    }
    trips = {
        "trip_id": locations,
        "started_at": pd.date_range("2020-01-01", periods=size, freq="s"),
        "finished_at": pd.date_range("2020-01-01", periods=size, freq="s") + pd.Timedelta(seconds=1),
        "origin_location_id": locations,
        "destination_location_id": np.roll(locations, -1),
        "distance_km": distance_pair[0],
    }
    from fastmob.core.flow_dataframe import FlowDataFrame
    from fastmob.core.trips_dataframe import Trips

    return {
        "array_pair": array_pair,
        "distance_pair": distance_pair,
        "flow_pair": (FlowDataFrame(flow), FlowDataFrame({**flow, "flow": flow["flow"] * 1.01})),
        "trips_pair": (Trips(pd.DataFrame(trips)), Trips(pd.DataFrame(trips))),
    }


def call_benchmark_func(
    func: Callable[..., Any],
    input_value: tuple[np.ndarray, np.ndarray],
    spec: BenchmarkSpec,
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        arr1, arr2 = input_value
        return func(arr1, arr2, **spec.kwargs)


def run_timed_call(
    func: Callable[..., Any],
    make_input: Callable[[], tuple[np.ndarray, np.ndarray]],
    spec: BenchmarkSpec,
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print("    Warming up...")
    call_benchmark_func(func, make_input(), spec)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        start = time.perf_counter()
        call_benchmark_func(func, make_input(), spec)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        print(f"    Round {i + 1}: {duration:.4f} seconds")

    summary = summarize_times(times)
    print(f"    Average Time: {summary['average_seconds']:.4f} s")
    print(f"    Minimum Time: {summary['minimum_seconds']:.4f} s")
    return {
        "status": "ok",
        "times_seconds": times,
        "iterations_completed": len(times),
        **summary,
    }


def run_memory_call(
    func: Callable[..., Any],
    make_input: Callable[[], tuple[np.ndarray, np.ndarray]],
    spec: BenchmarkSpec,
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    peak_memory_mb: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        gc.collect()
        rss_before = _BENCH_PROC.memory_info().rss
        call_benchmark_func(func, make_input(), spec)
        rss_after = _BENCH_PROC.memory_info().rss
        delta_mb = max(0.0, (rss_after - rss_before) / (1024 * 1024))
        peak_memory_mb.append(delta_mb)
        print(f"    Round {i + 1}: {delta_mb:.4f} MB peak")

    summary = summarize_memory(peak_memory_mb)
    print(f"    Average Peak Memory: {summary['average_peak_memory_mb']:.4f} MB")
    print(f"    Minimum Peak Memory: {summary['minimum_peak_memory_mb']:.4f} MB")
    print(f"    Maximum Peak Memory: {summary['maximum_peak_memory_mb']:.4f} MB")
    return {
        "status": "ok",
        "peak_memory_mb": peak_memory_mb,
        "current_memory_mb": [],
        "iterations_completed": len(peak_memory_mb),
        **summary,
    }


def empty_profile_result(status: str, reason: str, profile: str) -> dict[str, Any]:
    if profile == "memory":
        return {
            "status": status,
            "reason": reason,
            "peak_memory_mb": [],
            "current_memory_mb": [],
            "iterations_completed": 0,
            **summarize_memory([]),
        }
    return {
        "status": status,
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def skipped_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("skipped", reason, profile)


def error_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("error", reason, profile)


def benchmark_metric(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], tuple[np.ndarray, np.ndarray]],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = import_metric(spec, library)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    try:
        if profile == "memory":
            return run_memory_call(func, make_input, spec, iterations=iterations, sleep_seconds=sleep_seconds)
        return run_timed_call(func, make_input, spec, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:  # noqa: BLE001
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_size(
    size: int,
    *,
    library: str,
    seed: int,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    inputs_by_kind = make_inputs_for_size(size, seed=seed)

    print(f"\nSize {size_label(size)} ({size} values)")
    return {
        "size": size,
        "label": size_label(size),
        "values": size,
        "input_source": INPUT_SOURCE,
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                library,
                lambda spec=spec: inputs_by_kind[spec.input_kind],
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in EVALUATION_METRICS
        },
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    results = [
        benchmark_size(
            size,
            library=args.library,
            seed=args.seed,
            profile=args.profile,
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
        )
        for size in args.sizes
    ]

    metadata = {
        "suite": "evaluation",
        "library": args.library,
        "profile": args.profile,
        "input_source": INPUT_SOURCE,
        "input_type": "numpy.ndarray",
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "sizes": args.sizes,
        "seed": args.seed,
        **({"memory_method": "memray_peak_heap"} if args.profile == "memory" else {}),
    }
    return {"metadata": metadata, "results": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone evaluation speed benchmarks.")
    parser.add_argument("--library", choices=["fastmob", "skmob"], default="fastmob")
    parser.add_argument(
        "--backend",
        choices=["pandas", "polars", "both"],
        default="pandas",
        help="Backend label for output filename; evaluation uses numpy so computation is identical.",
    )
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    backends = ["pandas", "polars"] if args.backend == "both" else [args.backend]
    for backend in backends:
        output_path = build_output_path(Path(args.output_dir), args.library, args.profile, backend)
        write_json(payload, output_path)
        print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
