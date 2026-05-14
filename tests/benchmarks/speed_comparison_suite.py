"""Standalone comparison-metric speed benchmark suite.

Run from the repository root, for example:

    python tests/benchmarks/speed_comparison_suite.py --backend pandas --sizes 10000
    python tests/benchmarks/speed_comparison_suite.py --backend polars --sizes 1000 10000 100000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000]
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]

# Column name used in Brightkite for latitudes (the numeric column we compare).
_LAT_COL = "latitude"
# Column used as user ID in Brightkite.
_UID_COL = "user"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    module_path: str
    func_name: str
    input_kind: str  # "array_pair" | "df_pair" | "df_pair_column"
    kwargs: dict[str, Any] = field(default_factory=dict)
    # For "df_pair_column" the positional column argument to pass.
    column: str | None = None


COMPARISON_METRICS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "wasserstein_distance",
        "skmob2.comparison.metrics",
        "wasserstein_distance",
        "array_pair",
    ),
    BenchmarkSpec(
        "histogram_jensen_shannon_divergence",
        "skmob2.comparison.metrics",
        "histogram_jensen_shannon_divergence",
        "array_pair",
        kwargs={"bin_size": 1.0},
    ),
    BenchmarkSpec(
        "column_distribution_wasserstein_distance",
        "skmob2.comparison.distribution",
        "column_distribution_wasserstein_distance",
        "df_pair_column",
        column=_LAT_COL,
    ),
    BenchmarkSpec(
        "column_distribution_jensen_shannon_divergence",
        "skmob2.comparison.distribution",
        "column_distribution_jensen_shannon_divergence",
        "df_pair_column",
        column=_LAT_COL,
    ),
    BenchmarkSpec(
        "visits_per_user_wasserstein_distance",
        "skmob2.comparison.distribution",
        "visits_per_user_wasserstein_distance",
        "df_pair",
    ),
    BenchmarkSpec(
        "visits_per_user_jensen_shannon_divergence",
        "skmob2.comparison.distribution",
        "visits_per_user_jensen_shannon_divergence",
        "df_pair",
    ),
)


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


def build_output_path(output_dir: Path, backend: str, profile: str = "speed") -> Path:
    return output_dir / f"skmob2_comparison_{profile}_{backend}.json"


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


def split_in_half(df: Any) -> tuple[Any, Any]:
    """Split a DataFrame into two roughly equal halves."""
    mid = len(df) // 2
    try:
        return df.head(mid), df.tail(len(df) - mid)
    except TypeError:
        return df[:mid], df[mid:]


def import_metric(spec: BenchmarkSpec) -> Callable[..., Any]:
    import importlib

    try:
        module = importlib.import_module(spec.module_path)
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def call_benchmark_func(
    func: Callable[..., Any],
    input_value: Any,
    spec: BenchmarkSpec,
) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        if spec.input_kind == "array_pair":
            arr1, arr2 = input_value
            return func(arr1, arr2, **spec.kwargs)
        if spec.input_kind == "df_pair_column":
            df1, df2 = input_value
            return func(df1, df2, spec.column, **spec.kwargs)
        # "df_pair"
        df1, df2 = input_value
        return func(df1, df2, **spec.kwargs)


def make_array_pair_from_df(df: Any, col: str) -> tuple[Any, Any]:
    """Extract a column and split into two numpy arrays."""
    try:
        import narwhals as nw

        ndf = nw.from_native(df, eager_only=True)
        arr = ndf.get_column(col).to_numpy()
    except Exception:
        arr = df[col].to_numpy()
    mid = len(arr) // 2
    return arr[:mid], arr[mid:]


def make_input_for_spec(spec: BenchmarkSpec, df1: Any, df2: Any) -> Any:
    if spec.input_kind == "array_pair":
        return make_array_pair_from_df(df1, _LAT_COL)
    return df1, df2


def run_timed_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
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
    make_input: Callable[[], Any],
    spec: BenchmarkSpec,
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print("    Warming up...")
    call_benchmark_func(func, make_input(), spec)

    current_memory_mb: list[float] = []
    peak_memory_mb: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        tracemalloc.start()
        try:
            call_benchmark_func(func, make_input(), spec)
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        current_mb = current_bytes / (1024 * 1024)
        peak_mb = peak_bytes / (1024 * 1024)
        current_memory_mb.append(current_mb)
        peak_memory_mb.append(peak_mb)
        print(f"    Round {i + 1}: {peak_mb:.4f} MB peak")

    summary = summarize_memory(peak_memory_mb)
    print(f"    Average Peak Memory: {summary['average_peak_memory_mb']:.4f} MB")
    print(f"    Minimum Peak Memory: {summary['minimum_peak_memory_mb']:.4f} MB")
    print(f"    Maximum Peak Memory: {summary['maximum_peak_memory_mb']:.4f} MB")
    return {
        "status": "ok",
        "peak_memory_mb": peak_memory_mb,
        "current_memory_mb": current_memory_mb,
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
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = import_metric(spec)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    try:
        if profile == "memory":
            return run_memory_call(func, make_input, spec, iterations=iterations, sleep_seconds=sleep_seconds)
        return run_timed_call(func, make_input, spec, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_size(
    df: Any,
    size: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    try:
        sliced = df.head(size)
    except TypeError:
        sliced = df[:size]
    df1, df2 = split_in_half(sliced)

    print(f"\nSize {size_label(size)} ({len(sliced)} rows)")
    return {
        "size": size,
        "label": size_label(size),
        "rows": len(sliced),
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                lambda spec=spec, df1=df1, df2=df2: make_input_for_spec(spec, df1, df2),
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in COMPARISON_METRICS
        },
    }


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Place the Brightkite file there before running.")

    selected_backend = backend or args.backend
    if selected_backend == "pandas":
        print(f"Loading Brightkite into pandas from {data_path}...")
        df = load_brightkite_pandas(data_path)
        input_type = "pandas.DataFrame"
    else:
        print(f"Loading Brightkite into Polars from {data_path}...")
        df = load_brightkite_polars(data_path)
        input_type = "polars.DataFrame"

    results = [
        benchmark_size(
            df,
            size,
            profile=args.profile,
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
        )
        for size in args.sizes
    ]

    metadata = {
        "suite": "comparison",
        "library": "skmob2",
        "profile": args.profile,
        "backend": selected_backend,
        "input_type": input_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "sizes": args.sizes,
    }
    return {"metadata": metadata, "results": results}


def concrete_backends(args: argparse.Namespace) -> Iterable[str]:
    if args.backend == "both":
        return ("pandas", "polars")
    return (args.backend,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone comparison-metric speed benchmarks.")
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="pandas")
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for backend in concrete_backends(args):
        payload = run_suite(args, backend=backend)
        output_path = build_output_path(Path(args.output_dir), backend, args.profile)
        write_json(payload, output_path)
        print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
