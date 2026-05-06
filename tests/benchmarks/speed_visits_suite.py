"""Standalone visits speed benchmark suite.

Run from the repository root, for example:

    python tests/benchmarks/speed_visits_suite.py --library skmob2
    python tests/benchmarks/speed_visits_suite.py --library skmob
    python tests/benchmarks/speed_visits_suite.py --library skmob --timing-mode workflow_tdf
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    module_path: str
    func_name: str
    kwargs: dict[str, Any]


VISITS_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "frequency_rank",
        "skmob2.measures.visits.frequency_rank",
        "frequency_rank",
        {},
    ),
    MetricSpec(
        "location_frequency",
        "skmob2.measures.visits.location_frequency",
        "location_frequency",
        {},
    ),
    MetricSpec(
        "recency_rank",
        "skmob2.measures.visits.recency_rank",
        "recency_rank",
        {},
    ),
)


class SkippedMetric(Exception):
    """Raised when a metric cannot be benchmarked in the selected library."""


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


def build_output_path(output_dir: Path, library: str, timing_mode: str) -> Path:
    if library == "skmob2":
        filename = "skmob2_visits_speed.json"
    else:
        filename = f"skmob_visits_speed_{timing_mode}.json"
    return output_dir / filename


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


def import_metric(spec: MetricSpec, library: str) -> Callable[..., Any]:
    try:
        if library == "skmob2":
            module = importlib.import_module(spec.module_path)
        else:
            module = importlib.import_module("skmob.measures.individual")
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def metric_kwargs_for_library(spec: MetricSpec, library: str, func: Callable[..., Any]) -> dict[str, Any]:
    kwargs = dict(spec.kwargs)
    if library != "skmob":
        return kwargs

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    accepts_extra_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if "show_progress" in signature.parameters or accepts_extra_kwargs:
        kwargs.setdefault("show_progress", False)
    return kwargs


def load_brightkite_pandas(data_path: Path):
    import pandas as pd

    df = pd.read_csv(data_path, sep="\t", header=None, names=BRIGHTKITE_COLUMNS)
    df["check-in_time"] = pd.to_datetime(df["check-in_time"], errors="coerce")
    return df


def load_brightkite_polars(data_path: Path):
    import polars as pl

    return pl.read_csv(
        data_path,
        separator="\t",
        has_header=False,
        new_columns=BRIGHTKITE_COLUMNS,
        try_parse_dates=True,
    )


def make_skmob_tdf(skmob_module: Any, df: Any) -> Any:
    return skmob_module.TrajDataFrame(
        df.copy(),
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


def run_timed_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print("    Warming up...")
    func(make_input(), **kwargs)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        start = time.perf_counter()
        func(make_input(), **kwargs)
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


def benchmark_metric(
    spec: MetricSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = import_metric(spec, library)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return {
            "status": "skipped",
            "reason": str(exc),
            "times_seconds": [],
            "iterations_completed": 0,
            **summarize_times([]),
        }

    try:
        kwargs = metric_kwargs_for_library(spec, library, func)
        return run_timed_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:
        print(f"    error: {exc}")
        return {
            "status": "error",
            "reason": str(exc),
            "times_seconds": [],
            "iterations_completed": 0,
            **summarize_times([]),
        }


def benchmark_skmob2_size(df: Any, size: int, *, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    size_df = df.head(size)
    print(f"\nSize {size_label(size)} ({len(size_df)} rows)")
    return {
        "size": size,
        "label": size_label(size),
        "rows": len(size_df),
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                "skmob2",
                lambda size_df=size_df: size_df,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in VISITS_METRICS
        },
    }


def benchmark_skmob_size(
    raw_df: Any,
    skmob_module: Any,
    size: int,
    *,
    timing_mode: str,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    pandas_slice = raw_df.head(size).copy()
    print(f"\nSize {size_label(size)} ({len(pandas_slice)} rows)")

    if timing_mode == "prebuilt_tdf":
        tdf = make_skmob_tdf(skmob_module, pandas_slice)

        def make_input() -> Any:
            return tdf

    else:

        def make_input() -> Any:
            return make_skmob_tdf(skmob_module, pandas_slice)

    return {
        "size": size,
        "label": size_label(size),
        "rows": len(pandas_slice),
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                "skmob",
                make_input,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in VISITS_METRICS
        },
    }


def build_metadata(args: argparse.Namespace, *, input_type: str, timing_mode: str) -> dict[str, Any]:
    return {
        "library": args.library,
        "timing_mode": timing_mode,
        "input_type": input_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "sizes": args.sizes,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Place the Brightkite file there before running.")

    if args.library == "skmob2":
        print(f"Loading Brightkite into Polars from {data_path}...")
        df = load_brightkite_polars(data_path)
        results = [
            benchmark_skmob2_size(df, size, iterations=args.iterations, sleep_seconds=args.sleep_seconds)
            for size in args.sizes
        ]
        metadata = build_metadata(args, input_type="polars.DataFrame", timing_mode="measure_only")
        return {"metadata": metadata, "results": results}

    print(f"Loading Brightkite into pandas from {data_path}...")
    raw_df = load_brightkite_pandas(data_path)
    try:
        skmob_module = importlib.import_module("skmob")
    except Exception as exc:
        raise SystemExit(f"Unable to import original skmob: {exc}") from exc

    results = [
        benchmark_skmob_size(
            raw_df,
            skmob_module,
            size,
            timing_mode=args.timing_mode,
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
        )
        for size in args.sizes
    ]
    metadata = build_metadata(args, input_type="skmob.TrajDataFrame", timing_mode=args.timing_mode)
    return {"metadata": metadata, "results": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone visits speed benchmarks.")
    parser.add_argument("--library", choices=["skmob2", "skmob"], required=True)
    parser.add_argument("--timing-mode", choices=["prebuilt_tdf", "workflow_tdf"], default="prebuilt_tdf")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    output_path = build_output_path(Path(args.output_dir), args.library, args.timing_mode)
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
