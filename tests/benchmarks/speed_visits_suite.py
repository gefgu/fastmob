"""Standalone visits/collective speed benchmark suite.

Run from the repository root, for example:

    python tests/benchmarks/speed_visits_suite.py --library skmob2 --backend both
    python tests/benchmarks/speed_visits_suite.py --library skmob
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
SKMOB_CATALOG_PATH = Path(__file__).resolve().parent / "skmob_public_api_catalog.json"
MOVINGPANDAS_CATALOG_PATH = Path(__file__).resolve().parent / "movingpandas_skmob_api_catalog.json"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    skmob2_module_path: str
    skmob_module_path: str
    func_name: str
    kwargs: dict[str, Any]
    input_kind: str = "trajectory"


VISITS_METRICS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("random_entropy", "skmob2.measures.visits.random_entropy", "skmob.measures.individual", "random_entropy", {}),
    BenchmarkSpec(
        "uncorrelated_entropy",
        "skmob2.measures.visits.uncorrelated_entropy",
        "skmob.measures.individual",
        "uncorrelated_entropy",
        {},
    ),
    BenchmarkSpec("real_entropy", "skmob2.measures.visits.real_entropy", "skmob.measures.individual", "real_entropy", {}),
    BenchmarkSpec(
        "location_frequency",
        "skmob2.measures.visits.location_frequency",
        "skmob.measures.individual",
        "location_frequency",
        {},
    ),
    BenchmarkSpec(
        "individual_mobility_network",
        "skmob2.measures.visits.individual_mobility_network",
        "skmob.measures.individual",
        "individual_mobility_network",
        {},
    ),
    BenchmarkSpec("recency_rank", "skmob2.measures.visits.recency_rank", "skmob.measures.individual", "recency_rank", {}),
    BenchmarkSpec(
        "frequency_rank",
        "skmob2.measures.visits.frequency_rank",
        "skmob.measures.individual",
        "frequency_rank",
        {},
    ),
    BenchmarkSpec(
        "random_location_entropy",
        "skmob2.measures.flows.random_location_entropy",
        "skmob.measures.collective",
        "random_location_entropy",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "uncorrelated_location_entropy",
        "skmob2.measures.flows.uncorrelated_location_entropy",
        "skmob.measures.collective",
        "uncorrelated_location_entropy",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "mean_square_displacement",
        "skmob2.measures.flows.mean_square_displacement",
        "skmob.measures.collective",
        "mean_square_displacement",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "visits_per_location",
        "skmob2.measures.flows.visits_per_location",
        "skmob.measures.collective",
        "visits_per_location",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "homes_per_location",
        "skmob2.measures.flows.homes_per_location",
        "skmob.measures.collective",
        "homes_per_location",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "visits_per_time_unit",
        "skmob2.measures.flows.visits_per_time_unit",
        "skmob.measures.collective",
        "visits_per_time_unit",
        {},
        input_kind="collective",
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


def load_catalog(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_output_path(output_dir: Path, library: str, timing_mode: str, backend: str | None = None) -> Path:
    if library == "skmob2":
        if backend is None or backend == "both":
            raise ValueError("skmob2 output path requires a concrete backend")
        filename = f"skmob2_visits_speed_{backend}.json"
    elif library == "skmob":
        filename = f"skmob_visits_speed_{timing_mode}.json"
    else:
        filename = "movingpandas_visits_speed.json"
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


def import_metric(spec: BenchmarkSpec, library: str) -> Callable[..., Any]:
    try:
        module_path = spec.skmob2_module_path if library == "skmob2" else spec.skmob_module_path
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def metric_kwargs_for_library(spec: BenchmarkSpec, library: str, func: Callable[..., Any]) -> dict[str, Any]:
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
    call_benchmark_func(func, make_input(), kwargs)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        start = time.perf_counter()
        call_benchmark_func(func, make_input(), kwargs)
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


def call_benchmark_func(func: Callable[..., Any], input_value: Any, kwargs: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        return func(input_value, **kwargs)


def skipped_result(reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def error_result(reason: str) -> dict[str, Any]:
    return {
        "status": "error",
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def benchmark_metric(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    if library == "movingpandas":
        reason = "no benchmarkable MovingPandas analogue for visits/collective API"
        print(f"    skipped: {reason}")
        return skipped_result(reason)

    try:
        func = import_metric(spec, library)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc))

    try:
        kwargs = metric_kwargs_for_library(spec, library, func)
        return run_timed_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:
        print(f"    error: {exc}")
        return error_result(str(exc))


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


def benchmark_movingpandas_size(size: int) -> dict[str, Any]:
    return {
        "size": size,
        "label": size_label(size),
        "rows": 0,
        "metrics": {spec.name: skipped_result("no benchmarkable MovingPandas analogue") for spec in VISITS_METRICS},
    }


def build_metadata(args: argparse.Namespace, *, input_type: str, timing_mode: str, backend: str | None) -> dict[str, Any]:
    return {
        "suite": "visits",
        "library": args.library,
        "backend": backend,
        "timing_mode": timing_mode,
        "input_type": input_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "sizes": args.sizes,
        "skmob_catalog_path": str(SKMOB_CATALOG_PATH),
        "movingpandas_catalog_path": str(MOVINGPANDAS_CATALOG_PATH),
    }


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists() and args.library != "movingpandas":
        raise SystemExit(f"Dataset not found at {data_path}. Place the Brightkite file there before running.")

    if args.library == "skmob2":
        selected_backend = backend or args.backend
        if selected_backend == "both":
            raise ValueError("run_suite requires a concrete backend when library is skmob2")
        if selected_backend == "pandas":
            print(f"Loading Brightkite into pandas from {data_path}...")
            df = load_brightkite_pandas(data_path)
            input_type = "pandas.DataFrame"
        else:
            print(f"Loading Brightkite into Polars from {data_path}...")
            df = load_brightkite_polars(data_path)
            input_type = "polars.DataFrame"
        results = [
            benchmark_skmob2_size(df, size, iterations=args.iterations, sleep_seconds=args.sleep_seconds)
            for size in args.sizes
        ]
        metadata = build_metadata(args, input_type=input_type, timing_mode="measure_only", backend=selected_backend)
        return {"metadata": metadata, "results": results}

    if args.library == "movingpandas":
        results = [benchmark_movingpandas_size(size) for size in args.sizes]
        metadata = build_metadata(args, input_type="not_applicable", timing_mode="not_applicable", backend=None)
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
    metadata = build_metadata(args, input_type="skmob.TrajDataFrame", timing_mode=args.timing_mode, backend=None)
    return {"metadata": metadata, "results": results}


def concrete_backends(args: argparse.Namespace) -> Iterable[str | None]:
    if args.library != "skmob2":
        return (None,)
    if args.backend == "both":
        return ("pandas", "polars")
    return (args.backend,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone visits/collective speed benchmarks.")
    parser.add_argument("--library", choices=["skmob2", "skmob", "movingpandas"], required=True)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--timing-mode", choices=["prebuilt_tdf", "workflow_tdf"], default="prebuilt_tdf")
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
        output_path = build_output_path(Path(args.output_dir), args.library, args.timing_mode, backend)
        write_json(payload, output_path)
        print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
