"""Standalone collective speed benchmark suite.

Run from the repository root, for example:

    python benchmarks/collective/speed_suite.py --library fastmob --backend both
    python benchmarks/collective/speed_suite.py --library skmob
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import platform
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.sorted_input_cache import (
    DEFAULT_INPUT_CACHE_DIR,
    load_or_create_sorted_input,
)
from benchmarks.utils import (
    concrete_backends as iter_concrete_backends,
)
from benchmarks.utils import (
    concrete_input_orders as iter_concrete_input_orders,
)
from benchmarks.utils import (
    error_result,
    load_catalog,
    merge_payload,
    nonnegative_float,
    nonnegative_int,
    positive_int,
    run_profiled_call,
    run_profiled_call_isolated,
    size_label,
    skipped_result,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
SKMOB_CATALOG_PATH = Path(__file__).resolve().parents[1] / "skmob_public_api_catalog.json"

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir

MOVINGPANDAS_CATALOG_PATH = Path(__file__).resolve().parents[1] / "movingpandas_skmob_api_catalog.json"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]
BRIGHTKITE_TRAJECTORY_KWARGS = {
    "uid_col": "user",
    "datetime_col": "check-in_time",
    "lat_col": "latitude",
    "lng_col": "longitude",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fastmob_module_path: str
    skmob_module_path: str
    func_name: str
    kwargs: dict[str, Any]
    input_kind: str = "trajectory"


COLLECTIVE_METRICS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "random_location_entropy",
        "fastmob.measures.collective.random_location_entropy",
        "skmob.measures.collective",
        "random_location_entropy",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "uncorrelated_location_entropy",
        "fastmob.measures.collective.uncorrelated_location_entropy",
        "skmob.measures.collective",
        "uncorrelated_location_entropy",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "mean_square_displacement",
        "fastmob.measures.collective.mean_square_displacement",
        "skmob.measures.collective",
        "mean_square_displacement",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "visits_per_location",
        "fastmob.measures.collective.visits_per_location",
        "skmob.measures.collective",
        "visits_per_location",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "homes_per_location",
        "fastmob.measures.collective.homes_per_location",
        "skmob.measures.collective",
        "homes_per_location",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "visits_per_time_unit",
        "fastmob.measures.collective.visits_per_time_unit",
        "skmob.measures.collective",
        "visits_per_time_unit",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "od_matrix",
        "fastmob.measures.collective.od",
        "skmob.measures.collective",
        "od_matrix",
        {},
        input_kind="collective",
    ),
    BenchmarkSpec(
        "od_metrics_per_area",
        "fastmob.measures.collective.od",
        "skmob.measures.collective",
        "od_metrics_per_area",
        {},
        input_kind="collective",
    ),
)


class SkippedMetric(Exception):
    """Raised when a metric cannot be benchmarked in the selected library."""


def build_output_path(
    output_dir: Path,
    library: str,
    timing_mode: str,
    backend: str | None = None,
    profile: str = "speed",
    input_order: str = "raw",
) -> Path:
    order_part = "" if input_order == "raw" else f"{input_order}_"
    if library == "fastmob":
        if backend is None or backend == "both":
            raise ValueError("fastmob output path requires a concrete backend")
        filename = f"fastmob_collective_{profile}_{order_part}{backend}.json"
    elif library == "skmob":
        filename = f"skmob_collective_{profile}_{order_part}{timing_mode}.json"
    else:
        filename = (
            f"movingpandas_collective_{profile}_{input_order}.json"
            if input_order != "raw"
            else f"movingpandas_collective_{profile}.json"
        )
    return output_dir / filename


def import_metric(spec: BenchmarkSpec, library: str) -> Callable[..., Any]:
    try:
        module_path = spec.fastmob_module_path if library == "fastmob" else spec.skmob_module_path
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def metric_kwargs_for_library(spec: BenchmarkSpec, library: str, func: Callable[..., Any]) -> dict[str, Any]:
    kwargs = dict(spec.kwargs)
    if library == "fastmob" and spec.name == "homes_per_location":
        kwargs.update(BRIGHTKITE_TRAJECTORY_KWARGS)
        return kwargs

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


def benchmark_metric(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    if library == "movingpandas":
        reason = "no benchmarkable MovingPandas analogue for visits/collective API"
        print(f"    skipped: {reason}")
        return skipped_result(reason, profile)

    try:
        func = import_metric(spec, library)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    try:
        kwargs = metric_kwargs_for_library(spec, library, func)
        if retries > 0:
            return run_profiled_call_isolated(
                func,
                make_input,
                kwargs,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                retries=retries,
            )
        return run_profiled_call(
            func,
            make_input,
            kwargs,
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_specs(
    specs: tuple[BenchmarkSpec, ...],
    library: str,
    make_input_for_spec: Callable[[BenchmarkSpec], Any],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str,
    retries: int,
) -> dict[str, Any]:
    return {
        spec.name: benchmark_metric(
            spec,
            library,
            lambda spec=spec: make_input_for_spec(spec),
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
            retries=retries,
        )
        for spec in specs
    }


def make_fastmob_metric_input(df: Any, spec: BenchmarkSpec) -> Any:
    if spec.name == "od_metrics_per_area":
        return make_od_input(df, precomputed=True)
    if spec.name == "od_matrix":
        return make_od_input(df)
    return df


def benchmark_fastmob_size(
    df: Any,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
) -> dict[str, Any]:
    size_df = df.head(size)
    print(f"\nSize {size_label(size)} ({len(size_df)} rows)")

    return {
        "size": size,
        "label": size_label(size),
        "rows": len(size_df),
        "metrics": benchmark_specs(
            specs,
            "fastmob",
            lambda spec: make_fastmob_metric_input(size_df, spec),
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
            retries=retries,
        ),
    }


def make_collective_input(df: Any) -> Any:
    import pandas as pd

    is_polars = df.__class__.__module__.startswith("polars")
    pdf = df.to_pandas() if is_polars else df.copy()
    visits = pdf.rename(
        columns={
            "user": "user_id",
            "check-in_time": "start_timestamp",
            "location id": "location_id",
        }
    )
    visits["start_timestamp"] = pd.to_datetime(visits["start_timestamp"], errors="coerce")
    visits["datetime"] = visits["start_timestamp"]
    visits["area"] = visits["location_id"].astype(str)
    visits = visits.sort_values(["user_id", "start_timestamp"], kind="mergesort").reset_index(drop=True)
    if is_polars:
        import polars as pl

        return pl.from_pandas(visits)
    return visits


def make_od_input(df: Any, *, precomputed: bool = False) -> Any:
    visits = make_collective_input(df)
    is_polars = visits.__class__.__module__.startswith("polars")
    pdf = visits.to_pandas() if is_polars else visits.copy()
    pdf["origin_area"] = pdf["area"]
    pdf["destination_area"] = pdf.groupby("user_id", sort=False)["area"].shift(-1).fillna(pdf["area"])
    od = pdf[["origin_area", "destination_area"]]
    if precomputed:
        od = od.groupby(["origin_area", "destination_area"], as_index=False).size().rename(columns={"size": "count"})
    if is_polars:
        import polars as pl

        return pl.from_pandas(od)
    return od


def benchmark_skmob_size(
    raw_df: Any,
    skmob_module: Any,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    timing_mode: str,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
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
        "metrics": benchmark_specs(
            specs,
            "skmob",
            lambda _spec: make_input(),
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
            retries=retries,
        ),
    }


def benchmark_movingpandas_size(
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    profile: str = "speed",
) -> dict[str, Any]:
    return {
        "size": size,
        "label": size_label(size),
        "rows": 0,
        "metrics": benchmark_specs(
            specs,
            "movingpandas",
            lambda _spec: None,
            profile=profile,
            iterations=0,
            sleep_seconds=0,
            retries=0,
        ),
    }


def build_metadata(
    args: argparse.Namespace,
    *,
    input_type: str,
    timing_mode: str,
    backend: str | None,
    input_cache_path: Path | None = None,
    input_cache_status: str = "not_applicable",
) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "collective",
        "library": args.library,
        "profile": args.profile,
        "backend": backend,
        "timing_mode": timing_mode,
        "input_type": input_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "retries": args.retries,
        "sizes": args.sizes,
        "metrics": args.metrics,
        "input_order": args.input_order,
        "input_cache_path": None if input_cache_path is None else str(input_cache_path),
        "input_cache_status": input_cache_status,
        "skmob_catalog_path": str(SKMOB_CATALOG_PATH),
        "movingpandas_catalog_path": str(MOVINGPANDAS_CATALOG_PATH),
        **({"memory_method": "memray_peak_heap"} if args.profile == "memory" else {}),
    }


def load_brightkite_for_order(
    *,
    data_path: Path,
    backend: str,
    input_order: str,
    input_cache_dir: Path,
) -> tuple[Any, Path | None, str]:
    if input_order == "raw":
        if backend == "polars":
            return load_brightkite_polars(data_path), None, "not_applicable"
        return load_brightkite_pandas(data_path), None, "not_applicable"

    sorted_input = load_or_create_sorted_input(
        cache_dir=input_cache_dir,
        suite="collective",
        backend=backend,
        data_path=data_path,
        load_raw=lambda: (
            load_brightkite_polars(data_path) if backend == "polars" else load_brightkite_pandas(data_path)
        ),
        uid_col="user",
        datetime_col="check-in_time",
    )
    return sorted_input.data, sorted_input.path, sorted_input.status


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists() and args.library != "movingpandas":
        raise SystemExit(f"Dataset not found at {data_path}. Place the Brightkite file there before running.")
    specs = selected_specs(args)

    if args.library == "fastmob":
        selected_backend = backend or args.backend
        if selected_backend == "both":
            raise ValueError("run_suite requires a concrete backend when library is fastmob")
        if selected_backend == "pandas":
            input_type = "pandas.DataFrame"
        else:
            input_type = "polars.DataFrame"
        print(f"Loading {args.input_order} Brightkite into {selected_backend} from {data_path}...")
        df, input_cache_path, input_cache_status = load_brightkite_for_order(
            data_path=data_path,
            backend=selected_backend,
            input_order=args.input_order,
            input_cache_dir=Path(args.input_cache_dir),
        )
        results = [
            benchmark_fastmob_size(
                df,
                size,
                specs=specs,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
            )
            for size in args.sizes
        ]
        metadata = build_metadata(
            args,
            input_type=input_type,
            timing_mode="measure_only",
            backend=selected_backend,
            input_cache_path=input_cache_path,
            input_cache_status=input_cache_status,
        )
        return {"metadata": metadata, "results": results}

    if args.library == "movingpandas":
        results = [benchmark_movingpandas_size(size, specs=specs, profile=args.profile) for size in args.sizes]
        metadata = build_metadata(
            args,
            input_type="not_applicable",
            timing_mode="not_applicable",
            backend=None,
        )
        return {"metadata": metadata, "results": results}

    print(f"Loading {args.input_order} Brightkite into pandas from {data_path}...")
    raw_df, input_cache_path, input_cache_status = load_brightkite_for_order(
        data_path=data_path,
        backend="pandas",
        input_order=args.input_order,
        input_cache_dir=Path(args.input_cache_dir),
    )
    try:
        skmob_module = importlib.import_module("skmob")
    except Exception as exc:
        raise SystemExit(f"Unable to import original skmob: {exc}") from exc

    results = [
        benchmark_skmob_size(
            raw_df,
            skmob_module,
            size,
            specs=specs,
            timing_mode=args.timing_mode,
            profile=args.profile,
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
            retries=args.retries,
        )
        for size in args.sizes
    ]
    metadata = build_metadata(
        args,
        input_type="skmob.TrajDataFrame",
        timing_mode=args.timing_mode,
        backend=None,
        input_cache_path=input_cache_path,
        input_cache_status=input_cache_status,
    )
    return {"metadata": metadata, "results": results}


def concrete_backends(args: argparse.Namespace) -> Iterable[str | None]:
    return iter_concrete_backends(args.library, args.backend)


def concrete_input_orders(args: argparse.Namespace) -> Iterable[str]:
    return iter_concrete_input_orders(args.input_order)


def selected_specs(args: argparse.Namespace) -> tuple[BenchmarkSpec, ...]:
    requested = set(args.metrics)
    return tuple(spec for spec in COLLECTIVE_METRICS if spec.name in requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone collective speed benchmarks.")
    parser.add_argument("--library", choices=["fastmob", "skmob", "movingpandas"], required=True)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument(
        "--timing-mode",
        choices=["prebuilt_tdf", "workflow_tdf"],
        default="prebuilt_tdf",
    )
    parser.add_argument("--input-order", choices=["raw", "sorted", "both"], default="raw")
    parser.add_argument("--input-cache-dir", type=Path, default=DEFAULT_INPUT_CACHE_DIR)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument(
        "--retries",
        type=nonnegative_int,
        default=0,
        help="Retry a metric this many times after failure.",
    )
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument(
        "--metrics",
        choices=[spec.name for spec in COLLECTIVE_METRICS],
        nargs="+",
        default=[spec.name for spec in COLLECTIVE_METRICS],
        help="Only run the selected collective metrics.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge selected sizes/metrics into an existing output JSON instead of replacing the whole file.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for input_order in concrete_input_orders(args):
        order_args = argparse.Namespace(**vars(args))
        order_args.input_order = input_order
        for backend in concrete_backends(order_args):
            payload = run_suite(order_args, backend=backend)
            output_path = build_output_path(
                Path(order_args.output_dir),
                order_args.library,
                order_args.timing_mode,
                backend,
                order_args.profile,
                order_args.input_order,
            )
            if order_args.append and output_path.exists():
                payload = merge_payload(load_catalog(output_path), payload)
            write_json(payload, output_path)
            print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
