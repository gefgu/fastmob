"""Standalone privacy attack speed benchmark suite.

Run from the repository root, for example:

    python benchmarks/privacy/speed_suite.py --library fastmob --backend both
    python benchmarks/privacy/speed_suite.py --library skmob
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import importlib
import inspect
import json
import os
import platform
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.sorted_input_cache import DEFAULT_INPUT_CACHE_DIR, load_or_create_sorted_input

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "scikit-mobility" / "examples" / "privacy_toy.csv"

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    class_name: str
    init_kwargs: dict[str, Any]
    assess_kwargs: dict[str, Any]


PRIVACY_ATTACKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("location_kl2", "LocationAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("location_sequence_kl2", "LocationSequenceAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("location_time_kl2", "LocationTimeAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("unique_location_kl2", "UniqueLocationAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("location_frequency_kl2", "LocationFrequencyAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("location_probability_kl2", "LocationProbabilityAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("location_proportion_kl2", "LocationProportionAttack", {"knowledge_length": 2}, {}),
    BenchmarkSpec("home_work", "HomeWorkAttack", {}, {}),
)
PRIVACY_H3_RESOLUTION = 12


class SkippedAttack(Exception):
    """Raised when an attack cannot be benchmarked in the selected library."""


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


def build_output_path(
    output_dir: Path,
    library: str,
    timing_mode: str,
    backend: str | None = None,
    input_order: str = "raw",
    profile: str = "speed",
) -> Path:
    order_part = "" if input_order == "raw" else f"{input_order}_"
    if library == "fastmob":
        if backend is None or backend == "both":
            raise ValueError("fastmob output path requires a concrete backend")
        filename = f"fastmob_privacy_{profile}_{order_part}{backend}.json"
    else:
        filename = f"skmob_privacy_{profile}_{order_part}{timing_mode}.json"
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


def import_attack(spec: BenchmarkSpec, library: str) -> Callable[..., Any]:
    try:
        if library == "fastmob":
            function_names = {
                "LocationAttack": "location_risk",
                "LocationSequenceAttack": "location_sequence_risk",
                "LocationTimeAttack": "location_time_risk",
                "UniqueLocationAttack": "unique_location_risk",
                "LocationFrequencyAttack": "location_frequency_risk",
                "LocationProbabilityAttack": "location_probability_risk",
                "LocationProportionAttack": "location_proportion_risk",
                "HomeWorkAttack": "home_work_risk",
            }
            return functools.partial(
                getattr(importlib.import_module("fastmob.privacy"), function_names[spec.class_name]), **spec.init_kwargs
            )
        module = importlib.import_module("skmob.privacy.attacks")
        attack = getattr(module, spec.class_name)(**spec.init_kwargs)
    except Exception as exc:
        raise SkippedAttack(f"attack setup failed: {exc}") from exc
    return attack.assess_risk


def load_privacy_toy_pandas(data_path: Path):
    import pandas as pd

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def repeat_privacy_toy_pandas(df: Any, repeat_factor: int) -> Any:
    if repeat_factor == 1:
        return df.copy()

    import pandas as pd

    max_uid = int(df["uid"].max())
    pieces = []
    for repeat_index in range(repeat_factor):
        piece = df.copy()
        piece["uid"] = piece["uid"] + (repeat_index * max_uid)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def canonicalize_privacy_h3_cells(df: Any, resolution: int = PRIVACY_H3_RESOLUTION) -> Any:
    """Replace privacy benchmark coordinates with their H3 cell centers.

    This makes Fastmob's default H3 equivalence relation identical to
    scikit-mobility's exact-coordinate relation for fair comparisons.
    """
    import h3

    lats = df["lat"].to_list()
    lngs = df["lng"].to_list()
    centers = [h3.cell_to_latlng(h3.latlng_to_cell(lat, lng, resolution)) for lat, lng in zip(lats, lngs)]
    center_lats, center_lngs = zip(*centers) if centers else ([], [])
    if type(df).__module__.startswith("polars"):
        import polars as pl

        return df.with_columns(pl.Series("lat", center_lats), pl.Series("lng", center_lngs))
    return df.assign(lat=center_lats, lng=center_lngs)


def load_privacy_toy_polars(data_path: Path, repeat_factor: int):
    import polars as pl

    return pl.from_pandas(repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), repeat_factor))


def make_skmob_tdf(skmob_module: Any, df: Any) -> Any:
    return skmob_module.TrajDataFrame(df.copy(), latitude="lat", longitude="lng", datetime="datetime", user_id="uid")


def call_benchmark_func(func: Callable[..., Any], input_value: Any, kwargs: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        if "show_progress" in inspect.signature(func).parameters:
            kwargs = {**kwargs, "show_progress": False}
        return func(input_value, **kwargs)


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


def run_memory_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    import memray

    peak_memory_mb: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp_path = f.name
        os.unlink(tmp_path)
        try:
            with memray.Tracker(tmp_path, native_traces=False):
                call_benchmark_func(func, make_input(), kwargs)
            reader = memray.FileReader(tmp_path)
            peak_bytes = sum(record.size for record in reader.get_high_watermark_allocation_records(merge_threads=True))
            peak_mb = peak_bytes / (1024 * 1024)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
        peak_memory_mb.append(peak_mb)
        print(f"    Round {i + 1}: {peak_mb:.4f} MB peak (memray)")

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


def run_profiled_call(
    func: Callable[..., Any],
    make_input: Callable[[], Any],
    kwargs: dict[str, Any],
    *,
    profile: str,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    if profile == "memory":
        return run_memory_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)
    return run_timed_call(func, make_input, kwargs, iterations=iterations, sleep_seconds=sleep_seconds)


def empty_result(status: str, reason: str, profile: str = "speed") -> dict[str, Any]:
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
    return empty_result("skipped", reason, profile)


def error_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_result("error", reason, profile)


def benchmark_attack(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    input_order: str = "raw",
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = import_attack(spec, library)
    except SkippedAttack as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    assess_kwargs = dict(spec.assess_kwargs)

    try:
        return run_profiled_call(
            func,
            make_input,
            assess_kwargs,
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_fastmob(
    df: Any,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    input_order: str = "raw",
) -> dict[str, Any]:
    print(f"\nPrivacy toy ({len(df)} rows)")
    return {
        "rows": len(df),
        "metrics": {
            spec.name: benchmark_attack(
                spec,
                "fastmob",
                lambda df=df: df,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                profile=profile,
                input_order=input_order,
            )
            for spec in PRIVACY_ATTACKS
        },
    }


def benchmark_skmob(
    raw_df: Any,
    skmob_module: Any,
    *,
    timing_mode: str,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    print(f"\nPrivacy toy ({len(raw_df)} rows)")
    if timing_mode == "prebuilt_tdf":
        tdf = make_skmob_tdf(skmob_module, raw_df)

        def make_input() -> Any:
            return tdf

    else:

        def make_input() -> Any:
            return make_skmob_tdf(skmob_module, raw_df)

    return {
        "rows": len(raw_df),
        "metrics": {
            spec.name: benchmark_attack(
                spec,
                "skmob",
                make_input,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                profile=profile,
            )
            for spec in PRIVACY_ATTACKS
        },
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
        "suite": "privacy",
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
        "repeat_dataset": args.repeat_dataset,
        "input_order": args.input_order,
        "input_cache_path": None if input_cache_path is None else str(input_cache_path),
        "input_cache_status": input_cache_status,
        "h3_resolution": PRIVACY_H3_RESOLUTION,
        **({"memory_method": "memray_peak_heap"} if args.profile == "memory" else {}),
    }


def load_privacy_toy_for_order(
    *,
    data_path: Path,
    backend: str,
    input_order: str,
    input_cache_dir: Path,
    repeat_dataset: int,
) -> tuple[Any, Path | None, str]:
    if input_order == "raw":
        if backend == "polars":
            return load_privacy_toy_polars(data_path, repeat_dataset), None, "not_applicable"
        return repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), repeat_dataset), None, "not_applicable"

    sorted_input = load_or_create_sorted_input(
        cache_dir=input_cache_dir,
        suite="privacy",
        backend=backend,
        data_path=data_path,
        load_raw=(
            lambda: (
                load_privacy_toy_polars(data_path, repeat_dataset)
                if backend == "polars"
                else repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), repeat_dataset)
            )
        ),
        uid_col="uid",
        datetime_col="datetime",
        repeat_factor=repeat_dataset,
    )
    return sorted_input.data, sorted_input.path, sorted_input.status


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}.")

    if args.library == "fastmob":
        selected_backend = backend or args.backend
        if selected_backend == "both":
            raise ValueError("run_suite requires a concrete backend when library is fastmob")
        if selected_backend == "pandas":
            input_type = "pandas.DataFrame"
        else:
            input_type = "polars.DataFrame"
        print(f"Loading {args.input_order} privacy toy into {selected_backend} from {data_path}...")
        df, input_cache_path, input_cache_status = load_privacy_toy_for_order(
            data_path=data_path,
            backend=selected_backend,
            input_order=args.input_order,
            input_cache_dir=Path(args.input_cache_dir),
            repeat_dataset=args.repeat_dataset,
        )
        df = canonicalize_privacy_h3_cells(df)
        metadata = build_metadata(
            args,
            input_type=input_type,
            timing_mode="measure_only",
            backend=selected_backend,
            input_cache_path=input_cache_path,
            input_cache_status=input_cache_status,
        )
        return {
            "metadata": metadata,
            "results": [
                benchmark_fastmob(
                    df,
                    iterations=args.iterations,
                    sleep_seconds=args.sleep_seconds,
                    profile=args.profile,
                    input_order=args.input_order,
                )
            ],
        }

    print(f"Loading {args.input_order} privacy toy into pandas from {data_path}...")
    raw_df, input_cache_path, input_cache_status = load_privacy_toy_for_order(
        data_path=data_path,
        backend="pandas",
        input_order=args.input_order,
        input_cache_dir=Path(args.input_cache_dir),
        repeat_dataset=args.repeat_dataset,
    )
    raw_df = canonicalize_privacy_h3_cells(raw_df)
    try:
        skmob_module = importlib.import_module("skmob")
    except Exception as exc:
        raise SystemExit(f"Unable to import original skmob: {exc}") from exc

    metadata = build_metadata(
        args,
        input_type="skmob.TrajDataFrame",
        timing_mode=args.timing_mode,
        backend=None,
        input_cache_path=input_cache_path,
        input_cache_status=input_cache_status,
    )
    return {
        "metadata": metadata,
        "results": [
            benchmark_skmob(
                raw_df,
                skmob_module,
                timing_mode=args.timing_mode,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                profile=args.profile,
            )
        ],
    }


def concrete_backends(args: argparse.Namespace) -> Iterable[str | None]:
    if args.library != "fastmob":
        return (None,)
    if args.backend == "both":
        return ("pandas", "polars")
    return (args.backend,)


def concrete_input_orders(args: argparse.Namespace) -> Iterable[str]:
    if args.input_order == "both":
        return ("raw", "sorted")
    return (args.input_order,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone privacy attack speed benchmarks.")
    parser.add_argument("--library", choices=["fastmob", "skmob"], required=True)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--timing-mode", choices=["prebuilt_tdf", "workflow_tdf"], default="prebuilt_tdf")
    parser.add_argument("--input-order", choices=["raw", "sorted", "both"], default="raw")
    parser.add_argument("--input-cache-dir", type=Path, default=DEFAULT_INPUT_CACHE_DIR)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument(
        "--repeat-dataset",
        type=positive_int,
        default=1000,
        help="Repeat the privacy toy dataset this many times, forcing unique user ids for each repeat.",
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
                order_args.input_order,
                order_args.profile,
            )
            write_json(payload, output_path)
            print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
