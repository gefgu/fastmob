"""Standalone privacy attack speed benchmark suite.

Run from the repository root, for example:

    python tests/benchmarks/speed_privacy_suite.py --library skmob2 --backend both
    python tests/benchmarks/speed_privacy_suite.py --library skmob
"""

from __future__ import annotations

import argparse
import importlib
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
DEFAULT_DATA_PATH = REPO_ROOT / "scikit-mobility" / "examples" / "privacy_toy.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


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


class SkippedAttack(Exception):
    """Raised when an attack cannot be benchmarked in the selected library."""


def summarize_times(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"average_seconds": None, "minimum_seconds": None}
    return {"average_seconds": sum(times) / len(times), "minimum_seconds": min(times)}


def build_output_path(output_dir: Path, library: str, timing_mode: str, backend: str | None = None) -> Path:
    if library == "skmob2":
        if backend is None or backend == "both":
            raise ValueError("skmob2 output path requires a concrete backend")
        filename = f"skmob2_privacy_speed_{backend}.json"
    else:
        filename = f"skmob_privacy_speed_{timing_mode}.json"
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
    module_path = "skmob2.privacy.attacks" if library == "skmob2" else "skmob.privacy.attacks"
    try:
        module = importlib.import_module(module_path)
        attack_cls = getattr(module, spec.class_name)
        attack = attack_cls(**spec.init_kwargs)
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


def load_privacy_toy_polars(data_path: Path, repeat_factor: int):
    import polars as pl

    return pl.from_pandas(repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), repeat_factor))


def make_skmob_tdf(skmob_module: Any, df: Any) -> Any:
    return skmob_module.TrajDataFrame(df.copy(), latitude="lat", longitude="lng", datetime="datetime", user_id="uid")


def call_benchmark_func(func: Callable[..., Any], input_value: Any, kwargs: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        return func(input_value, show_progress=False, **kwargs)


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


def empty_result(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def skipped_result(reason: str) -> dict[str, Any]:
    return empty_result("skipped", reason)


def error_result(reason: str) -> dict[str, Any]:
    return empty_result("error", reason)


def benchmark_attack(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = import_attack(spec, library)
    except SkippedAttack as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc))

    try:
        return run_timed_call(
            func,
            make_input,
            spec.assess_kwargs,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:
        print(f"    error: {exc}")
        return error_result(str(exc))


def benchmark_skmob2(df: Any, *, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    print(f"\nPrivacy toy ({len(df)} rows)")
    return {
        "rows": len(df),
        "metrics": {
            spec.name: benchmark_attack(
                spec,
                "skmob2",
                lambda df=df: df,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
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
            )
            for spec in PRIVACY_ATTACKS
        },
    }


def build_metadata(args: argparse.Namespace, *, input_type: str, timing_mode: str, backend: str | None) -> dict[str, Any]:
    return {
        "suite": "privacy",
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
        "repeat_factor": args.repeat_factor,
    }


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}.")

    if args.library == "skmob2":
        selected_backend = backend or args.backend
        if selected_backend == "both":
            raise ValueError("run_suite requires a concrete backend when library is skmob2")
        if selected_backend == "pandas":
            print(f"Loading privacy toy into pandas from {data_path}...")
            df = repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), args.repeat_factor)
            input_type = "pandas.DataFrame"
        else:
            print(f"Loading privacy toy into Polars from {data_path}...")
            df = load_privacy_toy_polars(data_path, args.repeat_factor)
            input_type = "polars.DataFrame"
        metadata = build_metadata(args, input_type=input_type, timing_mode="measure_only", backend=selected_backend)
        return {"metadata": metadata, "results": [benchmark_skmob2(df, iterations=args.iterations, sleep_seconds=args.sleep_seconds)]}

    print(f"Loading privacy toy into pandas from {data_path}...")
    raw_df = repeat_privacy_toy_pandas(load_privacy_toy_pandas(data_path), args.repeat_factor)
    try:
        skmob_module = importlib.import_module("skmob")
    except Exception as exc:
        raise SystemExit(f"Unable to import original skmob: {exc}") from exc

    metadata = build_metadata(args, input_type="skmob.TrajDataFrame", timing_mode=args.timing_mode, backend=None)
    return {
        "metadata": metadata,
        "results": [
            benchmark_skmob(
                raw_df,
                skmob_module,
                timing_mode=args.timing_mode,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
            )
        ],
    }


def concrete_backends(args: argparse.Namespace) -> Iterable[str | None]:
    if args.library != "skmob2":
        return (None,)
    if args.backend == "both":
        return ("pandas", "polars")
    return (args.backend,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone privacy attack speed benchmarks.")
    parser.add_argument("--library", choices=["skmob2", "skmob"], required=True)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--timing-mode", choices=["prebuilt_tdf", "workflow_tdf"], default="prebuilt_tdf")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--repeat-factor", type=positive_int, default=1)
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
