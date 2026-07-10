"""Standalone MarkovDiaryGenerator speed benchmark.

Run from the repository root, for example:

    python benchmarks/markov_diary_speed_suite.py --n-agents 500
    python benchmarks/markov_diary_speed_suite.py --n-agents 100 500 --diary-length 168
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402
from benchmarks.models.speed_suite import nonnegative_float, positive_int, write_json  # noqa: E402

MODEL_SEED = 2
MODEL_START = pd.Timestamp("2020-01-01 08:00:00")
DEFAULT_AGENT_COUNTS = [500]
DEFAULT_DIARY_LENGTH = 24
DEFAULT_FIT_INDIVIDUALS = 3


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    build_call: Callable[[Any, int, int, int], Callable[[], Any]]


def load_diary_training(reference_dir: Path) -> pd.DataFrame:
    diary_path = reference_dir / "diary_training.parquet"
    if not diary_path.exists():
        raise SystemExit(
            f"Markov diary benchmark input not found at {diary_path}. "
            "Run 'bash scripts/populate_skmob_cache.sh --datasets models' first."
        )
    return pd.read_parquet(diary_path)


def fit_diary(diary_training: pd.DataFrame, n_individuals: int) -> Any:
    from fkmob.models import MarkovDiaryGenerator

    mdg = MarkovDiaryGenerator()
    mdg.fit(diary_training.copy(), n_individuals, lid="cluster")
    return mdg


def build_generate_public_call(mdg: Any, n_agents: int, diary_length: int, seed: int) -> Callable[[], int]:
    def _call() -> int:
        rows = 0
        for agent_id in range(n_agents):
            diary = mdg.generate(diary_length, MODEL_START, random_state=seed + agent_id)
            rows += len(diary)
        return rows

    return _call


def build_fit_call(diary_training: pd.DataFrame, n_individuals: int) -> Callable[[], Any]:
    return lambda: fit_diary(diary_training, n_individuals)


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("generate_public", build_generate_public_call),
)


def summarize_times(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"average_seconds": None, "minimum_seconds": None}
    return {"average_seconds": sum(times) / len(times), "minimum_seconds": min(times)}


def run_timed_call(func: Callable[[], Any], *, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    print("    Warming up...")
    func()

    times: list[float] = []
    for index in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        gc.collect()
        start = time.perf_counter()
        func()
        duration = time.perf_counter() - start
        times.append(duration)
        print(f"    Round {index + 1}: {duration:.4f} seconds")

    summary = summarize_times(times)
    print(f"    Average Time: {summary['average_seconds']:.4f} s")
    print(f"    Minimum Time: {summary['minimum_seconds']:.4f} s")
    return {"status": "ok", "times_seconds": times, "iterations_completed": len(times), **summary}


def benchmark_fit(
    diary_training: pd.DataFrame,
    *,
    fit_individuals: int,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print("  fit")
    return run_timed_call(
        build_fit_call(diary_training, fit_individuals),
        iterations=iterations,
        sleep_seconds=sleep_seconds,
    )


def benchmark_generation(
    spec: BenchmarkSpec,
    mdg: Any,
    *,
    n_agents: int,
    diary_length: int,
    iterations: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    return run_timed_call(
        spec.build_call(mdg, n_agents, diary_length, MODEL_SEED),
        iterations=iterations,
        sleep_seconds=sleep_seconds,
    )


def selected_specs(args: argparse.Namespace) -> tuple[BenchmarkSpec, ...]:
    requested = set(args.metrics)
    return tuple(spec for spec in BENCHMARKS if spec.name in requested)


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "markov_diary",
        "library": "fkmob",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(args.reference_dir),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "n_agents": args.n_agents,
        "diary_length": args.diary_length,
        "fit_individuals": args.fit_individuals,
        "metrics": args.metrics,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    diary_training = load_diary_training(Path(args.reference_dir))
    fitted_diary = fit_diary(diary_training, args.fit_individuals)

    results = []
    if "fit" in args.metrics:
        results.append(
            {
                "benchmark_group": "fit",
                "label": f"{args.fit_individuals} individuals",
                "n_agents": None,
                "diary_length": None,
                "metrics": {
                    "fit": benchmark_fit(
                        diary_training,
                        fit_individuals=args.fit_individuals,
                        iterations=args.iterations,
                        sleep_seconds=args.sleep_seconds,
                    )
                },
            }
        )

    generation_specs = selected_specs(args)
    for n_agents in args.n_agents:
        results.append(
            {
                "benchmark_group": "generate",
                "label": f"{n_agents} agents / {args.diary_length} hours",
                "n_agents": n_agents,
                "diary_length": args.diary_length,
                "metrics": {
                    spec.name: benchmark_generation(
                        spec,
                        fitted_diary,
                        n_agents=n_agents,
                        diary_length=args.diary_length,
                        iterations=args.iterations,
                        sleep_seconds=args.sleep_seconds,
                    )
                    for spec in generation_specs
                },
            }
        )

    return {"metadata": build_metadata(args), "results": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MarkovDiaryGenerator benchmarks.")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--n-agents", type=positive_int, nargs="+", default=DEFAULT_AGENT_COUNTS)
    parser.add_argument("--diary-length", type=positive_int, default=DEFAULT_DIARY_LENGTH)
    parser.add_argument("--fit-individuals", type=positive_int, default=DEFAULT_FIT_INDIVIDUALS)
    parser.add_argument(
        "--metrics",
        choices=["fit", *[spec.name for spec in BENCHMARKS]],
        nargs="+",
        default=["generate_public"],
        help="Metrics to run. 'generate_public' loops over MarkovDiaryGenerator.generate for each agent.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    output_path = Path(args.output_dir) / "fkmob_markov_diary_speed.json"
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
