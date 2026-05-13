"""Large-scale generation model speed benchmark suite.

Targets realistic user scenarios: 1 000 – 10 000 tile tessellations, up to 50 000 agents.
For skmob comparison (which cannot handle these scales), run at smaller sizes:

    # skmob2 at full scale
    python tests/benchmarks/speed_models_large_scale.py --library skmob2 --sizes 1000 5000 10000

    # skmob comparison baseline (smaller sizes only)
    python tests/benchmarks/speed_models_large_scale.py --library skmob --sizes 100 200 500

Output files:
    tests/benchmarks/results/skmob2_models_large_scale_speed.json
    tests/benchmarks/results/skmob_models_large_scale_speed.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Re-use all helpers from the existing suite
# ---------------------------------------------------------------------------

_SUITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SUITE_DIR))

from speed_models_suite import (  # noqa: E402
    SkippedBenchmark,
    build_metadata,
    build_output_path,
    expand_tessellation,
    fit_diary,
    import_model_classes,
    load_model_inputs,
    nonnegative_float,
    positive_int,
    prepare_library_inputs,
    run_memory_call,
    run_timed_call,
    size_label,
    to_timestamp,
    write_json,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SIZES = [1000, 5000, 10000]

MODEL_SEED = 2
MODEL_START_EPR = "2020-01-01 08:00:00"
MODEL_END_EPR = "2020-01-08 08:00:00"   # 7-day simulation for EPR
MODEL_START_SOCIAL = "2020-01-01 08:00:00"
MODEL_END_SOCIAL = "2020-01-02 08:00:00"  # 24h for social models (diary-driven)


@dataclass(frozen=True)
class LargeBenchmarkSpec:
    name: str
    kind: str
    kwargs: dict[str, Any]


LARGE_SCALE_BENCHMARKS: tuple[LargeBenchmarkSpec, ...] = (
    # EPR family — 7-day window, increasing agent counts
    LargeBenchmarkSpec("epr_100a",        "epr", {"model": "EPR",        "n_agents": 100,   "relevance_column": "population"}),
    LargeBenchmarkSpec("epr_1000a",       "epr", {"model": "EPR",        "n_agents": 1000,  "relevance_column": "population"}),
    LargeBenchmarkSpec("epr_10000a",      "epr", {"model": "EPR",        "n_agents": 10000, "relevance_column": "population"}),
    LargeBenchmarkSpec("epr_50000a",      "epr", {"model": "EPR",        "n_agents": 50000, "relevance_column": "population"}),
    LargeBenchmarkSpec("density_epr_1000a", "epr", {"model": "DensityEPR", "n_agents": 1000, "relevance_column": "population"}),
    LargeBenchmarkSpec("spatial_epr_1000a", "epr", {"model": "SpatialEPR", "n_agents": 1000}),
    # Social models — 24h window, smaller agent counts (social graph scales quadratically)
    LargeBenchmarkSpec("geosim_20a",      "geosim",  {"n_agents": 20}),
    LargeBenchmarkSpec("geosim_100a",     "geosim",  {"n_agents": 100}),
    LargeBenchmarkSpec("sts_epr_20a",     "sts_epr", {"n_agents": 20,  "relevance_column": "population"}),
    LargeBenchmarkSpec("sts_epr_100a",    "sts_epr", {"n_agents": 100, "relevance_column": "population"}),
)


def _starting_locs(n_agents: int, size: int, rng_seed: int) -> list[int]:
    """Sample n_agents starting locations (with replacement) from [0, size)."""
    rng = np.random.RandomState(rng_seed)
    return rng.choice(size, size=n_agents, replace=True).tolist()


def _social_graph_random(n_agents: int) -> str:
    """For large-scale social models use igraph random graph."""
    return "random"


def build_large_call(
    spec: LargeBenchmarkSpec,
    library: str,
    tessellation: Any,
    diary_training: Any,
    size: int,
) -> Callable[[], Any]:
    classes = import_model_classes(library)

    if spec.kind == "epr":
        model_name = spec.kwargs["model"]
        n_agents = spec.kwargs["n_agents"]
        generate_kwargs: dict[str, Any] = {}
        if "relevance_column" in spec.kwargs:
            generate_kwargs["relevance_column"] = spec.kwargs["relevance_column"]
        start = to_timestamp(MODEL_START_EPR)
        end = to_timestamp(MODEL_END_EPR)
        starting_locs = _starting_locs(n_agents, size, MODEL_SEED)

        def _epr_call() -> Any:
            return classes[model_name]().generate(
                start,
                end,
                tessellation,
                n_agents=n_agents,
                starting_locations=list(starting_locs),
                random_state=MODEL_SEED,
                show_progress=False,
                **generate_kwargs,
            )

        return _epr_call

    if spec.kind == "geosim":
        n_agents = spec.kwargs["n_agents"]
        start = to_timestamp(MODEL_START_SOCIAL)
        end = to_timestamp(MODEL_END_SOCIAL)

        def _geosim_call() -> Any:
            return classes["GeoSim"]().generate(
                start,
                end,
                tessellation,
                social_graph=_social_graph_random(n_agents),
                n_agents=n_agents,
                random_state=MODEL_SEED,
                show_progress=False,
            )

        return _geosim_call

    if spec.kind == "sts_epr":
        n_agents = spec.kwargs["n_agents"]
        start = to_timestamp(MODEL_START_SOCIAL)
        end = to_timestamp(MODEL_END_SOCIAL)

        def _sts_call() -> Any:
            return classes["STS_epr"]().generate(
                start,
                end,
                tessellation,
                fit_diary(classes["MarkovDiaryGenerator"], diary_training),
                social_graph=_social_graph_random(n_agents),
                n_agents=n_agents,
                rsl=False,
                relevance_column=spec.kwargs["relevance_column"],
                random_state=MODEL_SEED,
                show_progress=False,
            )

        return _sts_call

    raise SkippedBenchmark(f"unknown benchmark kind: {spec.kind}")


def benchmark_large_model(
    spec: LargeBenchmarkSpec,
    library: str,
    tessellation: Any,
    diary_training: Any,
    size: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = build_large_call(spec, library, tessellation, diary_training, size)
    except Exception as exc:
        print(f"    skipped: {exc}")
        from speed_models_suite import skipped_result
        return skipped_result(str(exc), profile)

    try:
        if profile == "memory":
            return run_memory_call(func, iterations=iterations, sleep_seconds=sleep_seconds)
        return run_timed_call(func, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:
        print(f"    error: {exc}")
        from speed_models_suite import error_result
        return error_result(str(exc), profile)


def benchmark_large_size(
    library: str,
    base_tessellation: Any,
    diary_training: Any,
    size: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    benchmarks: tuple[LargeBenchmarkSpec, ...] = LARGE_SCALE_BENCHMARKS,
) -> dict[str, Any]:
    size_tessellation = expand_tessellation(base_tessellation, size)
    print(f"\nSize {size_label(size)} ({len(size_tessellation)} locations)")
    try:
        tessellation, diary = prepare_library_inputs(library, size_tessellation, diary_training)
    except SkippedBenchmark as exc:
        from speed_models_suite import skipped_result
        return {
            "size": size,
            "label": size_label(size),
            "locations": len(size_tessellation),
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in benchmarks},
        }

    return {
        "size": size,
        "label": size_label(size),
        "locations": len(size_tessellation),
        "metrics": {
            spec.name: benchmark_large_model(
                spec,
                library,
                tessellation,
                diary,
                size,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in benchmarks
        },
    }


_SKMOB_SKIP_THRESHOLD_S = 1800.0  # 30 minutes


def _estimate_skmob_seconds(spec: LargeBenchmarkSpec, size: int) -> float:
    """Rough upper-bound estimate of skmob wall-clock seconds for one run.

    Fitted from observed timings:
      EPR  N=200  100 agents  7d: 3.15 s  → 31.5 ms/agent
      EPR  N=500  100 agents  7d: 3.44 s  → 34.4 ms/agent
      GeoSim N=200 100 agents 1d: 0.025 s → 0.26 ms/agent
      GeoSim N=500 100 agents 1d: 0.178 s → 1.78 ms/agent  (power law ~N^2.1)
      STS   N=500 100 agents  1d: 1.83 s  → 18.3 ms/agent
    """
    n = spec.kwargs.get("n_agents", 20)
    if spec.kind == "epr":
        # OD rows are lazily cached across agents; per-agent cost grows slowly with N.
        per_agent_s = 0.034 * (1.0 + size / 10_000.0)
        return per_agent_s * n
    if spec.kind == "geosim":
        # Time-step granularity scales super-linearly with N (exponent ~2.1).
        per_agent_s = 2.6e-6 * (size / 200.0) ** 2.1
        return per_agent_s * n
    if spec.kind == "sts_epr":
        # Similar per-agent cost to EPR (24h window, social graph overhead).
        per_agent_s = 0.020 * (1.0 + size / 5_000.0)
        return per_agent_s * n
    return 0.0


def run_large_suite(args: argparse.Namespace) -> dict[str, Any]:
    base_tessellation, diary_training = load_model_inputs(Path(args.reference_dir))

    results = []
    for size in args.sizes:
        if args.library == "skmob":
            kept, skipped = [], []
            for s in LARGE_SCALE_BENCHMARKS:
                est = _estimate_skmob_seconds(s, size)
                if est > _SKMOB_SKIP_THRESHOLD_S:
                    skipped.append((s, est))
                else:
                    kept.append(s)
            if skipped:
                print(f"\nSize {size}: skipping for skmob (estimated > 30 min):")
                for s, est in skipped:
                    print(f"  {s.name}: ~{est/60:.1f} min estimated")
            benchmarks = tuple(kept)
        else:
            benchmarks = LARGE_SCALE_BENCHMARKS

        results.append(
            benchmark_large_size(
                args.library,
                base_tessellation,
                diary_training,
                size,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                benchmarks=benchmarks,
            )
        )
    meta = build_metadata(args)
    meta["suite"] = "models_large_scale"
    return {"metadata": meta, "results": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-scale generation model benchmarks.")
    parser.add_argument("--library", choices=["skmob2", "skmob"], required=True)
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_large_suite(args)
    output_path = build_output_path(Path(args.output_dir), args.library, f"{args.profile}_large_scale")
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
