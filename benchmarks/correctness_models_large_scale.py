"""Large-scale statistical correctness baseline for skmob2 model generation.

Generates N_RUNS independent runs of each model spec at N=10000 tiles, computes all
pairwise Wasserstein-based comparison metrics between runs, and stores the resulting
percentile distributions as a JSON file.

The resulting baseline JSON is committed to tests/shared/ and used by
tests/correctness/models/test_large_scale_parity.py to verify skmob2 self-consistency
at realistic tessellation scale.

Each spec is saved incrementally so partial results are preserved on interruption.

Usage:
    python benchmarks/correctness_models_large_scale.py
    python benchmarks/correctness_models_large_scale.py --size 5000 --output my_baseline.json
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling benchmark modules
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

from model_statistical_baseline import (  # noqa: E402
    compute_trajectory_pair_metrics,
    fit_diary,
    summarize,
    _trajectory_to_std,
)
from speed_models_large_scale import (  # noqa: E402
    LARGE_SCALE_BENCHMARKS,
    LargeBenchmarkSpec,
    MODEL_END_EPR,
    MODEL_END_SOCIAL,
    MODEL_START_EPR,
    MODEL_START_SOCIAL,
)
from benchmarks.models.speed_suite import expand_tessellation, load_model_inputs  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "shared"
DEFAULT_OUTPUT_NAME = "model_statistical_baseline_large_scale.json"
DEFAULT_SIZE = 10000

# Per-spec N_RUNS from {100, 75, 50, 25, 10, 5}.
# Constraint: N_RUNS × seconds_per_run < 1800 s (30 min) for each spec.
LARGE_SPEC_N_RUNS: dict[str, int] = {
    "epr_100a":            100,   # 100 × 2.3 s  ≈  3.8 min
    "epr_1000a":           100,   # 100 × 3.7 s  ≈  6.1 min
    "epr_10000a":           75,   #  75 × 17 s   ≈ 21.6 min
    "epr_50000a":           10,   #  10 × 78 s   ≈ 13 min
    "density_epr_1000a":   100,   # 100 × 3.6 s  ≈  6 min
    "spatial_epr_1000a":   100,   # 100 × 3.7 s  ≈  6.2 min
    "geosim_20a":          100,   # trivially fast
    "geosim_100a":         100,   # trivially fast
    "sts_epr_20a":         100,   # 100 × 2 s    ≈  3.3 min
    "sts_epr_100a":        100,   # 100 × 4.9 s  ≈  8.1 min
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _to_timestamp(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _rng_starting_locs(n_agents: int, n_locs: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    return rng.choice(n_locs, size=n_agents, replace=True).tolist()


def _import_skmob2_models() -> dict[str, Any]:
    mod = importlib.import_module("skmob2.models")
    return {
        "EPR": mod.EPR,
        "DensityEPR": mod.DensityEPR,
        "SpatialEPR": mod.SpatialEPR,
        "GeoSim": mod.GeoSim,
        "STS_epr": mod.STS_epr,
        "MarkovDiaryGenerator": mod.MarkovDiaryGenerator,
    }


def generate_one_large_run(
    spec: LargeBenchmarkSpec,
    tessellation: pd.DataFrame,
    diary_training: pd.DataFrame,
    seed: int,
) -> pd.DataFrame | None:
    """Generate one run of a large-scale spec at a given seed.

    Uses random starting locations sampled from the tessellation (not sequential
    indices) so all agents start at valid tile IDs even when n_agents > n_locs.
    """
    classes = _import_skmob2_models()
    n_locs = len(tessellation)
    n_agents = spec.kwargs["n_agents"] if "n_agents" in spec.kwargs else 20

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if spec.kind == "epr":
                model_name = spec.kwargs["model"]
                generate_kwargs: dict[str, Any] = {}
                if "relevance_column" in spec.kwargs:
                    generate_kwargs["relevance_column"] = spec.kwargs["relevance_column"]
                start = _to_timestamp(MODEL_START_EPR)
                end = _to_timestamp(MODEL_END_EPR)
                starting_locs = _rng_starting_locs(n_agents, n_locs, seed)
                return pd.DataFrame(
                    classes[model_name]().generate(
                        start,
                        end,
                        tessellation,
                        n_agents=n_agents,
                        starting_locations=starting_locs,
                        random_state=seed,
                        show_progress=False,
                        **generate_kwargs,
                    )
                )

            if spec.kind == "geosim":
                importlib.import_module("igraph")
                social_graph = "random"
                start = _to_timestamp(MODEL_START_SOCIAL)
                end = _to_timestamp(MODEL_END_SOCIAL)
                return pd.DataFrame(
                    classes["GeoSim"]().generate(
                        start,
                        end,
                        tessellation,
                        social_graph=social_graph,
                        n_agents=n_agents,
                        random_state=seed,
                        show_progress=False,
                    )
                )

            if spec.kind == "sts_epr":
                importlib.import_module("igraph")
                social_graph = "random"
                diary = fit_diary(classes["MarkovDiaryGenerator"], diary_training)
                start = _to_timestamp(MODEL_START_SOCIAL)
                end = _to_timestamp(MODEL_END_SOCIAL)
                return pd.DataFrame(
                    classes["STS_epr"]().generate(
                        start,
                        end,
                        tessellation,
                        diary,
                        social_graph=social_graph,
                        n_agents=n_agents,
                        rsl=False,
                        relevance_column=spec.kwargs.get("relevance_column", "relevance"),
                        random_state=seed,
                        show_progress=False,
                    )
                )

        except ImportError as exc:
            print(f"    [skip] {spec.name} seed={seed}: missing dep — {exc}")
            return None
        except Exception as exc:
            print(f"    [error] {spec.name} seed={seed}: {type(exc).__name__}: {exc}")
            return None

    return None


# ---------------------------------------------------------------------------
# Baseline computation for a single spec
# ---------------------------------------------------------------------------


def run_large_spec_baseline(
    spec: LargeBenchmarkSpec,
    tessellation: pd.DataFrame,
    diary_training: pd.DataFrame,
    n_runs: int,
) -> dict[str, Any] | None:
    print(f"  Generating {n_runs} runs of {spec.name} (n_agents={spec.kwargs.get('n_agents', 20)})...", flush=True)

    runs: list[pd.DataFrame] = []
    for seed in range(n_runs):
        df = generate_one_large_run(spec, tessellation, diary_training, seed)
        if df is None:
            print(f"    [skip] {spec.name} — aborting after first failure")
            return None
        runs.append(df)
        if (seed + 1) % 10 == 0:
            print(f"    {seed + 1}/{n_runs} runs done", flush=True)

    n_pairs = len(runs) * (len(runs) - 1) // 2
    print(f"  Computing metrics for {n_pairs} pairs...", flush=True)

    std_runs = [_trajectory_to_std(df) for df in runs]
    metric_buckets: dict[str, list[float]] = {}
    for i, j in itertools.combinations(range(len(std_runs)), 2):
        metrics = compute_trajectory_pair_metrics(std_runs[i], std_runs[j])
        for k, v in metrics.items():
            metric_buckets.setdefault(k, []).append(v)

    result = {k: summarize(v) for k, v in metric_buckets.items()}
    for k, v in result.items():
        p50 = v.get("p50")
        p99 = v.get("p99")
        fmt_p50 = f"{p50:.4f}" if p50 is not None else "n/a"
        fmt_p99 = f"{p99:.4f}" if p99 is not None else "n/a"
        print(f"    {k}: p50={fmt_p50} p99={fmt_p99}")
    return result


# ---------------------------------------------------------------------------
# Incremental save helpers
# ---------------------------------------------------------------------------


def _load_existing(output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(output_path: Path, metadata: dict[str, Any], baseline: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "baseline": baseline}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute large-scale statistical baseline for skmob2 model generation."
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Tessellation tile count")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help="Output JSON path",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--specs",
        nargs="+",
        default=None,
        help="Subset of spec names to run (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    base_tessellation, diary_training = load_model_inputs(Path(args.reference_dir))
    tessellation = expand_tessellation(base_tessellation, args.size)
    print(f"Tessellation: {len(tessellation)} tiles (expanded from {len(base_tessellation)})")

    existing = _load_existing(args.output)
    baseline: dict[str, Any] = dict(existing.get("baseline", {}))

    metadata: dict[str, Any] = {
        "library": "skmob2",
        "tessellation_size": len(tessellation),
        "spec_n_runs": LARGE_SPEC_N_RUNS,
        "model_start_epr": MODEL_START_EPR,
        "model_end_epr": MODEL_END_EPR,
        "model_start_social": MODEL_START_SOCIAL,
        "model_end_social": MODEL_END_SOCIAL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    spec_filter = set(args.specs) if args.specs else None
    specs_to_run = [
        s for s in LARGE_SCALE_BENCHMARKS
        if (spec_filter is None or s.name in spec_filter) and s.name not in baseline
    ]

    if not specs_to_run:
        print("All specs already in baseline — nothing to do.")
        _save(args.output, metadata, baseline)
        print(f"Baseline at {args.output}")
        return 0

    for spec in specs_to_run:
        n_runs = LARGE_SPEC_N_RUNS.get(spec.name, 100)
        print(f"\n[{spec.name}]  n_runs={n_runs}")
        result = run_large_spec_baseline(spec, tessellation, diary_training, n_runs)
        if result is not None:
            baseline[spec.name] = result
        _save(args.output, metadata, baseline)
        print(f"  Saved incrementally → {args.output}")

    print(f"\nFinal baseline written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
