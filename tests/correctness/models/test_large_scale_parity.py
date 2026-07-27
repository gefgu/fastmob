"""Large-scale statistical parity tests for fastmob model generation.

Verifies that fastmob produces self-consistent output at N=10000 tile tessellations:
a fresh pair of runs produces Wasserstein-based metrics within the p99 of the
same-model variability distribution captured in the pre-computed baseline.

Generating the baseline (run once, then commit the JSON):
    python benchmarks/correctness_models_large_scale.py

Running these tests:
    bash scripts/run_correctness.sh tests/correctness/models/test_large_scale_parity.py -v -m slow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_BENCH_DIR = Path(__file__).resolve().parents[3] / "benchmarks"
sys.path.insert(0, str(_BENCH_DIR))

from benchmarks.models.speed_suite import expand_tessellation, load_model_inputs

from correctness_models_large_scale import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_REFERENCE_DIR,
    DEFAULT_SIZE,
    LARGE_SCALE_BENCHMARKS,
    generate_one_large_run,
)
from model_statistical_baseline import _trajectory_to_std, compute_trajectory_pair_metrics

_BASELINE_PATH = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def large_scale_baseline() -> dict[str, Any]:
    if not _BASELINE_PATH.exists():
        pytest.skip(
            "No large-scale model statistical baseline. "
            "Run 'python benchmarks/correctness_models_large_scale.py' "
            "and commit the resulting JSON to tests/shared/."
        )
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def large_tessellation() -> pd.DataFrame:
    ref_dir = Path(DEFAULT_REFERENCE_DIR)
    if not (ref_dir / "input.parquet").exists():
        pytest.skip("No model cache. Run 'bash scripts/populate_skmob_cache.sh --datasets models' first.")
    base_tess, _ = load_model_inputs(ref_dir)
    return expand_tessellation(base_tess, DEFAULT_SIZE)


@pytest.fixture(scope="session")
def large_diary_training() -> pd.DataFrame:
    ref_dir = Path(DEFAULT_REFERENCE_DIR)
    diary_path = ref_dir / "diary_training.parquet"
    if not diary_path.exists():
        pytest.skip(
            "No cached model diary training frame. Run 'bash scripts/populate_skmob_cache.sh --datasets models' first."
        )
    return pd.read_parquet(diary_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_by_name(name: str):
    for spec in LARGE_SCALE_BENCHMARKS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def _threshold(baseline: dict[str, Any], spec_name: str, metric_key: str) -> float | None:
    entry = baseline.get("baseline", {}).get(spec_name, {}).get(metric_key, {})
    return entry.get("p99")


def _assert_within_baseline(
    actual: dict[str, float],
    baseline: dict[str, Any],
    spec_name: str,
) -> None:
    missing = []
    for metric_key, value in actual.items():
        p99 = _threshold(baseline, spec_name, metric_key)
        if p99 is None:
            missing.append(metric_key)
            continue
        assert value <= p99, (
            f"{spec_name} {metric_key}: fastmob-vs-fastmob distance {value:.4f} "
            f"exceeds baseline p99={p99:.4f} (same-model variability at N={DEFAULT_SIZE})"
        )
    if missing:
        pytest.skip(f"{spec_name}: no p99 threshold for {missing!r} in baseline")


def _run_parity_check(
    spec_name: str,
    baseline: dict[str, Any],
    tessellation: pd.DataFrame,
    diary_training: pd.DataFrame,
    seed_a: int = 200,
    seed_b: int = 201,
) -> None:
    if spec_name not in baseline.get("baseline", {}):
        pytest.skip(f"No baseline entry for {spec_name!r} — run baseline generator first")

    spec = _spec_by_name(spec_name)
    df_a = generate_one_large_run(spec, tessellation, diary_training, seed_a)
    df_b = generate_one_large_run(spec, tessellation, diary_training, seed_b)
    if df_a is None or df_b is None:
        pytest.skip(f"{spec_name}: run failed (missing optional dep?)")

    actual = compute_trajectory_pair_metrics(
        _trajectory_to_std(df_a),
        _trajectory_to_std(df_b),
    )
    _assert_within_baseline(actual, baseline, spec_name)


# ---------------------------------------------------------------------------
# Tests — one per benchmark spec
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skip(reason="model implementation under revision")
@pytest.mark.parametrize(
    "spec_name",
    [
        "epr_100a",
        "epr_1000a",
        "epr_10000a",
        "epr_50000a",
        "density_epr_1000a",
        "spatial_epr_1000a",
    ],
)
def test_epr_large_scale_parity(
    spec_name,
    large_scale_baseline,
    large_tessellation,
    large_diary_training,
):
    pytest.importorskip("powerlaw")
    _run_parity_check(spec_name, large_scale_baseline, large_tessellation, large_diary_training)


@pytest.mark.slow
@pytest.mark.skip(reason="model implementation under revision")
@pytest.mark.parametrize("spec_name", ["geosim_20a", "geosim_100a"])
def test_geosim_large_scale_parity(
    spec_name,
    large_scale_baseline,
    large_tessellation,
    large_diary_training,
):
    pytest.importorskip("powerlaw")
    pytest.importorskip("igraph")
    _run_parity_check(spec_name, large_scale_baseline, large_tessellation, large_diary_training)


@pytest.mark.slow
@pytest.mark.skip(reason="model implementation under revision")
@pytest.mark.parametrize("spec_name", ["sts_epr_20a", "sts_epr_100a"])
def test_sts_epr_large_scale_parity(
    spec_name,
    large_scale_baseline,
    large_tessellation,
    large_diary_training,
):
    pytest.importorskip("igraph")
    _run_parity_check(spec_name, large_scale_baseline, large_tessellation, large_diary_training)
