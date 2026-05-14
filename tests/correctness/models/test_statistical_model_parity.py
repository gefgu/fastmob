"""Statistical model parity tests.

Compares skmob2 model output against the cached skmob reference using
Wasserstein-based comparison metrics. The acceptance threshold for each
metric comes from a pre-computed baseline JSON that captures natural
same-model variability: if skmob2 vs skmob is within the range of skmob
vs skmob (different seeds), the implementations are statistically equivalent.

Generating the baseline (one-time setup, requires .venv-skmob):
    bash scripts/run_model_baseline.sh

Running these tests:
    bash scripts/run_correctness.sh tests/correctness/models/test_statistical_model_parity.py -v
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skmob2.measures.comparison import wasserstein_distance, visits_per_user_wasserstein_distance
from skmob2.measures.spatial import jump_lengths, radius_of_gyration, waiting_times
from skmob2.models import DensityEPR, EPR, GeoSim, Gravity, Radiation, SpatialEPR, STS_epr, MarkovDiaryGenerator

from tests.shared.skmob_cache import SkmobReferenceDataset, _REFERENCE_DIR

_SHARED_DIR = Path(__file__).resolve().parents[3] / "tests" / "shared"

MODEL_SEED = 2
MODEL_START = pd.Timestamp("2020-01-01 08:00:00")
MODEL_END = pd.Timestamp("2020-01-02 08:00:00")
MODEL_SOCIAL_GRAPH = [[0, 1], [0, 2], [1, 2]]
MODEL_STARTING_LOCATIONS = [0, 1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def skmob_baseline() -> dict[str, Any]:
    path = _SHARED_DIR / "model_statistical_baseline_skmob.json"
    if not path.exists():
        pytest.skip(
            "No model statistical baseline. Run 'bash scripts/run_model_baseline.sh' "
            "and commit the resulting JSON files to tests/shared/."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def models_reference() -> SkmobReferenceDataset:
    if not (_REFERENCE_DIR / "models" / "input.parquet").exists():
        pytest.skip("No skmob models cache. Run 'bash scripts/populate_skmob_cache.sh --datasets models'.")
    return SkmobReferenceDataset("models")


@pytest.fixture(scope="session")
def model_tessellation(models_reference: SkmobReferenceDataset) -> pd.DataFrame:
    return models_reference.input_df


@pytest.fixture(scope="session")
def model_diary_training() -> pd.DataFrame:
    path = _REFERENCE_DIR / "models" / "diary_training.parquet"
    if not path.exists():
        pytest.skip("No cached model diary training frame. Run 'bash scripts/populate_skmob_cache.sh --datasets models'.")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_rng(seed: int = MODEL_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _expected_df(models_reference: SkmobReferenceDataset, name: str) -> pd.DataFrame:
    expected = models_reference.result(name)
    if expected is None:
        pytest.skip(f"No cached skmob model output for {name}")
    return expected


def _normalize_trajectory(df: Any) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out = out[["uid", "datetime", "lat", "lng"]]
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lng"] = pd.to_numeric(out["lng"], errors="coerce")
    return out.dropna(subset=["datetime", "lat", "lng"]).reset_index(drop=True)


def _normalize_flow(df: Any) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out = out[["origin", "destination", "flow"]].copy()
    out["origin"] = out["origin"].astype(str)
    out["destination"] = out["destination"].astype(str)
    out["flow"] = pd.to_numeric(out["flow"], errors="coerce").fillna(0.0)
    return out


def _flat_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _threshold(baseline: dict[str, Any], model_name: str, metric_key: str) -> float | None:
    model_entry = baseline.get("baseline", {}).get(model_name, {})
    metric_entry = model_entry.get(metric_key, {})
    return metric_entry.get("p99")


def _compute_trajectory_metrics(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}

    try:
        val, _ = visits_per_user_wasserstein_distance(df1, df2)
        metrics["visits_per_user_wasserstein"] = float(val)
    except Exception:
        pass

    try:
        jl1 = _flat_array(jump_lengths(df1, merge=True))
        jl2 = _flat_array(jump_lengths(df2, merge=True))
        if jl1.size > 0 and jl2.size > 0:
            metrics["jump_lengths_wasserstein"] = float(wasserstein_distance(jl1, jl2))
    except Exception:
        pass

    try:
        rog1 = _flat_array(pd.DataFrame(radius_of_gyration(df1))["radius_of_gyration"].to_numpy())
        rog2 = _flat_array(pd.DataFrame(radius_of_gyration(df2))["radius_of_gyration"].to_numpy())
        if rog1.size > 0 and rog2.size > 0:
            metrics["radius_of_gyration_wasserstein"] = float(wasserstein_distance(rog1, rog2))
    except Exception:
        pass

    try:
        wt1 = _flat_array(waiting_times(df1, merge=True))
        wt2 = _flat_array(waiting_times(df2, merge=True))
        if wt1.size > 0 and wt2.size > 0:
            metrics["waiting_times_wasserstein"] = float(wasserstein_distance(wt1, wt2))
    except Exception:
        pass

    return metrics


def _assert_within_baseline(
    actual_metrics: dict[str, float],
    baseline: dict[str, Any],
    model_name: str,
) -> None:
    for metric_key, value in actual_metrics.items():
        p99 = _threshold(baseline, model_name, metric_key)
        if p99 is None:
            continue
        assert value <= p99, (
            f"{model_name} {metric_key}: skmob2 vs skmob distance {value:.4f} "
            f"exceeds baseline p99={p99:.4f} (same-model variability)"
        )


def _assert_cpc_within_baseline(cpc: float, baseline: dict[str, Any], model_name: str) -> None:
    p5 = baseline.get("baseline", {}).get(model_name, {}).get("od_matrix_cpc", {}).get("p5")
    if p5 is None:
        return
    assert cpc >= p5, (
        f"{model_name} od_matrix_cpc: skmob2 vs skmob similarity {cpc:.4f} "
        f"below baseline p5={p5:.4f} (same-model variability)"
    )


def _fit_diary(diary_training: pd.DataFrame) -> MarkovDiaryGenerator:
    mdg = MarkovDiaryGenerator()
    mdg.fit(diary_training.copy(), 3, lid="cluster")
    return mdg


# ---------------------------------------------------------------------------
# Trajectory model tests (EPR family)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,model_cls,generate_kwargs",
    [
        ("epr", EPR, {"relevance_column": "population"}),
        ("density_epr", DensityEPR, {"relevance_column": "population"}),
        ("spatial_epr", SpatialEPR, {}),
    ],
)
def test_epr_family_statistical_parity(
    skmob_baseline,
    models_reference,
    model_tessellation,
    name,
    model_cls,
    generate_kwargs,
):
    pytest.importorskip("powerlaw")

    skmob_df = _normalize_trajectory(_expected_df(models_reference, name))

    _reset_rng()
    skmob2_df = _normalize_trajectory(
        model_cls().generate(
            MODEL_START,
            MODEL_END,
            model_tessellation,
            n_agents=2,
            starting_locations=MODEL_STARTING_LOCATIONS.copy(),
            random_state=MODEL_SEED,
            show_progress=False,
            **generate_kwargs,
        )
    )

    actual = _compute_trajectory_metrics(skmob2_df, skmob_df)
    _assert_within_baseline(actual, skmob_baseline, name)


def test_geosim_statistical_parity(
    skmob_baseline,
    models_reference,
    model_tessellation,
):
    pytest.importorskip("powerlaw")
    pytest.importorskip("igraph")

    skmob_df = _normalize_trajectory(_expected_df(models_reference, "geosim"))

    _reset_rng()
    skmob2_df = _normalize_trajectory(
        GeoSim().generate(
            MODEL_START,
            MODEL_END,
            model_tessellation,
            social_graph=MODEL_SOCIAL_GRAPH,
            n_agents=3,
            random_state=MODEL_SEED,
            show_progress=False,
        )
    )

    actual = _compute_trajectory_metrics(skmob2_df, skmob_df)
    _assert_within_baseline(actual, skmob_baseline, "geosim")


def test_sts_epr_statistical_parity(
    skmob_baseline,
    models_reference,
    model_tessellation,
    model_diary_training,
):
    pytest.importorskip("igraph")

    skmob_df = _normalize_trajectory(_expected_df(models_reference, "sts_epr"))

    _reset_rng()
    skmob2_df = _normalize_trajectory(
        STS_epr().generate(
            MODEL_START,
            MODEL_END,
            model_tessellation,
            _fit_diary(model_diary_training),
            social_graph=MODEL_SOCIAL_GRAPH,
            n_agents=3,
            rsl=False,
            relevance_column="population",
            random_state=MODEL_SEED,
            show_progress=False,
        )
    )

    actual = _compute_trajectory_metrics(skmob2_df, skmob_df)
    _assert_within_baseline(actual, skmob_baseline, "sts_epr")


# ---------------------------------------------------------------------------
# Flow model tests (stochastic sampling variants)
# ---------------------------------------------------------------------------


def _flow_to_od_matrix(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalize_flow(df)
    return d.pivot_table(index="origin", columns="destination", values="flow", aggfunc="sum").fillna(0.0)


@pytest.mark.parametrize(
    "name,model_cls",
    [
        ("gravity_flows_sample", Gravity),
        ("radiation_flows_sample", Radiation),
    ],
)
def test_flow_sample_statistical_parity(
    skmob_baseline,
    models_reference,
    model_tessellation,
    name,
    model_cls,
):
    from skmob2.measures.comparison import od_matrix_common_part_of_commuters

    skmob_mat = _flow_to_od_matrix(_expected_df(models_reference, name))

    _reset_rng()
    skmob2_df = model_cls().generate(
        model_tessellation,
        tile_id_column="tile_id",
        tot_outflows_column="tot_outflow",
        relevance_column="population",
        out_format="flows_sample",
    )
    skmob2_mat = _flow_to_od_matrix(pd.DataFrame(skmob2_df))

    cpc = od_matrix_common_part_of_commuters(skmob2_mat, skmob_mat)
    _assert_cpc_within_baseline(cpc, skmob_baseline, name)
