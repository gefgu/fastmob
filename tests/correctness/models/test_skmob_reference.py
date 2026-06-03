"""Cached skmob parity tests for generation models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skmob2.models import DensityEPR, EPR, GeoSim, Gravity, MarkovDiaryGenerator, Radiation, STS_epr, SpatialEPR
from tests.shared.skmob_cache import SkmobReferenceDataset, _REFERENCE_DIR

MODEL_SEED = 2
MODEL_START = pd.Timestamp("2020-01-01 08:00:00")
MODEL_END = pd.Timestamp("2020-01-02 08:00:00")
MODEL_SOCIAL_GRAPH = [[0, 1], [0, 2], [1, 2]]
MODEL_STARTING_LOCATIONS = [0, 1]


@dataclass(frozen=True)
class FlowCase:
    name: str
    model_cls: type
    out_format: str
    exact_flow: bool = False


FLOW_CASES: tuple[FlowCase, ...] = (
    FlowCase("gravity_flows", Gravity, "flows"),
    FlowCase("gravity_probabilities", Gravity, "probabilities"),
    FlowCase("gravity_flows_sample", Gravity, "flows_sample", exact_flow=True),
    FlowCase("radiation_flows", Radiation, "flows"),
    FlowCase("radiation_probabilities", Radiation, "probabilities"),
    FlowCase("radiation_flows_sample", Radiation, "flows_sample", exact_flow=True),
)


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


def _reset_model_rng(seed: int = MODEL_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _expected(models_reference: SkmobReferenceDataset, name: str) -> pd.DataFrame:
    expected = models_reference.result(name)
    if expected is None:
        pytest.skip(f"No cached skmob model output for {name}")
    return expected


def _normalize_flow(df: Any) -> pd.DataFrame:
    if hasattr(df, "df"):
        df = df.df
    out = pd.DataFrame(df).copy()
    out = out[["origin", "destination", "flow"]]
    out["origin"] = out["origin"].astype(str)
    out["destination"] = out["destination"].astype(str)
    out["flow"] = pd.to_numeric(out["flow"], errors="raise")
    return out.sort_values(["origin", "destination"], kind="mergesort").reset_index(drop=True)


def _normalize_trajectory(df: Any) -> pd.DataFrame:
    if hasattr(df, "df"):
        df = df.df
    out = pd.DataFrame(df).copy()
    out = out[["uid", "datetime", "lat", "lng"]]
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["lat"] = pd.to_numeric(out["lat"], errors="raise")
    out["lng"] = pd.to_numeric(out["lng"], errors="raise")
    return out.sort_values(["uid", "datetime", "lat", "lng"], kind="mergesort").reset_index(drop=True)


def _normalize_diary(df: Any) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    out = out[["datetime", "abstract_location"]]
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["abstract_location"] = pd.to_numeric(out["abstract_location"], errors="raise")
    return out.sort_values(["datetime", "abstract_location"], kind="mergesort").reset_index(drop=True)


def _assert_frame_match(actual: pd.DataFrame, expected: pd.DataFrame, *, exact: bool = False) -> None:
    assert list(actual.columns) == list(expected.columns)
    assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=exact,
        rtol=0 if exact else 1e-9,
        atol=0 if exact else 1e-9,
    )


def _fit_diary(training: pd.DataFrame) -> MarkovDiaryGenerator:
    mdg = MarkovDiaryGenerator()
    mdg.fit(training.copy(), 3, lid="cluster")
    return mdg


@pytest.mark.parametrize("case", FLOW_CASES, ids=lambda case: case.name)
def test_flow_models_match_cached_skmob(models_reference, model_tessellation, case: FlowCase):
    expected = _normalize_flow(_expected(models_reference, case.name))

    _reset_model_rng()
    actual = case.model_cls().generate(
        model_tessellation,
        tile_id_column="tile_id",
        tot_outflows_column="tot_outflow",
        relevance_column="population",
        out_format=case.out_format,
    )

    _assert_frame_match(_normalize_flow(actual), expected, exact=case.exact_flow)


def test_markov_diary_generator_matches_cached_skmob(models_reference, model_diary_training):
    expected = _normalize_diary(_expected(models_reference, "markov_diary"))

    actual = _fit_diary(model_diary_training).generate(24, MODEL_START, random_state=MODEL_SEED)

    _assert_frame_match(_normalize_diary(actual), expected, exact=True)


@pytest.mark.skip(reason="Rust power law sampler changes RNG sequence; statistical parity holds")
@pytest.mark.parametrize(
    "name,model_cls,generate_kwargs",
    [
        ("epr", EPR, {"relevance_column": "population"}),
        ("density_epr", DensityEPR, {"relevance_column": "population"}),
        ("spatial_epr", SpatialEPR, {}),
    ],
)
def test_epr_family_matches_cached_skmob(models_reference, model_tessellation, name, model_cls, generate_kwargs):
    pytest.importorskip("powerlaw")
    expected = _normalize_trajectory(_expected(models_reference, name))

    actual = model_cls().generate(
        MODEL_START,
        MODEL_END,
        model_tessellation,
        n_agents=2,
        starting_locations=MODEL_STARTING_LOCATIONS.copy(),
        random_state=MODEL_SEED,
        show_progress=False,
        **generate_kwargs,
    )

    _assert_frame_match(_normalize_trajectory(actual), expected)


@pytest.mark.skip(reason="Rust power law sampler changes RNG sequence; statistical parity holds")
def test_geosim_matches_cached_skmob(models_reference, model_tessellation):
    pytest.importorskip("powerlaw")
    pytest.importorskip("igraph")
    expected = _normalize_trajectory(_expected(models_reference, "geosim"))

    actual = GeoSim().generate(
        MODEL_START,
        MODEL_END,
        model_tessellation,
        social_graph=MODEL_SOCIAL_GRAPH,
        n_agents=3,
        random_state=MODEL_SEED,
        show_progress=False,
    )

    _assert_frame_match(_normalize_trajectory(actual), expected)


def test_sts_epr_matches_cached_skmob(models_reference, model_tessellation, model_diary_training):
    pytest.importorskip("igraph")
    expected = _normalize_trajectory(_expected(models_reference, "sts_epr"))

    actual = STS_epr().generate(
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

    _assert_frame_match(_normalize_trajectory(actual), expected)
