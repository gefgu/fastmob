"""Cached skmob parity tests for privacy attacks on the tutorial dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from fastmob.privacy import attacks
from pandas.testing import assert_frame_equal

from tests.shared.skmob_cache import _REFERENCE_DIR, SkmobReferenceDataset


@dataclass(frozen=True)
class PrivacyCase:
    name: str
    attack_cls: type
    init_kwargs: dict[str, Any]
    assess_kwargs: dict[str, Any]


PRIVACY_CASES: tuple[PrivacyCase, ...] = (
    PrivacyCase("location_kl2", attacks.LocationAttack, {"knowledge_length": 2}, {}),
    PrivacyCase("location_kl3", attacks.LocationAttack, {"knowledge_length": 3}, {}),
    PrivacyCase("location_targets_kl3", attacks.LocationAttack, {"knowledge_length": 3}, {"targets": [1, 2]}),
    PrivacyCase(
        "location_force_instances_kl3",
        attacks.LocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("location_sequence_kl2", attacks.LocationSequenceAttack, {"knowledge_length": 2}, {}),
    PrivacyCase("location_sequence_kl3", attacks.LocationSequenceAttack, {"knowledge_length": 3}, {}),
    PrivacyCase(
        "location_sequence_targets_kl3",
        attacks.LocationSequenceAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "location_sequence_force_instances_kl3",
        attacks.LocationSequenceAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("location_time_kl2", attacks.LocationTimeAttack, {"knowledge_length": 2}, {}),
    PrivacyCase(
        "location_time_month_kl2",
        attacks.LocationTimeAttack,
        {"knowledge_length": 2, "time_precision": "Month"},
        {},
    ),
    PrivacyCase(
        "location_time_month_kl3",
        attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {},
    ),
    PrivacyCase(
        "location_time_month_targets_kl3",
        attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "location_time_month_force_instances_kl3",
        attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("unique_location_kl2", attacks.UniqueLocationAttack, {"knowledge_length": 2}, {}),
    PrivacyCase("unique_location_kl3", attacks.UniqueLocationAttack, {"knowledge_length": 3}, {}),
    PrivacyCase(
        "unique_location_targets_kl3",
        attacks.UniqueLocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "unique_location_force_instances_kl3",
        attacks.UniqueLocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("location_frequency_kl2", attacks.LocationFrequencyAttack, {"knowledge_length": 2}, {}),
    PrivacyCase(
        "location_frequency_tol05_kl2",
        attacks.LocationFrequencyAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    PrivacyCase("location_frequency_kl3", attacks.LocationFrequencyAttack, {"knowledge_length": 3}, {}),
    PrivacyCase(
        "location_frequency_targets_kl3",
        attacks.LocationFrequencyAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "location_frequency_force_instances_kl3",
        attacks.LocationFrequencyAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("location_probability_kl2", attacks.LocationProbabilityAttack, {"knowledge_length": 2}, {}),
    PrivacyCase(
        "location_probability_tol05_kl2",
        attacks.LocationProbabilityAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    PrivacyCase("location_probability_kl3", attacks.LocationProbabilityAttack, {"knowledge_length": 3}, {}),
    PrivacyCase(
        "location_probability_targets_kl3",
        attacks.LocationProbabilityAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "location_probability_force_instances_kl3",
        attacks.LocationProbabilityAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("location_proportion_kl2", attacks.LocationProportionAttack, {"knowledge_length": 2}, {}),
    PrivacyCase(
        "location_proportion_tol05_kl2",
        attacks.LocationProportionAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    PrivacyCase("location_proportion_kl3", attacks.LocationProportionAttack, {"knowledge_length": 3}, {}),
    PrivacyCase(
        "location_proportion_targets_kl3",
        attacks.LocationProportionAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    PrivacyCase(
        "location_proportion_force_instances_kl3",
        attacks.LocationProportionAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    PrivacyCase("home_work", attacks.HomeWorkAttack, {}, {}),
    PrivacyCase("home_work_targets", attacks.HomeWorkAttack, {}, {"targets": [1, 2]}),
    PrivacyCase(
        "home_work_force_instances",
        attacks.HomeWorkAttack,
        {},
        {"targets": [1, 2], "force_instances": True},
    ),
)


@pytest.fixture(scope="session")
def privacy_toy_reference() -> SkmobReferenceDataset:
    if not (_REFERENCE_DIR / "privacy_toy" / "input.parquet").exists():
        pytest.skip(
            "No skmob privacy_toy cache. Run '.venv-skmob/bin/python tests/populate_skmob_cache.py --datasets privacy_toy'."
        )
    return SkmobReferenceDataset("privacy_toy")


@pytest.fixture(scope="session")
def privacy_toy_polars(privacy_toy_reference):
    pl = pytest.importorskip("polars")
    return pl.from_pandas(privacy_toy_reference.input_df)


def _native_to_pandas(df: Any) -> pd.DataFrame:
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df).copy()


def _normalized(df: Any) -> pd.DataFrame:
    out = _native_to_pandas(df)
    out = out[[str(column) for column in out.columns]]
    sort_columns = list(out.columns)
    return out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _assert_cached_match(result: Any, expected: pd.DataFrame) -> None:
    actual_df = _normalized(result)
    expected_df = _normalized(expected)
    actual_df = actual_df.astype(object).where(pd.notna(actual_df), None)
    expected_df = expected_df.astype(object).where(pd.notna(expected_df), None)
    assert list(actual_df.columns) == list(expected_df.columns)
    assert_frame_equal(actual_df, expected_df, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", PRIVACY_CASES, ids=lambda case: case.name)
def test_privacy_toy_matches_cached_skmob_pandas(privacy_toy_reference, case: PrivacyCase):
    expected = privacy_toy_reference.result(case.name)
    if expected is None:
        pytest.skip(f"No cached skmob output for {case.name}")

    attack = case.attack_cls(**case.init_kwargs)
    result = attack.assess_risk(privacy_toy_reference.input_df.copy(), show_progress=False, **case.assess_kwargs)

    _assert_cached_match(result, expected)


@pytest.mark.parametrize("case", PRIVACY_CASES, ids=lambda case: case.name)
def test_privacy_toy_matches_cached_skmob_polars(privacy_toy_reference, privacy_toy_polars, case: PrivacyCase):
    expected = privacy_toy_reference.result(case.name)
    if expected is None:
        pytest.skip(f"No cached skmob output for {case.name}")

    attack = case.attack_cls(**case.init_kwargs)
    result = attack.assess_risk(privacy_toy_polars.clone(), show_progress=False, **case.assess_kwargs)

    _assert_cached_match(result, expected)
