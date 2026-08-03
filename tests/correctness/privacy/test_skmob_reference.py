"""Cached scikit-mobility parity tests for privacy risk functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import pytest
from fastmob import privacy
from pandas.testing import assert_frame_equal

from tests.shared.skmob_cache import _REFERENCE_DIR, SkmobReferenceDataset


@dataclass(frozen=True)
class PrivacyCase:
    name: str
    function: Callable[..., Any]
    kwargs: dict[str, Any]


_BASE = {"knowledge_length": 2}
PRIVACY_CASES = (
    PrivacyCase("location_kl2", privacy.location_risk, _BASE),
    PrivacyCase("location_kl3", privacy.location_risk, {"knowledge_length": 3}),
    PrivacyCase("location_targets_kl3", privacy.location_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("location_force_instances_kl3", privacy.location_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("location_sequence_kl2", privacy.location_sequence_risk, _BASE),
    PrivacyCase("location_sequence_kl3", privacy.location_sequence_risk, {"knowledge_length": 3}),
    PrivacyCase("location_sequence_targets_kl3", privacy.location_sequence_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("location_sequence_force_instances_kl3", privacy.location_sequence_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("location_time_kl2", privacy.location_time_risk, _BASE),
    PrivacyCase("location_time_month_kl2", privacy.location_time_risk, {"knowledge_length": 2, "time_precision": "Month"}),
    PrivacyCase("location_time_month_kl3", privacy.location_time_risk, {"knowledge_length": 3, "time_precision": "Month"}),
    PrivacyCase("location_time_month_targets_kl3", privacy.location_time_risk, {"knowledge_length": 3, "time_precision": "Month", "targets": [1, 2]}),
    PrivacyCase("location_time_month_force_instances_kl3", privacy.location_time_risk, {"knowledge_length": 3, "time_precision": "Month", "targets": [1, 2], "force_instances": True}),
    PrivacyCase("unique_location_kl2", privacy.unique_location_risk, _BASE),
    PrivacyCase("unique_location_kl3", privacy.unique_location_risk, {"knowledge_length": 3}),
    PrivacyCase("unique_location_targets_kl3", privacy.unique_location_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("unique_location_force_instances_kl3", privacy.unique_location_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("location_frequency_kl2", privacy.location_frequency_risk, _BASE),
    PrivacyCase("location_frequency_tol05_kl2", privacy.location_frequency_risk, {"knowledge_length": 2, "tolerance": 0.5}),
    PrivacyCase("location_frequency_kl3", privacy.location_frequency_risk, {"knowledge_length": 3}),
    PrivacyCase("location_frequency_targets_kl3", privacy.location_frequency_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("location_frequency_force_instances_kl3", privacy.location_frequency_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("location_probability_kl2", privacy.location_probability_risk, _BASE),
    PrivacyCase("location_probability_tol05_kl2", privacy.location_probability_risk, {"knowledge_length": 2, "tolerance": 0.5}),
    PrivacyCase("location_probability_kl3", privacy.location_probability_risk, {"knowledge_length": 3}),
    PrivacyCase("location_probability_targets_kl3", privacy.location_probability_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("location_probability_force_instances_kl3", privacy.location_probability_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("location_proportion_kl2", privacy.location_proportion_risk, _BASE),
    PrivacyCase("location_proportion_tol05_kl2", privacy.location_proportion_risk, {"knowledge_length": 2, "tolerance": 0.5}),
    PrivacyCase("location_proportion_kl3", privacy.location_proportion_risk, {"knowledge_length": 3}),
    PrivacyCase("location_proportion_targets_kl3", privacy.location_proportion_risk, {"knowledge_length": 3, "targets": [1, 2]}),
    PrivacyCase("location_proportion_force_instances_kl3", privacy.location_proportion_risk, {"knowledge_length": 3, "targets": [1, 2], "force_instances": True}),
    PrivacyCase("home_work", privacy.home_work_risk, {}),
    PrivacyCase("home_work_targets", privacy.home_work_risk, {"targets": [1, 2]}),
    PrivacyCase("home_work_force_instances", privacy.home_work_risk, {"targets": [1, 2], "force_instances": True}),
)


@pytest.fixture(scope="session")
def privacy_toy_reference() -> SkmobReferenceDataset:
    if not (_REFERENCE_DIR / "privacy_toy" / "input.parquet").exists():
        pytest.skip("No skmob privacy_toy cache.")
    return SkmobReferenceDataset("privacy_toy")


def _normalized(df: Any) -> pd.DataFrame:
    out = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df).copy()
    if "datetime" in out and out["datetime"].isna().all():
        out = out.drop(columns="datetime")
    return out.sort_values(list(out.columns), kind="mergesort").reset_index(drop=True).astype(object).where(pd.notna(out), None)


@pytest.mark.parametrize("case", PRIVACY_CASES, ids=lambda case: case.name)
def test_privacy_toy_matches_cached_skmob_pandas(privacy_toy_reference, case: PrivacyCase):
    expected = privacy_toy_reference.result(case.name)
    if expected is None:
        pytest.skip(f"No cached skmob output for {case.name}")
    # The cache generator maps this input to resolution-12 H3 centers before
    # running scikit-mobility, making its exact comparisons H3-equivalent.
    actual = case.function(privacy_toy_reference.input_df.copy(), **case.kwargs)
    actual = _normalized(actual)
    expected = _normalized(expected)
    if "datetime" not in actual and "datetime" in expected:
        expected = expected.drop(columns="datetime").sort_values(list(expected.columns.drop("datetime")), kind="mergesort").reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
