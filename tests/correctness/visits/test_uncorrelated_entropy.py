"""Correctness tests for skmob2/measures/visits/uncorrelated_entropy.py."""

from __future__ import annotations

import math

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Known-value assertions.
#
# The synthetic fixture has 3 users, 5 GPS points each.  All points are at
# distinct (lat, lng) pairs, so each location is visited exactly once.
# The visit-probability distribution is uniform over 5 locations:
#   p_i = 1/5 for i in 1..5
#   S_unc = -5 * (1/5 * log2(1/5)) = log2(5) ≈ 2.3219
# ---------------------------------------------------------------------------

EXPECTED_ENTROPY: float = math.log2(5)


def _to_dict(df) -> dict:
    """Convert an uncorrelated_entropy result DataFrame to {uid: entropy_value}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["uncorrelated_entropy"] for row in nw_df.rows(named=True)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uncorrelated_entropy_known_values(synthetic_tdf):
    """Uniform distribution over 5 distinct locations gives S = log2(5)."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    result = uncorrelated_entropy(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY}"


def test_uncorrelated_entropy_unequal_visits():
    """Non-uniform visit distribution produces correct Shannon entropy."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    # User "a" visits loc1 3 times and loc2 1 time: p1=0.75, p2=0.25
    # S = -(0.75*log2(0.75) + 0.25*log2(0.25))
    p1, p2 = 0.75, 0.25
    expected = -(p1 * math.log2(p1) + p2 * math.log2(p2))

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = uncorrelated_entropy(df)
    mapping = _to_dict(result)
    assert math.isclose(mapping["a"], expected, rel_tol=1e-9)


def test_uncorrelated_entropy_single_location():
    """A user who visits only one location has entropy 0."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 1.0, 1.0],
            "lng": [2.0, 2.0, 2.0],
        }
    )
    result = uncorrelated_entropy(df)
    mapping = _to_dict(result)
    assert mapping["a"] == 0.0


def test_uncorrelated_entropy_normalize():
    """With normalize=True the result equals S_unc / log2(n_distinct_locs)."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    # Uniform over 5 locs: S_unc = log2(5), normalized = 1.0
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = uncorrelated_entropy(df, normalize=True)
    mapping = _to_dict(result)
    assert math.isclose(mapping["a"], 1.0, rel_tol=1e-9)


def test_uncorrelated_entropy_normalize_single_loc():
    """normalize=True with a single location still returns 0 (no division by 0)."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [1.0, 1.0],
            "lng": [0.0, 0.0],
        }
    )
    result = uncorrelated_entropy(df, normalize=True)
    mapping = _to_dict(result)
    assert mapping["a"] == 0.0


def test_uncorrelated_entropy_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    # 4 visits to 2 distinct locs equally: p=0.5, S=1.0
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = uncorrelated_entropy(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "uncorrelated_entropy" in nw_result.columns
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    assert math.isclose(row["uncorrelated_entropy"], 1.0, rel_tol=1e-9)


def test_uncorrelated_entropy_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy

    result = uncorrelated_entropy(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), (
            f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY} (Polars)"
        )


@pytest.mark.skmob
def test_uncorrelated_entropy_matches_skmob(comparison_skmob):
    """skmob2 result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import uncorrelated_entropy as skmob_ue
    from skmob2.measures.visits.uncorrelated_entropy import (
        uncorrelated_entropy as skmob2_ue,
    )

    skmob_result = skmob_ue(comparison_skmob)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_ue(skmob2_input)

    skmob_dict = dict(
        zip(
            skmob_result["uid"].tolist(),
            skmob_result["uncorrelated_entropy"].tolist(),
        )
    )
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert math.isclose(skmob_dict[uid], skmob2_dict[uid], rel_tol=1e-5), (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )


def test_uncorrelated_entropy_matches_cached_reference(comparison_skmob_reference):
    """uncorrelated_entropy matches the cached skmob baseline without requiring the skmob environment."""
    from skmob2.measures.visits.uncorrelated_entropy import uncorrelated_entropy as skmob2_ue

    ref = comparison_skmob_reference
    skmob_result = ref.result("uncorrelated_entropy")
    skmob2_result = skmob2_ue(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["uncorrelated_entropy"].tolist()))
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert math.isclose(skmob_dict[uid], skmob2_dict[uid], rel_tol=1e-5), (
            f"uid={uid}: cached={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
