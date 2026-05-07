"""Correctness tests for skmob2/measures/visits/real_entropy.py."""

from __future__ import annotations

import math

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Known-value assertions.
#
# The synthetic fixture has 3 users, 5 GPS points each. All 5 locations per
# user are distinct and visited exactly once. The value below matches
# scikit-mobility's private _true_entropy estimator for length-5 sequences.
# ---------------------------------------------------------------------------

EXPECTED_ENTROPY: float = 5 * math.log2(5) / 6


def _to_dict(df) -> dict:
    """Convert a real_entropy result DataFrame to {uid: entropy_value}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["real_entropy"] for row in nw_df.rows(named=True)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_real_entropy_known_values(synthetic_tdf):
    """Five distinct locations visited once each gives real_entropy == log2(5)."""
    from skmob2.measures.visits.real_entropy import real_entropy

    result = real_entropy(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY}"


def test_real_entropy_repeated_location():
    """A sequence with repeated visits captures temporal correlations."""
    from skmob2.measures.visits.real_entropy import real_entropy
    from skmob2.measures.visits.real_entropy import _skmob_true_entropy

    # User "a": alternates between two locations — strong temporal structure
    df = pd.DataFrame(
        {
            "uid": ["a"] * 6,
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(df)
    mapping = _to_dict(result)

    seq = ["1.0_0.0", "2.0_0.0", "1.0_0.0", "2.0_0.0", "1.0_0.0", "2.0_0.0"]
    expected = _skmob_true_entropy(seq)
    assert math.isclose(mapping["a"], expected, rel_tol=1e-9)


def test_real_entropy_single_location():
    """A user with one observation has real_entropy == 0 (length-1 sequence)."""
    from skmob2.measures.visits.real_entropy import real_entropy

    # The Kontoyiannis estimator returns 0.0 only for sequences of length <= 1.
    # Use a single observation to exercise that path.
    df = pd.DataFrame(
        {
            "uid": ["a"],
            "datetime": pd.date_range("2020-01-01", periods=1, freq="h"),
            "lat": [1.0],
            "lng": [2.0],
        }
    )
    result = real_entropy(df)
    mapping = _to_dict(result)
    assert mapping["a"] == 0.0


def test_real_entropy_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.real_entropy import real_entropy
    from skmob2.measures.visits.real_entropy import _skmob_true_entropy

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "real_entropy" in nw_result.columns
    assert len(nw_result) == 1

    seq = ["1.0_0.0", "2.0_0.0", "3.0_0.0", "4.0_0.0"]
    expected = _skmob_true_entropy(seq)
    row = nw_result.rows(named=True)[0]
    assert math.isclose(row["real_entropy"], expected, rel_tol=1e-9)


def test_real_entropy_multiple_users_independent():
    """Each user's entropy is computed independently from others."""
    from skmob2.measures.visits.real_entropy import real_entropy
    from skmob2.measures.visits.real_entropy import _skmob_true_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "b", "b", "b"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, 10.0, 10.0, 10.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(df)
    mapping = _to_dict(result)

    # Expected values from the skmob-compatible estimator applied per-user.
    expected_a = _skmob_true_entropy(["1.0_0.0", "2.0_0.0"])
    expected_b = _skmob_true_entropy(["10.0_0.0", "10.0_0.0", "10.0_0.0"])

    assert math.isclose(mapping["a"], expected_a, rel_tol=1e-9)
    assert math.isclose(mapping["b"], expected_b, rel_tol=1e-9)


def test_real_entropy_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from skmob2.measures.visits.real_entropy import real_entropy

    result = real_entropy(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), (
            f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY} (Polars)"
        )


def test_real_entropy_output_backend_matches_input(synthetic_tdf):
    """Result backend matches the input backend (pandas in, pandas out)."""
    from skmob2.measures.visits.real_entropy import real_entropy

    result = real_entropy(synthetic_tdf)
    assert isinstance(result, pd.DataFrame), f"Expected pandas DataFrame, got {type(result)}"


@pytest.mark.skmob
def test_real_entropy_matches_skmob(comparison_skmob):
    """skmob2 result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import real_entropy as skmob_re
    from skmob2.measures.visits.real_entropy import real_entropy as skmob2_re

    skmob_result = skmob_re(comparison_skmob)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_re(skmob2_input)

    skmob_dict = dict(
        zip(
            skmob_result["uid"].tolist(),
            skmob_result["real_entropy"].tolist(),
        )
    )
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert math.isclose(skmob_dict[uid], skmob2_dict[uid], rel_tol=1e-5), (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
