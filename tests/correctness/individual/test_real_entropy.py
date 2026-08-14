"""Correctness tests for fastmob/measures/visits/real_entropy.py."""

from __future__ import annotations

import math

import narwhals as nw
import pandas as pd


def _with_location_ids(df):
    """Add the explicit location IDs consumed by real_entropy."""
    nw_df = nw.from_native(df, eager_only=True)
    return nw_df.with_columns(
        (nw.col("lat").cast(nw.String) + nw.lit("_") + nw.col("lng").cast(nw.String)).alias("location_id")
    ).to_native()


# ---------------------------------------------------------------------------
# Known-value assertions.
#
# The synthetic fixture has 3 users, 5 GPS points each. All 5 locations per
# user are distinct and visited exactly once.
# ---------------------------------------------------------------------------

EXPECTED_ENTROPY: float = math.log2(5)


def _kontoyiannis_entropy(sequence: list) -> float:
    """Reference implementation of the Kontoyiannis estimator."""
    n = len(sequence)
    if n <= 1:
        return 0.0

    col_max = [1] * n
    prev_row = [1] * n
    for i in range(1, n):
        curr_row = [1] * n
        for j in range(i + 1, n):
            if sequence[i - 1] == sequence[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            col_max[j] = max(col_max[j], curr_row[j])
        prev_row = curr_row

    return float((n / sum(col_max)) * math.log2(n))


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
    from fastmob.measures.individual.real_entropy import real_entropy

    result = real_entropy(_with_location_ids(synthetic_tdf))
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY}"


def test_real_entropy_repeated_location():
    """A sequence with repeated visits captures temporal correlations."""
    from fastmob.measures.individual.real_entropy import real_entropy

    # User "a": alternates between two locations — strong temporal structure
    df = pd.DataFrame(
        {
            "uid": ["a"] * 6,
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(_with_location_ids(df))
    mapping = _to_dict(result)

    seq = ["1.0_0.0", "2.0_0.0", "1.0_0.0", "2.0_0.0", "1.0_0.0", "2.0_0.0"]
    expected = _kontoyiannis_entropy(seq)
    assert math.isclose(mapping["a"], expected, rel_tol=1e-9)


def test_real_entropy_single_location():
    """A user with one observation has real_entropy == 0 (length-1 sequence)."""
    from fastmob.measures.individual.real_entropy import real_entropy

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
    result = real_entropy(_with_location_ids(df))
    mapping = _to_dict(result)
    assert mapping["a"] == 0.0


def test_real_entropy_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from fastmob.measures.individual.real_entropy import real_entropy

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(_with_location_ids(df))
    nw_result = nw.from_native(result, eager_only=True)
    assert "real_entropy" in nw_result.columns
    assert len(nw_result) == 1

    seq = ["1.0_0.0", "2.0_0.0", "3.0_0.0", "4.0_0.0"]
    expected = _kontoyiannis_entropy(seq)
    row = nw_result.rows(named=True)[0]
    assert math.isclose(row["real_entropy"], expected, rel_tol=1e-9)


def test_real_entropy_multiple_users_independent():
    """Each user's entropy is computed independently from others."""
    from fastmob.measures.individual.real_entropy import real_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "b", "b", "b"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, 10.0, 10.0, 10.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = real_entropy(_with_location_ids(df))
    mapping = _to_dict(result)

    expected_a = _kontoyiannis_entropy(["1.0_0.0", "2.0_0.0"])
    expected_b = _kontoyiannis_entropy(["10.0_0.0", "10.0_0.0", "10.0_0.0"])

    assert math.isclose(mapping["a"], expected_a, rel_tol=1e-9)
    assert math.isclose(mapping["b"], expected_b, rel_tol=1e-9)


def test_real_entropy_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from fastmob.measures.individual.real_entropy import real_entropy

    result = real_entropy(_with_location_ids(synthetic_tdf_polars))
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), (
            f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY} (Polars)"
        )


def test_real_entropy_output_backend_matches_input(synthetic_tdf):
    """Result backend matches the input backend (pandas in, pandas out)."""
    from fastmob.measures.individual.real_entropy import real_entropy

    result = real_entropy(_with_location_ids(synthetic_tdf))
    assert isinstance(result, pd.DataFrame), f"Expected pandas DataFrame, got {type(result)}"
