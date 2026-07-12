"""Correctness tests for fastmob/measures/visits/random_entropy.py."""

from __future__ import annotations

import math

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Known-value assertions.
#
# The synthetic fixture has 3 users, 5 GPS points each — all distinct (lat, lng)
# pairs — so n_distinct_locations = 5 and random_entropy = log2(5) for every
# user.
# ---------------------------------------------------------------------------

EXPECTED_ENTROPY: float = math.log2(5)


def _to_dict(df) -> dict:
    """Convert a random_entropy result DataFrame to {uid: entropy_value}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["random_entropy"] for row in nw_df.rows(named=True)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_random_entropy_known_values(synthetic_tdf):
    """Each user in the synthetic fixture has 5 distinct locations -> log2(5)."""
    from fastmob.measures.individual.random_entropy import random_entropy

    result = random_entropy(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY}"


def test_random_entropy_repeated_locations():
    """When a user revisits locations, n is the distinct count, not total visits."""
    from fastmob.measures.individual.random_entropy import random_entropy

    # User "a" visits 2 distinct locations 4 times total -> log2(2) = 1.0
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [3.0, 4.0, 3.0, 4.0],
        }
    )
    result = random_entropy(df)
    mapping = _to_dict(result)
    assert math.isclose(mapping["a"], math.log2(2), rel_tol=1e-9)


def test_random_entropy_single_location():
    """A user who visits only one location has entropy 0."""
    from fastmob.measures.individual.random_entropy import random_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 1.0, 1.0],
            "lng": [2.0, 2.0, 2.0],
        }
    )
    result = random_entropy(df)
    mapping = _to_dict(result)
    assert mapping["a"] == 0.0


def test_random_entropy_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from fastmob.measures.individual.random_entropy import random_entropy

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [0.0, 1.0, 2.0, 3.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = random_entropy(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "random_entropy" in nw_result.columns
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    assert math.isclose(row["random_entropy"], math.log2(4), rel_tol=1e-9)


def test_random_entropy_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from fastmob.measures.individual.random_entropy import random_entropy

    result = random_entropy(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, val in mapping.items():
        assert math.isclose(val, EXPECTED_ENTROPY, rel_tol=1e-9), (
            f"uid={uid!r}: got {val}, expected {EXPECTED_ENTROPY} (Polars)"
        )


@pytest.mark.skmob
def test_random_entropy_matches_skmob(comparison_skmob):
    """fastmob result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import random_entropy as skmob_re
    from fastmob.measures.individual.random_entropy import random_entropy as fastmob_re

    skmob_result = skmob_re(comparison_skmob)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_re(fastmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["random_entropy"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert math.isclose(skmob_dict[uid], fastmob_dict[uid], rel_tol=1e-5), (
            f"uid={uid}: skmob={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
        )


def test_random_entropy_null_coordinates_ignored():
    """NaN/None coordinates are ignored; valid locations still counted."""
    import math

    from fastmob.measures.individual.random_entropy import random_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, float("nan"), None, 3.0],
            "lng": [1.0, 2.0, 3.0, 4.0, 3.0],
        }
    )
    result = random_entropy(df)
    mapping = _to_dict(result)
    # rows 0,1,4 are valid → lat/lng (1,1),(2,2),(3,3) → 3 distinct locations
    assert math.isclose(mapping["a"], math.log2(3), rel_tol=1e-9)


def test_random_entropy_all_invalid_user_dropped_with_uid():
    """Users where all coordinates are invalid are omitted from the result."""
    from fastmob.measures.individual.random_entropy import random_entropy

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "b", "b"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [float("nan"), float("nan"), 1.0, 2.0],
            "lng": [float("nan"), float("nan"), 1.0, 2.0],
        }
    )
    result = random_entropy(df)
    mapping = _to_dict(result)
    assert "a" not in mapping
    assert "b" in mapping


def test_random_entropy_all_invalid_no_uid_returns_empty():
    """All-invalid input with no uid column returns an empty result."""
    from fastmob.measures.individual.random_entropy import random_entropy

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [float("nan"), float("nan")],
            "lng": [float("nan"), float("nan")],
        }
    )
    result = random_entropy(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "random_entropy" in nw_result.columns
    assert len(nw_result) == 0


def test_random_entropy_polars_null_coordinates_ignored():
    """Polars Arrow-null coordinates are ignored; valid locations counted."""
    import math

    import pytest

    polars = pytest.importorskip("polars")
    from fastmob.measures.individual.random_entropy import random_entropy

    df = polars.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": [
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 02:00:00",
                "2020-01-01 03:00:00",
            ],
            "lat": [1.0, 2.0, None, 3.0],
            "lng": [1.0, 2.0, 3.0, 3.0],
        }
    ).with_columns(polars.col("datetime").str.to_datetime())
    result = random_entropy(df)
    nw_result = nw.from_native(result, eager_only=True)
    row = {r["uid"]: r["random_entropy"] for r in nw_result.rows(named=True)}
    # rows 0,1,3 are valid → (1,1),(2,2),(3,3) → 3 distinct locations
    assert math.isclose(row["a"], math.log2(3), rel_tol=1e-9)


def test_random_entropy_matches_cached_reference(comparison_skmob_reference):
    """random_entropy matches the cached skmob baseline without requiring the skmob environment."""
    from fastmob.measures.individual.random_entropy import random_entropy as fastmob_re

    ref = comparison_skmob_reference
    skmob_result = ref.result("random_entropy")
    fastmob_result = fastmob_re(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["random_entropy"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert math.isclose(skmob_dict[uid], fastmob_dict[uid], rel_tol=1e-5), (
            f"uid={uid}: cached={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
        )
