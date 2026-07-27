"""Correctness tests for fastmob/measures/spatial/number_of_visits.py."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Known-value assertions for the shared synthetic fixture (3 users, 5 points each).
# ---------------------------------------------------------------------------

EXPECTED_VISITS: dict[str, int] = {
    "user_a": 5,
    "user_b": 5,
    "user_c": 5,
}


def _to_dict(df) -> dict[str, int]:
    """Convert a number_of_visits result DataFrame to {uid: visit_count}.

    Uses Narwhals rows() to avoid the pyarrow dependency that polars .to_pandas()
    requires in this environment.
    """
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")

    return {row[uid_col]: row["number_of_visits"] for row in nw_df.rows(named=True)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_number_of_visits_known_values(synthetic_tdf):
    """Each user in the synthetic fixture has exactly 5 visit rows."""
    from fastmob.measures.individual.number_of_visits import number_of_visits

    result = number_of_visits(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_VISITS.keys())
    for uid, expected in EXPECTED_VISITS.items():
        assert mapping[uid] == expected, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_number_of_visits_repeated_locations():
    """Repeated (lat, lng) pairs still count as separate visits."""
    from fastmob.measures.individual.number_of_visits import number_of_visits

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [3.0, 4.0, 3.0, 4.0],
        }
    )
    result = number_of_visits(df)
    mapping = _to_dict(result)
    assert mapping["a"] == 4


def test_number_of_visits_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from fastmob.measures.individual.number_of_visits import number_of_visits

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = number_of_visits(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "number_of_visits" in nw_result.columns
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    assert row["number_of_visits"] == 3


def test_number_of_visits_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from fastmob.measures.individual.number_of_visits import number_of_visits

    result = number_of_visits(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_VISITS.keys())
    for uid, expected in EXPECTED_VISITS.items():
        assert mapping[uid] == expected, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


@pytest.mark.skmob
def test_number_of_visits_matches_skmob(comparison_skmob):
    """fastmob result matches skmob on each comparison dataset."""
    import pandas as pd
    from fastmob.measures.individual.number_of_visits import number_of_visits as fastmob_nov
    from skmob.measures.individual import number_of_visits as skmob_nov

    skmob_result = skmob_nov(comparison_skmob)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_nov(fastmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["number_of_visits"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert skmob_dict[uid] == fastmob_dict[uid], f"uid={uid}: skmob={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"


def test_number_of_visits_matches_cached_reference(comparison_skmob_reference):
    """number_of_visits matches the cached skmob baseline without requiring the skmob environment."""
    from fastmob.measures.individual.number_of_visits import number_of_visits as fastmob_nov

    ref = comparison_skmob_reference
    skmob_result = ref.result("number_of_visits")
    fastmob_result = fastmob_nov(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["number_of_visits"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert skmob_dict[uid] == fastmob_dict[uid], f"uid={uid}: cached={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
