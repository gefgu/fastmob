"""Correctness tests for fastmob/measures/spatial/number_of_locations.py."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Known-value assertions for the shared synthetic fixture (3 users, 5 points each,
# all distinct (lat, lng) pairs).
# ---------------------------------------------------------------------------

EXPECTED_LOCATIONS: dict[str, int] = {
    "user_a": 5,  # 5 distinct (0.0,0.0) (0.0,1.0) … (0.0,4.0)
    "user_b": 5,  # 5 distinct latitudes along meridian
    "user_c": 5,  # 5 distinct Paris-area points
}


def _to_dict(df) -> dict[str, int]:
    """Convert a number_of_locations result DataFrame to {uid: location_count}.

    Uses Narwhals rows() to avoid the pyarrow dependency that polars .to_pandas()
    requires in this environment.
    """
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")

    return {row[uid_col]: row["number_of_locations"] for row in nw_df.rows(named=True)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_number_of_locations_known_values(synthetic_tdf):
    """Each user in the synthetic fixture visits 5 distinct locations."""
    from fastmob.measures.individual.number_of_locations import number_of_locations

    result = number_of_locations(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_LOCATIONS.keys())
    for uid, expected in EXPECTED_LOCATIONS.items():
        assert mapping[uid] == expected, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_number_of_locations_deduplicates_repeated():
    """Repeated (lat, lng) pairs are counted only once per user."""
    from fastmob.measures.individual.number_of_locations import number_of_locations

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [3.0, 4.0, 3.0, 4.0],
        }
    )
    result = number_of_locations(df)
    mapping = _to_dict(result)
    # 4 visits but only 2 distinct (lat, lng) pairs
    assert mapping["a"] == 2


def test_number_of_locations_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from fastmob.measures.individual.number_of_locations import number_of_locations

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 3.0],
            "lng": [3.0, 4.0, 3.0, 5.0],
        }
    )
    result = number_of_locations(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "number_of_locations" in nw_result.columns
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    # (1,3), (2,4), (1,3), (3,5) — 3 distinct pairs
    assert row["number_of_locations"] == 3


def test_number_of_locations_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from fastmob.measures.individual.number_of_locations import number_of_locations

    result = number_of_locations(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_LOCATIONS.keys())
    for uid, expected in EXPECTED_LOCATIONS.items():
        assert mapping[uid] == expected, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


@pytest.mark.skmob
def test_number_of_locations_matches_skmob(comparison_skmob):
    """fastmob result matches skmob on each comparison dataset."""
    import pandas as pd
    from fastmob.measures.individual.number_of_locations import (
        number_of_locations as fastmob_nol,
    )
    from skmob.measures.individual import number_of_locations as skmob_nol

    skmob_result = skmob_nol(comparison_skmob)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_nol(fastmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["number_of_locations"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert skmob_dict[uid] == fastmob_dict[uid], f"uid={uid}: skmob={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"


def test_number_of_locations_matches_cached_reference(comparison_skmob_reference):
    """number_of_locations matches the cached skmob baseline without requiring the skmob environment."""
    from fastmob.measures.individual.number_of_locations import number_of_locations as fastmob_nol

    ref = comparison_skmob_reference
    skmob_result = ref.result("number_of_locations")
    fastmob_result = fastmob_nol(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["number_of_locations"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert skmob_dict[uid] == fastmob_dict[uid], f"uid={uid}: cached={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
