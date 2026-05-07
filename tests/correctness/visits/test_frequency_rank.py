"""Correctness tests for skmob2/measures/visits/frequency_rank.py."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(df) -> dict:
    """Convert a frequency_rank result DataFrame to {uid: {(lat, lng): rank}}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    lat_col = next((c for c in ("lat", "latitude") if c in columns), None)
    lng_col = next((c for c in ("lng", "lon", "longitude") if c in columns), None)

    result: dict = {}
    for row in nw_df.rows(named=True):
        uid = row[uid_col]
        loc = (row[lat_col], row[lng_col])
        rank = row["frequency_rank"]
        result.setdefault(uid, {})[loc] = rank
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_frequency_rank_known_values(synthetic_tdf):
    """All 5 locations visited once each; all get distinct ranks 1-5."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    result = frequency_rank(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_ranks in mapping.items():
        assert len(loc_ranks) == 5
        assert set(loc_ranks.values()) == {1, 2, 3, 4, 5}


def test_frequency_rank_unequal_visits():
    """The most frequently visited location gets rank 1."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    # User "a" visits (1.0, 0.0) 3 times and (2.0, 0.0) once.
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = frequency_rank(df)
    mapping = _to_dict(result)

    assert mapping["a"][(1.0, 0.0)] == 1
    assert mapping["a"][(2.0, 0.0)] == 2


def test_frequency_rank_ties_follow_skmob_pandas_order():
    """Equal-frequency ties match skmob's pandas groupby/sort behavior."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [5.0, 1.0, 3.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = frequency_rank(df)
    mapping = _to_dict(result)

    assert mapping["a"][(1.0, 0.0)] == 1
    assert mapping["a"][(3.0, 0.0)] == 2
    assert mapping["a"][(5.0, 0.0)] == 3


def test_frequency_rank_single_location():
    """A user who visits only one location gets that location rank 1."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 1.0, 1.0],
            "lng": [2.0, 2.0, 2.0],
        }
    )
    result = frequency_rank(df)
    mapping = _to_dict(result)
    assert len(mapping["a"]) == 1
    assert mapping["a"][(1.0, 2.0)] == 1


def test_frequency_rank_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    # 3 visits to loc1, 1 visit to loc2.
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = frequency_rank(df)
    nw_result = nw.from_native(result, eager_only=True)

    assert "frequency_rank" in nw_result.columns
    assert len(nw_result) == 2
    rows = {row["lat"]: row["frequency_rank"] for row in nw_result.rows(named=True)}
    assert rows[1.0] == 1
    assert rows[2.0] == 2


def test_frequency_rank_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from skmob2.measures.visits.frequency_rank import frequency_rank

    result = frequency_rank(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_ranks in mapping.items():
        assert len(loc_ranks) == 5
        assert set(loc_ranks.values()) == {1, 2, 3, 4, 5}


@pytest.mark.skmob
def test_frequency_rank_matches_skmob(comparison_skmob):
    """skmob2 result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import frequency_rank as skmob_fr
    from skmob2.measures.visits.frequency_rank import frequency_rank as skmob2_fr

    skmob_result = skmob_fr(comparison_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_fr(skmob2_input)

    # Build comparable {uid: {(lat, lng): rank}} dicts.
    skmob_dict: dict = {}
    skmob_result = skmob_result.reset_index()
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        loc = (row["lat"], row["lng"])
        skmob_dict.setdefault(uid, {})[loc] = int(row["frequency_rank"])

    skmob2_dict = _to_dict(skmob2_result)

    common_uids = set(skmob_dict) & set(skmob2_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_locs = set(skmob_dict[uid]) & set(skmob2_dict[uid])
        for loc in common_locs:
            assert skmob_dict[uid][loc] == skmob2_dict[uid][loc], (
                f"uid={uid}, loc={loc}: skmob={skmob_dict[uid][loc]}, skmob2={skmob2_dict[uid][loc]}"
            )
