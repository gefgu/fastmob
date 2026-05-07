"""Correctness tests for skmob2/measures/visits/recency_rank.py."""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(df) -> dict:
    """Convert a recency_rank result DataFrame to {uid: {(lat, lng): rank}}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    lat_col = next((c for c in ("lat", "latitude") if c in columns), None)
    lng_col = next((c for c in ("lng", "lon", "longitude") if c in columns), None)

    result: dict = {}
    for row in nw_df.rows(named=True):
        uid = row[uid_col]
        loc = (row[lat_col], row[lng_col])
        rank = row["recency_rank"]
        result.setdefault(uid, {})[loc] = rank
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recency_rank_known_values(synthetic_tdf):
    """All 5 locations are distinct; the most recently visited gets rank 1."""
    from skmob2.measures.visits.recency_rank import recency_rank

    result = recency_rank(synthetic_tdf)
    mapping = _to_dict(result)

    # Three users present.
    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}

    for uid, loc_ranks in mapping.items():
        # Each user visited 5 distinct locations.
        assert len(loc_ranks) == 5
        # Ranks form a contiguous 1..5 set.
        assert set(loc_ranks.values()) == {1, 2, 3, 4, 5}
        # Rank 1 is assigned to the latest visit.
        rank1_loc = min(loc_ranks, key=loc_ranks.get)
        assert loc_ranks[rank1_loc] == 1


def test_recency_rank_repeated_location():
    """When a user visits a location multiple times, it gets a single rank."""
    from skmob2.measures.visits.recency_rank import recency_rank

    # User "a" visits (1.0, 0.0) at t=0 and t=3, and (2.0, 0.0) at t=1.
    # After dedup: (1.0, 0.0) most recent (t=3) -> rank 1, (2.0, 0.0) -> rank 2.
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 03:00"]),
            "lat": [1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = recency_rank(df)
    mapping = _to_dict(result)

    assert set(mapping["a"].keys()) == {(1.0, 0.0), (2.0, 0.0)}
    assert mapping["a"][(1.0, 0.0)] == 1
    assert mapping["a"][(2.0, 0.0)] == 2


def test_recency_rank_single_location():
    """A user who visits only one location gets that location rank 1."""
    from skmob2.measures.visits.recency_rank import recency_rank

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 1.0, 1.0],
            "lng": [2.0, 2.0, 2.0],
        }
    )
    result = recency_rank(df)
    mapping = _to_dict(result)
    assert len(mapping["a"]) == 1
    assert mapping["a"][(1.0, 2.0)] == 1


def test_recency_rank_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.recency_rank import recency_rank

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 2.0, 3.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = recency_rank(df)
    nw_result = nw.from_native(result, eager_only=True)

    assert "recency_rank" in nw_result.columns
    # 3 distinct locations -> 3 rows.
    assert len(nw_result) == 3
    # Rank 1 is assigned to last-visited location (lat=3.0).
    rows = {row["lat"]: row["recency_rank"] for row in nw_result.rows(named=True)}
    assert rows[3.0] == 1
    assert rows[1.0] == 3


def test_recency_rank_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from skmob2.measures.visits.recency_rank import recency_rank

    result = recency_rank(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_ranks in mapping.items():
        assert len(loc_ranks) == 5
        assert set(loc_ranks.values()) == {1, 2, 3, 4, 5}


@pytest.mark.skmob
def test_recency_rank_matches_skmob(comparison_skmob):
    """skmob2 result matches skmob on each comparison dataset."""
    import pandas as pd
    from skmob.measures.individual import recency_rank as skmob_rr
    from skmob2.measures.visits.recency_rank import recency_rank as skmob2_rr

    skmob_result = skmob_rr(comparison_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_rr(skmob2_input)

    # Build comparable {uid: {(lat, lng): rank}} dicts.
    skmob_dict: dict = {}
    skmob_result = skmob_result.reset_index()
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        loc = (row["lat"], row["lng"])
        skmob_dict.setdefault(uid, {})[loc] = int(row["recency_rank"])

    skmob2_dict = _to_dict(skmob2_result)

    common_uids = set(skmob_dict) & set(skmob2_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_locs = set(skmob_dict[uid]) & set(skmob2_dict[uid])
        for loc in common_locs:
            assert skmob_dict[uid][loc] == skmob2_dict[uid][loc], (
                f"uid={uid}, loc={loc}: skmob={skmob_dict[uid][loc]}, skmob2={skmob2_dict[uid][loc]}"
            )
