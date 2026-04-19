"""Correctness tests for skmob2/measures/visits/location_frequency.py."""
from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_freq_dict(df) -> dict:
    """Convert a location_frequency DataFrame to {uid: {(lat, lng): freq}}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    lat_col = next((c for c in ("lat", "latitude") if c in columns), None)
    lng_col = next((c for c in ("lng", "lon", "longitude") if c in columns), None)

    result: dict = {}
    for row in nw_df.rows(named=True):
        uid = row[uid_col] if uid_col else "__single__"
        loc = (row[lat_col], row[lng_col])
        result.setdefault(uid, {})[loc] = row["frequency"]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_location_frequency_known_values(synthetic_tdf):
    """Each user visits 5 distinct locations once: all frequencies should be 1."""
    from skmob2.measures.visits.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf)
    freq_dict = _to_freq_dict(result)

    assert set(freq_dict.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_freqs in freq_dict.items():
        assert len(loc_freqs) == 5
        assert all(v == 1.0 for v in loc_freqs.values())


def test_location_frequency_normalized(synthetic_tdf):
    """With normalize=True each user's frequencies sum to 1.0."""
    from skmob2.measures.visits.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf, normalize=True)
    freq_dict = _to_freq_dict(result)

    for uid, loc_freqs in freq_dict.items():
        total = sum(loc_freqs.values())
        assert abs(total - 1.0) < 1e-9, f"uid={uid}: sum={total}"


def test_location_frequency_repeated_visits():
    """Locations visited multiple times accumulate correctly."""
    from skmob2.measures.visits.location_frequency import location_frequency

    # User "a" visits loc A 3 times and loc B once.
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df)
    freq_dict = _to_freq_dict(result)

    assert freq_dict["a"][(1.0, 0.0)] == 3.0
    assert freq_dict["a"][(2.0, 0.0)] == 1.0


def test_location_frequency_normalized_repeated():
    """Normalized frequencies sum to 1 when visits are unequal."""
    from skmob2.measures.visits.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df, normalize=True)
    freq_dict = _to_freq_dict(result)

    assert abs(freq_dict["a"][(1.0, 0.0)] - 0.75) < 1e-9
    assert abs(freq_dict["a"][(2.0, 0.0)] - 0.25) < 1e-9


def test_location_frequency_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.visits.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [1.0, 2.0, 1.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df)
    nw_result = nw.from_native(result, eager_only=True)

    assert "frequency" in nw_result.columns
    # 2 distinct locations.
    assert len(nw_result) == 2
    rows = {(row["lat"], row["lng"]): row["frequency"] for row in nw_result.rows(named=True)}
    assert rows[(1.0, 0.0)] == 2.0
    assert rows[(2.0, 0.0)] == 1.0


def test_location_frequency_sorted_descending():
    """Output is sorted by frequency descending within each user."""
    from skmob2.measures.visits.location_frequency import location_frequency

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [1.0, 2.0, 2.0, 3.0, 2.0],
            "lng": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    result = location_frequency(df)
    nw_result = nw.from_native(result, eager_only=True)
    freqs = nw_result.filter(nw.col("uid") == "a").get_column("frequency").to_list()
    # Frequencies must be non-increasing.
    assert freqs == sorted(freqs, reverse=True)


def test_location_frequency_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same frequency values as pandas."""
    from skmob2.measures.visits.location_frequency import location_frequency

    result = location_frequency(synthetic_tdf_polars)
    freq_dict = _to_freq_dict(result)

    assert set(freq_dict.keys()) == {"user_a", "user_b", "user_c"}
    for uid, loc_freqs in freq_dict.items():
        assert len(loc_freqs) == 5
        assert all(v == 1.0 for v in loc_freqs.values())


@pytest.mark.skmob
def test_location_frequency_matches_skmob(brightkite_skmob):
    """skmob2 counts match skmob on the Brightkite dataset."""
    import pandas as pd
    from skmob.measures.individual import location_frequency as skmob_lf
    from skmob2.measures.visits.location_frequency import location_frequency as skmob2_lf

    skmob_result = skmob_lf(brightkite_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_lf(skmob2_input)

    # Build comparable dicts: {uid: {(lat, lng): count}}.
    skmob_dict: dict = {}
    skmob_result = skmob_result.reset_index()
    for _, row in skmob_result.iterrows():
        uid = row["uid"]
        loc = (row["lat"], row["lng"])
        skmob_dict.setdefault(uid, {})[loc] = float(row["location_frequency"])

    skmob2_dict = _to_freq_dict(skmob2_result)

    common_uids = set(skmob_dict) & set(skmob2_dict)
    assert len(common_uids) > 0
    for uid in common_uids:
        common_locs = set(skmob_dict[uid]) & set(skmob2_dict[uid])
        for loc in common_locs:
            assert abs(skmob_dict[uid][loc] - skmob2_dict[uid][loc]) < 1e-5, (
                f"uid={uid}, loc={loc}: "
                f"skmob={skmob_dict[uid][loc]}, skmob2={skmob2_dict[uid][loc]}"
            )
