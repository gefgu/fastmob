"""Correctness tests for skmob2/measures/spatial/home_location.py."""

from __future__ import annotations

import pandas as pd
import pytest
import narwhals as nw


# Expected home locations for the shared synthetic fixture.
# All timestamps are in hours 0-4, which fall within the default nighttime
# window (22:00-07:00), so the home is the first visited location (each
# location appears once — tie-broken by first occurrence).
EXPECTED_HOME: dict[str, tuple[float, float]] = {
    "user_a": (0.0, 0.0),  # lat=0.0, lng=0.0
    "user_b": (10.0, 20.0),  # lat=10.0, lng=20.0
    "user_c": (48.8566, 2.3522),  # lat=48.8566, lng=2.3522
}


def _to_dict(df, lat_col: str = "lat", lng_col: str = "lng") -> dict[str, tuple[float, float]]:
    """Convert a home_location result DataFrame to {uid: (lat, lng)}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    lat = next(c for c in ("lat", "latitude") if c in columns)
    lng = next(c for c in ("lng", "lon", "longitude") if c in columns)
    return {row[uid_col]: (row[lat], row[lng]) for row in nw_df.rows(named=True)}


def test_home_location_known_values(synthetic_tdf):
    """Home location for each synthetic user matches the expected (lat, lng)."""
    from skmob2.measures.spatial.home_location import home_location

    result = home_location(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_HOME.keys())
    for uid, (exp_lat, exp_lng) in EXPECTED_HOME.items():
        got_lat, got_lng = mapping[uid]
        assert abs(got_lat - exp_lat) < 1e-6, f"uid={uid!r}: lat got {got_lat}, expected {exp_lat}"
        assert abs(got_lng - exp_lng) < 1e-6, f"uid={uid!r}: lng got {got_lng}, expected {exp_lng}"


def test_home_location_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.spatial.home_location import home_location

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01 22:00", periods=3, freq="h"),
            "lat": [1.0, 2.0, 1.0],
            "lng": [10.0, 20.0, 10.0],
        }
    )
    result = home_location(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    # lat=1.0, lng=10.0 appears twice; lat=2.0, lng=20.0 appears once.
    assert abs(row["lat"] - 1.0) < 1e-9
    assert abs(row["lng"] - 10.0) < 1e-9


def test_home_location_fallback_to_all_hours():
    """When no nighttime records exist, the most-visited overall location is used."""
    from skmob2.measures.spatial.home_location import home_location

    # All records at noon — no nighttime data.
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": [
                pd.Timestamp("2020-01-01 12:00"),
                pd.Timestamp("2020-01-01 13:00"),
                pd.Timestamp("2020-01-01 12:00"),
            ],
            "lat": [1.0, 2.0, 1.0],
            "lng": [10.0, 20.0, 10.0],
        }
    )
    result = home_location(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    # lat=1.0, lng=10.0 appears twice; should be selected as home.
    assert abs(row["lat"] - 1.0) < 1e-9
    assert abs(row["lng"] - 10.0) < 1e-9


def test_home_location_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same home locations as pandas."""
    from skmob2.measures.spatial.home_location import home_location

    result = home_location(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_HOME.keys())
    for uid, (exp_lat, exp_lng) in EXPECTED_HOME.items():
        got_lat, got_lng = mapping[uid]
        assert abs(got_lat - exp_lat) < 1e-6, f"uid={uid!r}: lat got {got_lat}, expected {exp_lat} (Polars)"
        assert abs(got_lng - exp_lng) < 1e-6, f"uid={uid!r}: lng got {got_lng}, expected {exp_lng} (Polars)"


@pytest.mark.skmob
def test_home_location_matches_skmob(brightkite_skmob):
    """skmob2 result matches skmob on the Brightkite dataset."""
    from skmob.measures.individual import home_location as skmob_hl
    from skmob2.measures.spatial.home_location import home_location as skmob2_hl

    skmob_result = skmob_hl(brightkite_skmob)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_hl(skmob2_input)

    skmob_dict = {row["uid"]: (row["lat"], row["lng"]) for row in skmob_result.to_dict(orient="records")}
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        exp_lat, exp_lng = skmob_dict[uid]
        got_lat, got_lng = skmob2_dict[uid]
        assert abs(got_lat - exp_lat) < 1e-6, f"uid={uid}: lat skmob={exp_lat}, skmob2={got_lat}"
        assert abs(got_lng - exp_lng) < 1e-6, f"uid={uid}: lng skmob={exp_lng}, skmob2={got_lng}"
