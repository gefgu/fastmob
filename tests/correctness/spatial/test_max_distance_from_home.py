"""Correctness tests for skmob2/measures/spatial/max_distance_from_home.py."""
from __future__ import annotations

import pandas as pd
import pytest
import narwhals as nw


# Expected max-distance-from-home for the shared synthetic fixture.
# Home is the first visited location for each user (all timestamps 00:00-04:00,
# all within nighttime window 22:00-07:00, each location visited once so first
# occurrence wins).  Distances are Haversine (km) from home to furthest point.
EXPECTED_MAX_DIST_FROM_HOME: dict[str, float] = {
    "user_a": 444.7803209341316,   # 4-degree equatorial step from (0,0) to (0,4)
    "user_b": 444.7803209341316,   # 4-degree meridional step from (10,20) to (14,20)
    "user_c": 3.439627269068569,   # furthest Paris cluster point from first location
}


def _to_dict(df) -> dict[str, float]:
    """Convert a max_distance_from_home result DataFrame to {uid: distance_km}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["max_distance_from_home"] for row in nw_df.rows(named=True)}


def test_max_distance_from_home_known_values(synthetic_tdf):
    """Known-value check for the shared 3-user synthetic fixture."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.max_distance_from_home import max_distance_from_home

    result = max_distance_from_home(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST_FROM_HOME.keys())
    for uid, expected in EXPECTED_MAX_DIST_FROM_HOME.items():
        assert abs(mapping[uid] - expected) < 1e-6, (
            f"uid={uid!r}: got {mapping[uid]}, expected {expected}"
        )


def test_max_distance_from_home_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.max_distance_from_home import max_distance_from_home

    # Nighttime records only; home = (0.0, 0.0); farthest = (0.0, 4.0).
    df = pd.DataFrame({
        "datetime": [
            pd.Timestamp("2020-01-01 00:00"),
            pd.Timestamp("2020-01-01 01:00"),
            pd.Timestamp("2020-01-01 02:00"),
        ],
        "lat": [0.0, 0.0, 0.0],
        "lng": [0.0, 2.0, 4.0],
    })
    result = max_distance_from_home(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "max_distance_from_home" in nw_result.columns
    assert len(nw_result) == 1
    val = nw_result.rows(named=True)[0]["max_distance_from_home"]
    # 4-degree equatorial step ≈ 444.780 km
    assert abs(val - 444.7803209341316) < 1e-5


def test_max_distance_from_home_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.max_distance_from_home import max_distance_from_home

    result = max_distance_from_home(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST_FROM_HOME.keys())
    for uid, expected in EXPECTED_MAX_DIST_FROM_HOME.items():
        assert abs(mapping[uid] - expected) < 1e-6, (
            f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"
        )


@pytest.mark.skmob
def test_max_distance_from_home_matches_skmob(brightkite_skmob):
    """skmob2 result matches skmob on the Brightkite dataset."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import max_distance_from_home as skmob_mdfh
    from skmob2.measures.spatial.max_distance_from_home import max_distance_from_home as skmob2_mdfh

    skmob_result = skmob_mdfh(brightkite_skmob)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_mdfh(skmob2_input)

    skmob_dict = dict(
        zip(skmob_result["uid"].tolist(), skmob_result["max_distance_from_home"].tolist())
    )
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert abs(skmob_dict[uid] - skmob2_dict[uid]) < skmob_dict[uid] * 1e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
