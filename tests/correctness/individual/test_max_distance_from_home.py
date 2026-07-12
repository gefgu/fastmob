"""Correctness tests for fastmob/measures/spatial/max_distance_from_home.py."""

from __future__ import annotations

import pandas as pd
import pytest
import narwhals as nw
import numpy as np


# Expected max-distance-from-home for the shared synthetic fixture.
# Home is the first visited location for each user (all timestamps 00:00-04:00,
# all within nighttime window 22:00-07:00, each location visited once so first
# occurrence wins).  Distances are Haversine (km) from home to furthest point.
EXPECTED_MAX_DIST_FROM_HOME: dict[str, float] = {
    "user_a": 444.7803209341316,  # 4-degree equatorial step from (0,0) to (0,4)
    "user_b": 444.7803209341316,  # 4-degree meridional step from (10,20) to (14,20)
    "user_c": 3.439627269068569,  # furthest Paris cluster point from first location
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
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home

    result = max_distance_from_home(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST_FROM_HOME.keys())
    for uid, expected in EXPECTED_MAX_DIST_FROM_HOME.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_max_distance_from_home_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home

    # Nighttime records only; home = (0.0, 0.0); farthest = (0.0, 4.0).
    df = pd.DataFrame(
        {
            "datetime": [
                pd.Timestamp("2020-01-01 00:00"),
                pd.Timestamp("2020-01-01 01:00"),
                pd.Timestamp("2020-01-01 02:00"),
            ],
            "lat": [0.0, 0.0, 0.0],
            "lng": [0.0, 2.0, 4.0],
        }
    )
    result = max_distance_from_home(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "max_distance_from_home" in nw_result.columns
    assert len(nw_result) == 1
    val = nw_result.rows(named=True)[0]["max_distance_from_home"]
    # 4-degree equatorial step ≈ 444.780 km
    assert abs(val - 444.7803209341316) < 1e-5


def test_max_distance_from_home_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home

    result = max_distance_from_home(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_MAX_DIST_FROM_HOME.keys())
    for uid, expected in EXPECTED_MAX_DIST_FROM_HOME.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


def test_max_distance_from_home_polars_null_coordinates_are_ignored():
    """Arrow-backed max-distance-from-home skips null coordinate rows."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home

    df = pl.DataFrame(
        {
            "uid": ["a", "a", "a"],
            "datetime": [
                pd.Timestamp("2020-01-01 00:00"),
                pd.Timestamp("2020-01-01 01:00"),
                pd.Timestamp("2020-01-01 02:00"),
            ],
            "lat": [0.0, None, 0.0],
            "lng": [0.0, 4.0, 2.0],
        }
    )

    result = max_distance_from_home(df)
    mapping = _to_dict(result)

    assert abs(mapping["a"] - 222.3901604670658) < 1e-6


def test_max_distance_from_point_indexed_arrow_null_coordinates_are_ignored():
    """Arrow helper accepts nullable coordinate arrays and skips invalid rows."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import max_distance_from_point_indexed_arrow

    home_lats = pa.array([0.0], type=pa.float64())
    home_lngs = pa.array([0.0], type=pa.float64())
    lats = pa.array([0.0, None, 0.0], type=pa.float64())
    lngs = pa.array([0.0, 4.0, 2.0], type=pa.float64())
    indices = np.array([0, 1, 2], dtype=np.uintp)
    ends = np.array([3], dtype=np.uintp)

    result = max_distance_from_point_indexed_arrow(home_lats, home_lngs, lats, lngs, indices, ends)

    np.testing.assert_allclose(np.asarray(result), [222.3901604670658], rtol=0.0, atol=1e-6)


@pytest.mark.skmob
def test_max_distance_from_home_matches_skmob(comparison_skmob):
    """fastmob result matches skmob on each comparison dataset."""
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from skmob.measures.individual import max_distance_from_home as skmob_mdfh
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home as fastmob_mdfh

    skmob_result = skmob_mdfh(comparison_skmob)
    fastmob_input = pd.DataFrame(comparison_skmob).copy()
    fastmob_result = fastmob_mdfh(fastmob_input)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["max_distance_from_home"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        assert abs(skmob_dict[uid] - fastmob_dict[uid]) < skmob_dict[uid] * 1e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, fastmob={fastmob_dict[uid]}"
        )


def test_max_distance_from_home_matches_cached_reference(comparison_skmob_reference):
    """max_distance_from_home matches the cached skmob baseline without requiring the skmob environment.

    Users with an ambiguous (tied) home location are allowed to differ because a different
    home point yields a different max_distance_from_home value.
    """
    pytest.importorskip("fastmob._core", reason="Run maturin develop first")
    from fastmob.measures.individual.max_distance_from_home import max_distance_from_home as fastmob_mdfh

    ref = comparison_skmob_reference
    skmob_result = ref.result("max_distance_from_home")
    fastmob_result = fastmob_mdfh(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["max_distance_from_home"].tolist()))
    fastmob_dict = _to_dict(fastmob_result)

    # Import the ambiguity helper from the home_location test module.
    from tests.correctness.individual.test_home_location import _nighttime_is_ambiguous

    common = set(skmob_dict) & set(fastmob_dict)
    assert len(common) > 0
    for uid in common:
        tol = skmob_dict[uid] * 1e-5 + 1e-5
        if abs(skmob_dict[uid] - fastmob_dict[uid]) >= tol:
            assert _nighttime_is_ambiguous(ref.input_df, uid), (
                f"uid={uid}: cached={skmob_dict[uid]}, fastmob={fastmob_dict[uid]} "
                f"(no tied home location, so this is a real correctness failure)"
            )
