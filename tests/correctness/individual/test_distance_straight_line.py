"""Correctness tests for skmob2/measures/spatial/distance_straight_line.py."""

from __future__ import annotations

import pytest
import narwhals as nw
import numpy as np
import pandas as pd

# Pre-computed expected total distances for the shared synthetic fixture
# (3 users, 5 GPS points each, 1-degree steps equator/meridian, Paris cluster).
EXPECTED_TOTAL_DIST: dict[str, float] = {
    "user_a": 444.7803209341316,  # 4 × ~111.195 km along equator
    "user_b": 444.7803209341316,  # 4 × ~111.195 km along meridian
    "user_c": 3.440787203829053,  # sum of 4 small Paris steps
}


def _to_dict(df) -> dict[str, float]:
    """Convert a distance_straight_line result DataFrame to {uid: distance_km}."""
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")
    return {row[uid_col]: row["distance_straight_line"] for row in nw_df.rows(named=True)}


def test_distance_straight_line_known_values(synthetic_tdf):
    """Known-value check against pre-computed summed Haversine results."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.individual.distance_straight_line import distance_straight_line

    result = distance_straight_line(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_TOTAL_DIST.keys())
    for uid, expected in EXPECTED_TOTAL_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_distance_straight_line_single_user():
    """Without a uid column the whole frame is treated as one individual."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.individual.distance_straight_line import distance_straight_line

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = distance_straight_line(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "distance_straight_line" in nw_result.columns
    assert len(nw_result) == 1
    val = nw_result.rows(named=True)[0]["distance_straight_line"]
    # Two jumps of ~111.195 km each = ~222.390 km
    assert abs(val - 2 * 111.1950802335329) < 1e-5


def test_distance_straight_line_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.individual.distance_straight_line import distance_straight_line

    result = distance_straight_line(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_TOTAL_DIST.keys())
    for uid, expected in EXPECTED_TOTAL_DIST.items():
        assert abs(mapping[uid] - expected) < 1e-6, f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"


def test_total_distance_numpy_and_arrow_helpers_match_batch_helper():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2._core import total_distance_arrow, total_distance_batch_km, total_distance_numpy

    lats = np.array([0.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    lngs = np.array([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float64)
    ranges = [(0, 3), (3, 5)]

    expected = total_distance_batch_km(lats.tolist(), lngs.tolist(), ranges)
    result_numpy = total_distance_numpy(lats, lngs, ranges)
    result_arrow = total_distance_arrow(pl.Series(lats).to_arrow(), pl.Series(lngs).to_arrow(), ranges)

    np.testing.assert_allclose(result_numpy, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result_arrow, expected, rtol=0.0, atol=1e-12)


def test_total_distance_numpy_helper_validation_errors():
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import total_distance_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="same length"):
        total_distance_numpy(arr, arr[:1], [(0, 1)])
    with pytest.raises(ValueError, match="range end"):
        total_distance_numpy(arr, arr, [(0, 3)])


@pytest.mark.skmob
def test_distance_straight_line_matches_skmob(comparison_skmob):
    """skmob2 result closely matches skmob on each comparison dataset."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import distance_straight_line as skmob_dsl
    from skmob2.measures.individual.distance_straight_line import (
        distance_straight_line as skmob2_dsl,
    )

    skmob_result = skmob_dsl(comparison_skmob)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
    skmob2_result = skmob2_dsl(skmob2_input)

    skmob_dict = dict(
        zip(
            skmob_result["uid"].tolist(),
            skmob_result["distance_straight_line"].tolist(),
        )
    )
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        # skmob uses skmob.utils.gislib with earth radius 6371.0 km; skmob2
        # uses Rust geo::Haversine. The accumulated total can differ slightly.
        assert abs(skmob_dict[uid] - skmob2_dict[uid]) < skmob_dict[uid] * 2e-5 + 1e-5, (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )


def test_distance_straight_line_matches_cached_reference(comparison_skmob_reference):
    """distance_straight_line matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.individual.distance_straight_line import distance_straight_line as skmob2_dsl

    ref = comparison_skmob_reference
    skmob_result = ref.result("distance_straight_line")
    skmob2_result = skmob2_dsl(ref.input_df)

    skmob_dict = dict(zip(skmob_result["uid"].tolist(), skmob_result["distance_straight_line"].tolist()))
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert abs(skmob_dict[uid] - skmob2_dict[uid]) < skmob_dict[uid] * 2e-5 + 1e-5, (
            f"uid={uid}: cached={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
