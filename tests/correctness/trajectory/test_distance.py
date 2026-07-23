"""Correctness tests for fastmob.trajectory.trajectory_distance.

Hand-crafted fixtures in ``conftest.py`` (``distance_seq_a`` /
``distance_seq_b_translated``) put every point on the same meridian
(``lng=0``), so a great-circle (haversine) distance along that meridian is
*exact* -- not an approximation -- when computed as
``R_earth_km * radians(delta_lat)``. That lets this file project the same
fixtures into a planar (0, km) coordinate space and feed them directly to
the third-party ``similaritymeasures`` package (DTW, Fréchet) and
``scipy.spatial.distance.directed_hausdorff`` (Hausdorff) as an independent,
exact-tolerance reference -- not just a structural/shape check. LCSS has no
off-the-shelf Python reference in this ecosystem, so it is checked against a
hand-written classical DP reimplementation instead (same precedent as this
test suite's existing hand-derived ``EXPECTED_JUMP_LENGTHS`` fixture).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fastmob.trajectory import trajectory_distance

EARTH_RADIUS_KM = 6371.0088  # matches geo::Haversine's default sphere radius (metres / 1000)


def _meridian_km(lat_deg: float) -> float:
    """Exact great-circle distance (km) from the equator along a meridian."""
    return EARTH_RADIUS_KM * math.radians(lat_deg)


def _projected_xy(lat_values: list[float]) -> np.ndarray:
    return np.array([[0.0, _meridian_km(lat)] for lat in lat_values])


def _reference_lcss(lats_a, lngs_a, lats_b, lngs_b, epsilon_km: float) -> float:
    """Independent classical LCSS DP (Vlachos et al. 2002), re-derived
    directly from the algorithm's textbook definition rather than by reading
    fastmob's own Rust implementation."""
    from fastmob._core import haversine_km

    n, m = len(lats_a), len(lats_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = haversine_km(lats_a[i - 1], lngs_a[i - 1], lats_b[j - 1], lngs_b[j - 1])
            if d <= epsilon_km:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    normalizer = min(n, m)
    return dp[n][m] / normalizer if normalizer > 0 else 0.0


def test_identical_sequences_are_maximally_similar(distance_seq_a):
    assert trajectory_distance(distance_seq_a, distance_seq_a, method="dtw") == pytest.approx(0.0, abs=1e-9)
    assert trajectory_distance(distance_seq_a, distance_seq_a, method="frechet") == pytest.approx(0.0, abs=1e-9)
    assert trajectory_distance(distance_seq_a, distance_seq_a, method="hausdorff") == pytest.approx(0.0, abs=1e-9)
    assert trajectory_distance(distance_seq_a, distance_seq_a, method="lcss", epsilon_km=0.01) == pytest.approx(1.0)


def test_translated_sequence_frechet_equals_hausdorff_equals_point_offset(distance_seq_a, distance_seq_b_translated):
    """A pure per-point translation with matched point counts and order: the
    identity alignment is optimal for both Fréchet and Hausdorff, so both
    collapse to the single point-to-point offset distance."""
    expected_offset_km = _meridian_km(1.0) - _meridian_km(0.0)
    frechet = trajectory_distance(distance_seq_a, distance_seq_b_translated, method="frechet")
    hausdorff = trajectory_distance(distance_seq_a, distance_seq_b_translated, method="hausdorff")
    assert frechet == pytest.approx(expected_offset_km, abs=1e-6)
    assert hausdorff == pytest.approx(expected_offset_km, abs=1e-6)


def test_lcss_similarity_is_in_unit_range(distance_seq_a, distance_seq_b_translated):
    similarity = trajectory_distance(distance_seq_a, distance_seq_b_translated, method="lcss", epsilon_km=1.0)
    assert 0.0 <= similarity <= 1.0


def test_multi_user_side_raises_value_error():
    df = pd.DataFrame(
        {
            "uid": ["a", "a", "b"],
            "lat": [0.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        }
    )
    single = pd.DataFrame(
        {"lat": [0.0, 1.0], "lng": [0.0, 0.0], "datetime": pd.date_range("2020-01-01", periods=2, freq="h")}
    )
    with pytest.raises(ValueError, match="single point-sequences"):
        trajectory_distance(df, single, method="dtw")


def test_unknown_method_raises(distance_seq_a, distance_seq_b_translated):
    with pytest.raises(ValueError, match="unknown distance method"):
        trajectory_distance(distance_seq_a, distance_seq_b_translated, method="bogus")


def test_polars_matches_pandas(distance_seq_a, distance_seq_b_translated):
    pl = pytest.importorskip("polars", reason="Polars not installed")
    seq_a_pl = pl.from_pandas(distance_seq_a)
    seq_b_pl = pl.from_pandas(distance_seq_b_translated)
    for method in ("dtw", "frechet", "hausdorff", "lcss"):
        pandas_value = trajectory_distance(distance_seq_a, distance_seq_b_translated, method=method)
        polars_value = trajectory_distance(seq_a_pl, seq_b_pl, method=method)
        assert pandas_value == pytest.approx(polars_value)


@pytest.mark.parametrize(
    "lat_values_a,lat_values_b",
    [
        ([0.0, 0.5, 1.2, 2.0, 2.1], [0.1, 0.4, 1.5, 1.9, 2.3]),
        ([10.0, 10.0, 11.0, 12.0], [10.2, 10.9, 11.9, 12.3]),
    ],
)
def test_dtw_and_frechet_match_similaritymeasures(lat_values_a, lat_values_b):
    """Independent numeric comparison against the third-party
    `similaritymeasures` package, using the exact meridian-km projection
    described in this module's docstring."""
    similaritymeasures = pytest.importorskip("similaritymeasures", reason="similaritymeasures not installed")

    seq_a = pd.DataFrame(
        {
            "lat": lat_values_a,
            "lng": [0.0] * len(lat_values_a),
            "datetime": pd.date_range("2020-01-01", periods=len(lat_values_a), freq="h"),
        }
    )
    seq_b = pd.DataFrame(
        {
            "lat": lat_values_b,
            "lng": [0.0] * len(lat_values_b),
            "datetime": pd.date_range("2020-01-01", periods=len(lat_values_b), freq="h"),
        }
    )

    xy_a = _projected_xy(lat_values_a)
    xy_b = _projected_xy(lat_values_b)

    expected_frechet = similaritymeasures.frechet_dist(xy_a, xy_b)
    expected_dtw, _ = similaritymeasures.dtw(xy_a, xy_b)

    assert trajectory_distance(seq_a, seq_b, method="frechet") == pytest.approx(expected_frechet, abs=1e-6)
    assert trajectory_distance(seq_a, seq_b, method="dtw") == pytest.approx(expected_dtw, abs=1e-6)


@pytest.mark.parametrize(
    "lat_values_a,lat_values_b",
    [
        ([0.0, 0.5, 1.2, 2.0, 2.1], [0.1, 0.4, 1.5, 1.9, 2.3]),
        ([10.0, 10.0, 11.0, 12.0], [10.2, 10.9, 11.9, 12.3]),
    ],
)
def test_hausdorff_matches_scipy(lat_values_a, lat_values_b):
    """Independent numeric comparison against
    `scipy.spatial.distance.directed_hausdorff`, using the same exact
    meridian-km projection."""
    from scipy.spatial.distance import directed_hausdorff

    seq_a = pd.DataFrame(
        {
            "lat": lat_values_a,
            "lng": [0.0] * len(lat_values_a),
            "datetime": pd.date_range("2020-01-01", periods=len(lat_values_a), freq="h"),
        }
    )
    seq_b = pd.DataFrame(
        {
            "lat": lat_values_b,
            "lng": [0.0] * len(lat_values_b),
            "datetime": pd.date_range("2020-01-01", periods=len(lat_values_b), freq="h"),
        }
    )

    xy_a = _projected_xy(lat_values_a)
    xy_b = _projected_xy(lat_values_b)

    a_to_b, _, _ = directed_hausdorff(xy_a, xy_b)
    b_to_a, _, _ = directed_hausdorff(xy_b, xy_a)
    expected = max(a_to_b, b_to_a)

    assert trajectory_distance(seq_a, seq_b, method="hausdorff") == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("epsilon_km", [0.1, 1.0, 50.0])
def test_lcss_matches_hand_written_reference(epsilon_km):
    lat_values_a = [0.0, 0.5, 1.2, 2.0, 2.1, 3.5]
    lat_values_b = [0.1, 0.4, 1.5, 1.9, 2.3, 3.4]
    lngs = [0.0] * len(lat_values_a)

    seq_a = pd.DataFrame(
        {"lat": lat_values_a, "lng": lngs, "datetime": pd.date_range("2020-01-01", periods=len(lat_values_a), freq="h")}
    )
    seq_b = pd.DataFrame(
        {"lat": lat_values_b, "lng": lngs, "datetime": pd.date_range("2020-01-01", periods=len(lat_values_b), freq="h")}
    )

    expected = _reference_lcss(lat_values_a, lngs, lat_values_b, lngs, epsilon_km)
    assert trajectory_distance(seq_a, seq_b, method="lcss", epsilon_km=epsilon_km) == pytest.approx(expected)
