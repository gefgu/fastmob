"""Correctness tests for skmob2/measures/wasserstein_distance.py."""
from __future__ import annotations

import math

import pandas as pd
import pytest


@pytest.fixture
def identical_traj():
    """Two identical small trajectories — distance should be 0."""
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    return df, df.copy()


@pytest.fixture
def separated_traj():
    """Two trajectories 100 km apart — distance should be roughly 100 km."""
    df_a = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [0.0, 0.0],
            "lng": [0.0, 0.0],
        }
    )
    # Roughly 111 km per degree latitude at equator.
    df_b = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [1.0, 1.0],
            "lng": [0.0, 0.0],
        }
    )
    return df_a, df_b


def _skip_if_no_core():
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )


def test_identical_trajectories_zero_distance(identical_traj):
    """Wasserstein distance between identical point clouds must be 0."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a, df_b = identical_traj
    dist = wasserstein_distance(df_a, df_b)
    assert dist == pytest.approx(0.0, abs=1e-4), (
        f"Expected ~0 for identical trajectories, got {dist}"
    )


def test_symmetry(separated_traj):
    """Wasserstein distance must be symmetric: d(A, B) == d(B, A)."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a, df_b = separated_traj
    d_ab = wasserstein_distance(df_a, df_b)
    d_ba = wasserstein_distance(df_b, df_a)
    assert d_ab == pytest.approx(d_ba, rel=1e-5), (
        f"Symmetry violated: d(A,B)={d_ab}, d(B,A)={d_ba}"
    )


def test_nonzero_for_different_locations(separated_traj):
    """Wasserstein distance must be positive for non-overlapping clouds."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a, df_b = separated_traj
    dist = wasserstein_distance(df_a, df_b)
    assert dist > 0.0, f"Expected positive distance, got {dist}"


def test_distance_in_reasonable_km_range(separated_traj):
    """Distance between clouds 1° apart should be in the 50-150 km range."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a, df_b = separated_traj
    dist = wasserstein_distance(df_a, df_b)
    assert 50.0 < dist < 150.0, (
        f"Expected distance in (50, 150) km for 1° separation, got {dist}"
    )


def test_invalid_reg_raises():
    """reg <= 0 must raise ValueError."""
    _skip_if_no_core()
    import pandas as pd
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
            "lat": [0.0, 1.0],
            "lng": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="reg must be positive"):
        wasserstein_distance(df, df.copy(), reg=0.0)


def test_returns_float(identical_traj):
    """Return type must be a plain Python float."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a, df_b = identical_traj
    result = wasserstein_distance(df_a, df_b)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_top_level_import():
    """wasserstein_distance must be importable from skmob2 top-level."""
    _skip_if_no_core()
    import skmob2

    assert hasattr(skmob2, "wasserstein_distance"), (
        "wasserstein_distance not exported from skmob2 top-level"
    )


def test_measures_import():
    """wasserstein_distance must be importable from skmob2.measures."""
    _skip_if_no_core()
    import skmob2.measures

    assert hasattr(skmob2.measures, "wasserstein_distance"), (
        "wasserstein_distance not exported from skmob2.measures"
    )


def test_polars_input(separated_traj):
    """Polars DataFrames must be accepted without error."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a_pd, df_b_pd = separated_traj
    df_a_pl = polars.from_pandas(df_a_pd)
    df_b_pl = polars.from_pandas(df_b_pd)

    dist_pd = wasserstein_distance(df_a_pd, df_b_pd)
    dist_pl = wasserstein_distance(df_a_pl, df_b_pl)
    assert dist_pd == pytest.approx(dist_pl, rel=1e-5), (
        f"Pandas and Polars results differ: {dist_pd} vs {dist_pl}"
    )


def test_single_point_each():
    """Two single-point trajectories — distance equals Haversine between them."""
    _skip_if_no_core()
    from skmob2._core import haversine_km
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a = pd.DataFrame(
        {"datetime": [pd.Timestamp("2020-01-01")], "lat": [0.0], "lng": [0.0]}
    )
    df_b = pd.DataFrame(
        {"datetime": [pd.Timestamp("2020-01-01")], "lat": [1.0], "lng": [0.0]}
    )
    expected = haversine_km(0.0, 0.0, 1.0, 0.0)
    dist = wasserstein_distance(df_a, df_b)
    assert dist == pytest.approx(expected, rel=1e-3), (
        f"Single-point distance should match Haversine: expected={expected}, got={dist}"
    )
