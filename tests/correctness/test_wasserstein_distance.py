"""Correctness tests for skmob2/measures/wasserstein_distance.py."""
from __future__ import annotations

import pandas as pd
import pytest


def _skip_if_no_core():
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )


def _dist_df(centroid: str, time_bin: str, mean_volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"centroid": [centroid], "time_bin": [time_bin], "mean_volume": [mean_volume]}
    )


def test_identical_single_point_zero_distance():
    """Identical single-cell distributions must have distance 0."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    dist = wasserstein_distance(df, df.copy())
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_spatial_displacement_100m():
    """Single-point distributions 100 m apart (same time) → distance ≈ 100."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (100 0)", "12:00", 1.0)
    dist = wasserstein_distance(df_a, df_b, alpha=10.0)
    assert dist == pytest.approx(100.0, rel=1e-3)


def test_temporal_displacement_with_alpha():
    """Same location, 10 min apart, alpha=10 → distance ≈ 100 m."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (0 0)", "12:10", 1.0)
    dist = wasserstein_distance(df_a, df_b, alpha=10.0)
    assert dist == pytest.approx(100.0, rel=1e-3)


def test_cyclical_time_wraps_around():
    """23:55 vs 00:05 should be treated as 10 min apart (not 1430 min)."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a = _dist_df("POINT (0 0)", "23:55", 1.0)
    df_b = _dist_df("POINT (0 0)", "00:05", 1.0)
    dist_cyclic = wasserstein_distance(df_a, df_b, alpha=10.0, cyclical_period=1440.0)
    # Cyclical: 10 min * alpha=10 = 100 m
    # Linear would be: 1430 min * alpha=10 = 14300 m
    assert dist_cyclic == pytest.approx(100.0, rel=1e-3)


def test_symmetry():
    """d(A, B) must equal d(B, A)."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a = pd.DataFrame(
        {
            "centroid": ["POINT (0 0)", "POINT (200 0)"],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b = pd.DataFrame(
        {
            "centroid": ["POINT (100 0)", "POINT (300 0)"],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )
    d_ab = wasserstein_distance(df_a, df_b)
    d_ba = wasserstein_distance(df_b, df_a)
    # Entropic Sinkhorn is approximately symmetric; allow 2% tolerance
    assert d_ab == pytest.approx(d_ba, rel=2e-2)


def test_explicit_column_names():
    """Explicit column overrides should work."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df = pd.DataFrame(
        {
            "geo": ["POINT (0 0)"],
            "ts": ["12:00"],
            "vol": [1.0],
        }
    )
    dist = wasserstein_distance(df, df.copy(), centroid_col="geo", time_col="ts", weight_col="vol")
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_returns_float():
    """Return value must be a plain Python float."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    result = wasserstein_distance(df, df.copy())
    assert isinstance(result, float)


def test_invalid_reg_raises():
    """reg <= 0 must raise ValueError."""
    _skip_if_no_core()
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    with pytest.raises(ValueError, match="reg must be positive"):
        wasserstein_distance(df, df.copy(), reg=0.0)


def test_polars_parity():
    """Pandas and Polars inputs must produce the same result."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2.measures.wasserstein_distance import wasserstein_distance

    df_a_pd = pd.DataFrame(
        {
            "centroid": ["POINT (0 0)", "POINT (200 0)"],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b_pd = pd.DataFrame(
        {
            "centroid": ["POINT (100 0)", "POINT (300 0)"],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )
    df_a_pl = polars.DataFrame(
        {
            "centroid": ["POINT (0 0)", "POINT (200 0)"],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b_pl = polars.DataFrame(
        {
            "centroid": ["POINT (100 0)", "POINT (300 0)"],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )

    dist_pd = wasserstein_distance(df_a_pd, df_b_pd)
    dist_pl = wasserstein_distance(df_a_pl, df_b_pl)
    assert dist_pd == pytest.approx(dist_pl, rel=1e-5)


def test_top_level_import():
    """wasserstein_distance must be importable from skmob2."""
    _skip_if_no_core()
    import skmob2

    assert hasattr(skmob2, "wasserstein_distance")


def test_measures_import():
    """wasserstein_distance must be importable from skmob2.measures."""
    _skip_if_no_core()
    import skmob2.measures

    assert hasattr(skmob2.measures, "wasserstein_distance")
