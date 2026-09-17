"""Correctness tests for fastmob.measures.evaluation.spatial (OD matrix + stvd_emd)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _skip_if_no_core():
    pytest.importorskip(
        "fastmob._core",
        reason="Build the fastmob extension first (maturin develop)",
    )


def _dist_df(lat: float, lng: float, time_bin: str, mean_volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"center_lat": [lat], "center_lng": [lng], "time_bin": [time_bin], "mean_volume": [mean_volume]}
    )


# ---------------------------------------------------------------------------
# stvd_emd — single-point and basic properties
# ---------------------------------------------------------------------------


def test_identical_single_point_zero_distance():
    """Identical single-cell distributions must have distance 0."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    dist = stvd_emd(df, df.copy())
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_spatial_displacement_100m():
    """Single-point distributions ~100 m apart (same time) → positive distance."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = _dist_df(0.0, 0.0, "12:00", 1.0)
    df_b = _dist_df(0.0, 0.0009, "12:00", 1.0)  # ~100 m eastward at the equator
    dist_100 = stvd_emd(df_a, df_b, alpha=10.0)
    dist_0 = stvd_emd(df_a, df_a.copy(), alpha=10.0)
    assert dist_100 > dist_0
    assert dist_100 > 0.0


def test_temporal_displacement_with_alpha():
    """Same location, 10 min apart, alpha=10 → positive distance."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = _dist_df(0.0, 0.0, "12:00", 1.0)
    df_b = _dist_df(0.0, 0.0, "12:10", 1.0)
    dist = stvd_emd(df_a, df_b, alpha=10.0)
    assert dist > 0.0


def test_cyclical_time_wraps_around():
    """23:55 vs 00:05 must give the same distance as 00:00 vs 00:10 (10 min)."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a_cyclic = _dist_df(0.0, 0.0, "23:55", 1.0)
    df_b_cyclic = _dist_df(0.0, 0.0, "00:05", 1.0)
    df_a_linear = _dist_df(0.0, 0.0, "00:00", 1.0)
    df_b_linear = _dist_df(0.0, 0.0, "00:10", 1.0)

    dist_cyclic = stvd_emd(df_a_cyclic, df_b_cyclic, alpha=10.0, cyclical_period=1440.0)
    dist_linear = stvd_emd(df_a_linear, df_b_linear, alpha=10.0, cyclical_period=1440.0)
    assert dist_cyclic == pytest.approx(dist_linear, rel=1e-3)


def test_symmetry():
    """d(A, B) must equal d(B, A)."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.0, 0.002],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.001, 0.003],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )
    d_ab = stvd_emd(df_a, df_b)
    d_ba = stvd_emd(df_b, df_a)
    assert d_ab == pytest.approx(d_ba, rel=2e-2)


def test_explicit_column_names():
    """Explicit column overrides should work."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = pd.DataFrame(
        {
            "glat": [0.0],
            "glng": [0.0],
            "ts": ["12:00"],
            "vol": [1.0],
        }
    )
    dist = stvd_emd(df, df.copy(), lat_col="glat", lng_col="glng", time_col="ts", weight_col="vol")
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_returns_float():
    """Return value must be a plain Python float."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    result = stvd_emd(df, df.copy())
    assert isinstance(result, float)


def test_invalid_cyclical_period_raises():
    """cyclical_period <= 0 must raise ValueError."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    with pytest.raises(ValueError, match="cyclical_period must be positive"):
        stvd_emd(df, df.copy(), cyclical_period=0.0)


# ---------------------------------------------------------------------------
# reg / max_iter / tol plumbing
# ---------------------------------------------------------------------------


def test_custom_reg_max_iter_accepted():
    """Explicit reg/max_iter must be accepted and still return a float."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.0, 0.002],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.001, 0.003],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )
    dist = stvd_emd(df_a, df_b, reg=0.5, max_iter=50)
    assert isinstance(dist, float)
    assert dist > 0.0


def test_too_few_iterations_changes_result():
    """A deliberately tiny max_iter must measurably differ from a converged run.

    At the library's default ``reg=0.01`` the dual updates already saturate
    within a single iteration on small problems (the same numerical
    near-degeneracy motivating the sparse-Sinkhorn rewrite: cost values are
    in metres, so ``exp(-cost/reg)`` underflows to 0.0 for all but a point's
    closest match almost immediately) -- so this test uses a larger ``reg``
    where convergence genuinely takes multiple iterations, confirmed
    empirically against this exact fixture before writing the assertion.
    """
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0, 0.1],
            "center_lng": [0.0, 0.002, 0.05],
            "time_bin": ["10:00", "14:00", "18:00"],
            "mean_volume": [0.3, 0.7, 0.4],
        }
    )
    df_b = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0, 0.12],
            "center_lng": [0.001, 0.003, 0.06],
            "time_bin": ["11:00", "15:00", "19:30"],
            "mean_volume": [0.5, 0.5, 0.4],
        }
    )
    dist_converged = stvd_emd(df_a, df_b, reg=10.0, max_iter=200)
    dist_one_iter = stvd_emd(df_a, df_b, reg=10.0, max_iter=1)
    assert dist_converged != pytest.approx(dist_one_iter, rel=1e-3)


def test_tol_accepted_and_converges_for_trivial_case():
    """tol must be accepted and not change the (already-converged) zero-distance case."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    dist = stvd_emd(df, df.copy(), tol=1e-6)
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_invalid_max_iter_raises():
    """max_iter == 0 must raise ValueError."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        stvd_emd(df, df.copy(), max_iter=0)


def test_invalid_reg_raises():
    """reg <= 0 must raise ValueError."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    with pytest.raises(ValueError, match="reg must be positive"):
        stvd_emd(df, df.copy(), reg=0.0)


# ---------------------------------------------------------------------------
# Dense-path memory guard
# ---------------------------------------------------------------------------


def test_oversized_dense_input_raises_clear_error_not_oom():
    """n*m past the dense-path safety budget must raise ValueError, not OOM.

    n and m are each kept small/cheap to build (~24k rows, a few hundred KB);
    only their *product* needs to cross the guard's byte threshold.
    """
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    n = 24_000
    df_a = pd.DataFrame(
        {
            "center_lat": [0.0] * n,
            "center_lng": [0.0] * n,
            "time_bin": ["12:00"] * n,
            "mean_volume": [1.0] * n,
        }
    )
    df_b = df_a.copy()
    with pytest.raises(ValueError, match="too large"):
        stvd_emd(df_a, df_b)


def test_polars_parity():
    """Pandas and Polars inputs must produce the same result."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from fastmob.measures.evaluation import stvd_emd

    df_a_pd = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.0, 0.002],
            "time_bin": ["10:00", "14:00"],
            "mean_volume": [0.3, 0.7],
        }
    )
    df_b_pd = pd.DataFrame(
        {
            "center_lat": [0.0, 0.0],
            "center_lng": [0.001, 0.003],
            "time_bin": ["11:00", "15:00"],
            "mean_volume": [0.5, 0.5],
        }
    )
    df_a_pl = polars.from_pandas(df_a_pd)
    df_b_pl = polars.from_pandas(df_b_pd)

    dist_pd = stvd_emd(df_a_pd, df_b_pd)
    dist_pl = stvd_emd(df_a_pl, df_b_pl)
    assert dist_pd == pytest.approx(dist_pl, rel=1e-5)


def test_top_level_import():
    """stvd_emd must be importable from fastmob."""
    _skip_if_no_core()
    import fastmob

    assert hasattr(fastmob, "stvd_emd")


def test_measures_import():
    """stvd_emd must be importable from fastmob.measures."""
    _skip_if_no_core()
    import fastmob.measures

    assert hasattr(fastmob.measures, "stvd_emd")


# ---------------------------------------------------------------------------
# Zero-copy numpy helper tests
# ---------------------------------------------------------------------------


def test_stvd_emd_helper_accepts_numpy():
    """stvd_emd in _core must accept numpy arrays and return the same result."""
    _skip_if_no_core()
    from fastmob._core import stvd_emd as stvd_emd_core
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df(0.0, 0.0, "12:00", 1.0)
    expected = stvd_emd(df, df.copy())

    lats = np.array([0.0], dtype=np.float64)
    lngs = np.array([0.0], dtype=np.float64)
    ts = np.array([720.0], dtype=np.float64)
    ws = np.array([1.0], dtype=np.float64)

    result = stvd_emd_core(lats, lngs, ts, ws, lats, lngs, ts, ws, 10.0, 1440.0)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Zero-copy arrow helper tests
# ---------------------------------------------------------------------------


def test_stvd_emd_helper_accepts_arrow():
    """stvd_emd in _core must accept PyArrow float64 arrays."""
    _skip_if_no_core()
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from fastmob._core import stvd_emd as stvd_emd_core

    lats = pa.array([0.0], type=pa.float64())
    lngs = pa.array([0.0], type=pa.float64())
    ts = pa.array([720.0], type=pa.float64())
    ws = pa.array([1.0], type=pa.float64())

    result_arrow = stvd_emd_core(lats, lngs, ts, ws, lats, lngs, ts, ws, 10.0, 1440.0)
    assert isinstance(result_arrow, float)

    lats_np = np.array([0.0], dtype=np.float64)
    lngs_np = np.array([0.0], dtype=np.float64)
    ts_np = np.array([720.0], dtype=np.float64)
    ws_np = np.array([1.0], dtype=np.float64)
    result_numpy = stvd_emd_core(lats_np, lngs_np, ts_np, ws_np, lats_np, lngs_np, ts_np, ws_np, 10.0, 1440.0)

    assert result_arrow == pytest.approx(result_numpy, abs=1e-10)


def test_stvd_emd_helper_non_contiguous_numpy_raises():
    """Non-contiguous numpy arrays must raise ValueError."""
    _skip_if_no_core()
    from fastmob._core import stvd_emd as stvd_emd_core

    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    non_contig = arr[::2]

    with pytest.raises((ValueError, BufferError, TypeError)):
        stvd_emd_core(
            non_contig,
            non_contig,
            non_contig,
            non_contig,
            non_contig,
            non_contig,
            non_contig,
            non_contig,
            10.0,
            1440.0,
        )
