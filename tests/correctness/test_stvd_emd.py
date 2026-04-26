"""Correctness tests for skmob2/measures/stvd_emd.py."""
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
    from skmob2.measures.stvd_emd import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    dist = stvd_emd(df, df.copy())
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_spatial_displacement_100m():
    """Single-point distributions 100 m apart (same time) → positive distance."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (100 0)", "12:00", 1.0)
    dist_100 = stvd_emd(df_a, df_b, alpha=10.0)
    dist_0 = stvd_emd(df_a, df_a.copy(), alpha=10.0)
    # Sliced Wasserstein on 4D embedding: value < exact EMD but strictly > 0
    assert dist_100 > dist_0
    assert dist_100 > 0.0


def test_temporal_displacement_with_alpha():
    """Same location, 10 min apart, alpha=10 → positive distance."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (0 0)", "12:10", 1.0)
    dist = stvd_emd(df_a, df_b, alpha=10.0)
    assert dist > 0.0


def test_cyclical_time_wraps_around():
    """23:55 vs 00:05 must give the same distance as 00:00 vs 00:10 (10 min)."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df_a_cyclic = _dist_df("POINT (0 0)", "23:55", 1.0)
    df_b_cyclic = _dist_df("POINT (0 0)", "00:05", 1.0)
    df_a_linear = _dist_df("POINT (0 0)", "00:00", 1.0)
    df_b_linear = _dist_df("POINT (0 0)", "00:10", 1.0)

    # The circular embedding maps both pairs to the same chord length
    dist_cyclic = stvd_emd(df_a_cyclic, df_b_cyclic, alpha=10.0, cyclical_period=1440.0)
    dist_linear = stvd_emd(df_a_linear, df_b_linear, alpha=10.0, cyclical_period=1440.0)
    assert dist_cyclic == pytest.approx(dist_linear, rel=1e-3)


def test_symmetry():
    """d(A, B) must equal d(B, A)."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

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
    d_ab = stvd_emd(df_a, df_b)
    d_ba = stvd_emd(df_b, df_a)
    # Entropic Sinkhorn is approximately symmetric; allow 2% tolerance
    assert d_ab == pytest.approx(d_ba, rel=2e-2)


def test_explicit_column_names():
    """Explicit column overrides should work."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df = pd.DataFrame(
        {
            "geo": ["POINT (0 0)"],
            "ts": ["12:00"],
            "vol": [1.0],
        }
    )
    dist = stvd_emd(df, df.copy(), centroid_col="geo", time_col="ts", weight_col="vol")
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_returns_float():
    """Return value must be a plain Python float."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    result = stvd_emd(df, df.copy())
    assert isinstance(result, float)


def test_invalid_num_projections_raises():
    """num_projections <= 0 must raise ValueError."""
    _skip_if_no_core()
    from skmob2.measures.stvd_emd import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    with pytest.raises(ValueError, match="num_projections must be positive"):
        stvd_emd(df, df.copy(), num_projections=0)


def test_polars_parity():
    """Pandas and Polars inputs must produce the same result."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from skmob2.measures.stvd_emd import stvd_emd

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

    dist_pd = stvd_emd(df_a_pd, df_b_pd)
    dist_pl = stvd_emd(df_a_pl, df_b_pl)
    assert dist_pd == pytest.approx(dist_pl, rel=1e-5)


def test_top_level_import():
    """stvd_emd must be importable from skmob2."""
    _skip_if_no_core()
    import skmob2

    assert hasattr(skmob2, "stvd_emd")


def test_measures_import():
    """stvd_emd must be importable from skmob2.measures."""
    _skip_if_no_core()
    import skmob2.measures

    assert hasattr(skmob2.measures, "stvd_emd")


# ---------------------------------------------------------------------------
# Zero-copy numpy helper tests
# ---------------------------------------------------------------------------

def test_stvd_emd_numpy_helper():
    """stvd_emd_numpy in _core must accept numpy arrays and return the same result."""
    _skip_if_no_core()
    import numpy as np
    from skmob2._core import stvd_emd_numpy
    from skmob2.measures.stvd_emd import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    expected = stvd_emd(df, df.copy())

    xs = np.array([0.0], dtype=np.float64)
    ys = np.array([0.0], dtype=np.float64)
    ts = np.array([720.0], dtype=np.float64)  # 12:00 → 720 min
    ws = np.array([1.0], dtype=np.float64)

    result = stvd_emd_numpy(xs, ys, ts, ws, xs, ys, ts, ws, 10.0, 1440.0, 50)
    assert isinstance(result, float)
    assert result == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Zero-copy arrow helper tests
# ---------------------------------------------------------------------------

def test_stvd_emd_arrow_helper():
    """stvd_emd_arrow in _core must accept PyArrow float64 arrays."""
    _skip_if_no_core()
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    import numpy as np
    from skmob2._core import stvd_emd_arrow, stvd_emd_numpy

    xs = pa.array([0.0], type=pa.float64())
    ys = pa.array([0.0], type=pa.float64())
    ts = pa.array([720.0], type=pa.float64())
    ws = pa.array([1.0], type=pa.float64())

    result_arrow = stvd_emd_arrow(xs, ys, ts, ws, xs, ys, ts, ws, 10.0, 1440.0, 50)
    assert isinstance(result_arrow, float)

    xs_np = np.array([0.0], dtype=np.float64)
    ys_np = np.array([0.0], dtype=np.float64)
    ts_np = np.array([720.0], dtype=np.float64)
    ws_np = np.array([1.0], dtype=np.float64)
    result_numpy = stvd_emd_numpy(xs_np, ys_np, ts_np, ws_np, xs_np, ys_np, ts_np, ws_np, 10.0, 1440.0, 50)

    assert result_arrow == pytest.approx(result_numpy, abs=1e-10)


def test_stvd_emd_numpy_non_contiguous_raises():
    """Non-contiguous numpy arrays must raise ValueError."""
    _skip_if_no_core()
    import numpy as np
    from skmob2._core import stvd_emd_numpy

    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    non_contig = arr[::2]  # stride=2, not C-contiguous

    with pytest.raises((ValueError, BufferError, TypeError)):
        stvd_emd_numpy(
            non_contig, non_contig, non_contig, non_contig,
            non_contig, non_contig, non_contig, non_contig,
            10.0, 1440.0, 50,
        )
