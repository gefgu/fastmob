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


def _dist_df(centroid: str, time_bin: str, mean_volume: float) -> pd.DataFrame:
    return pd.DataFrame({"centroid": [centroid], "time_bin": [time_bin], "mean_volume": [mean_volume]})


# ---------------------------------------------------------------------------
# OD matrix
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="superseded by sparse FlowDataFrame CPC")
def test_od_matrix_common_part_aligns_origins_and_destinations():
    left = pd.DataFrame([[10, 0], [0, 5]], index=["a", "b"], columns=["x", "y"])
    right = pd.DataFrame([[5, 5], [0, 5]], index=["a", "c"], columns=["x", "z"])
    assert od_matrix_common_part_of_commuters(left, right) == pytest.approx(1.0 / 3.0)


def _trajectory_od_matrix(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    h3 = pytest.importorskip("h3")

    points = df[["uid", "datetime", "lat", "lng"]].copy()
    points["_datetime"] = pd.to_datetime(points["datetime"], errors="coerce")
    points["_lat"] = pd.to_numeric(points["lat"], errors="coerce")
    points["_lng"] = pd.to_numeric(points["lng"], errors="coerce")
    points = points.dropna(subset=["uid", "_datetime", "_lat", "_lng"])
    points = points[points["_lat"].between(-90, 90) & points["_lng"].between(-180, 180)]
    points = points.sort_values(["uid", "_datetime"], kind="mergesort")
    points["origin"] = [h3.latlng_to_cell(lat, lng, resolution) for lat, lng in zip(points["_lat"], points["_lng"])]
    points["destination"] = points.groupby("uid")["origin"].shift(-1)
    trips = points.dropna(subset=["destination"])
    trips = trips[trips["origin"] != trips["destination"]]
    if trips.empty:
        return pd.DataFrame(dtype=float)
    return trips.groupby(["origin", "destination"]).size().unstack(fill_value=0).astype(float)


def _reference_trajectory_cpc(left: pd.DataFrame, right: pd.DataFrame, resolution: int) -> float:
    return od_matrix_common_part_of_commuters(
        _trajectory_od_matrix(left, resolution),
        _trajectory_od_matrix(right, resolution),
    )


def _trajectory_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(
        {
            "uid": ["u1", "u2", "u1", "u1", "u2", "u1", "u3", "u3"],
            "datetime": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                ]
            ),
            "lat": [48.90, 48.90, 48.85, 48.85, 48.85, 999.0, 48.80, 48.80],
            "lng": [2.45, 2.45, 2.35, 2.35, 2.35, 2.50, 2.30, 2.30],
        }
    )
    right = pd.DataFrame(
        {
            "uid": ["u1", "u1", "u2", "u2", "u4"],
            "datetime": pd.to_datetime(
                [
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 09:00:00",
                    "2026-01-01 09:00:00",
                ]
            ),
            "lat": [48.85, 48.90, 48.85, 48.90, 48.70],
            "lng": [2.35, 2.45, 2.35, 2.45, 2.20],
        }
    )
    return left, right


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_matches_od_matrix_reference():
    _skip_if_no_core()
    left, right = _trajectory_fixture()

    result = trajectory_common_part_of_commuters(left, right, resolution=9)

    assert result == pytest.approx(_reference_trajectory_cpc(left, right, 9))


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_identical_is_one():
    _skip_if_no_core()
    left, _ = _trajectory_fixture()

    assert trajectory_common_part_of_commuters(left, left, resolution=9) == pytest.approx(1.0)


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_multi_matches_single_resolution_calls():
    _skip_if_no_core()
    left, right = _trajectory_fixture()

    multi_result = trajectory_common_part_of_commuters_multi(left, right, resolutions=(7, 8, 9))
    single_results = [(r, trajectory_common_part_of_commuters(left, right, resolution=r)) for r in (7, 8, 9)]

    assert multi_result == pytest.approx(single_results)


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_multi_prepares_inputs_only_once(monkeypatch):
    _skip_if_no_core()
    import fastmob.measures.evaluation.spatial as spatial_module

    left, right = _trajectory_fixture()
    calls = []
    original = spatial_module._trajectory_cpc_inputs

    def counting_wrapper(traj, **kwargs):
        calls.append(traj)
        return original(traj, **kwargs)

    monkeypatch.setattr(spatial_module, "_trajectory_cpc_inputs", counting_wrapper)

    trajectory_common_part_of_commuters_multi(left, right, resolutions=(7, 8, 9))

    # Exactly one prep call per trajectory side, regardless of how many
    # resolutions were requested.
    assert len(calls) == 2


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_empty_or_self_loop_only_returns_zero():
    _skip_if_no_core()
    loops = pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.to_datetime(["2026-01-01 08:00:00", "2026-01-01 09:00:00"]),
            "lat": [48.85, 48.85],
            "lng": [2.35, 2.35],
        }
    )

    assert trajectory_common_part_of_commuters(loops, loops, resolution=9) == pytest.approx(0.0)


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_dataframe_method_and_public_exports():
    _skip_if_no_core()
    import fastmob

    left, right = _trajectory_fixture()
    left_tdf = fastmob.TrajDataFrame(left)
    right_tdf = fastmob.TrajDataFrame(right)

    assert hasattr(fastmob, "trajectory_common_part_of_commuters")
    assert hasattr(fastmob.measures, "trajectory_common_part_of_commuters")
    assert left_tdf.common_part_of_commuters(right_tdf, resolution=9) == pytest.approx(
        trajectory_common_part_of_commuters(left, right, resolution=9)
    )


@pytest.mark.skip(reason="trajectory CPC was removed in favor of Trips CPC")
def test_trajectory_common_part_polars_smoke():
    _skip_if_no_core()
    pl = pytest.importorskip("polars")
    left, right = _trajectory_fixture()

    result = trajectory_common_part_of_commuters(
        pl.from_pandas(left),
        pl.from_pandas(right),
        resolution=9,
    )

    assert result == pytest.approx(_reference_trajectory_cpc(left, right, 9))


# ---------------------------------------------------------------------------
# stvd_emd — single-point and basic properties
# ---------------------------------------------------------------------------


def test_identical_single_point_zero_distance():
    """Identical single-cell distributions must have distance 0."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    dist = stvd_emd(df, df.copy())
    assert dist == pytest.approx(0.0, abs=1e-3)


def test_spatial_displacement_100m():
    """Single-point distributions 100 m apart (same time) → positive distance."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (100 0)", "12:00", 1.0)
    dist_100 = stvd_emd(df_a, df_b, alpha=10.0)
    dist_0 = stvd_emd(df_a, df_a.copy(), alpha=10.0)
    assert dist_100 > dist_0
    assert dist_100 > 0.0


def test_temporal_displacement_with_alpha():
    """Same location, 10 min apart, alpha=10 → positive distance."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a = _dist_df("POINT (0 0)", "12:00", 1.0)
    df_b = _dist_df("POINT (0 0)", "12:10", 1.0)
    dist = stvd_emd(df_a, df_b, alpha=10.0)
    assert dist > 0.0


def test_cyclical_time_wraps_around():
    """23:55 vs 00:05 must give the same distance as 00:00 vs 00:10 (10 min)."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df_a_cyclic = _dist_df("POINT (0 0)", "23:55", 1.0)
    df_b_cyclic = _dist_df("POINT (0 0)", "00:05", 1.0)
    df_a_linear = _dist_df("POINT (0 0)", "00:00", 1.0)
    df_b_linear = _dist_df("POINT (0 0)", "00:10", 1.0)

    dist_cyclic = stvd_emd(df_a_cyclic, df_b_cyclic, alpha=10.0, cyclical_period=1440.0)
    dist_linear = stvd_emd(df_a_linear, df_b_linear, alpha=10.0, cyclical_period=1440.0)
    assert dist_cyclic == pytest.approx(dist_linear, rel=1e-3)


def test_symmetry():
    """d(A, B) must equal d(B, A)."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

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
    assert d_ab == pytest.approx(d_ba, rel=2e-2)


def test_explicit_column_names():
    """Explicit column overrides should work."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

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
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    result = stvd_emd(df, df.copy())
    assert isinstance(result, float)


def test_invalid_num_projections_raises():
    """num_projections <= 0 must raise ValueError."""
    _skip_if_no_core()
    from fastmob.measures.evaluation import stvd_emd

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    with pytest.raises(ValueError, match="num_projections must be positive"):
        stvd_emd(df, df.copy(), num_projections=0)


def test_polars_parity():
    """Pandas and Polars inputs must produce the same result."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars to run this test")
    from fastmob.measures.evaluation import stvd_emd

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

    df = _dist_df("POINT (0 0)", "12:00", 1.0)
    expected = stvd_emd(df, df.copy())

    xs = np.array([0.0], dtype=np.float64)
    ys = np.array([0.0], dtype=np.float64)
    ts = np.array([720.0], dtype=np.float64)
    ws = np.array([1.0], dtype=np.float64)

    result = stvd_emd_core(xs, ys, ts, ws, xs, ys, ts, ws, 10.0, 1440.0, 50)
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

    xs = pa.array([0.0], type=pa.float64())
    ys = pa.array([0.0], type=pa.float64())
    ts = pa.array([720.0], type=pa.float64())
    ws = pa.array([1.0], type=pa.float64())

    result_arrow = stvd_emd_core(xs, ys, ts, ws, xs, ys, ts, ws, 10.0, 1440.0, 50)
    assert isinstance(result_arrow, float)

    xs_np = np.array([0.0], dtype=np.float64)
    ys_np = np.array([0.0], dtype=np.float64)
    ts_np = np.array([720.0], dtype=np.float64)
    ws_np = np.array([1.0], dtype=np.float64)
    result_numpy = stvd_emd_core(xs_np, ys_np, ts_np, ws_np, xs_np, ys_np, ts_np, ws_np, 10.0, 1440.0, 50)

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
            50,
        )
