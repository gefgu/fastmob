"""Correctness tests for skmob2/measures/radius_of_gyration.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Expected RoG for the synthetic 3-user fixture (user_a/b/c, 5 pts each).
EXPECTED_ROG: dict[str, float] = {
    "user_a": 157.25337332781638,
    "user_b": 157.25337332781643,
    "user_c": 1.2345928209968795,
}

# Reference values from the skmob test suite (9 points, 3 users: 1,2,3)
_SKMOB_TEST_LATS_LNGS = np.array(
    [
        [39.978253, 116.3272755],
        [40.013819, 116.306532],
        [39.878987, 116.1266865],
        [40.013819, 116.306532],
        [39.97958, 116.313649],
        [39.978696, 116.3262205],
        [39.98153775, 116.31079],
        [39.978161, 116.3272425],
        [38.978161, 115.3272425],
    ]
)
_SKMOB_TEST_UIDS = [1, 1, 1, 1, 1, 2, 2, 2, 3]
_SKMOB_TEST_DTS = [
    "2013-01-01 08:34:04",
    "2013-01-01 10:34:08",
    "2013-01-05 10:34:08",
    "2013-01-10 12:34:15",
    "2013-01-01 01:34:28",
    "2013-01-01 03:34:54",
    "2013-01-01 04:34:55",
    "2013-01-05 05:29:12",
    "2013-01-15 00:29:12",
]


def _haversine_km_py(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Pure-Python Haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _expected_rog(lats_lngs: np.ndarray) -> float:
    cm = np.mean(lats_lngs, axis=0)
    dists_sq = [_haversine_km_py(lat, lon, cm[0], cm[1]) ** 2 for lat, lon in lats_lngs]
    return float(np.sqrt(np.mean(dists_sq)))


@pytest.fixture(scope="module")
def skmob_ref_traj_pd():
    """Pandas DataFrame matching the original skmob radius_of_gyration test."""
    return pd.DataFrame(
        {
            "lat": _SKMOB_TEST_LATS_LNGS[:, 0].tolist(),
            "lng": _SKMOB_TEST_LATS_LNGS[:, 1].tolist(),
            "datetime": pd.to_datetime(_SKMOB_TEST_DTS),
            "uid": _SKMOB_TEST_UIDS,
        }
    )


@pytest.fixture(scope="module")
def skmob_ref_traj_pl(skmob_ref_traj_pd):
    """Polars DataFrame matching the original skmob radius_of_gyration test."""
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_pandas(skmob_ref_traj_pd)


def test_radius_of_gyration_known_values_pandas(synthetic_tdf):
    """RoG on synthetic fixture (pandas backend), hardcoded expected values."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    result = radius_of_gyration(synthetic_tdf)
    assert "radius_of_gyration" in result.columns

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)

    for uid, expected in EXPECTED_ROG.items():
        row = result[result[uid_col] == uid]
        assert len(row) == 1, f"Expected exactly one row for uid={uid!r}"
        actual = float(row["radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"RoG mismatch for uid={uid!r}",
        )


def test_radius_of_gyration_known_values_polars(synthetic_tdf_polars):
    """RoG on synthetic fixture (Polars backend), same expected values."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pytest.importorskip("polars", reason="Polars not installed")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    result = radius_of_gyration(synthetic_tdf_polars)
    assert "radius_of_gyration" in result.columns

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)

    result_pd = result.to_pandas()
    for uid, expected in EXPECTED_ROG.items():
        row = result_pd[result_pd[uid_col] == uid]
        assert len(row) == 1, f"Expected exactly one row for uid={uid!r}"
        actual = float(row["radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"RoG mismatch for uid={uid!r} (Polars)",
        )


@pytest.mark.parametrize("fixture_name", ["skmob_ref_traj_pd", "skmob_ref_traj_pl"])
def test_radius_of_gyration_skmob_reference_values(request, fixture_name):
    """RoG matches values computed by the Rust kernel directly."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_km
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    traj = request.getfixturevalue(fixture_name)
    result = radius_of_gyration(traj)

    if hasattr(result, "to_pandas"):
        result = result.to_pandas()

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)

    uids_arr = np.array(_SKMOB_TEST_UIDS)
    for uid in (1, 2, 3):
        pts = _SKMOB_TEST_LATS_LNGS[uids_arr == uid]
        expected = radius_of_gyration_km(list(map(tuple, pts)))
        row = result[result[uid_col] == uid]
        assert len(row) == 1, f"Expected one row for uid={uid}"
        actual = float(row["radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"RoG mismatch for uid={uid} (fixture={fixture_name})",
        )


def test_radius_of_gyration_no_uid_column():
    """When uid column is absent the result has a single RoG value."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=5, freq="h"),
            "lat": [0.0, 0.0, 0.0, 0.0, 0.0],
            "lng": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    result = radius_of_gyration(df)
    assert "radius_of_gyration" in result.columns
    assert len(result) == 1
    pts = np.array([(0.0, lon) for lon in [0.0, 1.0, 2.0, 3.0, 4.0]])
    expected = _expected_rog(pts)
    np.testing.assert_allclose(
        float(result["radius_of_gyration"].iloc[0]),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_radius_of_gyration_single_point_user():
    """A user with one point should have RoG = 0."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df = pd.DataFrame(
        {
            "uid": [1, 2, 2],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "lat": [10.0, 20.0, 21.0],
            "lng": [50.0, 60.0, 61.0],
        }
    )
    result = radius_of_gyration(df)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()

    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)
    rog_u1 = float(result[result[uid_col] == 1]["radius_of_gyration"].iloc[0])
    assert rog_u1 == pytest.approx(0.0, abs=1e-10), "Single-point user must have RoG = 0"


def test_radius_of_gyration_polars_pandas_agree():
    """Polars and pandas backends must produce identical RoG values."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Polars not installed")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df_pd = pd.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2, 2],
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "lng": [0.0, 0.0, 0.0, 20.0, 20.0, 20.0],
        }
    )
    df_pl = pl.from_pandas(df_pd)

    res_pd = radius_of_gyration(df_pd)
    res_pl = radius_of_gyration(df_pl).to_pandas()

    uid_col = next(c for c in ("uid", "user", "user_id") if c in res_pd.columns)
    for uid in (1, 2):
        val_pd = float(res_pd[res_pd[uid_col] == uid]["radius_of_gyration"].iloc[0])
        val_pl = float(res_pl[res_pl[uid_col] == uid]["radius_of_gyration"].iloc[0])
        np.testing.assert_allclose(
            val_pd, val_pl, rtol=1e-12, atol=1e-12, err_msg=f"pandas/polars disagree for uid={uid}"
        )


def test_radius_of_gyration_numpy_helper_matches_batch_helper():
    """Zero-copy numpy helper must match the compatibility batch helper."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_batch_km, radius_of_gyration_numpy

    lats = _SKMOB_TEST_LATS_LNGS[:, 0].astype(np.float64)
    lngs = _SKMOB_TEST_LATS_LNGS[:, 1].astype(np.float64)
    ranges = [(0, 5), (5, 8), (8, 9)]

    result = radius_of_gyration_numpy(lats, lngs, ranges)
    expected = radius_of_gyration_batch_km(lats.tolist(), lngs.tolist(), ranges)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_arrow_helper_matches_numpy_helper():
    """Arrow helper must match the numpy helper."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import radius_of_gyration_arrow, radius_of_gyration_numpy

    lats_np = _SKMOB_TEST_LATS_LNGS[:, 0].astype(np.float64)
    lngs_np = _SKMOB_TEST_LATS_LNGS[:, 1].astype(np.float64)
    ranges = [(0, 5), (5, 8), (8, 9)]

    result_arrow = radius_of_gyration_arrow(
        pa.array(lats_np, type=pa.float64()),
        pa.array(lngs_np, type=pa.float64()),
        ranges,
    )
    result_numpy = radius_of_gyration_numpy(lats_np, lngs_np, ranges)

    np.testing.assert_allclose(result_arrow, result_numpy, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_numpy_non_contiguous_raises():
    """Non-contiguous numpy arrays must raise before copying."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_numpy

    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    non_contig = arr[::2]

    with pytest.raises((ValueError, BufferError, TypeError)):
        radius_of_gyration_numpy(non_contig, non_contig, [(0, len(non_contig))])


def test_radius_of_gyration_numpy_mismatched_lengths_raise():
    """Latitude/longitude arrays must have matching lengths."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_numpy

    lats = np.array([0.0, 1.0], dtype=np.float64)
    lngs = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="same length"):
        radius_of_gyration_numpy(lats, lngs, [(0, 1)])


def test_radius_of_gyration_arrow_nulls_raise():
    """Arrow helper rejects nulls because the Rust slice path cannot represent them."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import radius_of_gyration_arrow

    lats = pa.array([0.0, None], type=pa.float64())
    lngs = pa.array([0.0, 1.0], type=pa.float64())

    with pytest.raises(ValueError, match="must not contain nulls"):
        radius_of_gyration_arrow(lats, lngs, [(0, 2)])


@pytest.mark.skmob
def test_radius_of_gyration_matches_skmob(brightkite_skmob):
    """skmob2 RoG must agree with skmob's reference implementation within 0.02 km."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import radius_of_gyration as skmob_rog
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as skmob2_rog

    skmob_result = skmob_rog(brightkite_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_rog(skmob2_input)

    skmob_uid = next(c for c in ("uid", "user", "user_id") if c in skmob_result.columns)
    skmob2_uid = next(c for c in ("uid", "user", "user_id") if c in skmob2_result.columns)

    skmob_map = dict(zip(skmob_result[skmob_uid], skmob_result["radius_of_gyration"]))
    skmob2_map = dict(zip(skmob2_result[skmob2_uid], skmob2_result["radius_of_gyration"]))

    assert set(skmob_map.keys()) == set(skmob2_map.keys()), "User sets differ"
    for uid in skmob_map:
        assert abs(skmob_map[uid] - skmob2_map[uid]) < 0.02, (
            f"RoG mismatch for uid={uid}: skmob={skmob_map[uid]:.6f}, skmob2={skmob2_map[uid]:.6f}"
        )
