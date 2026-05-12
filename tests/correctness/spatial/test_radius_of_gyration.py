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


def _as_pandas_result(result):
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    return result


def _rog_map(result) -> dict:
    result = _as_pandas_result(result)
    uid_col = next(c for c in ("uid", "user", "user_id") if c in result.columns)
    return dict(zip(result[uid_col], result["radius_of_gyration"]))


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


def test_radius_of_gyration_indexed_handles_interleaved_users():
    """RoG groups by uid without sorting the full dataframe."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_km
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df = pd.DataFrame(
        {
            "uid": ["b", "a", "b", "c", "a", "c"],
            "datetime": pd.date_range("2020-01-01", periods=6, freq="h"),
            "lat": [10.0, 0.0, 11.0, 20.0, 1.0, 21.0],
            "lng": [30.0, 0.0, 31.0, 40.0, 1.0, 41.0],
        }
    )

    result = radius_of_gyration(df)
    actual = _rog_map(result)

    assert list(result["uid"]) == ["a", "b", "c"]
    expected = {
        uid: radius_of_gyration_km(list(map(tuple, df.loc[df["uid"] == uid, ["lat", "lng"]].to_numpy())))
        for uid in ["a", "b", "c"]
    }
    for uid, expected_value in expected.items():
        np.testing.assert_allclose(actual[uid], expected_value, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_pandas_filters_invalid_coordinates_in_rust():
    """RoG ignores invalid coordinates without Python-side trajectory null drops."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_km
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df = pd.DataFrame(
        {
            "uid": ["b", "a", "b", "c", "a", "c"],
            "datetime": [pd.NaT] * 6,
            "lat": [10.0, 0.0, np.nan, 20.0, 1.0, None],
            "lng": [30.0, 0.0, 31.0, 40.0, 1.0, 41.0],
        }
    )

    result = radius_of_gyration(df)
    actual = _rog_map(result)

    assert list(result["uid"]) == ["a", "b", "c"]
    expected = {
        uid: radius_of_gyration_km(list(map(tuple, rows[["lat", "lng"]].to_numpy())))
        for uid, rows in df.dropna(subset=["lat", "lng"]).groupby("uid", sort=True)
    }
    assert set(actual) == set(expected)
    for uid, expected_value in expected.items():
        np.testing.assert_allclose(actual[uid], expected_value, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_polars_filters_invalid_coordinates_in_rust():
    """Arrow RoG path ignores null coordinates."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Polars not installed")
    from skmob2._core import radius_of_gyration_km
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df_pd = pd.DataFrame(
        {
            "uid": ["b", "a", "b", "c", "a", "c"],
            "datetime": [pd.NaT] * 6,
            "lat": [10.0, 0.0, None, 20.0, 1.0, None],
            "lng": [30.0, 0.0, 31.0, 40.0, 1.0, 41.0],
        }
    )
    result = radius_of_gyration(pl.from_pandas(df_pd)).to_pandas()
    actual = _rog_map(result)

    assert list(result["uid"]) == ["a", "b", "c"]
    expected = {
        uid: radius_of_gyration_km(list(map(tuple, rows[["lat", "lng"]].to_numpy())))
        for uid, rows in df_pd.dropna(subset=["lat", "lng"]).groupby("uid", sort=True)
    }
    assert set(actual) == set(expected)
    for uid, expected_value in expected.items():
        np.testing.assert_allclose(actual[uid], expected_value, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_no_uid_filters_invalid_coordinates():
    """No-uid RoG computes over valid coordinate rows only."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_km
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    df = pd.DataFrame(
        {
            "datetime": [pd.NaT] * 4,
            "lat": [0.0, np.nan, 1.0, None],
            "lng": [0.0, 1.0, 1.0, 2.0],
        }
    )

    result = radius_of_gyration(df)
    expected = radius_of_gyration_km(list(map(tuple, df.dropna(subset=["lat", "lng"])[["lat", "lng"]].to_numpy())))
    np.testing.assert_allclose(float(result["radius_of_gyration"].iloc[0]), expected, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_all_invalid_coordinates():
    """All-invalid inputs produce the empty-coordinate behavior."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration

    no_uid = pd.DataFrame(
        {
            "datetime": [pd.NaT, pd.NaT],
            "lat": [np.nan, None],
            "lng": [0.0, 1.0],
        }
    )
    no_uid_result = radius_of_gyration(no_uid)
    assert float(no_uid_result["radius_of_gyration"].iloc[0]) == 0.0

    with_uid = no_uid.assign(uid=["a", "b"])
    with_uid_result = radius_of_gyration(with_uid)
    assert len(with_uid_result) == 0


def test_build_indexed_user_ranges_sorts_stable_row_indices():
    """Indexed grouping keeps original row order within each sorted uid group."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    import narwhals as nw
    from skmob2.measures._common import _build_indexed_user_ranges

    df = nw.from_native(pd.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}), eager_only=True)

    uid_values, indices, ranges = _build_indexed_user_ranges(df, "uid")

    assert uid_values == ["a", "b", "c"]
    assert indices == [1, 4, 0, 2, 3, 5]
    assert ranges == [(0, 2), (2, 4), (4, 6)]


def test_build_indexed_user_ranges_handles_empty_dataframe():
    """Indexed grouping handles empty inputs without touching backend sort."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    import narwhals as nw
    from skmob2.measures._common import _build_indexed_user_ranges

    df = nw.from_native(pd.DataFrame({"uid": []}), eager_only=True)

    assert _build_indexed_user_ranges(df, "uid") == ([], [], [])


def test_radius_of_gyration_user_indices_numpy_helper():
    """Rust NumPy uid-index helper groups sorted stable row indexes."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_user_indices_numpy

    indices, starts, ends = radius_of_gyration_user_indices_numpy(np.array([2, 1, 2, 3, 1, 3], dtype=np.int64))

    assert isinstance(indices, np.ndarray)
    assert indices.tolist() == [1, 4, 0, 2, 3, 5]
    assert starts.tolist() == [0, 2, 4]
    assert ends.tolist() == [2, 4, 6]


def test_radius_of_gyration_user_indices_arrow_helper():
    """Rust Arrow uid-index helper supports string uid arrays."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import radius_of_gyration_user_indices_arrow

    indices, starts, ends = radius_of_gyration_user_indices_arrow(pa.array(["b", "a", "b", "c", "a", "c"]))

    assert isinstance(indices, np.ndarray)
    assert indices.tolist() == [1, 4, 0, 2, 3, 5]
    assert starts.tolist() == [0, 2, 4]
    assert ends.tolist() == [2, 4, 6]


def test_build_indexed_user_ranges_uses_arrow_for_polars_strings():
    """Indexed grouping keeps string uid support on the Arrow-backed path."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pl = pytest.importorskip("polars", reason="Polars not installed")
    import narwhals as nw
    from skmob2.measures._common import _build_indexed_user_ranges

    df = nw.from_native(pl.DataFrame({"uid": ["b", "a", "b", "c", "a", "c"]}), eager_only=True)

    uid_values, indices, ranges = _build_indexed_user_ranges(df, "uid")

    assert uid_values == ["a", "b", "c"]
    assert indices == [1, 4, 0, 2, 3, 5]
    assert ranges == [(0, 2), (2, 4), (4, 6)]


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


def test_radius_of_gyration_indexed_numpy_helper_matches_contiguous_helper():
    """Indexed helper must match contiguous helper when indexes encode the groups."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_indexed_numpy, radius_of_gyration_numpy

    lats = np.array([10.0, 0.0, 11.0, 20.0, 1.0, 21.0], dtype=np.float64)
    lngs = np.array([30.0, 0.0, 31.0, 40.0, 1.0, 41.0], dtype=np.float64)
    indices = np.array([1, 4, 0, 2, 3, 5], dtype=np.uintp)
    starts = np.array([0, 2, 4], dtype=np.uintp)
    ends = np.array([2, 4, 6], dtype=np.uintp)
    indexed_ranges = [(0, 2), (2, 4), (4, 6)]

    result = radius_of_gyration_indexed_numpy(lats, lngs, indices, starts, ends)
    expected_lats = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0], dtype=np.float64)
    expected_lngs = np.array([0.0, 1.0, 30.0, 31.0, 40.0, 41.0], dtype=np.float64)
    expected = radius_of_gyration_numpy(expected_lats, expected_lngs, indexed_ranges)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)


def test_radius_of_gyration_indexed_arrow_helper_matches_numpy_helper():
    """Arrow indexed helper must match the NumPy indexed helper."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import radius_of_gyration_indexed_arrow, radius_of_gyration_indexed_numpy

    lats = np.array([10.0, 0.0, 11.0, 20.0, 1.0, 21.0], dtype=np.float64)
    lngs = np.array([30.0, 0.0, 31.0, 40.0, 1.0, 41.0], dtype=np.float64)
    indices = np.array([1, 4, 0, 2, 3, 5], dtype=np.uintp)
    starts = np.array([0, 2, 4], dtype=np.uintp)
    ends = np.array([2, 4, 6], dtype=np.uintp)

    result_arrow = radius_of_gyration_indexed_arrow(
        pa.array(lats, type=pa.float64()),
        pa.array(lngs, type=pa.float64()),
        indices,
        starts,
        ends,
    )
    result_numpy = radius_of_gyration_indexed_numpy(lats, lngs, indices, starts, ends)

    np.testing.assert_allclose(np.asarray(result_arrow), result_numpy, rtol=0.0, atol=1e-12)


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


def test_radius_of_gyration_indexed_numpy_mismatched_lengths_raise():
    """Indexed helper validates latitude/longitude lengths."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_indexed_numpy

    lats = np.array([0.0, 1.0], dtype=np.float64)
    lngs = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="same length"):
        radius_of_gyration_indexed_numpy(
            lats,
            lngs,
            np.array([0], dtype=np.uintp),
            np.array([0], dtype=np.uintp),
            np.array([1], dtype=np.uintp),
        )


def test_radius_of_gyration_indexed_numpy_range_bounds_raise():
    """Indexed helper validates that ranges address the index array."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_indexed_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="index array bounds"):
        radius_of_gyration_indexed_numpy(
            arr,
            arr,
            np.array([0], dtype=np.uintp),
            np.array([0], dtype=np.uintp),
            np.array([2], dtype=np.uintp),
        )


def test_radius_of_gyration_indexed_numpy_index_bounds_raise():
    """Indexed helper validates that each index addresses coordinates."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2._core import radius_of_gyration_indexed_numpy

    arr = np.array([0.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="coordinate array bounds"):
        radius_of_gyration_indexed_numpy(
            arr,
            arr,
            np.array([0, 2], dtype=np.uintp),
            np.array([0], dtype=np.uintp),
            np.array([2], dtype=np.uintp),
        )


def test_radius_of_gyration_arrow_nulls_are_filtered():
    """Arrow helper skips null coordinates."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    pa = pytest.importorskip("pyarrow", reason="Install pyarrow to run this test")
    from skmob2._core import radius_of_gyration_arrow

    lats = pa.array([0.0, None], type=pa.float64())
    lngs = pa.array([0.0, 1.0], type=pa.float64())

    np.testing.assert_allclose(np.asarray(radius_of_gyration_arrow(lats, lngs, [(0, 2)])), [0.0], rtol=0.0, atol=1e-12)


@pytest.mark.skmob
def test_radius_of_gyration_matches_skmob(comparison_skmob):
    """skmob2 RoG must agree with skmob's reference implementation within 0.02 km."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob.measures.individual import radius_of_gyration as skmob_rog
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as skmob2_rog

    skmob_result = skmob_rog(comparison_skmob, show_progress=False)
    skmob2_input = pd.DataFrame(comparison_skmob).copy()
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


def test_radius_of_gyration_matches_cached_reference(comparison_skmob_reference):
    """RoG matches the cached skmob baseline without requiring the skmob environment."""
    pytest.importorskip("skmob2._core", reason="Run maturin develop first")
    from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as skmob2_rog

    ref = comparison_skmob_reference
    skmob_result = ref.result("radius_of_gyration")
    skmob2_result = skmob2_rog(ref.input_df)

    skmob_map = dict(zip(skmob_result["uid"], skmob_result["radius_of_gyration"]))
    skmob2_uid = next(c for c in ("uid", "user", "user_id") if c in skmob2_result.columns)
    skmob2_map = dict(zip(skmob2_result[skmob2_uid], skmob2_result["radius_of_gyration"]))

    assert set(skmob_map.keys()) == set(skmob2_map.keys()), "User sets differ"
    for uid in skmob_map:
        assert abs(skmob_map[uid] - skmob2_map[uid]) < 0.02, (
            f"RoG mismatch for uid={uid}: cached={skmob_map[uid]:.6f}, skmob2={skmob2_map[uid]:.6f}"
        )
