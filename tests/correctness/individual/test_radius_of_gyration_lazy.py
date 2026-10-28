"""Correctness tests for fastmob/measures/individual/radius_of_gyration_lazy.py."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastmob import radius_of_gyration, radius_of_gyration_lazy
from fastmob._core import RadiusOfGyrationStream

from ...shared.brightkite import _BRIGHTKITE_PATH

CHUNK_SIZES = [1, 7, 1000, None]
# Users whose points all coincide have a true radius of 0; the eager and
# streamed sums round differently there, so compare near-zero values absolutely.
ATOL_KM = 1e-6


def _random_traj(n: int = 5000, users: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "uid": rng.integers(0, users, n),
            "datetime": np.arange(n),
            "lat": rng.uniform(-60, 60, n),
            "lng": rng.uniform(-170, 170, n),
        }
    )


def _assert_matches_eager(df: pl.DataFrame, chunk_size: int | None, **kwargs) -> pl.DataFrame:
    expected = radius_of_gyration(df, **kwargs)
    result = radius_of_gyration_lazy(df.lazy(), chunk_size=chunk_size, **kwargs)
    assert isinstance(result, pl.DataFrame)
    assert result.columns == expected.columns
    for column in expected.columns[:-1]:
        assert result[column].to_list() == expected[column].to_list()
    np.testing.assert_allclose(
        result["radius_of_gyration"].to_numpy(),
        expected["radius_of_gyration"].to_numpy(),
        rtol=1e-12,
        atol=ATOL_KM,
    )
    return result


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_interleaved_users_match_eager(chunk_size):
    _assert_matches_eager(_random_traj(), chunk_size)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_grouped_users_match_eager(chunk_size):
    # Grouped batches take the contiguous per-batch path; users still straddle batches.
    _assert_matches_eager(_random_traj().sort("uid", maintain_order=True), chunk_size)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_invalid_coordinates_match_eager(chunk_size):
    df = _random_traj().with_columns(
        pl.when(pl.int_range(pl.len()) % 13 == 0).then(None).otherwise(pl.col("lat")).alias("lat"),
        pl.when(pl.int_range(pl.len()) % 17 == 0).then(float("nan")).otherwise(pl.col("lng")).alias("lng"),
    )
    _assert_matches_eager(df, chunk_size)


@pytest.mark.parametrize(
    "uids",
    [
        ["b", "a", "b", "c", "a", "c"],
        [3.5, 1.0, 3.5, 2.0, 1.0, 2.0],
        [True, False, True, True, False, False],
    ],
    ids=["string", "float", "bool"],
)
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_uid_dtypes_match_eager(uids, chunk_size):
    df = pl.DataFrame(
        {
            "uid": uids,
            "datetime": range(6),
            "lat": [45.0, 10.0, 45.3, 30.0, 10.1, 30.5],
            "lng": [9.0, 20.0, 9.4, 1.0, 20.2, 1.5],
        }
    )
    _assert_matches_eager(df, chunk_size)


def test_null_uid_is_its_own_user():
    df = pl.DataFrame(
        {
            "uid": [1, None, 1, None],
            "datetime": range(4),
            "lat": [45.0, 10.0, 45.3, 10.1],
            "lng": [9.0, 20.0, 9.4, 20.2],
        }
    )
    _assert_matches_eager(df, 1)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_no_uid_column(chunk_size):
    _assert_matches_eager(_random_traj().drop("uid"), chunk_size)


def test_all_invalid_user_is_filtered_with_warning():
    df = pl.DataFrame(
        {
            "uid": [1, 2, 1, 2],
            "datetime": range(4),
            "lat": [45.0, None, 45.3, float("nan")],
            "lng": [9.0, 20.0, 9.4, 20.2],
        }
    )
    with pytest.warns(RuntimeWarning, match="filtered out 1 user"):
        result = radius_of_gyration_lazy(df.lazy(), chunk_size=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = radius_of_gyration(df)
    assert result["uid"].to_list() == expected["uid"].to_list() == [1]


def test_explicit_columns_and_autodetected_aliases():
    df = _random_traj(500, 10).rename({"uid": "user", "lat": "latitude", "lng": "lon", "datetime": "time"})
    _assert_matches_eager(df, 64)
    renamed = df.rename({"user": "who", "latitude": "y", "lon": "x"})
    _assert_matches_eager(renamed, 64, uid_col="who", lat_col="y", lng_col="x")


def test_empty_lazy_frame():
    df = _random_traj(10, 2).filter(pl.col("lat") > 1000)
    result = radius_of_gyration_lazy(df.lazy())
    assert result.shape == (0, 2)
    assert result.columns == ["uid", "radius_of_gyration"]


def test_scan_parquet_source(tmp_path):
    df = _random_traj(20_000, 400, seed=3)
    path = tmp_path / "traj.parquet"
    df.write_parquet(path, row_group_size=1500)
    expected = radius_of_gyration(df)
    result = radius_of_gyration_lazy(pl.scan_parquet(path), chunk_size=2048)
    assert result["uid"].to_list() == expected["uid"].to_list()
    np.testing.assert_allclose(result["radius_of_gyration"], expected["radius_of_gyration"], rtol=1e-12, atol=ATOL_KM)


@pytest.mark.skipif(not _BRIGHTKITE_PATH.exists(), reason="Brightkite dataset not cached")
def test_brightkite_slice_matches_eager(tmp_path):
    df = pl.read_csv(
        _BRIGHTKITE_PATH,
        separator="\t",
        has_header=False,
        new_columns=["uid", "datetime", "lat", "lng", "location_id"],
        n_rows=200_000,
    )
    path = tmp_path / "brightkite.parquet"
    df.write_parquet(path)
    expected = radius_of_gyration(df)
    result = radius_of_gyration_lazy(pl.scan_parquet(path), chunk_size=25_000)
    assert result["uid"].to_list() == expected["uid"].to_list()
    np.testing.assert_allclose(result["radius_of_gyration"], expected["radius_of_gyration"], rtol=1e-12, atol=ATOL_KM)


@pytest.mark.parametrize(
    "traj",
    [pd.DataFrame({"uid": [1], "datetime": [0], "lat": [1.0], "lng": [1.0]}), pl.DataFrame({"lat": [1.0]})],
    ids=["pandas", "polars-eager"],
)
def test_non_lazy_input_raises(traj):
    with pytest.raises(TypeError, match="Polars LazyFrame"):
        radius_of_gyration_lazy(traj)


def test_stream_rejects_rows_unseen_in_first_pass():
    import pyarrow as pa

    stream = RadiusOfGyrationStream()
    uids = pa.chunked_array([[1, 1]])
    lats = pa.chunked_array([[1.0, 2.0]])
    lngs = pa.chunked_array([[1.0, 2.0]])
    stream.accumulate_moments(uids, lats, lngs)
    stream.finalize_centers()
    with pytest.raises(ValueError, match="not seen in the first pass"):
        stream.accumulate_sq(pa.chunked_array([[2, 2]]), lats, lngs)
    stream.accumulate_sq(uids[:1], lats[:1], lngs[:1])
    with pytest.raises(ValueError, match="different rows"):
        stream.finalize()


def test_stream_aligns_misaligned_chunks_and_locks_uid_type():
    import pyarrow as pa

    uids = pa.chunked_array([["a", "b", "a"], ["b"]])
    lats = pa.chunked_array([[45.0], [10.0, 45.3, 10.1]])
    lngs = pa.chunked_array([[9.0, 20.0], [9.4, 20.2]])
    stream = RadiusOfGyrationStream()
    stream.accumulate_moments(uids, lats, lngs)
    with pytest.raises(TypeError, match="changed between batches"):
        stream.accumulate_moments(pa.chunked_array([[1]]), lats[:1], lngs[:1])
    stream.finalize_centers()
    stream.accumulate_sq(uids, lats, lngs)
    labels, values, valid = stream.finalize()
    expected = radius_of_gyration(
        pl.DataFrame(
            {"uid": ["a", "b", "a", "b"], "datetime": range(4), "lat": lats.to_pylist(), "lng": lngs.to_pylist()}
        )
    )
    assert labels.to_pylist() == expected["uid"].to_list()
    np.testing.assert_allclose(values.to_numpy(), expected["radius_of_gyration"].to_numpy(), rtol=1e-12)
    assert valid.all()


@pytest.mark.parametrize(
    "uid",
    [
        pl.Series([3, 1, 3, 1], dtype=pl.UInt8),
        pl.Series([3, 1, 3, 1], dtype=pl.Int32),
        pl.Series(["2020-01-02", "2020-01-01"] * 2).str.to_date(),
        pl.Series([b"x", b"y", b"x", b"y"]),
    ],
    ids=["uint8", "int32", "date", "binary"],
)
def test_uid_dtype_round_trips(uid):
    df = pl.DataFrame(
        {"uid": uid, "datetime": range(4), "lat": [45.0, 10.0, 45.3, 10.1], "lng": [9.0, 20.0, 9.4, 20.2]}
    )
    result = _assert_matches_eager(df, 3)
    assert result["uid"].dtype == df["uid"].dtype
