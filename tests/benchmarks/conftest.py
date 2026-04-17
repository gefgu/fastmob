"""Fixtures for benchmark tests.

Data directory: tests/benchmarks/data/
Shared with correctness suite so the Brightkite file is downloaded only once.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
import pytest
from ..shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL


# ---------------------------------------------------------------------------
# Raw dataset — downloaded once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def brightkite_raw() -> pd.DataFrame:
    """Download/cache Brightkite; return full 100k-row plain pandas DataFrame."""
    if not _BRIGHTKITE_PATH.exists():
        print(f"\nDownloading Brightkite dataset to {_BRIGHTKITE_PATH}...")
        urllib.request.urlretrieve(_BRIGHTKITE_URL, _BRIGHTKITE_PATH)
    else:
        print(f"\nUsing cached dataset at {_BRIGHTKITE_PATH}")

    return pd.read_csv(
        _BRIGHTKITE_PATH,
        sep="\t",
        header=0,
        # nrows=100_000,
        names=["user", "check-in_time", "latitude", "longitude", "location id"],
    )


@pytest.fixture(scope="session")
def brightkite_tdf(brightkite_raw):
    """Wrap full dataset as skmob.TrajDataFrame; skipped when skmob is absent."""
    skmob = pytest.importorskip("skmob")
    return skmob.TrajDataFrame(
        brightkite_raw,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


# ---------------------------------------------------------------------------
# Parametrized size / slice fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1_000, 10_000, 100_000, 1_000_000, 4_000_000], ids=["1k", "10k", "100k", "1M", "4M"])
def dataset_size(request):
    return request.param


@pytest.fixture
def bench_slice_pandas(brightkite_raw, dataset_size):
    """Slice of the raw pandas DataFrame for the current dataset_size."""
    return brightkite_raw.head(dataset_size).copy()


@pytest.fixture
def bench_slice_skmob(brightkite_tdf, dataset_size):
    """Slice of the skmob TrajDataFrame; skipped when skmob is absent."""
    skmob = pytest.importorskip("skmob")
    return skmob.TrajDataFrame(
        brightkite_tdf.head(dataset_size).copy(),
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


@pytest.fixture
def bench_slice_polars(brightkite_raw, dataset_size):
    """Slice of the raw data as a Polars DataFrame for the current dataset_size."""
    polars = pytest.importorskip("polars", reason="Install polars to run this benchmark")

    pandas_slice = brightkite_raw.head(dataset_size).copy()
    return polars.from_pandas(pandas_slice)


# ---------------------------------------------------------------------------
# movingpandas fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def brightkite_tc(brightkite_raw):
    """Build a movingpandas TrajectoryCollection from the full Brightkite dataset.

    Constructs Shapely Point geometries from longitude/latitude columns, sets the
    datetime index, and groups trajectories by the "user" column. Construction is
    session-scoped because building a GeoDataFrame + TrajectoryCollection at 4M rows
    is expensive (10–60 s) and must not be included in benchmark timing.

    @usedBy bench_slice_movingpandas_1k, bench_slice_movingpandas_10k,
             bench_slice_movingpandas_100k, bench_slice_movingpandas_1M,
             bench_slice_movingpandas_4M
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    gpd = pytest.importorskip("geopandas", reason="Install geopandas to run this benchmark")
    import pandas as pd

    df = brightkite_raw.copy()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf["check-in_time"] = pd.to_datetime(gdf["check-in_time"])
    gdf = gdf.set_index("check-in_time")
    return mpd.TrajectoryCollection(gdf, traj_id_col="user")


@pytest.fixture
def bench_slice_movingpandas_1k(brightkite_tc):
    """TrajectoryCollection slice of the first 1 000 rows.

    @usedBy test_jump_lengths_movingpandas (1k tier),
             test_radius_of_gyration_movingpandas (1k tier)
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    point_gdf = brightkite_tc.to_point_gdf().head(1_000)
    return mpd.TrajectoryCollection(point_gdf, traj_id_col="user")


@pytest.fixture
def bench_slice_movingpandas_10k(brightkite_tc):
    """TrajectoryCollection slice of the first 10 000 rows.

    @usedBy test_jump_lengths_movingpandas (10k tier),
             test_radius_of_gyration_movingpandas (10k tier)
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    point_gdf = brightkite_tc.to_point_gdf().head(10_000)
    return mpd.TrajectoryCollection(point_gdf, traj_id_col="user")


@pytest.fixture
def bench_slice_movingpandas_100k(brightkite_tc):
    """TrajectoryCollection slice of the first 100 000 rows.

    @usedBy test_jump_lengths_movingpandas (100k tier),
             test_radius_of_gyration_movingpandas (100k tier)
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    point_gdf = brightkite_tc.to_point_gdf().head(100_000)
    return mpd.TrajectoryCollection(point_gdf, traj_id_col="user")


@pytest.fixture
def bench_slice_movingpandas_1M(brightkite_tc):
    """TrajectoryCollection slice of the first 1 000 000 rows.

    @usedBy test_jump_lengths_movingpandas (1M tier),
             test_radius_of_gyration_movingpandas (1M tier)
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    point_gdf = brightkite_tc.to_point_gdf().head(1_000_000)
    return mpd.TrajectoryCollection(point_gdf, traj_id_col="user")


@pytest.fixture
def bench_slice_movingpandas_4M(brightkite_tc):
    """TrajectoryCollection slice of the first 4 000 000 rows (full dataset).

    @usedBy test_jump_lengths_movingpandas (4M tier),
             test_radius_of_gyration_movingpandas (4M tier)
    """
    mpd = pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")
    point_gdf = brightkite_tc.to_point_gdf().head(4_000_000)
    return mpd.TrajectoryCollection(point_gdf, traj_id_col="user")


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the movingpandas pytest marker for benchmark tests."""
    config.addinivalue_line(
        "markers",
        "movingpandas: benchmarks that require the movingpandas package to be installed",
    )
