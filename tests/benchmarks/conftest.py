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
