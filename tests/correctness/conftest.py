"""Fixtures and constants for correctness tests."""
from __future__ import annotations

import pandas as pd
import pytest

from ..shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL

# ---------------------------------------------------------------------------
# Pre-computed expected values
# ---------------------------------------------------------------------------

# Synthetic trajectory: 3 users, 5 GPS points each.
# Values computed by running skmob2._core.jump_lengths_km on the same inputs.
EXPECTED_JUMP_LENGTHS: dict[str, list[float]] = {
    "user_a": [
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
    ],
    "user_b": [
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
        111.1950802335329,
    ],
    "user_c": [
        0.6845085181899847,
        0.9188177472926384,
        0.9187595626478804,
        0.9187013756985495,
    ],
}


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def _generate_synthetic_rows():
    """Centralized logic for generating test trajectory data."""
    rows = []
    
    # user_a: moves along the equator (1-degree steps)
    for i, lon in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
        rows.append({
            "uid": "user_a",
            "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
            "lat": 0.0,
            "lng": float(lon),
        })

    # user_b: moves along a meridian (1-degree steps)
    for i, lat in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
        rows.append({
            "uid": "user_b",
            "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
            "lat": float(lat),
            "lng": 20.0,
        })

    # user_c: small movements in Paris area
    lats = [48.8566, 48.8600, 48.8650, 48.8700, 48.8750]
    lons = [2.3522, 2.3600, 2.3700, 2.3800, 2.3900]
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        rows.append({
            "uid": "user_c",
            "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
            "lat": lat,
            "lng": lon,
        })
    return rows

@pytest.fixture(scope="session")
def synthetic_rows():
    """Raw data as a list of dicts to be shared across frameworks."""
    return _generate_synthetic_rows()

@pytest.fixture(scope="session")
def synthetic_tdf(synthetic_rows) -> pd.DataFrame:
    """Small Pandas DataFrame with 3 users, 5 GPS points each."""
    return pd.DataFrame(synthetic_rows)

@pytest.fixture(scope="session")
def synthetic_tdf_polars(synthetic_rows):
    """Small Polars DataFrame with 3 users, 5 GPS points each."""
    pl = pytest.importorskip("polars", reason="Polars not installed")
    # Using from_dicts is more direct than going through Pandas
    return pl.from_dicts(synthetic_rows)


# ---------------------------------------------------------------------------
# Optional skmob comparison fixture (skipped when skmob is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def brightkite_skmob():
    """Load Brightkite data as skmob.TrajDataFrame; skipped when skmob is absent."""
    import urllib.request

    skmob = pytest.importorskip("skmob")

    _BRIGHTKITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _BRIGHTKITE_PATH.exists():
        print(f"\nDownloading Brightkite dataset to {_BRIGHTKITE_PATH}...")
        urllib.request.urlretrieve(_BRIGHTKITE_URL, _BRIGHTKITE_PATH)
    else:
        print(f"\nUsing cached dataset at {_BRIGHTKITE_PATH}")

    import pandas as pd

    df = pd.read_csv(
        _BRIGHTKITE_PATH,
        sep="\t",
        header=0,
        nrows=100_000,
        names=["user", "check-in_time", "latitude", "longitude", "location id"],
    )
    return skmob.TrajDataFrame(
        df,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skmob: tests that require the skmob package to be installed",
    )
