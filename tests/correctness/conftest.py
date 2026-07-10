"""Fixtures and constants for correctness tests."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from ..shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from ..shared.foursquare import FOURSQUARE_DEFAULT_ROWS, load_foursquare_pandas
from ..shared.geolife import GEOLIFE_DEFAULT_ROWS, load_geolife_pandas

# ---------------------------------------------------------------------------
# Pre-computed expected values
# ---------------------------------------------------------------------------

# Synthetic trajectory: 3 users, 5 GPS points each.
# Values computed by running fkmob._core.jump_lengths_km on the same inputs.
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
        rows.append(
            {
                "uid": "user_a",
                "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
                "lat": 0.0,
                "lng": float(lon),
            }
        )

    # user_b: moves along a meridian (1-degree steps)
    for i, lat in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
        rows.append(
            {
                "uid": "user_b",
                "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
                "lat": float(lat),
                "lng": 20.0,
            }
        )

    # user_c: small movements in Paris area
    lats = [48.8566, 48.8600, 48.8650, 48.8700, 48.8750]
    lons = [2.3522, 2.3600, 2.3700, 2.3800, 2.3900]
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        rows.append(
            {
                "uid": "user_c",
                "datetime": pd.Timestamp(f"2020-01-01 0{i}:00:00"),
                "lat": lat,
                "lng": lon,
            }
        )
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


def _import_skmob_or_skip():
    try:
        return importlib.import_module("skmob")
    except Exception as exc:
        pytest.skip(f"skmob is not importable: {exc}")


@pytest.fixture(scope="session")
def brightkite_skmob():
    """Load Brightkite data as skmob.TrajDataFrame; skipped when skmob is absent."""
    import urllib.request

    skmob = _import_skmob_or_skip()

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
    # skmob sorts per-user trajectories by datetime only. Rows that share the
    # same user and timestamp therefore have backend/version-dependent order in
    # order-sensitive measures, so remove that ambiguity for exact comparisons.
    df = df.sort_values(["user", "check-in_time", "latitude", "longitude", "location id"]).drop_duplicates(
        ["user", "check-in_time"],
        keep="first",
    )
    return skmob.TrajDataFrame(
        df,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


@pytest.fixture(scope="session")
def geolife_pd(pytestconfig):
    """Load GeoLife data as a normalized pandas DataFrame."""
    mode = pytestconfig.getoption("--geolife-mode")
    rows = pytestconfig.getoption("--geolife-rows")
    try:
        return load_geolife_pandas(mode=mode, rows=rows)
    except Exception as exc:
        pytest.skip(f"GeoLife dataset is not available: {exc}")


@pytest.fixture(scope="session")
def geolife_skmob(pytestconfig):
    """Load GeoLife data as skmob.TrajDataFrame; skipped when skmob is absent."""
    skmob = _import_skmob_or_skip()
    mode = pytestconfig.getoption("--geolife-mode")
    rows = pytestconfig.getoption("--geolife-rows")
    try:
        geolife_pd = load_geolife_pandas(mode=mode, rows=rows)
    except Exception as exc:
        pytest.skip(f"GeoLife dataset is not available: {exc}")
    return skmob.TrajDataFrame(
        geolife_pd,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


@pytest.fixture(scope="session")
def foursquare_pd(pytestconfig):
    """Load Foursquare NYC data as a normalized pandas DataFrame."""
    mode = pytestconfig.getoption("--foursquare-mode")
    rows = pytestconfig.getoption("--foursquare-rows")
    try:
        return load_foursquare_pandas(mode=mode, rows=rows)
    except Exception as exc:
        pytest.skip(f"Foursquare NYC dataset is not available: {exc}")


@pytest.fixture(scope="session")
def foursquare_skmob(pytestconfig):
    """Load Foursquare NYC data as skmob.TrajDataFrame; skipped when skmob is absent."""
    skmob = _import_skmob_or_skip()
    mode = pytestconfig.getoption("--foursquare-mode")
    rows = pytestconfig.getoption("--foursquare-rows")
    try:
        foursquare_pd = load_foursquare_pandas(mode=mode, rows=rows)
    except Exception as exc:
        pytest.skip(f"Foursquare NYC dataset is not available: {exc}")
    return skmob.TrajDataFrame(
        foursquare_pd,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


@pytest.fixture(scope="session", params=["brightkite", "geolife", "foursquare"])
def comparison_skmob(request):
    """Dataset-backed skmob.TrajDataFrame for skmob/fkmob parity tests."""
    if request.param == "brightkite":
        return request.getfixturevalue("brightkite_skmob")
    if request.param == "geolife":
        return request.getfixturevalue("geolife_skmob")
    if request.param == "foursquare":
        return request.getfixturevalue("foursquare_skmob")
    raise AssertionError(f"Unknown comparison dataset: {request.param}")


@pytest.fixture(scope="session", params=["brightkite", "geolife", "foursquare"])
def comparison_skmob_reference(request):
    """Cached skmob baseline; auto-skips when the cache is absent.

    Run ``bash scripts/populate_skmob_cache.sh`` (inside .venv-skmob) once to
    populate the cache, then commit tests/shared/skmob_reference/ to git.
    After that, these tests run in the normal .venv without skmob installed.
    """
    from tests.shared.skmob_cache import SkmobReferenceDataset, _REFERENCE_DIR

    dataset = request.param
    if not (_REFERENCE_DIR / dataset / "input.parquet").exists():
        pytest.skip(
            f"No skmob reference cache for '{dataset}'. "
            "Run 'bash scripts/populate_skmob_cache.sh' first."
        )
    return SkmobReferenceDataset(dataset)


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    group = parser.getgroup("fkmob correctness")
    group.addoption(
        "--geolife-mode",
        action="store",
        choices=("slice", "full"),
        default="slice",
        help="GeoLife comparison mode for skmob correctness tests. Defaults to a deterministic slice.",
    )
    group.addoption(
        "--geolife-rows",
        action="store",
        type=int,
        default=GEOLIFE_DEFAULT_ROWS,
        help=f"Maximum GeoLife rows to load in slice mode. Defaults to {GEOLIFE_DEFAULT_ROWS}.",
    )
    group.addoption(
        "--foursquare-mode",
        action="store",
        choices=("slice", "full"),
        default="slice",
        help="Foursquare NYC comparison mode for skmob correctness tests. Defaults to a deterministic slice.",
    )
    group.addoption(
        "--foursquare-rows",
        action="store",
        type=int,
        default=FOURSQUARE_DEFAULT_ROWS,
        help=f"Maximum Foursquare NYC rows to load in slice mode. Defaults to {FOURSQUARE_DEFAULT_ROWS}.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skmob: tests that require the skmob package to be installed",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that require long runtimes (skip with -m 'not slow')",
    )
