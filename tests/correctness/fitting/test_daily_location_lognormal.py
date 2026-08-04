"""Correctness tests for fastmob.measures.fitting.fit_daily_location_lognormal."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastmob.core import Locations, Staypoints
from fastmob.measures.fitting import fit_daily_location_lognormal

# user1/day1: 2 distinct locations (home, work)
# user1/day2: 1 distinct location (work)
# user2/day1: 2 distinct locations (home, park)
# -> daily distinct-location counts = [2, 1, 2]
# log values = [ln2, 0, ln2]; mean = 2*ln2/3; population std (ddof=0) computed below.
_ROWS = [
    {"user_id": "u1", "location_id": "home", "timestamp": "2020-01-01"},
    {"user_id": "u1", "location_id": "work", "timestamp": "2020-01-01"},
    {"user_id": "u1", "location_id": "work", "timestamp": "2020-01-02"},
    {"user_id": "u2", "location_id": "home", "timestamp": "2020-01-01"},
    {"user_id": "u2", "location_id": "park", "timestamp": "2020-01-01"},
]

_EXPECTED_MU = 2 * math.log(2) / 3
_variance = ((math.log(2) - _EXPECTED_MU) ** 2 * 2 + (0.0 - _EXPECTED_MU) ** 2) / 3
_EXPECTED_SIGMA = math.sqrt(_variance)


def _pandas_fixture() -> pd.DataFrame:
    df = pd.DataFrame(_ROWS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_hand_computed_mu_sigma_pandas():
    x_points, y_points, mu, sigma = fit_daily_location_lognormal(_pandas_fixture())
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)
    assert sigma == pytest.approx(_EXPECTED_SIGMA, abs=1e-12)
    assert list(x_points) == [1.0, 2.0]
    assert y_points[0] == pytest.approx(1 / 3)
    assert y_points[1] == pytest.approx(2 / 3)


def test_hand_computed_mu_sigma_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pl.DataFrame(_ROWS).with_columns(pl.col("timestamp").str.to_datetime())
    x_points, _y_points, mu, sigma = fit_daily_location_lognormal(df)
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)
    assert sigma == pytest.approx(_EXPECTED_SIGMA, abs=1e-12)
    assert list(x_points) == [1.0, 2.0]


def test_explicit_column_overrides():
    df = _pandas_fixture().rename(columns={"user_id": "uid", "location_id": "loc"})
    _x_points, _y_points, mu, _sigma = fit_daily_location_lognormal(
        df, user_id_col="uid", location_id_col="loc", timestamp_col="timestamp"
    )
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)


def test_accepts_staypoints_and_uses_its_metadata():
    frame = _pandas_fixture().rename(columns={"timestamp": "started_at"})
    frame["finished_at"] = frame["started_at"] + pd.Timedelta(minutes=10)
    result = fit_daily_location_lognormal(Staypoints(frame))
    assert result[2] == pytest.approx(_EXPECTED_MU, abs=1e-12)
    assert result[3] == pytest.approx(_EXPECTED_SIGMA, abs=1e-12)


def test_validates_global_locations_catalogue():
    locations = Locations(
        pd.DataFrame(
            {
                "location_id": ["home", "work", "park"],
                "center_lat": [0.0, 0.0, 0.0],
                "center_lng": [0.0, 1.0, 2.0],
            }
        ),
        scope="global",
    )
    result = fit_daily_location_lognormal(_pandas_fixture(), locations=locations)
    assert result[2] == pytest.approx(_EXPECTED_MU, abs=1e-12)

    unknown = _pandas_fixture()
    unknown.loc[0, "location_id"] = "unknown"
    with pytest.raises(ValueError, match="absent from the global"):
        fit_daily_location_lognormal(unknown, locations=locations)


def test_validates_user_scoped_locations_with_overlapping_ids():
    visits = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "location_id": [0, 1, 0, 0],
            "timestamp": pd.to_datetime(["2020-01-01"] * 4),
        }
    )
    locations = Locations(
        pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "location_id": [0, 1, 0],
                "center_lat": [0.0, 1.0, 2.0],
                "center_lng": [0.0, 1.0, 2.0],
            }
        ),
        uid_col="user_id",
        scope="user",
    )
    x_points, probabilities, _mu, _sigma = fit_daily_location_lognormal(visits, locations=locations)
    assert x_points == [1.0, 2.0]
    assert probabilities == pytest.approx([0.5, 0.5])


def test_timezone_aware_inputs_bucket_by_local_wall_clock_day():
    local = _pandas_fixture()
    local["timestamp"] = local["timestamp"].dt.tz_localize("Europe/Lisbon")
    expected = fit_daily_location_lognormal(_pandas_fixture())
    result = fit_daily_location_lognormal(local)
    assert result[0] == expected[0]
    assert result[1] == pytest.approx(expected[1])
    assert result[2] == pytest.approx(expected[2])
    assert result[3] == pytest.approx(expected[3])


def test_too_few_points_raises():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["home"],
            "timestamp": pd.to_datetime(["2020-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="At least two"):
        fit_daily_location_lognormal(df)


def test_degenerate_zero_variance_raises():
    # Every user/day has exactly 1 distinct location -> log(1) == 0 for all -> sigma == 0.
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "location_id": ["home", "home", "home"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="positive log variance"):
        fit_daily_location_lognormal(df)


def test_missing_required_column_raises():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError):
        fit_daily_location_lognormal(df)
