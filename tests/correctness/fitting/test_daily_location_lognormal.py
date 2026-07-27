"""Correctness tests for fastmob.measures.fitting.daily_location_lognormal_fit."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastmob.measures.fitting import daily_location_lognormal_fit

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
    x_points, y_points, mu, sigma = daily_location_lognormal_fit(_pandas_fixture())
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)
    assert sigma == pytest.approx(_EXPECTED_SIGMA, abs=1e-12)
    assert list(x_points) == [1.0, 2.0]
    assert y_points[0] == pytest.approx(1 / 3)
    assert y_points[1] == pytest.approx(2 / 3)


def test_hand_computed_mu_sigma_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pl.DataFrame(_ROWS).with_columns(pl.col("timestamp").str.to_datetime())
    x_points, _y_points, mu, sigma = daily_location_lognormal_fit(df)
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)
    assert sigma == pytest.approx(_EXPECTED_SIGMA, abs=1e-12)
    assert list(x_points) == [1.0, 2.0]


def test_explicit_column_overrides():
    df = _pandas_fixture().rename(columns={"user_id": "uid", "location_id": "loc"})
    _x_points, _y_points, mu, _sigma = daily_location_lognormal_fit(
        df, user_id_col="uid", location_id_col="loc", timestamp_col="timestamp"
    )
    assert mu == pytest.approx(_EXPECTED_MU, abs=1e-12)


def test_too_few_points_raises():
    df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "location_id": ["home"],
            "timestamp": pd.to_datetime(["2020-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="At least two"):
        daily_location_lognormal_fit(df)


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
        daily_location_lognormal_fit(df)


def test_missing_required_column_raises():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError):
        daily_location_lognormal_fit(df)
