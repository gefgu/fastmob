"""Fixtures for fastmob.trajectory correctness tests."""

from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# interpolate() fixtures
# ---------------------------------------------------------------------------
# user "gap": points at hours 0, 2, 3 (lat 0, 1, 2; lng 0). With
#   sampling_rate_s=3600 (1h): the 0h->2h gap (7200s > 3600s) gets exactly one
#   inserted point at hour 1; the 2h->3h gap (3600s, not strictly greater) is
#   left untouched.
# user "no_gap": points spaced exactly at the sampling rate (0,1,2,3h) -> no
#   insertions at all, regardless of method.
# user "accelerating": 4 points (t=0,1,2,4h; lat=0,1,1,7) chosen so the
#   velocity feeding kinematic's boundary condition (0 deg/h, from point 1 to
#   point 2) differs sharply from the gap's own secant slope (3 deg/h, from
#   point 2 to point 3) -- this makes kinematic's Hermite-cubic fit visibly
#   diverge from a plain linear interpolation at the inserted point.


def _interpolate_rows():
    rows = []
    for i, (hour, lat) in enumerate(zip([0, 2, 3], [0.0, 1.0, 2.0])):
        rows.append(
            {
                "uid": "gap",
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hour),
                "lat": lat,
                "lng": 0.0,
            }
        )
    for hour in range(4):
        rows.append(
            {
                "uid": "no_gap",
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hour),
                "lat": float(hour),
                "lng": 0.0,
            }
        )
    for hour, lat in zip([0, 1, 2, 4], [0.0, 1.0, 1.0, 7.0]):
        rows.append(
            {
                "uid": "accelerating",
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hour),
                "lat": lat,
                "lng": 0.0,
            }
        )
    return rows


@pytest.fixture(scope="session")
def interpolate_tdf() -> pd.DataFrame:
    return pd.DataFrame(_interpolate_rows())


@pytest.fixture(scope="session")
def interpolate_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_interpolate_rows())


# ---------------------------------------------------------------------------
# interpolate_at() fixtures
# ---------------------------------------------------------------------------
# Single user, 3 points 1 hour apart, constant velocity of 1 deg lat/hour.


def _interpolate_at_rows():
    return [
        {
            "uid": "u1",
            "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hour),
            "lat": float(hour),
            "lng": 0.0,
        }
        for hour in range(3)
    ]


@pytest.fixture(scope="session")
def interpolate_at_tdf() -> pd.DataFrame:
    return pd.DataFrame(_interpolate_at_rows())


# ---------------------------------------------------------------------------
# trajectory_distance() fixtures
# ---------------------------------------------------------------------------
# seq_translated: seq_b is seq_a translated by exactly +1 degree latitude
#   everywhere, same longitudes, same point count/order -> the identity
#   alignment is optimal for every metric, so Fréchet == Hausdorff == the
#   single point-to-point haversine distance at that latitude offset, and DTW
#   == that same per-point distance summed over every point (since deviating
#   from the identity alignment can only add extra, non-zero-cost pairings).
# seq_identical: seq_b == seq_a exactly -> every metric collapses to its own
#   identity value (0 km distance, or 1.0 LCSS similarity).


def _line_km_rows(lat_values, lng_values, hours):
    return pd.DataFrame(
        {
            "lat": lat_values,
            "lng": lng_values,
            "datetime": [pd.Timestamp("2020-01-01") + pd.Timedelta(hours=h) for h in hours],
        }
    )


@pytest.fixture(scope="session")
def distance_seq_a() -> pd.DataFrame:
    return _line_km_rows([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0, 1, 2])


@pytest.fixture(scope="session")
def distance_seq_b_translated() -> pd.DataFrame:
    return _line_km_rows([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0, 1, 2])
