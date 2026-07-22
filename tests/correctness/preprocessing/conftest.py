"""Fixtures for preprocessing correctness tests."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# filter fixtures
# ---------------------------------------------------------------------------
# user_clean: 5 points moving slowly (~0.5 km apart, 1h spacing) → all survive
# user_noisy: same but point at index 2 is teleported to lat=0,lng=0 → speed >> 500 km/h → filtered


def _filter_rows():
    rows = []
    # user_clean: Paris → Paris+0.005° lat per step (≈0.55 km)
    for i in range(5):
        rows.append(
            {
                "uid": "user_clean",
                "datetime": pd.Timestamp(f"2020-01-01 {i:02d}:00:00"),
                "lat": 48.8566 + i * 0.005,
                "lng": 2.3522,
            }
        )
    # user_noisy: same trajectory but index 2 teleported to null island
    for i in range(5):
        lat = 0.0 if i == 2 else (48.8566 + i * 0.005)
        lng = 0.0 if i == 2 else 2.3522
        rows.append(
            {
                "uid": "user_noisy",
                "datetime": pd.Timestamp(f"2020-01-01 {i:02d}:00:00"),
                "lat": lat,
                "lng": lng,
            }
        )
    return rows


@pytest.fixture(scope="session")
def filter_tdf() -> pd.DataFrame:
    return pd.DataFrame(_filter_rows())


@pytest.fixture(scope="session")
def filter_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_filter_rows())


# ---------------------------------------------------------------------------
# outlier-detection fixtures (filter(method=...) for hampel/greedy/smart_greedy/zheng)
# ---------------------------------------------------------------------------
# hampel_case: 9 points, 1h spacing, slow ~55 m/h drift, except a single huge
#   teleport at index 4 (to lat=60.0). With the default window_size=5
#   (half_window=2) and n_sigma=3.0, the two speed values touching the
#   teleport (indices 4 and 5: the jump out to lat=60 and the jump back) are
#   each the sole extreme value in their 5-point window, so both are flagged;
#   every other index's own speed is unaffected and stays within threshold.
# greedy_case: 5 points, 1h spacing, slow drift, except index 2 teleported to
#   null island (lat=0, lng=0). Greedy tests fixed original-adjacent pairs
#   with max_speed_kmh=100.0, so *both* index 2 (arriving at null island) and
#   index 3 (leaving null island) fail the speed test and are dropped, even
#   though index 3 itself is a perfectly normal point.
# smart_greedy_case: same points as greedy_case. SmartGreedy tracks multiple
#   candidate chains instead of fixed adjacent pairs, so it finds that
#   {0, 1, 3, 4} is a single mutually-consistent chain (index 1 -> index 3 is
#   a normal 2-hour hop) and only drops the true outlier, index 2.
# zheng_case: 8 points, 1h spacing: a clean run of 4, a single teleport, then
#   a clean run of 3. With min_seg_size=1 (default), the isolated 1-point
#   "segment" at the teleport is dropped but both surrounding runs are kept
#   in full — including the point right after the teleport, which Greedy's
#   fixed-adjacent-pair test would have dropped (see greedy_case above).


def _hampel_rows():
    rows = []
    for i in range(9):
        lat = 60.0 if i == 4 else 48.8560 + i * 0.0005
        rows.append(
            {
                "uid": "hampel_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(hours=i),
                "lat": lat,
                "lng": 2.3520,
            }
        )
    return rows


def _greedy_rows():
    rows = []
    for i in range(5):
        lat = 0.0 if i == 2 else (48.8566 + i * 0.005)
        lng = 0.0 if i == 2 else 2.3522
        rows.append(
            {
                "uid": "greedy_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(hours=i),
                "lat": lat,
                "lng": lng,
            }
        )
    return rows


def _zheng_rows():
    rows = []
    lats = [0.0, 0.001, 0.002, 0.003, 5.0, 0.004, 0.005, 0.006]
    for i, lat in enumerate(lats):
        rows.append(
            {
                "uid": "zheng_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(hours=i),
                "lat": lat,
                "lng": 0.0,
            }
        )
    return rows


@pytest.fixture(scope="session")
def hampel_tdf() -> pd.DataFrame:
    return pd.DataFrame(_hampel_rows())


@pytest.fixture(scope="session")
def hampel_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_hampel_rows())


@pytest.fixture(scope="session")
def greedy_tdf() -> pd.DataFrame:
    return pd.DataFrame(_greedy_rows())


@pytest.fixture(scope="session")
def greedy_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_greedy_rows())


@pytest.fixture(scope="session")
def zheng_tdf() -> pd.DataFrame:
    return pd.DataFrame(_zheng_rows())


@pytest.fixture(scope="session")
def zheng_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_zheng_rows())


# ---------------------------------------------------------------------------
# compress fixtures
# ---------------------------------------------------------------------------
# user_dense: 10 points. First 5 clustered within 0.05 km of (48.86, 2.35),
#   then a 5 km jump, then 5 points clustered near (48.90, 2.40).
#   At spatial_radius=0.2 km → compresses to 2 output points.
# user_sparse: 5 points each ~0.5 km apart → no compression, 5 rows out.


def _compress_rows():
    rows = []
    # user_dense group 1: small jitter around (48.860, 2.350)
    base = [(48.8600, 2.3500), (48.8601, 2.3502), (48.8599, 2.3498), (48.8602, 2.3501), (48.8598, 2.3499)]
    for i, (lat, lng) in enumerate(base):
        rows.append(
            {
                "uid": "user_dense",
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=i),
                "lat": lat,
                "lng": lng,
            }
        )
    # user_dense group 2: ~5 km jump north, small jitter around (48.905, 2.400)
    base2 = [(48.9050, 2.4000), (48.9051, 2.4002), (48.9049, 2.3998), (48.9052, 2.4001), (48.9048, 2.3999)]
    for i, (lat, lng) in enumerate(base2):
        rows.append(
            {
                "uid": "user_dense",
                "datetime": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=10 + i),
                "lat": lat,
                "lng": lng,
            }
        )
    # user_sparse: 5 points ~0.55 km apart
    for i in range(5):
        rows.append(
            {
                "uid": "user_sparse",
                "datetime": pd.Timestamp(f"2020-01-01 {i:02d}:00:00"),
                "lat": 48.8566 + i * 0.005,
                "lng": 2.3522,
            }
        )
    return rows


@pytest.fixture(scope="session")
def compress_tdf() -> pd.DataFrame:
    return pd.DataFrame(_compress_rows())


@pytest.fixture(scope="session")
def compress_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_compress_rows())


# ---------------------------------------------------------------------------
# simplify fixtures
# ---------------------------------------------------------------------------
# near_collinear: 3 points almost on a straight line (~0.7 m lateral offset);
#   used by douglas_peucker/max_distance/chan_chin/imai_iri "drop redundant
#   middle point" tests.
# time_ratio: 3 points spatially collinear (middle point exactly halfway
#   between the endpoints), but the middle timestamp arrives after only 1%
#   of the elapsed time between the endpoints. Douglas-Peucker only looks at
#   space, so it drops the middle point; top_down_time_ratio also checks the
#   time-interpolated position, so it keeps it.
# min_dist: 3 points where the middle point is ~5.5 m from the first
#   (below a 20 m threshold) and the last point is ~33 m from the first
#   (above it).
# min_td: 3 points where the middle point arrives 10s after the first
#   (below a 60s threshold) and the last point arrives 120s after the first
#   (above it).
# corner: 3 points forming a 90-degree turn; every algorithm must keep the
#   corner vertex or the deviation would be ~1 km, far above any reasonable
#   epsilon.


def _simplify_rows():
    rows = []
    rows.extend(
        [
            {
                "uid": "near_collinear",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "lat": 48.8560,
                "lng": 2.3520,
            },
            {
                "uid": "near_collinear",
                "datetime": pd.Timestamp("2020-01-01 00:01:00"),
                "lat": 48.8565,
                "lng": 2.3520 + 0.00001,
            },
            {
                "uid": "near_collinear",
                "datetime": pd.Timestamp("2020-01-01 00:02:00"),
                "lat": 48.8570,
                "lng": 2.3520,
            },
        ]
    )
    rows.extend(
        [
            {
                "uid": "time_ratio",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "lat": 48.8560,
                "lng": 2.3520,
            },
            {
                "uid": "time_ratio",
                "datetime": pd.Timestamp("2020-01-01 00:00:01"),
                "lat": 48.8565,
                "lng": 2.3520,
            },
            {
                "uid": "time_ratio",
                "datetime": pd.Timestamp("2020-01-01 00:01:40"),
                "lat": 48.8570,
                "lng": 2.3520,
            },
        ]
    )
    rows.extend(
        [
            {
                "uid": "min_dist",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "lat": 48.8560,
                "lng": 2.3520,
            },
            {
                "uid": "min_dist",
                "datetime": pd.Timestamp("2020-01-01 00:01:00"),
                "lat": 48.8560 + 0.00005,
                "lng": 2.3520,
            },
            {
                "uid": "min_dist",
                "datetime": pd.Timestamp("2020-01-01 00:02:00"),
                "lat": 48.8560 + 0.0003,
                "lng": 2.3520,
            },
        ]
    )
    rows.extend(
        [
            {
                "uid": "min_td",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "lat": 48.8560,
                "lng": 2.3520,
            },
            {
                "uid": "min_td",
                "datetime": pd.Timestamp("2020-01-01 00:00:10"),
                "lat": 48.8561,
                "lng": 2.3520,
            },
            {
                "uid": "min_td",
                "datetime": pd.Timestamp("2020-01-01 00:02:00"),
                "lat": 48.8562,
                "lng": 2.3520,
            },
        ]
    )
    rows.extend(
        [
            {
                "uid": "corner",
                "datetime": pd.Timestamp("2020-01-01 00:00:00"),
                "lat": 48.85,
                "lng": 2.35,
            },
            {
                "uid": "corner",
                "datetime": pd.Timestamp("2020-01-01 00:01:00"),
                "lat": 48.86,
                "lng": 2.35,
            },
            {
                "uid": "corner",
                "datetime": pd.Timestamp("2020-01-01 00:02:00"),
                "lat": 48.86,
                "lng": 2.45,
            },
        ]
    )
    return rows


@pytest.fixture(scope="session")
def simplify_tdf() -> pd.DataFrame:
    return pd.DataFrame(_simplify_rows())


@pytest.fixture(scope="session")
def simplify_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_simplify_rows())


# ---------------------------------------------------------------------------
# stay_locations fixtures
# ---------------------------------------------------------------------------
# user_stationary: 10 points within 0.1 km of (48.856, 2.352), over 60 min
#   → 1 stop at spatial_radius=0.2, minutes_for_a_stop=20
# user_moving: 5 points 5 km apart, 5-min spacing → 0 stops


def _stops_rows():
    rows = []
    # user_stationary: tiny jitter around Paris centre, 1 per 6 minutes
    base = [
        (48.8560, 2.3520),
        (48.8561, 2.3522),
        (48.8559, 2.3518),
        (48.8562, 2.3521),
        (48.8558, 2.3519),
        (48.8563, 2.3523),
        (48.8557, 2.3517),
        (48.8564, 2.3524),
        (48.8556, 2.3516),
        (48.8565, 2.3525),
    ]
    for i, (lat, lng) in enumerate(base):
        rows.append(
            {
                "uid": "user_stationary",
                "datetime": pd.Timestamp("2020-01-01 08:00:00") + pd.Timedelta(minutes=6 * i),
                "lat": lat,
                "lng": lng,
            }
        )
    # user_moving: 5 km steps, 5-min spacing → never accumulates 20 min within 0.2 km
    for i in range(5):
        rows.append(
            {
                "uid": "user_moving",
                "datetime": pd.Timestamp("2020-01-01 08:00:00") + pd.Timedelta(minutes=5 * i),
                "lat": 48.8566 + i * 0.045,  # ~5 km per step
                "lng": 2.3522,
            }
        )
    return rows


@pytest.fixture(scope="session")
def stops_tdf() -> pd.DataFrame:
    return pd.DataFrame(_stops_rows())


@pytest.fixture(scope="session")
def stops_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_stops_rows())


# ---------------------------------------------------------------------------
# cluster fixtures
# ---------------------------------------------------------------------------
# Pre-built stops for clustering:
#   A and B within 0.05 km → one cluster (2 visits → label 0)
#   C is 5 km away → separate cluster (1 visit → label 1)


def _cluster_rows():
    return [
        # visit 1 to A/B cluster
        {
            "uid": "user1",
            "datetime": pd.Timestamp("2020-01-01 09:00:00"),
            "lat": 48.8566,
            "lng": 2.3522,
        },
        # visit 2 to A/B cluster (B, very close to A)
        {
            "uid": "user1",
            "datetime": pd.Timestamp("2020-01-02 09:00:00"),
            "lat": 48.8567,  # ~0.011 km from A
            "lng": 2.3523,
        },
        # visit to C (5 km away)
        {
            "uid": "user1",
            "datetime": pd.Timestamp("2020-01-03 09:00:00"),
            "lat": 48.9066,  # ~5.5 km north
            "lng": 2.3522,
        },
    ]


@pytest.fixture(scope="session")
def cluster_tdf() -> pd.DataFrame:
    return pd.DataFrame(_cluster_rows())


@pytest.fixture(scope="session")
def cluster_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_cluster_rows())


# ---------------------------------------------------------------------------
# segment fixtures
# ---------------------------------------------------------------------------
# gap_case: 5 points, a 2-hour gap between points 2 and 3 (well above a
#   10-minute gap_s threshold) → 2 segments.
# value_case: 5 points, a "poi" column with values [a, a, b, b, c] → 3
#   segments on every value change.
# angle_case: 3 points; 0->1 heads due north, 1->2 heads due east (a 90 deg
#   turn) → the turn point starts a new segment at the default min_angle=45.
# speed_case: 6 points; a fast hop, then 2 points "parked" at the same spot
#   for 300s (at or above a 300s duration_s threshold, with speed_kmh=5.0
#   raising the "moving" floor above their ~0 km/h parked speed), then 2 more
#   points moving again → 3 segments.
# short_parked_case: 4 points; a single parked point below the 300s duration
#   threshold → no split at all (folded into the surrounding segment), the
#   fastmob row-preservation analogue of MovingPandas' SpeedSplitter/
#   ObservationGapSplitter dropping short "non-moving"/gap rows outright.
# stop_case: 7 points; 2 points moving away from an anchor, then 4 points
#   held within a 0.2 km radius for a 40-minute stop (above the default
#   20-minute threshold), then 1 further point moving away → the stop's own
#   rows get their own segment, bracketed by the moving segments before and
#   after it.


def _segment_rows():
    rows = []
    for i, (lat, lng, seconds) in enumerate(
        [
            (48.0, 2.0, 0),
            (48.01, 2.0, 60),
            (48.02, 2.0, 120),
            (48.03, 2.0, 7320),  # +2h gap from the previous point
            (48.04, 2.0, 7380),
        ]
    ):
        rows.append(
            {
                "uid": "gap_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(seconds=seconds),
                "lat": lat,
                "lng": lng,
                "poi": None,
            }
        )
    for i, poi in enumerate(["a", "a", "b", "b", "c"]):
        rows.append(
            {
                "uid": "value_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(minutes=i),
                "lat": 48.0 + i * 0.001,
                "lng": 2.0,
                "poi": poi,
            }
        )
    for lat, lng, seconds in [(48.0, 2.0, 0), (48.01, 2.0, 60), (48.01, 2.02, 120)]:
        rows.append(
            {
                "uid": "angle_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(seconds=seconds),
                "lat": lat,
                "lng": lng,
                "poi": None,
            }
        )
    for lat, lng, seconds in [
        (48.0, 2.0, 0),
        (48.02, 2.0, 60),
        (48.02, 2.0, 180),
        (48.02, 2.0, 480),
        (48.03, 2.0, 540),
        (48.04, 2.0, 600),
    ]:
        rows.append(
            {
                "uid": "speed_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(seconds=seconds),
                "lat": lat,
                "lng": lng,
                "poi": None,
            }
        )
    for lat, lng, seconds in [
        (48.0, 2.0, 0),
        (48.01, 2.0, 60),
        (48.01, 2.0, 120),
        (48.02, 2.0, 180),
    ]:
        rows.append(
            {
                "uid": "short_parked_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(seconds=seconds),
                "lat": lat,
                "lng": lng,
                "poi": None,
            }
        )
    for lat, lng, seconds in [
        (48.85, 2.0, 0),
        (48.86, 2.1, 600),
        (48.90, 2.5, 1200),
        (48.9001, 2.5001, 1800),
        (48.9002, 2.5002, 3000),
        (49.0, 3.0, 3600),
        (49.5, 3.5, 4200),
    ]:
        rows.append(
            {
                "uid": "stop_case",
                "datetime": pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(seconds=seconds),
                "lat": lat,
                "lng": lng,
                "poi": None,
            }
        )
    return rows


@pytest.fixture(scope="session")
def segment_tdf() -> pd.DataFrame:
    return pd.DataFrame(_segment_rows())


@pytest.fixture(scope="session")
def segment_tdf_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    return pl.from_dicts(_segment_rows())
