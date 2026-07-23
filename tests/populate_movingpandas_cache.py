# ruff: noqa: E402
"""Populate the MovingPandas reference cache.

Run this script inside .venv-movingpandas via the shell wrapper:
    bash scripts/populate_movingpandas_cache.sh

Or directly (after activating the movingpandas env):
    .venv-movingpandas/bin/python tests/populate_movingpandas_cache.py

Covers the 5 simplification methods MovingPandas has a direct equivalent
for: DouglasPeucker, TopDownTimeRatio, MinDistance, MinTimeDelta, and
MaxDistance (Chan-Chin and Imai-Iri are validated separately via the MoveTK
C++ driver oracle; see the project plan), plus all 6 segmentation methods
(AngleChange, ObservationGap, Speed, Stop, ValueChange, Temporal).

MinDistance/MinTimeDelta measure distance/time directly on the unprojected
lat/lng trajectory (MovingPandas converts to metres internally for
geographic CRSes). DouglasPeucker/TopDownTimeRatio/MaxDistance measure raw
shapely distance in the trajectory's native CRS units, so each user's
trajectory is reprojected to its own estimated local UTM CRS first — the
closest MovingPandas-native equivalent of fastmob's own local
equirectangular planar-km projection — so that `tolerance` (in metres)
lines up with fastmob's `epsilon_km * 1000`.

Segmentation methodology note: ObservationGap/Speed/Stop are run through
MovingPandas' real, public `.split()` API directly — none of their splitters
duplicate a boundary row across two adjacent output sub-trajectories, so
each row unambiguously belongs to exactly one sub-trajectory and the mapping
to `segment_id` is exact. AngleChange/ValueChange/Temporal, however, *do*
duplicate a boundary row for LineString-connectivity reasons (each
`_split_traj` copies one endpoint from an adjacent group into the current
group after the group-key assignment is already decided — see
`trajectory_splitter.py`'s `AngleChangeSplitter`/`ValueChangeSplitter`/
`TemporalSplitter`), so a row can appear in two different sub-trajectories
of the same `.split()` result. That is purely a presentation nicety (it
keeps the rendered line unbroken at the seam) and is not part of the actual
segmentation *decision*, so for those 3 methods this script ports the same
public grouping key (`add_direction`/`add_speed` + `angular_difference` for
AngleChange, `pandas.Grouper` for Temporal, a value-change cumulative sum
for ValueChange — the exact logic `trajectory_splitter.py` itself uses)
computed directly against MovingPandas' own real per-row heading/speed/stop
features, but read *before* that duplication step runs, giving an
unambiguous one-row-one-segment mapping directly comparable to fastmob's
own convention.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import movingpandas as mpd
from movingpandas.geometry_utils import angular_difference
from movingpandas.trajectory_generalizer import (
    DouglasPeuckerGeneralizer,
    MaxDistanceGeneralizer,
    MinDistanceGeneralizer,
    MinTimeDeltaGeneralizer,
    TopDownTimeRatioGeneralizer,
)
from movingpandas.trajectory_splitter import (
    ObservationGapSplitter,
    SpeedSplitter,
    StopSplitter,
)

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from tests.shared.movingpandas_cache import _REFERENCE_DIR

DATASET_NAME = "brightkite"
MAX_USERS = 8
MAX_POINTS_PER_USER = 60
EPSILON_KM = 0.05
MIN_DISTANCE_KM = 0.2
MIN_TIME_DELTA_S = 600.0

# Segmentation method parameters, chosen to match
# tests/correctness/preprocessing/test_segment.py's fixtures/defaults where
# there is a natural fastmob equivalent.
SEGMENT_GAP_S = 3600.0
# 0.0 (matching MovingPandas' own SpeedSplitter default) rather than
# test_segment.py's hand-crafted speed_kmh=5.0 fixture value: on this sparse
# check-in dataset (consecutive points are rarely a continuous GPS trace),
# any stricter floor filters out almost every point, leaving no multi-point
# "moving" runs at all for `ObservationGapSplitter`'s `len(df) > 1` cutoff
# (see this module's docstring's segmentation methodology note).
SEGMENT_SPEED_KMH = 0.0
SEGMENT_MAX_SPEED_KMH = float("inf")
SEGMENT_DURATION_S = 300.0
SEGMENT_STOP_RADIUS_KM = 0.2
SEGMENT_MINUTES_FOR_A_STOP = 20.0
SEGMENT_MIN_ANGLE_DEG = 45.0
SEGMENT_ANGLE_MIN_SPEED_KMH = 0.0
SEGMENT_VALUE_CHANGE_COL = "location_id"
SEGMENT_TEMPORAL_MODE = "day"


def _load_brightkite_slice() -> pd.DataFrame:
    """Load and return a small, deterministic multi-user Brightkite slice.

    @usedBy `main()`. Downloads the raw Brightkite check-in dump to
    `tests/shared/data/` on first run (network I/O), then writes nothing
    itself (the caller saves `input.parquet`).
    """
    _BRIGHTKITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _BRIGHTKITE_PATH.exists():
        print(f"  Downloading Brightkite to {_BRIGHTKITE_PATH} ...")
        urllib.request.urlretrieve(_BRIGHTKITE_URL, _BRIGHTKITE_PATH)

    df = pd.read_csv(
        _BRIGHTKITE_PATH,
        sep="\t",
        header=0,
        nrows=200_000,
        names=["uid", "datetime", "lat", "lng", "location_id"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["location_id"] = df["location_id"].astype("string")
    df = df.dropna(subset=["uid", "datetime", "lat", "lng"])
    df = df.sort_values(["uid", "datetime"], kind="mergesort")
    df = df.drop_duplicates(["uid", "datetime"], keep="first")

    # A handful of users with a manageable number of points each so the
    # (partly recursive) reference algorithms run quickly and deterministically.
    counts = df.groupby("uid").size()
    eligible = counts[(counts >= 20) & (counts <= 500)].index[:MAX_USERS]
    df = df[df["uid"].isin(eligible)].copy()
    df = df.groupby("uid", group_keys=False).head(MAX_POINTS_PER_USER)
    df = df.sort_values(["uid", "datetime"], kind="mergesort").reset_index(drop=True)
    df["row_index"] = df.groupby("uid").cumcount()
    # location_id is kept only for the segment(method="value_change") oracle
    # (a categorical column to watch for consecutive-value changes); every
    # other method/generalizer ignores it.
    return df[["uid", "datetime", "lat", "lng", "location_id", "row_index"]]


# fastmob method name -> MovingPandas' own `method=` value for
# `Trajectory.get_position_at`.
INTERPOLATE_AT_METHODS = {"linear": "interpolated", "nearest": "nearest"}


def _build_trajectory(user_df: pd.DataFrame, uid: object) -> mpd.Trajectory:
    """Build a MovingPandas Trajectory for one user's chronologically-sorted slice.

    @usedBy `_run_projected`, `_run_latlon`.
    """
    frame = user_df[["datetime", "lat", "lng", "row_index"]].copy()
    return mpd.Trajectory(frame, traj_id=uid, t="datetime", x="lng", y="lat", crs="epsg:4326")


def _query_times_for_user(user_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Deterministic query timestamps for one user: two points (25% and 75%
    of the way) between every consecutive pair of points (always in this
    user's time range) plus one timestamp strictly before the first point
    (always out of range, to exercise the invalid/out-of-bounds path).

    Deliberately avoids the exact 50% midpoint: at that point `method=
    "nearest"` has a genuine tie between the two surrounding points, and
    fastmob's and MovingPandas' tie-breaking conventions are not guaranteed
    to agree (nor does either commit to one) -- 25%/75% keeps every query
    unambiguously closer to one specific neighbor.

    @usedBy `main()`.
    """
    times = user_df["datetime"].tolist()
    queries = []
    for t0, t1 in zip(times, times[1:]):
        span = t1 - t0
        queries.append(t0 + span * 0.25)
        queries.append(t0 + span * 0.75)
    queries.append(times[0] - pd.Timedelta(hours=1))
    return queries


def _run_interpolate_at_for_user(user_df: pd.DataFrame, uid: object, mpd_method: str) -> list[dict]:
    """Query MovingPandas' real `Trajectory.get_position_at` at each of this
    user's deterministic query timestamps.

    @usedBy `main()`.
    """
    traj = _build_trajectory(user_df, uid)
    rows = []
    for query_time in _query_times_for_user(user_df):
        try:
            point = traj.get_position_at(query_time, method=mpd_method)
            rows.append({"uid": uid, "query_time": query_time, "lat": point.y, "lon": point.x, "valid": True})
        except ValueError:
            rows.append(
                {
                    "uid": uid,
                    "query_time": query_time,
                    "lat": float("nan"),
                    "lon": float("nan"),
                    "valid": False,
                }
            )
    return rows


def _run_projected(user_df: pd.DataFrame, uid: object, generalizer_cls, tolerance_km: float) -> list[int]:
    """Run a shapely-distance-based generalizer after reprojecting to a local UTM CRS.

    @usedBy `METHODS["douglas_peucker"]`, `METHODS["top_down_time_ratio"]`,
    `METHODS["max_distance"]`.
    """
    traj = _build_trajectory(user_df, uid)
    traj.populate_geometry_column()
    utm_crs = traj.df.estimate_utm_crs()
    traj = traj.to_crs(utm_crs)
    result = generalizer_cls(traj).generalize(tolerance=tolerance_km * 1000.0)
    return result.df["row_index"].tolist()


def _run_latlon(user_df: pd.DataFrame, uid: object, generalizer_cls, tolerance) -> list[int]:
    """Run a haversine-distance-based generalizer directly on unprojected lat/lng.

    @usedBy `METHODS["min_distance"]`, `METHODS["min_time_delta"]`.
    """
    traj = _build_trajectory(user_df, uid)
    result = generalizer_cls(traj).generalize(tolerance=tolerance)
    return result.df["row_index"].tolist()


METHODS = {
    "douglas_peucker": lambda user_df, uid: _run_projected(user_df, uid, DouglasPeuckerGeneralizer, EPSILON_KM),
    "top_down_time_ratio": lambda user_df, uid: _run_projected(user_df, uid, TopDownTimeRatioGeneralizer, EPSILON_KM),
    "max_distance": lambda user_df, uid: _run_projected(user_df, uid, MaxDistanceGeneralizer, EPSILON_KM),
    "min_distance": lambda user_df, uid: _run_latlon(user_df, uid, MinDistanceGeneralizer, MIN_DISTANCE_KM * 1000.0),
    "min_time_delta": lambda user_df, uid: _run_latlon(
        user_df, uid, MinTimeDeltaGeneralizer, timedelta(seconds=MIN_TIME_DELTA_S)
    ),
}


def _segments_from_collection(collection) -> list[tuple[int, int]]:
    """Return ``[(row_index, segment_id), ...]`` from a `TrajectoryCollection`.

    ``segment_id`` is simply the sub-trajectory's position in
    ``collection.trajectories`` (`.split()`'s own output order). Only safe
    to use directly for splitters that never duplicate a boundary row across
    two sub-trajectories (see this module's docstring) — `ObservationGap`,
    `Speed`, and `Stop`.

    @usedBy `_run_observation_gap`, `_run_speed`, `_run_stop`.
    """
    rows = []
    for seg_id, sub_traj in enumerate(collection.trajectories):
        for idx in sub_traj.df["row_index"].tolist():
            rows.append((int(idx), seg_id))
    return rows


def _run_observation_gap(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Run the real `ObservationGapSplitter` (no boundary-row duplication).

    @usedBy `SEGMENT_METHODS["observation_gap"]`.
    """
    traj = _build_trajectory(user_df, uid)
    result = ObservationGapSplitter(traj).split(gap=timedelta(seconds=SEGMENT_GAP_S))
    return _segments_from_collection(result)


def _run_speed(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Run the real `SpeedSplitter` (no boundary-row duplication).

    MovingPandas measures speed in metres/second for a geographic (lat/lon)
    CRS, so `SEGMENT_SPEED_KMH`/`SEGMENT_MAX_SPEED_KMH` are converted from
    km/h to m/s to match fastmob's own `speed_kmh`/`max_speed_kmh`.

    @usedBy `SEGMENT_METHODS["speed"]`.
    """
    traj = _build_trajectory(user_df, uid)
    speed_ms = SEGMENT_SPEED_KMH * 1000.0 / 3600.0
    max_speed_ms = float("inf") if SEGMENT_MAX_SPEED_KMH == float("inf") else SEGMENT_MAX_SPEED_KMH * 1000.0 / 3600.0
    result = SpeedSplitter(traj).split(
        speed=speed_ms, duration=timedelta(seconds=SEGMENT_DURATION_S), max_speed=max_speed_ms
    )
    return _segments_from_collection(result)


def _run_stop(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Run the real `StopSplitter` (no boundary-row duplication).

    MovingPandas' `TrajectoryStopDetector` uses a cluster-diameter criterion
    (``max_diameter``), a different (if related) stop-detection algorithm
    from fastmob's own anchor-point expanding-window criterion
    (`stay_locations`/`segment(method="stop")`'s `stop_radius_km`). This
    picks the closest natural conversion (``max_diameter = 2 *
    stop_radius_km * 1000`` metres) rather than claiming algorithmic
    equivalence; see `test_segment.py`'s cached parity test for the
    resulting (documented, wide) tolerance.

    @usedBy `SEGMENT_METHODS["stop"]`.
    """
    traj = _build_trajectory(user_df, uid)
    result = StopSplitter(traj).split(
        max_diameter=SEGMENT_STOP_RADIUS_KM * 1000.0 * 2.0,
        min_duration=timedelta(minutes=SEGMENT_MINUTES_FOR_A_STOP),
    )
    return _segments_from_collection(result)


def _run_angle_change(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Port `AngleChangeSplitter`'s exact pre-duplication grouping key.

    Uses MovingPandas' own `add_direction`/`add_speed` (the real per-row
    heading/speed features) and `angular_difference`, replaying the same
    `comp_dir`/`dir_group` state machine `AngleChangeSplitter._split_traj`
    itself runs — before its own post-hoc step that copies each group's
    trailing point into the next group's start for line-connectivity
    (see this module's docstring for why that duplication makes the
    row-to-segment mapping ambiguous if read from `.split()`'s own output
    instead).

    @usedBy `SEGMENT_METHODS["angle_change"]`.
    """
    traj = _build_trajectory(user_df, uid)
    traj.add_direction(overwrite=True)
    traj.add_speed(overwrite=True)
    direction_col = traj.get_direction_col()
    speed_col = traj.get_speed_col()

    min_speed_ms = SEGMENT_ANGLE_MIN_SPEED_KMH * 1000.0 / 3600.0
    directions = traj.df[direction_col].tolist()
    speeds = traj.df[speed_col].tolist()
    row_indices = traj.df["row_index"].tolist()

    comp_dir = directions[0] if directions else 0.0
    dir_group = 0
    rows = []
    for row_index, direction, speed in zip(row_indices, directions, speeds):
        if speed >= min_speed_ms and angular_difference(comp_dir, direction) >= SEGMENT_MIN_ANGLE_DEG:
            comp_dir = direction
            dir_group += 1
        rows.append((int(row_index), dir_group))
    return rows


def _run_value_change(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Port `ValueChangeSplitter`'s exact pre-duplication grouping key.

    Replays the same ``shift() != value`` cumulative-sum group key
    `ValueChangeSplitter._split_traj` computes on `SEGMENT_VALUE_CHANGE_COL`,
    before its own post-hoc trailing-row-duplication step (see this module's
    docstring). Cast to plain ``object`` first so real Brightkite rows with
    a missing (nullable-NA) location_id compare with regular Python ``None``
    semantics instead of pandas' NA-propagating boolean comparison (`NA !=
    NA` is itself `NA`, which cannot cast to `int`).

    @usedBy `SEGMENT_METHODS["value_change"]`.
    """
    ordered = user_df.sort_values("row_index")
    values = ordered[SEGMENT_VALUE_CHANGE_COL].astype(object)
    changed = (values.shift() != values).astype(int).cumsum()
    return list(zip(ordered["row_index"].astype(int).tolist(), changed.tolist()))


def _run_temporal(user_df: pd.DataFrame, uid: object) -> list[tuple[int, int]]:
    """Port `TemporalSplitter`'s exact pre-duplication grouping key.

    Replays the same ``pandas.Grouper(freq=...)`` bucketing
    `TemporalSplitter._split_traj` computes, skipping empty buckets exactly
    as the real splitter does, before its own post-hoc trailing-row-
    duplication step (see this module's docstring).

    @usedBy `SEGMENT_METHODS["temporal"]`.
    """
    modes = {"hour": "h", "day": "D", "month": "ME", "year": "YE"}
    freq = modes[SEGMENT_TEMPORAL_MODE]
    frame = user_df.set_index("datetime")
    rows = []
    group_id = -1
    for _, values in frame.groupby(pd.Grouper(freq=freq)):
        if len(values) == 0:
            continue
        group_id += 1
        rows.extend((int(idx), group_id) for idx in values["row_index"].tolist())
    return rows


SEGMENT_METHODS = {
    "observation_gap": _run_observation_gap,
    "speed": _run_speed,
    "stop": _run_stop,
    "angle_change": _run_angle_change,
    "value_change": _run_value_change,
    "temporal": _run_temporal,
}


def main() -> None:
    """Populate `tests/shared/movingpandas_reference/brightkite/`.

    @usedBy `scripts/populate_movingpandas_cache.sh`. Writes `input.parquet`,
    one `simplify_<method>.parquet` per entry in `METHODS`, one
    `segment_<method>.parquet` per entry in `SEGMENT_METHODS`, and one
    `interpolate_at_<method>.parquet` per entry in `INTERPOLATE_AT_METHODS`
    to the dataset's cache directory.
    """
    parser = argparse.ArgumentParser(description="Populate MovingPandas reference cache")
    parser.parse_args()

    out_dir = _REFERENCE_DIR / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Loading Brightkite slice ...")
    df = _load_brightkite_slice()
    print(f"    {df['uid'].nunique()} users, {len(df)} rows")
    print("    input.parquet")
    df.to_parquet(out_dir / "input.parquet", index=False)

    for method, run_fn in METHODS.items():
        print(f"==> {method}")
        rows = []
        for uid, user_df in df.groupby("uid", sort=False):
            try:
                kept = run_fn(user_df.reset_index(drop=True), uid)
            except Exception as exc:
                print(f"    SKIP user={uid} ({type(exc).__name__}: {exc})")
                continue
            rows.extend({"uid": uid, "row_index": int(idx)} for idx in kept)
        result_df = pd.DataFrame(rows, columns=["uid", "row_index"])
        path = out_dir / f"simplify_{method}.parquet"
        result_df.to_parquet(path, index=False)
        n_users = result_df["uid"].nunique() if len(result_df) else 0
        print(f"    {path.name} ({len(result_df)} kept rows across {n_users} users)")

    for method, run_fn in SEGMENT_METHODS.items():
        print(f"==> segment: {method}")
        rows = []
        for uid, user_df in df.groupby("uid", sort=False):
            try:
                segment_pairs = run_fn(user_df.reset_index(drop=True), uid)
            except Exception as exc:
                print(f"    SKIP user={uid} ({type(exc).__name__}: {exc})")
                continue
            rows.extend(
                {"uid": uid, "row_index": row_index, "segment_id": segment_id}
                for row_index, segment_id in segment_pairs
            )
        result_df = pd.DataFrame(rows, columns=["uid", "row_index", "segment_id"])
        path = out_dir / f"segment_{method}.parquet"
        result_df.to_parquet(path, index=False)
        n_users = result_df["uid"].nunique() if len(result_df) else 0
        n_segments = result_df.groupby("uid")["segment_id"].nunique().sum() if len(result_df) else 0
        print(f"    {path.name} ({len(result_df)} rows, {n_segments} segments across {n_users} users)")

    for method, mpd_method in INTERPOLATE_AT_METHODS.items():
        print(f"==> interpolate_at: {method}")
        rows = []
        for uid, user_df in df.groupby("uid", sort=False):
            try:
                rows.extend(_run_interpolate_at_for_user(user_df.reset_index(drop=True), uid, mpd_method))
            except Exception as exc:
                print(f"    SKIP user={uid} method={method} ({type(exc).__name__}: {exc})")
                continue
        result_df = pd.DataFrame(rows, columns=["uid", "query_time", "lat", "lon", "valid"])
        path = out_dir / f"interpolate_at_{method}.parquet"
        result_df.to_parquet(path, index=False)
        n_users = result_df["uid"].nunique() if len(result_df) else 0
        print(f"    {path.name} ({len(result_df)} rows across {n_users} users)")

    print(f"\nDone. Commit {out_dir} to git to track the snapshot.")


if __name__ == "__main__":
    main()
