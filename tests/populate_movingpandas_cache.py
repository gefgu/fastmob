# ruff: noqa: E402
"""Populate the MovingPandas reference cache.

Run this script inside .venv-movingpandas via the shell wrapper:
    bash scripts/populate_movingpandas_cache.sh

Or directly (after activating the movingpandas env):
    .venv-movingpandas/bin/python tests/populate_movingpandas_cache.py

Covers the 5 simplification methods MovingPandas has a direct equivalent
for: DouglasPeucker, TopDownTimeRatio, MinDistance, MinTimeDelta, and
MaxDistance (Chan-Chin and Imai-Iri are validated separately via the MoveTK
C++ driver oracle; see the project plan).

MinDistance/MinTimeDelta measure distance/time directly on the unprojected
lat/lng trajectory (MovingPandas converts to metres internally for
geographic CRSes). DouglasPeucker/TopDownTimeRatio/MaxDistance measure raw
shapely distance in the trajectory's native CRS units, so each user's
trajectory is reprojected to its own estimated local UTM CRS first — the
closest MovingPandas-native equivalent of fastmob's own local
equirectangular planar-km projection — so that `tolerance` (in metres)
lines up with fastmob's `epsilon_km * 1000`.
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
from movingpandas.trajectory_generalizer import (
    DouglasPeuckerGeneralizer,
    MaxDistanceGeneralizer,
    MinDistanceGeneralizer,
    MinTimeDeltaGeneralizer,
    TopDownTimeRatioGeneralizer,
)

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from tests.shared.movingpandas_cache import _REFERENCE_DIR

DATASET_NAME = "brightkite"
MAX_USERS = 8
MAX_POINTS_PER_USER = 60
EPSILON_KM = 0.05
MIN_DISTANCE_KM = 0.2
MIN_TIME_DELTA_S = 600.0


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
    return df[["uid", "datetime", "lat", "lng", "row_index"]]


def _build_trajectory(user_df: pd.DataFrame, uid: object) -> mpd.Trajectory:
    """Build a MovingPandas Trajectory for one user's chronologically-sorted slice.

    @usedBy `_run_projected`, `_run_latlon`.
    """
    frame = user_df[["datetime", "lat", "lng", "row_index"]].copy()
    return mpd.Trajectory(frame, traj_id=uid, t="datetime", x="lng", y="lat", crs="epsg:4326")


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


def main() -> None:
    """Populate `tests/shared/movingpandas_reference/brightkite/`.

    @usedBy `scripts/populate_movingpandas_cache.sh`. Writes `input.parquet`
    and one `simplify_<method>.parquet` per entry in `METHODS` to the
    dataset's cache directory.
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

    print(f"\nDone. Commit {out_dir} to git to track the snapshot.")


if __name__ == "__main__":
    main()
