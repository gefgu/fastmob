"""Populate the PTRAIL reference cache.

Run this script inside .venv-ptrail via the shell wrapper:
    bash scripts/populate_ptrail_cache.sh

Or directly (after activating the ptrail env):
    .venv-ptrail/bin/python tests/populate_ptrail_cache.py

Covers PTRAIL's only outlier-detection method fastmob has a direct named
equivalent for: Hampel (``Filters.hampel_outlier_detection``). PTRAIL's
Hampel filter is generic over a named column, so this script first computes
PTRAIL's own ``Speed`` kinematic feature column via
``ptrail.features.kinematic_features.KinematicFeatures.create_speed_column``
(consecutive-point speed, matching fastmob's own Hampel kernel's input
series) and runs the real ``Filters.hampel_outlier_detection`` on it.

Known version-compatibility note
---------------------------------
PTRAIL 1.0's ``Filters.hampel_outlier_detection`` calls ``hampel(col)`` from
the separate ``hampel`` PyPI package and uses the return value directly as a
list of outlier indices (``df.drop(df.index[outlier_indices])``). This is
only valid for ``hampel<1.0`` (which returned a plain list); ``hampel>=1.0``
returns a ``Result`` namedtuple instead, which crashes that line with an
``IndexError``. ``pyproject.toml``'s ``dev-ptrail`` extra pins
``hampel==0.0.5`` (a version PTRAIL's own ``hampel>=0.0.5`` requirement
already allows) so the real, unmodified ``Filters.hampel_outlier_detection``
runs successfully — this script never patches or reimplements PTRAIL's own
code.

This also means the cached reference here reflects ``hampel==0.0.5``'s
older, coarser algorithm (a centered pandas ``.rolling(window_size * 2)``
with backward/forward-filled boundary values), not the newer Cython kernel
(``hampel==1.0.2``) fastmob's own ``hampel.rs`` was ported from — see
``tests/correctness/preprocessing/test_filter.py``'s cache-based parity test
for the resulting documented tolerance.

Because ``hampel==0.0.5``'s rolling window only produces non-NaN
median/MAD values once at least ``window_size * 2`` points are available,
each user's slice must be reasonably long (unlike the tiny hand-crafted
correctness fixtures) for the comparison to be meaningful at all.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures
from ptrail.preprocessing.filters import Filters
from ptrail.preprocessing.helpers import Helpers
from ptrail.utilities import constants as const

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from tests.shared.ptrail_cache import _REFERENCE_DIR

DATASET_NAME = "brightkite"
MAX_USERS = 8
MAX_POINTS_PER_USER = 60
MIN_POINTS_PER_USER = 20

# Smaller than Brightkite check-ins' typical multi-hour cadence, so most
# consecutive gaps exceed it and get exactly one PTRAIL-inserted point.
INTERPOLATE_SAMPLING_RATE_S = 1800.0
INTERPOLATE_METHODS = ("linear", "cubic", "kinematic")


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

    # A handful of users with enough points each for PTRAIL's
    # window_size*2-wide rolling window to produce meaningful, non-NaN
    # medians/MADs across most of the slice.
    counts = df.groupby("uid").size()
    eligible = counts[(counts >= MIN_POINTS_PER_USER) & (counts <= 500)].index[:MAX_USERS]
    df = df[df["uid"].isin(eligible)].copy()
    df = df.groupby("uid", group_keys=False).head(MAX_POINTS_PER_USER)
    df = df.sort_values(["uid", "datetime"], kind="mergesort").reset_index(drop=True)
    df["row_index"] = df.groupby("uid").cumcount()
    return df[["uid", "datetime", "lat", "lng", "row_index"]]


def _run_hampel_for_user(user_df: pd.DataFrame) -> list[int]:
    """Run PTRAIL's real Hampel outlier detector for one user's slice and
    return the kept local ``row_index`` values.

    @usedBy `main()`.
    """
    frame = user_df[["uid", "datetime", "lat", "lng", "row_index"]].copy()
    tdf = PTRAILDataFrame(
        frame,
        latitude="lat",
        longitude="lng",
        datetime="datetime",
        traj_id="uid",
        rest_of_columns=["row_index"],
    )
    tdf = KinematicFeatures.create_speed_column(tdf)
    result = Filters.hampel_outlier_detection(tdf, "Speed")
    return result.reset_index()["row_index"].tolist()


def _run_interpolate_for_user(user_df: pd.DataFrame, uid: object, ip_type: str) -> pd.DataFrame:
    """Run PTRAIL's real per-user interpolation helper directly (bypassing the
    multiprocessing-based `Interpolation.interpolate_position` wrapper, same
    convention `_run_hampel_for_user` above already uses) and return a
    chronologically-sorted `(datetime, lat, lon)` frame -- original points
    plus any point PTRAIL inserted.

    The installed ptrail==0.7.1 Beta's `Helpers.linear_help`/`cubic_help`/
    `kinematic_help` expect the DateTime-indexing step their own
    `_linear_ip`/`_cubic_ip`/`_kinematic_ip` wrappers normally perform before
    splitting by trajectory ID (`dataframe.reset_index()[[...]].set_index(
    DateTime)`) to have already happened -- calling the helper directly on a
    plain-columns frame (as this ptrail version's own docstrings describe)
    raises ``ValueError: cannot set a row with mismatched columns`` from
    inside `.loc[new_timestamp] = [...]`, since the DateTime column is still
    present as a regular column, changing the target row's column count.
    Pre-indexing here (mirroring what `_linear_ip` etc. do internally, one
    version-generation earlier) avoids that.

    @usedBy `main()`.
    """
    frame = pd.DataFrame(
        {
            const.DateTime: user_df["datetime"].to_numpy(),
            const.TRAJECTORY_ID: uid,
            const.LAT: user_df["lat"].to_numpy(),
            const.LONG: user_df["lng"].to_numpy(),
        }
    ).set_index(const.DateTime)
    helper_fn = {
        "linear": Helpers.linear_help,
        "cubic": Helpers.cubic_help,
        "kinematic": Helpers.kinematic_help,
    }[ip_type]
    result = helper_fn(frame, uid, INTERPOLATE_SAMPLING_RATE_S, "")
    result = result.reset_index().sort_values(const.DateTime, kind="mergesort").reset_index(drop=True)
    return result[[const.DateTime, const.LAT, const.LONG]]


def main() -> None:
    """Populate `tests/shared/ptrail_reference/brightkite/`.

    @usedBy `scripts/populate_ptrail_cache.sh`. Writes `input.parquet`,
    `outlier_hampel.parquet`, and one `interpolate_<method>.parquet` per
    entry in `INTERPOLATE_METHODS` to the dataset's cache directory.
    """
    parser = argparse.ArgumentParser(description="Populate PTRAIL reference cache")
    parser.parse_args()

    out_dir = _REFERENCE_DIR / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Loading Brightkite slice ...")
    df = _load_brightkite_slice()
    print(f"    {df['uid'].nunique()} users, {len(df)} rows")
    print("    input.parquet")
    df.to_parquet(out_dir / "input.parquet", index=False)

    print("==> hampel")
    rows = []
    for uid, user_df in df.groupby("uid", sort=False):
        try:
            kept = _run_hampel_for_user(user_df.reset_index(drop=True))
        except Exception as exc:  # noqa: BLE001
            print(f"    SKIP user={uid} ({type(exc).__name__}: {exc})")
            continue
        rows.extend({"uid": uid, "row_index": int(idx)} for idx in kept)
    result_df = pd.DataFrame(rows, columns=["uid", "row_index"])
    path = out_dir / "outlier_hampel.parquet"
    result_df.to_parquet(path, index=False)
    n_users = result_df["uid"].nunique() if len(result_df) else 0
    print(f"    {path.name} ({len(result_df)} kept rows across {n_users} users)")

    for method in INTERPOLATE_METHODS:
        print(f"==> interpolate: {method}")
        rows = []
        for uid, user_df in df.groupby("uid", sort=False):
            try:
                result = _run_interpolate_for_user(user_df.reset_index(drop=True), uid, method)
            except Exception as exc:  # noqa: BLE001
                print(f"    SKIP user={uid} method={method} ({type(exc).__name__}: {exc})")
                continue
            for row in result.itertuples(index=False):
                rows.append(
                    {
                        "uid": uid,
                        "datetime": getattr(row, const.DateTime),
                        "lat": getattr(row, const.LAT),
                        "lon": getattr(row, const.LONG),
                    }
                )
        result_df = pd.DataFrame(rows, columns=["uid", "datetime", "lat", "lon"])
        path = out_dir / f"interpolate_{method}.parquet"
        result_df.to_parquet(path, index=False)
        n_users = result_df["uid"].nunique() if len(result_df) else 0
        print(f"    {path.name} ({len(result_df)} rows across {n_users} users)")

    print(f"\nDone. Commit {out_dir} to git to track the snapshot.")


if __name__ == "__main__":
    main()
