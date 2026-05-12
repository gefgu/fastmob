# ruff: noqa: E402
"""Populate the skmob reference cache.

Run this script inside .venv-skmob via the shell wrapper:
    bash tests/populate_skmob_cache.sh

Or directly (after activating the skmob env):
    .venv-skmob/bin/python tests/populate_skmob_cache.py [options]

Options:
    --datasets   Comma-separated list of datasets to populate.
                 Default: brightkite,geolife,foursquare
    --geolife-rows N      Max GeoLife rows (default: 10000)
    --foursquare-rows N   Max Foursquare rows (default: 10000)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import skmob
from skmob.measures import individual as skmob_individual
from skmob.preprocessing import clustering as skmob_clustering
from skmob.preprocessing import compression as skmob_compression
from skmob.preprocessing import detection as skmob_detection
from skmob.preprocessing import filtering as skmob_filtering

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from tests.shared.foursquare import FOURSQUARE_DEFAULT_ROWS, load_foursquare_pandas
from tests.shared.geolife import GEOLIFE_DEFAULT_ROWS, load_geolife_pandas
from tests.shared.skmob_cache import _REFERENCE_DIR


# ---------------------------------------------------------------------------
# Dataset loaders (replicate the conftest fixture logic exactly)
# ---------------------------------------------------------------------------


def _load_brightkite() -> skmob.TrajDataFrame:
    _BRIGHTKITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _BRIGHTKITE_PATH.exists():
        print(f"  Downloading Brightkite to {_BRIGHTKITE_PATH} ...")
        urllib.request.urlretrieve(_BRIGHTKITE_URL, _BRIGHTKITE_PATH)
    else:
        print(f"  Using cached Brightkite at {_BRIGHTKITE_PATH}")

    df = pd.read_csv(
        _BRIGHTKITE_PATH,
        sep="\t",
        header=0,
        nrows=100_000,
        names=["user", "check-in_time", "latitude", "longitude", "location id"],
    )
    df = (
        df.sort_values(["user", "check-in_time", "latitude", "longitude", "location id"])
        .drop_duplicates(["user", "check-in_time"], keep="first")
    )
    return skmob.TrajDataFrame(
        df,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


def _load_geolife(rows: int) -> skmob.TrajDataFrame:
    geolife_pd = load_geolife_pandas(mode="slice", rows=rows)
    return skmob.TrajDataFrame(
        geolife_pd,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


def _load_foursquare(rows: int) -> skmob.TrajDataFrame:
    foursquare_pd = load_foursquare_pandas(mode="slice", rows=rows)
    return skmob.TrajDataFrame(
        foursquare_pd,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


# ---------------------------------------------------------------------------
# Cache writer
# ---------------------------------------------------------------------------


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _save_count(n: int, path: Path) -> None:
    path.write_text(json.dumps({"count": n}))


def _run_all(tdf: skmob.TrajDataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- standardised input for skmob2 ---
    print("    input.parquet")
    _save(pd.DataFrame(tdf).copy(), out_dir / "input.parquet")

    # --- scalar-per-user measures ---
    scalar: list[tuple[str, object]] = [
        ("radius_of_gyration", lambda t: skmob_individual.radius_of_gyration(t, show_progress=False)),
        ("home_location", lambda t: skmob_individual.home_location(t)),
        ("maximum_distance", lambda t: skmob_individual.maximum_distance(t)),
        ("distance_straight_line", lambda t: skmob_individual.distance_straight_line(t)),
        ("number_of_visits", lambda t: skmob_individual.number_of_visits(t)),
        ("number_of_locations", lambda t: skmob_individual.number_of_locations(t)),
        ("max_distance_from_home", lambda t: skmob_individual.max_distance_from_home(t)),
        ("random_entropy", lambda t: skmob_individual.random_entropy(t)),
        ("real_entropy", lambda t: skmob_individual.real_entropy(t)),
        ("uncorrelated_entropy", lambda t: skmob_individual.uncorrelated_entropy(t)),
    ]
    for name, fn in scalar:
        print(f"    {name}.parquet")
        result = fn(tdf)
        _save(result.reset_index(drop=True), out_dir / f"{name}.parquet")

    # --- list-per-user measures ---
    print("    jump_lengths.parquet")
    _save(
        skmob_individual.jump_lengths(tdf, show_progress=False, merge=False).reset_index(drop=True),
        out_dir / "jump_lengths.parquet",
    )

    print("    k_radius_of_gyration_k2.parquet")
    _save(
        skmob_individual.k_radius_of_gyration(tdf, k=2, show_progress=False).reset_index(drop=True),
        out_dir / "k_radius_of_gyration_k2.parquet",
    )

    print("    waiting_times.parquet")
    _save(
        skmob_individual.waiting_times(tdf).reset_index(drop=True),
        out_dir / "waiting_times.parquet",
    )

    # --- location-keyed measures (MultiIndex → reset_index) ---
    location_keyed: list[tuple[str, object]] = [
        ("location_frequency", lambda t: skmob_individual.location_frequency(t, show_progress=False)),
        ("frequency_rank", lambda t: skmob_individual.frequency_rank(t, show_progress=False)),
        ("recency_rank", lambda t: skmob_individual.recency_rank(t, show_progress=False)),
    ]
    for name, fn in location_keyed:
        print(f"    {name}.parquet")
        _save(fn(tdf).reset_index(), out_dir / f"{name}.parquet")

    # --- network measure ---
    print("    individual_mobility_network.parquet")
    _save(
        skmob_individual.individual_mobility_network(tdf, show_progress=False).reset_index(),
        out_dir / "individual_mobility_network.parquet",
    )

    # --- preprocessing: row counts only ---
    print("    filter_count.json")
    _save_count(len(skmob_filtering.filter(tdf, max_speed_kmh=500.0)), out_dir / "filter_count.json")

    print("    compress_count.json")
    _save_count(len(skmob_compression.compress(tdf, spatial_radius_km=0.2)), out_dir / "compress_count.json")

    print("    stay_locations_count.json + cluster_count.json")
    stops = skmob_detection.stay_locations(tdf, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
    _save_count(len(stops), out_dir / "stay_locations_count.json")
    clustered = skmob_clustering.cluster(stops, cluster_radius_km=0.1)
    _save_count(len(clustered), out_dir / "cluster_count.json")

    print(f"    → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate skmob reference cache")
    parser.add_argument(
        "--datasets",
        default="brightkite,geolife,foursquare",
        help="Comma-separated dataset names (default: all three)",
    )
    parser.add_argument("--geolife-rows", type=int, default=GEOLIFE_DEFAULT_ROWS)
    parser.add_argument("--foursquare-rows", type=int, default=FOURSQUARE_DEFAULT_ROWS)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    loaders = {
        "brightkite": _load_brightkite,
        "geolife": lambda: _load_geolife(args.geolife_rows),
        "foursquare": lambda: _load_foursquare(args.foursquare_rows),
    }

    for dataset in datasets:
        if dataset not in loaders:
            print(f"Unknown dataset: {dataset!r} — skip")
            continue
        print(f"\n==> {dataset}")
        tdf = loaders[dataset]()
        _run_all(tdf, _REFERENCE_DIR / dataset)

    print("\nAll done. Commit tests/shared/skmob_reference/ to git to track the snapshots.")


if __name__ == "__main__":
    main()
