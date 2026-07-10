# ruff: noqa: E402
"""Populate the skmob reference cache.

Run this script inside .venv-skmob via the shell wrapper:
    bash scripts/populate_skmob_cache.sh

Or directly (after activating the skmob env):
    .venv-skmob/bin/python tests/populate_skmob_cache.py [options]

Options:
    --datasets   Comma-separated list of datasets to populate.
                 Default: brightkite,geolife,foursquare,privacy_toy,models
    --geolife-rows N      Max GeoLife rows (default: 10000)
    --foursquare-rows N   Max Foursquare rows (default: 10000)
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import urllib.request
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# skmob 1.3.1 uses np.NaN which was removed in NumPy 2.0.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas as pd
import skmob
from skmob.models.epr import DensityEPR as SkmobDensityEPR
from skmob.models.epr import EPR as SkmobEPR
from skmob.models.epr import SpatialEPR as SkmobSpatialEPR
from skmob.models.geosim import GeoSim as SkmobGeoSim
from skmob.models.gravity import Gravity as SkmobGravity
from skmob.models.markov_diary_generator import MarkovDiaryGenerator as SkmobMarkovDiaryGenerator
from skmob.models.radiation import Radiation as SkmobRadiation
from skmob.models.sts_epr import STS_epr as SkmobSTSEPR
from skmob.measures import individual as skmob_individual
from skmob.privacy import attacks as skmob_privacy_attacks
from skmob.preprocessing import clustering as skmob_clustering
from skmob.preprocessing import compression as skmob_compression
from skmob.preprocessing import detection as skmob_detection
from skmob.preprocessing import filtering as skmob_filtering

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL
from tests.shared.foursquare import FOURSQUARE_DEFAULT_ROWS, load_foursquare_pandas
from tests.shared.geolife import GEOLIFE_DEFAULT_ROWS, load_geolife_pandas
from tests.shared.skmob_cache import _REFERENCE_DIR

PRIVACY_TOY_PATH = REPO_ROOT / "scikit-mobility" / "examples" / "privacy_toy.csv"
SKMOB_EXAMPLES_DIR = REPO_ROOT / "scikit-mobility" / "examples"
MODEL_SEED = 2
MODEL_START = pd.Timestamp("2020-01-01 08:00:00")
MODEL_END = pd.Timestamp("2020-01-02 08:00:00")
MODEL_SOCIAL_GRAPH = [[0, 1], [0, 2], [1, 2]]
MODEL_STARTING_LOCATIONS = [0, 1]

PRIVACY_ATTACK_CASES: tuple[tuple[str, type, dict[str, object], dict[str, object]], ...] = (
    ("location_kl2", skmob_privacy_attacks.LocationAttack, {"knowledge_length": 2}, {}),
    ("location_kl3", skmob_privacy_attacks.LocationAttack, {"knowledge_length": 3}, {}),
    ("location_targets_kl3", skmob_privacy_attacks.LocationAttack, {"knowledge_length": 3}, {"targets": [1, 2]}),
    (
        "location_force_instances_kl3",
        skmob_privacy_attacks.LocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("location_sequence_kl2", skmob_privacy_attacks.LocationSequenceAttack, {"knowledge_length": 2}, {}),
    ("location_sequence_kl3", skmob_privacy_attacks.LocationSequenceAttack, {"knowledge_length": 3}, {}),
    (
        "location_sequence_targets_kl3",
        skmob_privacy_attacks.LocationSequenceAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    (
        "location_sequence_force_instances_kl3",
        skmob_privacy_attacks.LocationSequenceAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("location_time_kl2", skmob_privacy_attacks.LocationTimeAttack, {"knowledge_length": 2}, {}),
    (
        "location_time_month_kl2",
        skmob_privacy_attacks.LocationTimeAttack,
        {"knowledge_length": 2, "time_precision": "Month"},
        {},
    ),
    (
        "location_time_month_kl3",
        skmob_privacy_attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {},
    ),
    (
        "location_time_month_targets_kl3",
        skmob_privacy_attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {"targets": [1, 2]},
    ),
    (
        "location_time_month_force_instances_kl3",
        skmob_privacy_attacks.LocationTimeAttack,
        {"knowledge_length": 3, "time_precision": "Month"},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("unique_location_kl2", skmob_privacy_attacks.UniqueLocationAttack, {"knowledge_length": 2}, {}),
    ("unique_location_kl3", skmob_privacy_attacks.UniqueLocationAttack, {"knowledge_length": 3}, {}),
    (
        "unique_location_targets_kl3",
        skmob_privacy_attacks.UniqueLocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    (
        "unique_location_force_instances_kl3",
        skmob_privacy_attacks.UniqueLocationAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("location_frequency_kl2", skmob_privacy_attacks.LocationFrequencyAttack, {"knowledge_length": 2}, {}),
    (
        "location_frequency_tol05_kl2",
        skmob_privacy_attacks.LocationFrequencyAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    ("location_frequency_kl3", skmob_privacy_attacks.LocationFrequencyAttack, {"knowledge_length": 3}, {}),
    (
        "location_frequency_targets_kl3",
        skmob_privacy_attacks.LocationFrequencyAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    (
        "location_frequency_force_instances_kl3",
        skmob_privacy_attacks.LocationFrequencyAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("location_probability_kl2", skmob_privacy_attacks.LocationProbabilityAttack, {"knowledge_length": 2}, {}),
    (
        "location_probability_tol05_kl2",
        skmob_privacy_attacks.LocationProbabilityAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    ("location_probability_kl3", skmob_privacy_attacks.LocationProbabilityAttack, {"knowledge_length": 3}, {}),
    (
        "location_probability_targets_kl3",
        skmob_privacy_attacks.LocationProbabilityAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    (
        "location_probability_force_instances_kl3",
        skmob_privacy_attacks.LocationProbabilityAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("location_proportion_kl2", skmob_privacy_attacks.LocationProportionAttack, {"knowledge_length": 2}, {}),
    (
        "location_proportion_tol05_kl2",
        skmob_privacy_attacks.LocationProportionAttack,
        {"knowledge_length": 2, "tolerance": 0.5},
        {},
    ),
    ("location_proportion_kl3", skmob_privacy_attacks.LocationProportionAttack, {"knowledge_length": 3}, {}),
    (
        "location_proportion_targets_kl3",
        skmob_privacy_attacks.LocationProportionAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2]},
    ),
    (
        "location_proportion_force_instances_kl3",
        skmob_privacy_attacks.LocationProportionAttack,
        {"knowledge_length": 3},
        {"targets": [1, 2], "force_instances": True},
    ),
    ("home_work", skmob_privacy_attacks.HomeWorkAttack, {}, {}),
    ("home_work_targets", skmob_privacy_attacks.HomeWorkAttack, {}, {"targets": [1, 2]}),
    (
        "home_work_force_instances",
        skmob_privacy_attacks.HomeWorkAttack,
        {},
        {"targets": [1, 2], "force_instances": True},
    ),
)


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
    df = df.sort_values(["user", "check-in_time", "latitude", "longitude", "location id"]).drop_duplicates(
        ["user", "check-in_time"], keep="first"
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


def _load_privacy_toy() -> skmob.TrajDataFrame:
    if not PRIVACY_TOY_PATH.exists():
        raise FileNotFoundError(f"Privacy toy dataset not found at {PRIVACY_TOY_PATH}")
    df = pd.read_csv(PRIVACY_TOY_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return skmob.TrajDataFrame(df, latitude="lat", longitude="lng", datetime="datetime", user_id="uid")


def _load_model_tessellation():
    geojson_path = SKMOB_EXAMPLES_DIR / "NY_counties_2011.geojson"
    flows_path = SKMOB_EXAMPLES_DIR / "NY_commuting_flows_2011.csv"
    if not geojson_path.exists() or not flows_path.exists():
        raise FileNotFoundError(f"Missing scikit-mobility model examples under {SKMOB_EXAMPLES_DIR}")

    import geopandas as gpd
    from shapely.geometry import shape

    _patch_geopandas_apply_for_numpy2(gpd)

    payload = json.loads(geojson_path.read_text())
    tessellation = gpd.GeoDataFrame(
        [feature["properties"] for feature in payload["features"]],
        geometry=[shape(feature["geometry"]) for feature in payload["features"]],
        crs="EPSG:4269",
    )
    tessellation["tile_id"] = tessellation["tile_id"].astype(str)
    flows = pd.read_csv(flows_path, dtype={"origin": str, "destination": str})
    outflows = (
        flows[flows["origin"] != flows["destination"]]
        .groupby("origin", as_index=False)["flow"]
        .sum()
        .rename(columns={"origin": "tile_id", "flow": "tot_outflow"})
    )
    tessellation = tessellation.merge(outflows, on="tile_id", how="left")
    tessellation["tot_outflow"] = tessellation["tot_outflow"].fillna(0).astype(int)
    tessellation = tessellation.sort_values("tile_id", kind="mergesort").head(12).reset_index(drop=True)
    tessellation["tot_outflow"] = tessellation["tot_outflow"].clip(upper=250).astype(int)
    return tessellation


def _patch_geopandas_apply_for_numpy2(gpd) -> None:
    """Avoid old GeoPandas ``copy=False`` paths that fail under NumPy 2."""
    if getattr(gpd.GeoSeries.apply, "_fkmob_numpy2_patch", False):
        return
    original_apply = gpd.GeoSeries.apply

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        try:
            return original_apply(self, func, convert_dtype=convert_dtype, args=args, **kwargs)
        except ValueError as exc:
            if "Unable to avoid copy" not in str(exc):
                raise
            series = pd.Series(list(self), index=self.index)
            return series.apply(lambda geom: func(geom, *args), **kwargs)

    apply._fkmob_numpy2_patch = True
    gpd.GeoSeries.apply = apply


def _model_tessellation_input(tessellation) -> pd.DataFrame:
    centroids = tessellation.geometry.centroid
    return pd.DataFrame(
        {
            "tile_id": tessellation["tile_id"].astype(str).to_numpy(),
            "lat": [point.y for point in centroids],
            "lng": [point.x for point in centroids],
            "population": pd.to_numeric(tessellation["population"], errors="coerce").fillna(0).to_numpy(dtype=float),
            "tot_outflow": pd.to_numeric(tessellation["tot_outflow"], errors="coerce").fillna(0).to_numpy(dtype=int),
        }
    )


def _load_model_diary_training() -> pd.DataFrame:
    geolife_path = SKMOB_EXAMPLES_DIR / "geolife_sample.txt.gz"
    if not geolife_path.exists():
        raise FileNotFoundError(f"Missing scikit-mobility GeoLife sample at {geolife_path}")

    with gzip.open(geolife_path, "rt", encoding="utf-8") as handle:
        df = pd.read_csv(handle)
    tdf = skmob.TrajDataFrame(df, latitude="lat", longitude="lng", datetime="datetime", user_id="uid")
    clustered = skmob_clustering.cluster(tdf)
    return pd.DataFrame(clustered)[["uid", "datetime", "cluster"]].copy()


# ---------------------------------------------------------------------------
# Cache writer
# ---------------------------------------------------------------------------


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def _save_count(n: int, path: Path) -> None:
    path.write_text(json.dumps({"count": n}))


def _run_measure(name: str, fn, tdf: skmob.TrajDataFrame, path: Path, *, reset_index: bool = False) -> bool:
    """Run one skmob measure, save to path, return True on success."""
    print(f"    {path.name}")
    try:
        result = fn(tdf)
        if reset_index:
            # skmob 1.3.1 bug: calling reset_index() on the TrajDataFrame subclass
            # for datasets with string UIDs mangles the columns (uid and the measure
            # column disappear).  Convert to a plain pandas DataFrame first so that
            # MultiIndex levels become regular columns correctly.
            plain = pd.DataFrame(result)
            if isinstance(plain.index, pd.MultiIndex):
                df = plain.reset_index()
            else:
                df = plain.reset_index(drop=True)
        else:
            df = result.reset_index(drop=True)
        _save(df, path)
        return True
    except Exception as exc:
        print(f"      SKIP ({type(exc).__name__}: {exc})")
        return False


def _run_all(tdf: skmob.TrajDataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- standardised input for fkmob ---
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
        _run_measure(name, fn, tdf, out_dir / f"{name}.parquet")

    # --- list-per-user measures ---
    _run_measure(
        "jump_lengths",
        lambda t: skmob_individual.jump_lengths(t, show_progress=False, merge=False),
        tdf,
        out_dir / "jump_lengths.parquet",
    )
    _run_measure(
        "k_radius_of_gyration_k2",
        lambda t: skmob_individual.k_radius_of_gyration(t, k=2, show_progress=False),
        tdf,
        out_dir / "k_radius_of_gyration_k2.parquet",
    )
    _run_measure(
        "waiting_times",
        lambda t: skmob_individual.waiting_times(t),
        tdf,
        out_dir / "waiting_times.parquet",
    )

    # --- location-keyed measures (MultiIndex → reset_index) ---
    location_keyed: list[tuple[str, object]] = [
        ("location_frequency", lambda t: skmob_individual.location_frequency(t, show_progress=False)),
        ("frequency_rank", lambda t: skmob_individual.frequency_rank(t, show_progress=False)),
        ("recency_rank", lambda t: skmob_individual.recency_rank(t, show_progress=False)),
    ]
    for name, fn in location_keyed:
        _run_measure(name, fn, tdf, out_dir / f"{name}.parquet", reset_index=True)

    # --- network measure ---
    _run_measure(
        "individual_mobility_network",
        lambda t: skmob_individual.individual_mobility_network(t, show_progress=False),
        tdf,
        out_dir / "individual_mobility_network.parquet",
        reset_index=True,
    )

    # --- preprocessing: row counts only ---
    for count_name, count_fn in [
        ("filter_count", lambda t: len(skmob_filtering.filter(t, max_speed_kmh=500.0))),
        ("compress_count", lambda t: len(skmob_compression.compress(t, spatial_radius_km=0.2))),
    ]:
        print(f"    {count_name}.json")
        try:
            _save_count(count_fn(tdf), out_dir / f"{count_name}.json")
        except Exception as exc:
            print(f"      SKIP ({type(exc).__name__}: {exc})")

    print("    stay_locations_count.json + cluster_count.json")
    try:
        stops = skmob_detection.stay_locations(tdf, spatial_radius_km=0.2, minutes_for_a_stop=20.0)
        _save_count(len(stops), out_dir / "stay_locations_count.json")
        try:
            clustered = skmob_clustering.cluster(stops, cluster_radius_km=0.1)
            _save_count(len(clustered), out_dir / "cluster_count.json")
        except Exception as exc:
            print(f"      cluster SKIP ({type(exc).__name__}: {exc})")
    except Exception as exc:
        print(f"      stay_locations SKIP ({type(exc).__name__}: {exc})")

    print(f"    → {out_dir}")


def _run_privacy_toy(tdf: skmob.TrajDataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("    input.parquet")
    _save(pd.DataFrame(tdf).copy(), out_dir / "input.parquet")

    for name, attack_cls, init_kwargs, assess_kwargs in PRIVACY_ATTACK_CASES:
        print(f"    {name}.parquet")
        try:
            attack = attack_cls(**init_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                result = attack.assess_risk(tdf.copy(), show_progress=False, **assess_kwargs)
            _save(result.reset_index(drop=True), out_dir / f"{name}.parquet")
        except Exception as exc:
            print(f"      SKIP ({type(exc).__name__}: {exc})")

    print(f"    → {out_dir}")


def _reset_model_rng(seed: int = MODEL_SEED) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import igraph

        if hasattr(igraph, "set_random_number_generator"):
            igraph.set_random_number_generator(random)
    except Exception:
        pass


def _run_model_case(name: str, fn, out_dir: Path) -> None:
    print(f"    {name}.parquet")
    try:
        _reset_model_rng()
        result = fn()
        df = (
            pd.DataFrame(result.to_numpy(), columns=list(result.columns))
            if isinstance(result, pd.DataFrame)
            else pd.DataFrame(result)
        )
        _save(df.reset_index(drop=True), out_dir / f"{name}.parquet")
    except Exception as exc:
        print(f"      SKIP ({type(exc).__name__}: {exc})")


def _fit_model_diary_generator(training: pd.DataFrame):
    mdg = SkmobMarkovDiaryGenerator()
    mdg.fit(training.copy(), 3, lid="cluster")
    return mdg


def _run_models(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tessellation = _load_model_tessellation()
    tessellation_input = _model_tessellation_input(tessellation)
    diary_training = _load_model_diary_training()

    print("    tessellation.parquet")
    _save(tessellation_input, out_dir / "tessellation.parquet")
    print("    input.parquet")
    _save(tessellation_input, out_dir / "input.parquet")
    print("    diary_training.parquet")
    _save(diary_training, out_dir / "diary_training.parquet")

    for out_format in ("flows", "probabilities", "flows_sample"):
        _run_model_case(
            f"gravity_{out_format}",
            lambda out_format=out_format: SkmobGravity().generate(
                tessellation,
                tile_id_column="tile_id",
                tot_outflows_column="tot_outflow",
                relevance_column="population",
                out_format=out_format,
            ),
            out_dir,
        )

    for out_format in ("flows", "probabilities", "flows_sample"):
        _run_model_case(
            f"radiation_{out_format}",
            lambda out_format=out_format: SkmobRadiation().generate(
                tessellation,
                tile_id_column="tile_id",
                tot_outflows_column="tot_outflow",
                relevance_column="population",
                out_format=out_format,
            ),
            out_dir,
        )

    _run_model_case(
        "markov_diary",
        lambda: _fit_model_diary_generator(diary_training).generate(24, MODEL_START, random_state=MODEL_SEED),
        out_dir,
    )

    for name, model_cls, relevance_column in (
        ("epr", SkmobEPR, "population"),
        ("density_epr", SkmobDensityEPR, "population"),
        ("spatial_epr", SkmobSpatialEPR, None),
    ):
        generate_kwargs = (
            {}
            if relevance_column is None
            else {
                "relevance_column": relevance_column,
            }
        )
        _run_model_case(
            name,
            lambda model_cls=model_cls, generate_kwargs=generate_kwargs: model_cls().generate(
                MODEL_START,
                MODEL_END,
                tessellation,
                n_agents=2,
                starting_locations=MODEL_STARTING_LOCATIONS.copy(),
                random_state=MODEL_SEED,
                show_progress=False,
                **generate_kwargs,
            ),
            out_dir,
        )

    _run_model_case(
        "geosim",
        lambda: SkmobGeoSim().generate(
            MODEL_START,
            MODEL_END,
            tessellation,
            social_graph=MODEL_SOCIAL_GRAPH,
            n_agents=3,
            random_state=MODEL_SEED,
            show_progress=False,
        ),
        out_dir,
    )
    _run_model_case(
        "sts_epr",
        lambda: SkmobSTSEPR().generate(
            MODEL_START,
            MODEL_END,
            tessellation,
            _fit_model_diary_generator(diary_training),
            social_graph=MODEL_SOCIAL_GRAPH,
            n_agents=3,
            rsl=False,
            relevance_column="population",
            random_state=MODEL_SEED,
            show_progress=False,
        ),
        out_dir,
    )

    print(f"    → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate skmob reference cache")
    parser.add_argument(
        "--datasets",
        default="brightkite,geolife,foursquare,privacy_toy,models",
        help="Comma-separated dataset names (default: all cached datasets)",
    )
    parser.add_argument("--geolife-rows", type=int, default=GEOLIFE_DEFAULT_ROWS)
    parser.add_argument("--foursquare-rows", type=int, default=FOURSQUARE_DEFAULT_ROWS)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    loaders = {
        "brightkite": _load_brightkite,
        "geolife": lambda: _load_geolife(args.geolife_rows),
        "foursquare": lambda: _load_foursquare(args.foursquare_rows),
        "privacy_toy": _load_privacy_toy,
        "models": lambda: None,
    }

    for dataset in datasets:
        if dataset not in loaders:
            print(f"Unknown dataset: {dataset!r} — skip")
            continue
        print(f"\n==> {dataset}")
        if dataset == "models":
            _run_models(_REFERENCE_DIR / dataset)
            continue
        tdf = loaders[dataset]()
        if dataset == "privacy_toy":
            _run_privacy_toy(tdf, _REFERENCE_DIR / dataset)
        else:
            _run_all(tdf, _REFERENCE_DIR / dataset)

    print("\nAll done. Commit tests/shared/skmob_reference/ to git to track the snapshots.")


if __name__ == "__main__":
    main()
