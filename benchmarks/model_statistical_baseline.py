"""Standalone statistical baseline for mobility model generation comparison.

Generates N runs of each stochastic model with seeds 0..N-1, computes all
pairwise comparison metrics between runs of the same model, and stores the
resulting percentile distributions as a JSON file.

The resulting baseline JSON is committed to tests/shared/ and used by
tests/correctness/models/test_statistical_model_parity.py to check whether
fastmob and skmob generate statistically equivalent trajectories.

Usage:
    # fastmob baseline (run in .venv)
    python benchmarks/model_statistical_baseline.py --library fastmob

    # skmob baseline (run in .venv-skmob)
    python benchmarks/model_statistical_baseline.py --library skmob

    # Custom output path
    python benchmarks/model_statistical_baseline.py \\
        --library fastmob --n-runs 50 --output results/my_baseline.json
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import platform
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"
DEFAULT_SHARED_DIR = REPO_ROOT / "tests" / "shared"

MODEL_SEED = 0
MODEL_START = "2020-01-01 08:00:00"
MODEL_END = "2020-01-02 08:00:00"
MODEL_SOCIAL_GRAPH = [[0, 1], [0, 2], [1, 2]]
MODEL_STARTING_LOCATIONS = [0, 1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    output_kind: str  # "trajectory" or "flow"
    kwargs: dict


TRAJECTORY_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("epr", "epr", "trajectory", {"model": "EPR", "relevance_column": "population"}),
    ModelSpec("density_epr", "epr", "trajectory", {"model": "DensityEPR", "relevance_column": "population"}),
    ModelSpec("spatial_epr", "epr", "trajectory", {"model": "SpatialEPR"}),
    ModelSpec("geosim", "geosim", "trajectory", {}),
    ModelSpec("sts_epr", "sts_epr", "trajectory", {"relevance_column": "population"}),
)

FLOW_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("gravity_flows_sample", "gravity", "flow", {"out_format": "flows_sample"}),
    ModelSpec("radiation_flows_sample", "radiation", "flow", {"out_format": "flows_sample"}),
)

ALL_SPECS = TRAJECTORY_SPECS + FLOW_SPECS


# ---------------------------------------------------------------------------
# RNG + library helpers (shared with benchmarks.models.speed_suite)
# ---------------------------------------------------------------------------


def reset_rng(seed: int) -> None:
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import igraph

        if hasattr(igraph, "set_random_number_generator"):
            igraph.set_random_number_generator(random)
    except Exception:
        pass


def import_model_classes(library: str) -> dict[str, Any]:
    if library == "fastmob":
        module = importlib.import_module("fastmob.models")
        return {
            "Gravity": module.Gravity,
            "Radiation": module.Radiation,
            "MarkovDiaryGenerator": module.MarkovDiaryGenerator,
            "EPR": module.EPR,
            "DensityEPR": module.DensityEPR,
            "SpatialEPR": module.SpatialEPR,
            "GeoSim": module.GeoSim,
            "STS_epr": module.STS_epr,
        }

    try:
        import numpy as _np

        if not hasattr(_np, "NaN"):
            _np.NaN = _np.nan  # type: ignore[attr-defined]
    except Exception:
        pass

    return {
        "Gravity": importlib.import_module("skmob.models.gravity").Gravity,
        "Radiation": importlib.import_module("skmob.models.radiation").Radiation,
        "MarkovDiaryGenerator": importlib.import_module("skmob.models.markov_diary_generator").MarkovDiaryGenerator,
        "EPR": importlib.import_module("skmob.models.epr").EPR,
        "DensityEPR": importlib.import_module("skmob.models.epr").DensityEPR,
        "SpatialEPR": importlib.import_module("skmob.models.epr").SpatialEPR,
        "GeoSim": importlib.import_module("skmob.models.geosim").GeoSim,
        "STS_epr": importlib.import_module("skmob.models.sts_epr").STS_epr,
    }


def fit_diary(mdg_cls: type, diary_training: Any) -> Any:
    mdg = mdg_cls()
    mdg.fit(diary_training.copy(), 3, lid="cluster")
    return mdg


def load_model_inputs(reference_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tessellation_path = reference_dir / "input.parquet"
    diary_path = reference_dir / "diary_training.parquet"
    if not tessellation_path.exists() or not diary_path.exists():
        raise SystemExit(
            f"Model benchmark inputs not found in {reference_dir}. "
            "Run 'bash scripts/populate_skmob_cache.sh --datasets models' first."
        )
    return pd.read_parquet(tessellation_path), pd.read_parquet(diary_path)


def _patch_geopandas_apply_for_numpy2(gpd: Any) -> None:
    if getattr(gpd.GeoSeries.apply, "_fastmob_numpy2_patch", False):
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

    apply._fastmob_numpy2_patch = True
    gpd.GeoSeries.apply = apply


def prepare_skmob_tessellation(tessellation: pd.DataFrame) -> Any:
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:
        raise RuntimeError(f"skmob requires GeoPandas/Shapely: {exc}") from exc

    _patch_geopandas_apply_for_numpy2(gpd)
    gdf = gpd.GeoDataFrame(
        tessellation.drop(columns=["lat", "lng"]).copy(),
        geometry=[Point(lng, lat) for lat, lng in zip(tessellation["lat"], tessellation["lng"])],
        crs="EPSG:4326",
    )
    return gdf


# ---------------------------------------------------------------------------
# Single-run generation
# ---------------------------------------------------------------------------


def generate_one_run(
    spec: ModelSpec,
    library: str,
    tessellation: Any,
    diary_training: pd.DataFrame,
    seed: int,
    *,
    n_agents: int,
) -> pd.DataFrame | None:
    """Generate one run of a model at a given seed. Returns None if the model cannot run."""
    classes = import_model_classes(library)
    start = pd.Timestamp(MODEL_START)
    end = pd.Timestamp(MODEL_END)
    starting_locs = list(range(n_agents))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if spec.kind == "gravity":
                reset_rng(seed)
                return pd.DataFrame(
                    classes["Gravity"]().generate(
                        tessellation,
                        tile_id_column="tile_id",
                        tot_outflows_column="tot_outflow",
                        relevance_column="population",
                        out_format=spec.kwargs["out_format"],
                    )
                )

            if spec.kind == "radiation":
                reset_rng(seed)
                return pd.DataFrame(
                    classes["Radiation"]().generate(
                        tessellation,
                        tile_id_column="tile_id",
                        tot_outflows_column="tot_outflow",
                        relevance_column="population",
                        out_format=spec.kwargs["out_format"],
                    )
                )

            if spec.kind == "epr":
                model_name = spec.kwargs["model"]
                generate_kwargs = {}
                if "relevance_column" in spec.kwargs:
                    generate_kwargs["relevance_column"] = spec.kwargs["relevance_column"]
                return pd.DataFrame(
                    classes[model_name]().generate(
                        start,
                        end,
                        tessellation,
                        n_agents=n_agents,
                        starting_locations=starting_locs.copy(),
                        random_state=seed,
                        show_progress=False,
                        **generate_kwargs,
                    )
                )

            if spec.kind == "geosim":
                classes["GeoSim"]  # availability check; raises KeyError if missing
                importlib.import_module("igraph")
                social_graph = [[i, j] for i in range(n_agents) for j in range(i + 1, n_agents)]
                return pd.DataFrame(
                    classes["GeoSim"]().generate(
                        start,
                        end,
                        tessellation,
                        social_graph=social_graph,
                        n_agents=n_agents,
                        random_state=seed,
                        show_progress=False,
                    )
                )

            if spec.kind == "sts_epr":
                importlib.import_module("igraph")
                social_graph = [[i, j] for i in range(n_agents) for j in range(i + 1, n_agents)]
                diary = fit_diary(classes["MarkovDiaryGenerator"], diary_training)
                return pd.DataFrame(
                    classes["STS_epr"]().generate(
                        start,
                        end,
                        tessellation,
                        diary,
                        social_graph=social_graph,
                        n_agents=n_agents,
                        rsl=False,
                        relevance_column=spec.kwargs.get("relevance_column", "relevance"),
                        random_state=seed,
                        show_progress=False,
                    )
                )

        except ImportError:
            return None
        except Exception as exc:
            print(f"    [error] {spec.name} seed={seed}: {type(exc).__name__}: {exc}")
            return None

    return None


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def _flat_numpy(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _import_comparison() -> Any:
    return importlib.import_module("fastmob.measures.evaluation")


def _import_spatial() -> Any:
    return importlib.import_module("fastmob.measures.individual")


def _trajectory_to_std(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to uid/datetime/lat/lng."""
    if hasattr(df, "df"):
        df = df.df
    out = df.copy()
    renames = {}
    for col in out.columns:
        lc = col.lower()
        if col != "uid" and lc in ("uid", "user", "user_id"):
            renames[col] = "uid"
        elif col != "datetime" and lc in ("datetime", "timestamp", "time"):
            renames[col] = "datetime"
        elif col != "lat" and lc in ("lat", "latitude"):
            renames[col] = "lat"
        elif col != "lng" and lc in ("lng", "lon", "longitude"):
            renames[col] = "lng"
    if renames:
        out = out.rename(columns=renames)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lng"] = pd.to_numeric(out["lng"], errors="coerce")
    return out.dropna(subset=["datetime", "lat", "lng"])


def compute_trajectory_pair_metrics(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    cmp = _import_comparison()
    spatial = _import_spatial()

    out: dict[str, float] = {}

    # visits per user
    try:
        val, _ = cmp.visits_per_user_wasserstein_distance(df1, df2)
        out["visits_per_user_wasserstein"] = float(val)
    except Exception:
        pass

    # jump lengths
    try:
        jl1 = _flat_numpy(spatial.jump_lengths(df1, merge=True))
        jl2 = _flat_numpy(spatial.jump_lengths(df2, merge=True))
        if jl1.size > 0 and jl2.size > 0:
            out["jump_lengths_wasserstein"] = float(cmp.wasserstein_distance(jl1, jl2))
    except Exception:
        pass

    # radius of gyration
    try:
        rog_df1 = pd.DataFrame(spatial.radius_of_gyration(df1))
        rog_df2 = pd.DataFrame(spatial.radius_of_gyration(df2))
        rog1 = _flat_numpy(rog_df1["radius_of_gyration"].to_numpy())
        rog2 = _flat_numpy(rog_df2["radius_of_gyration"].to_numpy())
        if rog1.size > 0 and rog2.size > 0:
            out["radius_of_gyration_wasserstein"] = float(cmp.wasserstein_distance(rog1, rog2))
    except Exception:
        pass

    # waiting times
    try:
        wt1 = _flat_numpy(spatial.waiting_times(df1, merge=True))
        wt2 = _flat_numpy(spatial.waiting_times(df2, merge=True))
        if wt1.size > 0 and wt2.size > 0:
            out["waiting_times_wasserstein"] = float(cmp.wasserstein_distance(wt1, wt2))
    except Exception:
        pass

    return out


def compute_flow_pair_metrics(df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, float]:
    cmp = _import_comparison()
    out: dict[str, float] = {}

    try:
        def to_od_matrix(df: pd.DataFrame) -> pd.DataFrame:
            if hasattr(df, "df"):
                df = df.df
            d = pd.DataFrame(df)[["origin", "destination", "flow"]].copy()
            d["origin"] = d["origin"].astype(str)
            d["destination"] = d["destination"].astype(str)
            d["flow"] = pd.to_numeric(d["flow"], errors="coerce").fillna(0.0)
            return d.pivot_table(index="origin", columns="destination", values="flow", aggfunc="sum").fillna(0.0)

        mat1 = to_od_matrix(df1)
        mat2 = to_od_matrix(df2)
        out["od_matrix_cpc"] = float(cmp.od_matrix_common_part_of_commuters(mat1, mat2))
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Pairwise aggregation and summary
# ---------------------------------------------------------------------------


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n_pairs": 0,
            "mean": None, "std": None,
            "p5": None, "p25": None, "p50": None,
            "p75": None, "p95": None, "p99": None,
        }
    a = np.array(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n_pairs": len(values),
            "mean": None, "std": None,
            "p5": None, "p25": None, "p50": None,
            "p75": None, "p95": None, "p99": None,
        }
    return {
        "n_pairs": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p5":  float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def run_model_baseline(
    spec: ModelSpec,
    library: str,
    tessellation: Any,
    diary_training: pd.DataFrame,
    *,
    n_runs: int,
    n_agents: int,
) -> dict[str, Any] | None:
    print(f"  Generating {n_runs} runs of {spec.name}...", flush=True)

    runs: list[pd.DataFrame] = []
    for seed in range(n_runs):
        df = generate_one_run(spec, library, tessellation, diary_training, seed, n_agents=n_agents)
        if df is None:
            print(f"    [skip] {spec.name} — skipped (see error above or missing optional dep)")
            return None
        runs.append(df)
        if (seed + 1) % 10 == 0:
            print(f"    {seed + 1}/{n_runs} runs done", flush=True)

    n_pairs = len(runs) * (len(runs) - 1) // 2
    print(f"  Computing metrics for {n_pairs} pairs...", flush=True)

    if spec.output_kind == "trajectory":
        std_runs = [_trajectory_to_std(df) for df in runs]
        metric_buckets: dict[str, list[float]] = {}
        for i, j in itertools.combinations(range(len(std_runs)), 2):
            metrics = compute_trajectory_pair_metrics(std_runs[i], std_runs[j])
            for k, v in metrics.items():
                metric_buckets.setdefault(k, []).append(v)
    else:
        metric_buckets = {}
        for i, j in itertools.combinations(range(len(runs)), 2):
            metrics = compute_flow_pair_metrics(runs[i], runs[j])
            for k, v in metrics.items():
                metric_buckets.setdefault(k, []).append(v)

    result = {k: summarize(v) for k, v in metric_buckets.items()}
    for k, v in result.items():
        p50 = v.get("p50")
        p95 = v.get("p95")
        fmt_p50 = f"{p50:.4f}" if p50 is not None else "n/a"
        fmt_p95 = f"{p95:.4f}" if p95 is not None else "n/a"
        print(f"    {k}: p50={fmt_p50} p95={fmt_p95}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute statistical baseline for model generation comparison.")
    parser.add_argument("--library", choices=["fastmob", "skmob"], required=True)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: tests/shared/model_statistical_baseline_{library}.json)",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    return parser.parse_args(argv)


def build_tessellation(library: str, tessellation: pd.DataFrame) -> Any:
    if library == "fastmob":
        return tessellation.copy()
    return prepare_skmob_tessellation(tessellation)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    output_path = args.output or DEFAULT_SHARED_DIR / f"model_statistical_baseline_{args.library}.json"
    tessellation_raw, diary_training = load_model_inputs(Path(args.reference_dir))

    try:
        tessellation = build_tessellation(args.library, tessellation_raw)
    except RuntimeError as exc:
        print(f"Cannot prepare tessellation for {args.library}: {exc}")
        return 1

    baseline: dict[str, Any] = {}
    n_pairs = args.n_runs * (args.n_runs - 1) // 2

    for spec in ALL_SPECS:
        print(f"\n[{spec.name}]")
        result = run_model_baseline(
            spec,
            args.library,
            tessellation,
            diary_training,
            n_runs=args.n_runs,
            n_agents=args.n_agents,
        )
        if result is not None:
            baseline[spec.name] = result

    payload = {
        "metadata": {
            "library": args.library,
            "n_runs": args.n_runs,
            "n_pairs": n_pairs,
            "n_agents": args.n_agents,
            "seeds": list(range(args.n_runs)),
            "model_start": MODEL_START,
            "model_end": MODEL_END,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "baseline": baseline,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote baseline to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
