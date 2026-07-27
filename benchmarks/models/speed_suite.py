"""Standalone generation model speed benchmark suite.

Run from the repository root, for example:

    python benchmarks/models/speed_suite.py --library fastmob
    python benchmarks/models/speed_suite.py --library skmob --n-agents 2 10 50 --n-locations 12 50 200
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import os
import platform
import random
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir

DEFAULT_AGENT_COUNTS = [2, 10, 50]
DEFAULT_LOCATION_COUNTS = [12, 50, 200]
DEFAULT_SIZES = DEFAULT_LOCATION_COUNTS
MODEL_SEED = 2
MODEL_START = "2020-01-01 08:00:00"
MODEL_END = "2020-01-01 14:00:00"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    kind: str
    kwargs: dict[str, Any]


MODEL_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("gravity_flows", "gravity", {"out_format": "flows"}),
    BenchmarkSpec("gravity_probabilities", "gravity", {"out_format": "probabilities"}),
    BenchmarkSpec("radiation_flows", "radiation", {"out_format": "flows"}),
    BenchmarkSpec("radiation_probabilities", "radiation", {"out_format": "probabilities"}),
    BenchmarkSpec("markov_diary", "markov_diary", {"diary_length": 24}),
    BenchmarkSpec("epr", "epr", {"model": "EPR", "relevance_column": "population"}),
    BenchmarkSpec("density_epr", "epr", {"model": "DensityEPR", "relevance_column": "population"}),
    BenchmarkSpec("spatial_epr", "epr", {"model": "SpatialEPR"}),
    BenchmarkSpec("geosim", "geosim", {}),
    BenchmarkSpec("sts_epr", "sts_epr", {"relevance_column": "population"}),
)
LOCATION_MODEL_BENCHMARKS = tuple(spec for spec in MODEL_BENCHMARKS if spec.kind in {"gravity", "radiation"})
DIARY_MODEL_BENCHMARKS = tuple(spec for spec in MODEL_BENCHMARKS if spec.kind == "markov_diary")
TRAJECTORY_MODEL_BENCHMARKS = tuple(spec for spec in MODEL_BENCHMARKS if spec.kind in {"epr", "geosim", "sts_epr"})


class SkippedBenchmark(Exception):
    """Raised when a model benchmark cannot run in the selected library."""


def size_label(size: int) -> str:
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def summarize_times(times: list[float]) -> dict[str, float | None]:
    if not times:
        return {"average_seconds": None, "minimum_seconds": None}
    return {"average_seconds": sum(times) / len(times), "minimum_seconds": min(times)}


def summarize_memory(peak_memory_mb: list[float]) -> dict[str, float | None]:
    if not peak_memory_mb:
        return {
            "average_peak_memory_mb": None,
            "minimum_peak_memory_mb": None,
            "maximum_peak_memory_mb": None,
        }
    return {
        "average_peak_memory_mb": sum(peak_memory_mb) / len(peak_memory_mb),
        "minimum_peak_memory_mb": min(peak_memory_mb),
        "maximum_peak_memory_mb": max(peak_memory_mb),
    }


def build_output_path(output_dir: Path, library: str, profile: str = "speed") -> Path:
    return output_dir / f"{library}_models_{profile}.json"


def write_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_model_inputs(reference_dir: Path) -> tuple[Any, Any]:
    import pandas as pd

    tessellation_path = reference_dir / "input.parquet"
    diary_path = reference_dir / "diary_training.parquet"
    if not tessellation_path.exists() or not diary_path.exists():
        raise SystemExit(
            f"Model benchmark inputs not found in {reference_dir}. "
            "Run 'bash scripts/populate_skmob_cache.sh --datasets models' first."
        )
    return pd.read_parquet(tessellation_path), pd.read_parquet(diary_path)


def expand_tessellation(tessellation: Any, size: int) -> Any:
    import pandas as pd

    base = tessellation.reset_index(drop=True)
    if size <= len(base):
        return base.head(size).copy()

    frames = []
    repeats = (size + len(base) - 1) // len(base)
    for repeat in range(repeats):
        frame = base.copy()
        frame["tile_id"] = frame["tile_id"].astype(str) + f"_{repeat}"
        frame["lat"] = frame["lat"].astype(float) + repeat * 0.01
        frame["lng"] = frame["lng"].astype(float) + repeat * 0.01
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).head(size)


def model_case_label(n_agents: int, n_locations: int) -> str:
    return f"{n_agents} agents / {n_locations} locations"


def location_case_label(n_locations: int) -> str:
    return f"{n_locations} locations"


def starting_locations(n_agents: int, n_locations: int) -> list[int]:
    return [agent_index % n_locations for agent_index in range(n_agents)]


def social_graph(n_agents: int) -> list[list[int]]:
    return [[i, j] for i in range(n_agents) for j in range(i + 1, n_agents)]


def prepare_library_inputs(library: str, tessellation: Any, diary_training: Any) -> tuple[Any, Any]:
    if library == "fastmob":
        return tessellation.copy(), diary_training.copy()

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as exc:
        raise SkippedBenchmark(f"original skmob model inputs require GeoPandas/Shapely: {exc}") from exc

    patch_geopandas_apply_for_numpy2(gpd)
    gdf = gpd.GeoDataFrame(
        tessellation.drop(columns=["lat", "lng"]).copy(),
        geometry=[Point(lng, lat) for lat, lng in zip(tessellation["lat"], tessellation["lng"])],
        crs="EPSG:4326",
    )
    return gdf, diary_training.copy()


def patch_geopandas_apply_for_numpy2(gpd: Any) -> None:
    if getattr(gpd.GeoSeries.apply, "_fastmob_numpy2_patch", False):
        return
    original_apply = gpd.GeoSeries.apply

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        try:
            return original_apply(self, func, convert_dtype=convert_dtype, args=args, **kwargs)
        except ValueError as exc:
            if "Unable to avoid copy" not in str(exc):
                raise
            import pandas as pd

            series = pd.Series(list(self), index=self.index)
            return series.apply(lambda geom: func(geom, *args), **kwargs)

    apply._fastmob_numpy2_patch = True
    gpd.GeoSeries.apply = apply


def reset_rng(seed: int = MODEL_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import igraph

        if hasattr(igraph, "set_random_number_generator"):
            igraph.set_random_number_generator(random)
    except Exception:  # noqa: BLE001, S110
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
        import numpy as np

        if not hasattr(np, "NaN"):
            np.NaN = np.nan  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001, S110
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


def build_call(
    spec: BenchmarkSpec,
    library: str,
    tessellation: Any,
    diary_training: Any,
    *,
    n_agents: int | None = None,
) -> Callable[[], Any]:
    classes = import_model_classes(library)
    start = to_timestamp(MODEL_START)
    end = to_timestamp(MODEL_END)

    if spec.kind == "gravity":
        return lambda: classes["Gravity"]().generate(
            tessellation,
            tile_id_column="tile_id",
            tot_outflows_column="tot_outflow",
            relevance_column="population",
            out_format=spec.kwargs["out_format"],
        )

    if spec.kind == "radiation":
        return lambda: classes["Radiation"]().generate(
            tessellation,
            tile_id_column="tile_id",
            tot_outflows_column="tot_outflow",
            relevance_column="population",
            out_format=spec.kwargs["out_format"],
        )

    if spec.kind == "markov_diary":
        return lambda: fit_diary(classes["MarkovDiaryGenerator"], diary_training).generate(
            spec.kwargs["diary_length"], start, random_state=MODEL_SEED
        )

    if spec.kind == "epr":
        if n_agents is None:
            raise SkippedBenchmark("EPR benchmarks require n_agents")
        model_name = spec.kwargs["model"]
        generate_kwargs = {}
        if "relevance_column" in spec.kwargs:
            generate_kwargs["relevance_column"] = spec.kwargs["relevance_column"]
        start_locs = starting_locations(n_agents, len(tessellation))
        return lambda: classes[model_name]().generate(
            start,
            end,
            tessellation,
            n_agents=n_agents,
            starting_locations=start_locs.copy(),
            random_state=MODEL_SEED,
            show_progress=False,
            **generate_kwargs,
        )

    if spec.kind == "geosim":
        if n_agents is None:
            raise SkippedBenchmark("GeoSim benchmarks require n_agents")
        graph = social_graph(n_agents)
        return lambda: classes["GeoSim"]().generate(
            start,
            end,
            tessellation,
            social_graph=[edge.copy() for edge in graph],
            n_agents=n_agents,
            random_state=MODEL_SEED,
            show_progress=False,
        )

    if spec.kind == "sts_epr":
        if n_agents is None:
            raise SkippedBenchmark("STS_epr benchmarks require n_agents")
        graph = social_graph(n_agents)
        return lambda: classes["STS_epr"]().generate(
            start,
            end,
            tessellation,
            fit_diary(classes["MarkovDiaryGenerator"], diary_training),
            social_graph=[edge.copy() for edge in graph],
            n_agents=n_agents,
            rsl=False,
            relevance_column=spec.kwargs["relevance_column"],
            random_state=MODEL_SEED,
            show_progress=False,
        )

    raise SkippedBenchmark(f"unknown benchmark kind: {spec.kind}")


def to_timestamp(value: str) -> Any:
    import pandas as pd

    return pd.Timestamp(value)


def fit_diary(mdg_cls: type, diary_training: Any) -> Any:
    mdg = mdg_cls()
    mdg.fit(diary_training.copy(), 3, lid="cluster")
    return mdg


def run_timed_call(func: Callable[[], Any], *, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    print("    Warming up...")
    reset_rng()
    call_benchmark_func(func)

    times: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        reset_rng()
        start = time.perf_counter()
        call_benchmark_func(func)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        print(f"    Round {i + 1}: {duration:.4f} seconds")

    summary = summarize_times(times)
    print(f"    Average Time: {summary['average_seconds']:.4f} s")
    print(f"    Minimum Time: {summary['minimum_seconds']:.4f} s")
    return {"status": "ok", "times_seconds": times, "iterations_completed": len(times), **summary}


def run_memory_call(func: Callable[[], Any], *, iterations: int, sleep_seconds: float) -> dict[str, Any]:
    import memray

    peak_memory_mb: list[float] = []
    for i in range(iterations):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        reset_rng()
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp_path = f.name
        os.unlink(tmp_path)
        try:
            with memray.Tracker(tmp_path, native_traces=False):
                call_benchmark_func(func)
            reader = memray.FileReader(tmp_path)
            peak_bytes = sum(record.size for record in reader.get_high_watermark_allocation_records(merge_threads=True))
            peak_mb = peak_bytes / (1024 * 1024)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
        peak_memory_mb.append(peak_mb)
        print(f"    Round {i + 1}: {peak_mb:.4f} MB peak (memray)")

    summary = summarize_memory(peak_memory_mb)
    print(f"    Average Peak Memory: {summary['average_peak_memory_mb']:.4f} MB")
    print(f"    Minimum Peak Memory: {summary['minimum_peak_memory_mb']:.4f} MB")
    print(f"    Maximum Peak Memory: {summary['maximum_peak_memory_mb']:.4f} MB")
    return {
        "status": "ok",
        "peak_memory_mb": peak_memory_mb,
        "current_memory_mb": [],
        "iterations_completed": len(peak_memory_mb),
        **summary,
    }


def call_benchmark_func(func: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return func()


def empty_profile_result(status: str, reason: str, profile: str) -> dict[str, Any]:
    if profile == "memory":
        return {
            "status": status,
            "reason": reason,
            "peak_memory_mb": [],
            "current_memory_mb": [],
            "iterations_completed": 0,
            **summarize_memory([]),
        }
    return {
        "status": status,
        "reason": reason,
        "times_seconds": [],
        "iterations_completed": 0,
        **summarize_times([]),
    }


def skipped_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("skipped", reason, profile)


def error_result(reason: str, profile: str = "speed") -> dict[str, Any]:
    return empty_profile_result("error", reason, profile)


def benchmark_model(
    spec: BenchmarkSpec,
    library: str,
    tessellation: Any,
    diary_training: Any,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    n_agents: int | None = None,
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        func = build_call(spec, library, tessellation, diary_training, n_agents=n_agents)
    except Exception as exc:  # noqa: BLE001
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    try:
        if profile == "memory":
            return run_memory_call(func, iterations=iterations, sleep_seconds=sleep_seconds)
        return run_timed_call(func, iterations=iterations, sleep_seconds=sleep_seconds)
    except Exception as exc:  # noqa: BLE001
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_size(
    library: str,
    base_tessellation: Any,
    diary_training: Any,
    size: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    specs: tuple[BenchmarkSpec, ...] = MODEL_BENCHMARKS,
) -> dict[str, Any]:
    size_tessellation = expand_tessellation(base_tessellation, size)
    n_agents = DEFAULT_AGENT_COUNTS[0]
    print(f"\nSize {size_label(size)} ({len(size_tessellation)} locations)")
    try:
        tessellation, diary = prepare_library_inputs(library, size_tessellation, diary_training)
    except SkippedBenchmark as exc:
        return {
            "size": size,
            "label": size_label(size),
            "locations": len(size_tessellation),
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in specs},
        }

    return {
        "size": size,
        "label": size_label(size),
        "locations": len(size_tessellation),
        "metrics": {
            spec.name: benchmark_model(
                spec,
                library,
                tessellation,
                diary,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                n_agents=n_agents if spec in TRAJECTORY_MODEL_BENCHMARKS else None,
            )
            for spec in specs
        },
    }


def benchmark_location_case(
    library: str,
    base_tessellation: Any,
    diary_training: Any,
    n_locations: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    specs: tuple[BenchmarkSpec, ...] = LOCATION_MODEL_BENCHMARKS,
) -> dict[str, Any]:
    size_tessellation = expand_tessellation(base_tessellation, n_locations)
    print(f"\nLocation-only models: {location_case_label(n_locations)}")
    try:
        tessellation, diary = prepare_library_inputs(library, size_tessellation, diary_training)
    except SkippedBenchmark as exc:
        return {
            "benchmark_group": "location_models",
            "label": location_case_label(n_locations),
            "n_agents": None,
            "n_locations": len(size_tessellation),
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in specs},
        }

    return {
        "benchmark_group": "location_models",
        "label": location_case_label(n_locations),
        "n_agents": None,
        "n_locations": len(size_tessellation),
        "metrics": {
            spec.name: benchmark_model(
                spec,
                library,
                tessellation,
                diary,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in specs
        },
    }


def benchmark_diary_case(
    library: str,
    diary_training: Any,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    specs: tuple[BenchmarkSpec, ...] = DIARY_MODEL_BENCHMARKS,
) -> dict[str, Any]:
    print("\nDiary-only models")
    return {
        "benchmark_group": "diary_models",
        "label": "diary only",
        "n_agents": None,
        "n_locations": None,
        "metrics": {
            spec.name: benchmark_model(
                spec,
                library,
                None,
                diary_training,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
            )
            for spec in specs
        },
    }


def benchmark_trajectory_case(
    library: str,
    base_tessellation: Any,
    diary_training: Any,
    n_agents: int,
    n_locations: int,
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    specs: tuple[BenchmarkSpec, ...] = TRAJECTORY_MODEL_BENCHMARKS,
) -> dict[str, Any]:
    size_tessellation = expand_tessellation(base_tessellation, n_locations)
    print(f"\nTrajectory models: {model_case_label(n_agents, n_locations)}")
    try:
        tessellation, diary = prepare_library_inputs(library, size_tessellation, diary_training)
    except SkippedBenchmark as exc:
        return {
            "benchmark_group": "trajectory_models",
            "label": model_case_label(n_agents, n_locations),
            "n_agents": n_agents,
            "n_locations": len(size_tessellation),
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in specs},
        }

    return {
        "benchmark_group": "trajectory_models",
        "label": model_case_label(n_agents, len(size_tessellation)),
        "n_agents": n_agents,
        "n_locations": len(size_tessellation),
        "metrics": {
            spec.name: benchmark_model(
                spec,
                library,
                tessellation,
                diary,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                n_agents=n_agents,
            )
            for spec in specs
        },
    }


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "models",
        "library": args.library,
        "profile": args.profile,
        "input_type": "cached_skmob_model_reference",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dir": str(args.reference_dir),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "n_agents": args.n_agents,
        "n_locations": args.n_locations,
        "metrics": args.metrics,
        **({"memory_method": "memray_peak_heap"} if args.profile == "memory" else {}),
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    base_tessellation, diary_training = load_model_inputs(Path(args.reference_dir))
    results = []
    requested = {spec.name for spec in selected_specs(args)}
    location_specs = tuple(spec for spec in LOCATION_MODEL_BENCHMARKS if spec.name in requested)
    diary_specs = tuple(spec for spec in DIARY_MODEL_BENCHMARKS if spec.name in requested)
    trajectory_specs = tuple(spec for spec in TRAJECTORY_MODEL_BENCHMARKS if spec.name in requested)
    if location_specs:
        for n_locations in args.n_locations:
            results.append(
                benchmark_location_case(
                    args.library,
                    base_tessellation,
                    diary_training,
                    n_locations,
                    profile=args.profile,
                    iterations=args.iterations,
                    sleep_seconds=args.sleep_seconds,
                    specs=location_specs,
                )
            )
    if diary_specs:
        results.append(
            benchmark_diary_case(
                args.library,
                diary_training,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                specs=diary_specs,
            )
        )
    if trajectory_specs:
        for n_locations in args.n_locations:
            for n_agents in args.n_agents:
                results.append(
                    benchmark_trajectory_case(
                        args.library,
                        base_tessellation,
                        diary_training,
                        n_agents,
                        n_locations,
                        profile=args.profile,
                        iterations=args.iterations,
                        sleep_seconds=args.sleep_seconds,
                        specs=trajectory_specs,
                    )
                )
    return {"metadata": build_metadata(args), "results": results}


def selected_specs(args: argparse.Namespace) -> tuple[BenchmarkSpec, ...]:
    requested = set(args.metrics)
    return tuple(spec for spec in MODEL_BENCHMARKS if spec.name in requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone generation model benchmarks.")
    parser.add_argument("--library", choices=["fastmob", "skmob"], required=True)
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument("--n-agents", type=positive_int, nargs="+", default=DEFAULT_AGENT_COUNTS)
    parser.add_argument("--n-locations", type=positive_int, nargs="+", default=None)
    parser.add_argument(
        "--metrics",
        choices=[spec.name for spec in MODEL_BENCHMARKS],
        nargs="+",
        default=[spec.name for spec in MODEL_BENCHMARKS],
        help="Only run the selected generation model metrics.",
    )
    parser.add_argument(
        "--sizes",
        type=positive_int,
        nargs="+",
        default=None,
        help="Deprecated alias for --n-locations.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    args = parser.parse_args(argv)
    if args.n_locations is None:
        args.n_locations = args.sizes if args.sizes is not None else DEFAULT_LOCATION_COUNTS
    elif args.sizes is not None:
        parser.error("--sizes is a deprecated alias for --n-locations; pass only one of them")
    args.sizes = args.n_locations
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_suite(args)
    output_path = build_output_path(Path(args.output_dir), args.library, args.profile)
    write_json(payload, output_path)
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
