"""Standalone preprocessing speed benchmark suite.

Run from the repository root, for example:

    python benchmarks/preprocessing/speed_suite.py --library fastmob --backend both
    python benchmarks/preprocessing/speed_suite.py --library skmob
    python benchmarks/preprocessing/speed_suite.py --library movingpandas
    python benchmarks/preprocessing/speed_suite.py --library ptrail
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmarks.sorted_input_cache import DEFAULT_INPUT_CACHE_DIR, load_or_create_sorted_input
from benchmarks.utils import (
    concrete_backends as iter_concrete_backends,
    concrete_input_orders as iter_concrete_input_orders,
    error_result,
    load_catalog,
    merge_payload,
    nonnegative_float,
    nonnegative_int,
    positive_int,
    run_profiled_call,
    run_profiled_call_isolated,
    size_label,
    skipped_result,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
SKMOB_CATALOG_PATH = Path(__file__).resolve().parents[1] / "skmob_public_api_catalog.json"

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))
from benchmark_env import detect_cpu_info, get_default_output_dir  # noqa: E402

MOVINGPANDAS_CATALOG_PATH = Path(__file__).resolve().parents[1] / "movingpandas_skmob_api_catalog.json"
DEFAULT_SIZES = [1_000, 10_000, 100_000, 1_000_000, 4_000_000]
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    fastmob_module_path: str
    skmob_module_path: str
    func_name: str
    kwargs: dict[str, Any]
    input_kind: str = "trajectory"
    movingpandas_api: str | None = None
    ptrail_api: str | None = None


PREPROCESSING_METRICS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "filter",
        "fastmob.preprocessing",
        "skmob.preprocessing.filtering",
        "filter",
        {},
        input_kind="preprocessing",
    ),
    BenchmarkSpec(
        "compress",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "compress",
        {},
        input_kind="preprocessing",
        movingpandas_api="MinDistanceGeneralizer.generalize",
    ),
    BenchmarkSpec(
        "stay_locations",
        "fastmob.preprocessing",
        "skmob.preprocessing.detection",
        "stay_locations",
        {},
        input_kind="preprocessing",
        movingpandas_api="TrajectoryStopDetector.get_stop_points",
    ),
    BenchmarkSpec(
        "cluster",
        "fastmob.preprocessing",
        "skmob.preprocessing.clustering",
        "cluster",
        {},
        input_kind="preprocessing",
    ),
    BenchmarkSpec(
        "simplify_douglas_peucker",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "simplify",
        {"method": "douglas_peucker", "epsilon_km": 0.05},
        input_kind="preprocessing",
        movingpandas_api="DouglasPeuckerGeneralizer.generalize",
    ),
    BenchmarkSpec(
        "simplify_top_down_time_ratio",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "simplify",
        {"method": "top_down_time_ratio", "epsilon_km": 0.05},
        input_kind="preprocessing",
        movingpandas_api="TopDownTimeRatioGeneralizer.generalize",
    ),
    BenchmarkSpec(
        "simplify_min_distance",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "simplify",
        {"method": "min_distance", "min_distance_km": 0.2},
        input_kind="preprocessing",
        movingpandas_api="MinDistanceGeneralizer.generalize",
    ),
    BenchmarkSpec(
        "simplify_min_time_delta",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "simplify",
        {"method": "min_time_delta", "min_time_delta_s": 600.0},
        input_kind="preprocessing",
        movingpandas_api="MinTimeDeltaGeneralizer.generalize",
    ),
    BenchmarkSpec(
        "simplify_max_distance",
        "fastmob.preprocessing",
        "skmob.preprocessing.compression",
        "simplify",
        {"method": "max_distance", "epsilon_km": 0.05},
        input_kind="preprocessing",
        movingpandas_api="MaxDistanceGeneralizer.generalize",
    ),
    # Outlier-detection methods: no skmob or MovingPandas analogue exists for
    # any of these, so skmob_module_path deliberately points at a submodule
    # that does not exist (a clean import failure -> SkippedMetric, matching
    # this file's established "no live comparison library" convention) and
    # none of these spec names appear in movingpandas_callable_for_spec's
    # dispatch (falls through to its own "no benchmarkable analogue" skip).
    # outlier_hampel is the one exception: it has a real PTRAIL analogue,
    # wired through ptrail_api + ptrail_callable_for_spec below.
    BenchmarkSpec(
        "outlier_hampel",
        "fastmob.preprocessing",
        "skmob.preprocessing.no_hampel_analogue",
        "filter",
        {"method": "hampel"},
        input_kind="preprocessing",
        ptrail_api="Filters.hampel_outlier_detection",
    ),
    BenchmarkSpec(
        "outlier_greedy",
        "fastmob.preprocessing",
        "skmob.preprocessing.no_greedy_analogue",
        "filter",
        {"method": "greedy", "max_speed_kmh": 100.0},
        input_kind="preprocessing",
    ),
    BenchmarkSpec(
        "outlier_smart_greedy",
        "fastmob.preprocessing",
        "skmob.preprocessing.no_smart_greedy_analogue",
        "filter",
        {"method": "smart_greedy", "max_speed_kmh": 100.0},
        input_kind="preprocessing",
    ),
    BenchmarkSpec(
        "outlier_zheng",
        "fastmob.preprocessing",
        "skmob.preprocessing.no_zheng_analogue",
        "filter",
        {"method": "zheng", "max_speed_kmh": 100.0, "min_seg_size": 1},
        input_kind="preprocessing",
    ),
)


class SkippedMetric(Exception):
    """Raised when a metric cannot be benchmarked in the selected library."""


def build_output_path(
    output_dir: Path,
    library: str,
    timing_mode: str,
    backend: str | None = None,
    profile: str = "speed",
    input_order: str = "raw",
) -> Path:
    order_part = "" if input_order == "raw" else f"{input_order}_"
    if library == "fastmob":
        if backend is None or backend == "both":
            raise ValueError("fastmob output path requires a concrete backend")
        filename = f"fastmob_preprocessing_{profile}_{order_part}{backend}.json"
    elif library == "skmob":
        filename = f"skmob_preprocessing_{profile}_{order_part}{timing_mode}.json"
    else:
        filename = (
            f"{library}_preprocessing_{profile}_{input_order}.json"
            if input_order != "raw"
            else f"{library}_preprocessing_{profile}.json"
        )
    return output_dir / filename


def import_metric(spec: BenchmarkSpec, library: str) -> Callable[..., Any]:
    if library == "skmob":
        patch_numpy_nan_for_skmob()

    try:
        module_path = spec.fastmob_module_path if library == "fastmob" else spec.skmob_module_path
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise SkippedMetric(f"import failed: {exc}") from exc

    try:
        return getattr(module, spec.func_name)
    except AttributeError as exc:
        raise SkippedMetric(f"metric not available: {spec.func_name}") from exc


def patch_numpy_nan_for_skmob() -> None:
    """Restore the NumPy alias still referenced by scikit-mobility 1.3.x."""
    try:
        import numpy as np
    except Exception:
        return

    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]


def metric_kwargs_for_library(
    spec: BenchmarkSpec,
    library: str,
    func: Callable[..., Any],
    *,
    input_order: str = "raw",
) -> dict[str, Any]:
    kwargs = dict(spec.kwargs)
    if library == "fastmob" and input_order == "sorted" and spec.input_kind == "trajectory":
        kwargs["presorted"] = True
    if library != "skmob":
        return kwargs

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    accepts_extra_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if "show_progress" in signature.parameters or accepts_extra_kwargs:
        kwargs.setdefault("show_progress", False)
    return kwargs


def load_brightkite_pandas(data_path: Path):
    import pandas as pd

    df = pd.read_csv(data_path, sep="\t", header=None, names=BRIGHTKITE_COLUMNS)
    df["check-in_time"] = pd.to_datetime(df["check-in_time"], errors="coerce")
    return df


def repeat_brightkite_pandas(df: Any, repeat_factor: int) -> Any:
    if repeat_factor <= 1:
        return df

    import pandas as pd

    repeated_chunks = []
    user_as_text = df["user"].astype(str)
    for repeat_idx in range(repeat_factor):
        chunk = df.copy()
        chunk["user"] = user_as_text + f"__rep{repeat_idx}"
        repeated_chunks.append(chunk)
    return pd.concat(repeated_chunks, ignore_index=True)


def load_brightkite_polars(data_path: Path):
    import polars as pl

    return pl.read_csv(
        data_path,
        separator="\t",
        has_header=False,
        new_columns=BRIGHTKITE_COLUMNS,
        try_parse_dates=True,
    )


def repeat_brightkite_polars(df: Any, repeat_factor: int) -> Any:
    if repeat_factor <= 1:
        return df

    import polars as pl

    repeated_chunks = []
    for repeat_idx in range(repeat_factor):
        repeated_chunks.append(
            df.with_columns((pl.col("user").cast(pl.Utf8) + pl.lit(f"__rep{repeat_idx}")).alias("user"))
        )
    return pl.concat(repeated_chunks, how="vertical")


def load_brightkite_movingpandas(data_path: Path, size: int, *, repeat_factor: int = 1) -> Any:
    try:
        import geopandas as gpd
        import movingpandas as mpd
        import pandas as pd
    except Exception as exc:
        raise SkippedMetric(f"movingpandas input setup failed: {exc}") from exc

    df = load_brightkite_pandas(data_path)
    df = repeat_brightkite_pandas(df, repeat_factor).head(size).copy()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf["check-in_time"] = pd.to_datetime(gdf["check-in_time"])
    gdf = gdf.set_index("check-in_time")
    return mpd.TrajectoryCollection(gdf, traj_id_col="user")


def load_brightkite_ptrail(data_path: Path, size: int, *, repeat_factor: int = 1) -> Any:
    """Build a PTRAILDataFrame with a precomputed ``Speed`` kinematic column.

    @usedBy `benchmark_ptrail_size`. The ``Speed`` column is computed once
    here (not inside the timed call) to match how a real caller would use
    PTRAIL: `Filters.hampel_outlier_detection` operates on an
    already-annotated column, not on raw lat/lng.
    """
    try:
        from ptrail.core.TrajectoryDF import PTRAILDataFrame
        from ptrail.features.kinematic_features import KinematicFeatures
    except Exception as exc:
        raise SkippedMetric(f"ptrail input setup failed: {exc}") from exc

    df = load_brightkite_pandas(data_path)
    df = repeat_brightkite_pandas(df, repeat_factor).head(size).copy()
    df = df.dropna(subset=["user", "check-in_time", "latitude", "longitude"])
    if df["check-in_time"].dt.tz is not None:
        df["check-in_time"] = df["check-in_time"].dt.tz_localize(None)
    tdf = PTRAILDataFrame(
        df,
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        traj_id="user",
    )
    return KinematicFeatures.create_speed_column(tdf)


def make_skmob_tdf(skmob_module: Any, df: Any) -> Any:
    return skmob_module.TrajDataFrame(
        df.copy(),
        latitude="latitude",
        longitude="longitude",
        datetime="check-in_time",
        user_id="user",
    )


def make_skmob_cluster_tdf(skmob_module: Any, df: Any) -> Any:
    valid_df = df.dropna(subset=["latitude", "longitude"]).copy()
    dropped = len(df) - len(valid_df)
    if dropped:
        print(f"    dropped {dropped} rows with missing coordinates for skmob cluster")
    return make_skmob_tdf(skmob_module, valid_df)


def movingpandas_radius_of_gyration(tc: Any) -> list[float]:
    results: list[float] = []
    for traj in tc.trajectories:
        coords = [(point.y, point.x) for point in traj.df.geometry]
        if not coords:
            results.append(math.nan)
            continue
        center_lat = sum(lat for lat, _ in coords) / len(coords)
        center_lng = sum(lng for _, lng in coords) / len(coords)
        squared = [(lat - center_lat) ** 2 + (lng - center_lng) ** 2 for lat, lng in coords]
        results.append(math.sqrt(sum(squared) / len(squared)))
    return results


def movingpandas_callable_for_spec(spec: BenchmarkSpec) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    if spec.name in {"distance_straight_line", "jump_lengths"}:
        return lambda tc, **kwargs: tc.add_distance(**kwargs), {"overwrite": True, "units": "km"}
    if spec.name == "waiting_times":
        return lambda tc, **kwargs: tc.add_timedelta(**kwargs), {"overwrite": True}
    if spec.name == "radius_of_gyration":
        return movingpandas_radius_of_gyration, {}
    if spec.name == "compress":
        try:
            import movingpandas as mpd
        except Exception as exc:
            raise SkippedMetric(f"movingpandas import failed: {exc}") from exc
        return lambda tc, **kwargs: mpd.MinDistanceGeneralizer(tc).generalize(**kwargs), {"tolerance": 200}
    if spec.name == "stay_locations":
        try:
            import movingpandas as mpd
        except Exception as exc:
            raise SkippedMetric(f"movingpandas import failed: {exc}") from exc
        return (
            lambda tc, **kwargs: mpd.TrajectoryStopDetector(tc).get_stop_points(**kwargs),
            {"max_diameter": 200, "min_duration": timedelta(minutes=20)},
        )
    if spec.name in {
        "simplify_douglas_peucker",
        "simplify_top_down_time_ratio",
        "simplify_min_distance",
        "simplify_min_time_delta",
        "simplify_max_distance",
    }:
        try:
            import movingpandas as mpd
        except Exception as exc:
            raise SkippedMetric(f"movingpandas import failed: {exc}") from exc
        # This benchmark's TrajectoryCollection stays in unprojected EPSG:4326
        # (see load_brightkite_movingpandas), so DouglasPeucker/TopDownTimeRatio/
        # MaxDistance -- which measure raw shapely distance in the trajectory's
        # native CRS units -- take a tolerance in *degrees*, not metres.
        # MinDistance/MinTimeDelta convert lat/lng distance to metres/timedelta
        # internally regardless of CRS, so they keep metre/timedelta tolerances.
        # Only relative timing is compared here, not output row counts, so an
        # approximate degrees<->metres correspondence is sufficient.
        generalizer_cls = {
            "simplify_douglas_peucker": mpd.DouglasPeuckerGeneralizer,
            "simplify_top_down_time_ratio": mpd.TopDownTimeRatioGeneralizer,
            "simplify_min_distance": mpd.MinDistanceGeneralizer,
            "simplify_min_time_delta": mpd.MinTimeDeltaGeneralizer,
            "simplify_max_distance": mpd.MaxDistanceGeneralizer,
        }[spec.name]
        tolerance = {
            "simplify_douglas_peucker": 0.0005,
            "simplify_top_down_time_ratio": 0.0005,
            "simplify_min_distance": 200.0,
            "simplify_min_time_delta": timedelta(minutes=10),
            "simplify_max_distance": 0.0005,
        }[spec.name]
        return (
            lambda tc, **kwargs: generalizer_cls(tc).generalize(**kwargs),
            {"tolerance": tolerance},
        )
    raise SkippedMetric("no benchmarkable MovingPandas analogue")


def ptrail_callable_for_spec(spec: BenchmarkSpec) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Return (callable, kwargs) for one spec's real PTRAIL analogue.

    @usedBy `benchmark_ptrail_size`. Only ``outlier_hampel`` has a real
    PTRAIL analogue (``Filters.hampel_outlier_detection``); every other
    spec cleanly skips.
    """
    if spec.name == "outlier_hampel":
        try:
            from ptrail.preprocessing.filters import Filters
        except Exception as exc:
            raise SkippedMetric(f"ptrail import failed: {exc}") from exc
        return lambda tdf, **kwargs: Filters.hampel_outlier_detection(tdf, **kwargs), {"column_name": "Speed"}
    raise SkippedMetric("no benchmarkable PTRAIL analogue")


def benchmark_metric(
    spec: BenchmarkSpec,
    library: str,
    make_input: Callable[[], Any],
    *,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
    input_order: str = "raw",
) -> dict[str, Any]:
    print(f"  {spec.name}")
    try:
        if library == "movingpandas":
            func, kwargs = movingpandas_callable_for_spec(spec)
        elif library == "ptrail":
            func, kwargs = ptrail_callable_for_spec(spec)
        else:
            func = import_metric(spec, library)
            kwargs = metric_kwargs_for_library(spec, library, func, input_order=input_order)
    except SkippedMetric as exc:
        print(f"    skipped: {exc}")
        return skipped_result(str(exc), profile)

    try:
        if retries > 0:
            return run_profiled_call_isolated(
                func,
                make_input,
                kwargs,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                retries=retries,
            )
        return run_profiled_call(
            func,
            make_input,
            kwargs,
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:
        print(f"    error: {exc}")
        return error_result(str(exc), profile)


def benchmark_fastmob_size(
    df: Any,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
    input_order: str = "raw",
) -> dict[str, Any]:
    size_df = df.head(size)
    print(f"\nSize {size_label(size)} ({len(size_df)} rows)")
    return {
        "size": size,
        "label": size_label(size),
        "rows": len(size_df),
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                "fastmob",
                lambda size_df=size_df: size_df,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                retries=retries,
                input_order=input_order,
            )
            for spec in specs
        },
    }


def benchmark_skmob_size(
    raw_df: Any,
    skmob_module: Any,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    timing_mode: str,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
    input_order: str = "raw",
) -> dict[str, Any]:
    pandas_slice = raw_df.head(size).copy()
    print(f"\nSize {size_label(size)} ({len(pandas_slice)} rows)")

    if timing_mode == "prebuilt_tdf":
        tdf = make_skmob_tdf(skmob_module, pandas_slice)

        def make_input() -> Any:
            return tdf

    else:

        def make_input() -> Any:
            return make_skmob_tdf(skmob_module, pandas_slice)

    metrics = {}
    for spec in specs:
        metric_make_input = make_input
        if spec.name == "cluster":
            if timing_mode == "prebuilt_tdf":
                cluster_tdf = make_skmob_cluster_tdf(skmob_module, pandas_slice)

                def metric_make_input(cluster_tdf=cluster_tdf) -> Any:
                    return cluster_tdf

            else:

                def metric_make_input() -> Any:
                    return make_skmob_cluster_tdf(skmob_module, pandas_slice)

        metrics[spec.name] = benchmark_metric(
            spec,
            "skmob",
            metric_make_input,
            profile=profile,
            iterations=iterations,
            sleep_seconds=sleep_seconds,
            retries=retries,
            input_order=input_order,
        )

    return {
        "size": size,
        "label": size_label(size),
        "rows": len(pandas_slice),
        "metrics": metrics,
    }


def benchmark_movingpandas_size(
    data_path: Path,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    repeat_factor: int = 1,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
    input_order: str = "raw",
) -> dict[str, Any]:
    print(f"\nSize {size_label(size)}")
    try:
        tc = load_brightkite_movingpandas(data_path, size, repeat_factor=repeat_factor)
    except SkippedMetric as exc:
        return {
            "size": size,
            "label": size_label(size),
            "rows": 0,
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in specs},
        }

    rows = len(tc.to_point_gdf())
    print(f"  MovingPandas rows: {rows}")
    return {
        "size": size,
        "label": size_label(size),
        "rows": rows,
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                "movingpandas",
                lambda tc=tc: tc,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                retries=retries,
                input_order=input_order,
            )
            for spec in specs
        },
    }


def benchmark_ptrail_size(
    data_path: Path,
    size: int,
    *,
    specs: tuple[BenchmarkSpec, ...],
    repeat_factor: int = 1,
    iterations: int,
    sleep_seconds: float,
    profile: str = "speed",
    retries: int = 0,
    input_order: str = "raw",
) -> dict[str, Any]:
    print(f"\nSize {size_label(size)}")
    try:
        tdf = load_brightkite_ptrail(data_path, size, repeat_factor=repeat_factor)
    except SkippedMetric as exc:
        return {
            "size": size,
            "label": size_label(size),
            "rows": 0,
            "metrics": {spec.name: skipped_result(str(exc), profile) for spec in specs},
        }

    rows = len(tdf)
    print(f"  PTRAIL rows: {rows}")
    return {
        "size": size,
        "label": size_label(size),
        "rows": rows,
        "metrics": {
            spec.name: benchmark_metric(
                spec,
                "ptrail",
                lambda tdf=tdf: tdf,
                profile=profile,
                iterations=iterations,
                sleep_seconds=sleep_seconds,
                retries=retries,
                input_order=input_order,
            )
            for spec in specs
        },
    }


def build_metadata(
    args: argparse.Namespace,
    *,
    input_type: str,
    timing_mode: str,
    backend: str | None,
    input_cache_path: Path | None = None,
    input_cache_status: str = "not_applicable",
) -> dict[str, Any]:
    cpu = detect_cpu_info()
    return {
        "suite": "preprocessing",
        "library": args.library,
        "profile": args.profile,
        "backend": backend,
        "timing_mode": timing_mode,
        "input_type": input_type,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_cores": cpu["cores"],
        "cpu_vendor": cpu["vendor_slug"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "iterations": args.iterations,
        "sleep_seconds": args.sleep_seconds,
        "retries": args.retries,
        "repeat_dataset": args.repeat_dataset,
        "sizes": args.sizes,
        "metrics": args.metrics,
        "input_order": args.input_order,
        "input_cache_path": None if input_cache_path is None else str(input_cache_path),
        "input_cache_status": input_cache_status,
        "skmob_catalog_path": str(SKMOB_CATALOG_PATH),
        "movingpandas_catalog_path": str(MOVINGPANDAS_CATALOG_PATH),
        **({"memory_method": "memray_peak_heap"} if args.profile == "memory" else {}),
    }


def load_brightkite_for_order(
    *,
    data_path: Path,
    backend: str,
    input_order: str,
    input_cache_dir: Path,
    repeat_factor: int,
) -> tuple[Any, Path | None, str]:
    if input_order == "raw":
        if backend == "polars":
            return repeat_brightkite_polars(load_brightkite_polars(data_path), repeat_factor), None, "not_applicable"
        return repeat_brightkite_pandas(load_brightkite_pandas(data_path), repeat_factor), None, "not_applicable"

    sorted_input = load_or_create_sorted_input(
        cache_dir=input_cache_dir,
        suite="preprocessing",
        backend=backend,
        data_path=data_path,
        load_raw=lambda: (
            repeat_brightkite_polars(load_brightkite_polars(data_path), repeat_factor)
            if backend == "polars"
            else repeat_brightkite_pandas(load_brightkite_pandas(data_path), repeat_factor)
        ),
        uid_col="user",
        datetime_col="check-in_time",
        repeat_factor=repeat_factor if repeat_factor > 1 else None,
    )
    return sorted_input.data, sorted_input.path, sorted_input.status


def run_suite(args: argparse.Namespace, *, backend: str | None = None) -> dict[str, Any]:
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Place the Brightkite file there before running.")
    specs = selected_specs(args)

    if args.library == "fastmob":
        selected_backend = backend or args.backend
        if selected_backend == "both":
            raise ValueError("run_suite requires a concrete backend when library is fastmob")
        if selected_backend == "pandas":
            input_type = "pandas.DataFrame"
        else:
            input_type = "polars.DataFrame"
        print(f"Loading {args.input_order} Brightkite into {selected_backend} from {data_path}...")
        df, input_cache_path, input_cache_status = load_brightkite_for_order(
            data_path=data_path,
            backend=selected_backend,
            input_order=args.input_order,
            input_cache_dir=Path(args.input_cache_dir),
            repeat_factor=args.repeat_dataset,
        )
        results = [
            benchmark_fastmob_size(
                df,
                size,
                specs=specs,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
                input_order=args.input_order,
            )
            for size in args.sizes
        ]
        metadata = build_metadata(
            args,
            input_type=input_type,
            timing_mode="measure_only",
            backend=selected_backend,
            input_cache_path=input_cache_path,
            input_cache_status=input_cache_status,
        )
        return {"metadata": metadata, "results": results}

    if args.library == "movingpandas":
        results = [
            benchmark_movingpandas_size(
                data_path,
                size,
                specs=specs,
                repeat_factor=args.repeat_dataset,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
            )
            for size in args.sizes
        ]
        metadata = build_metadata(
            args,
            input_type="movingpandas.TrajectoryCollection",
            timing_mode="measure_only",
            backend=None,
        )
        return {"metadata": metadata, "results": results}

    if args.library == "ptrail":
        results = [
            benchmark_ptrail_size(
                data_path,
                size,
                specs=specs,
                repeat_factor=args.repeat_dataset,
                profile=args.profile,
                iterations=args.iterations,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
            )
            for size in args.sizes
        ]
        metadata = build_metadata(
            args,
            input_type="ptrail.PTRAILDataFrame",
            timing_mode="measure_only",
            backend=None,
        )
        return {"metadata": metadata, "results": results}

    print(f"Loading {args.input_order} Brightkite into pandas from {data_path}...")
    raw_df, input_cache_path, input_cache_status = load_brightkite_for_order(
        data_path=data_path,
        backend="pandas",
        input_order=args.input_order,
        input_cache_dir=Path(args.input_cache_dir),
        repeat_factor=args.repeat_dataset,
    )
    try:
        skmob_module = importlib.import_module("skmob")
    except Exception as exc:
        raise SystemExit(f"Unable to import original skmob: {exc}") from exc

    results = [
        benchmark_skmob_size(
            raw_df,
            skmob_module,
            size,
            specs=specs,
            timing_mode=args.timing_mode,
            profile=args.profile,
            iterations=args.iterations,
            sleep_seconds=args.sleep_seconds,
            retries=args.retries,
            input_order=args.input_order,
        )
        for size in args.sizes
    ]
    metadata = build_metadata(
        args,
        input_type="skmob.TrajDataFrame",
        timing_mode=args.timing_mode,
        backend=None,
        input_cache_path=input_cache_path,
        input_cache_status=input_cache_status,
    )
    return {"metadata": metadata, "results": results}


def concrete_backends(args: argparse.Namespace) -> Iterable[str | None]:
    return iter_concrete_backends(args.library, args.backend)


def concrete_input_orders(args: argparse.Namespace) -> Iterable[str]:
    return iter_concrete_input_orders(args.input_order)


def selected_specs(args: argparse.Namespace) -> tuple[BenchmarkSpec, ...]:
    requested = set(args.metrics)
    return tuple(spec for spec in PREPROCESSING_METRICS if spec.name in requested)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone preprocessing speed benchmarks.")
    parser.add_argument("--library", choices=["fastmob", "skmob", "movingpandas", "ptrail"], required=True)
    parser.add_argument("--backend", choices=["pandas", "polars", "both"], default="both")
    parser.add_argument("--profile", choices=["speed", "memory"], default="speed")
    parser.add_argument("--timing-mode", choices=["prebuilt_tdf", "workflow_tdf"], default="prebuilt_tdf")
    parser.add_argument("--input-order", choices=["raw", "sorted", "both"], default="raw")
    parser.add_argument("--input-cache-dir", type=Path, default=DEFAULT_INPUT_CACHE_DIR)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--sleep", dest="sleep_seconds", type=nonnegative_float, default=0.5)
    parser.add_argument(
        "--retries", type=nonnegative_int, default=0, help="Retry a metric this many times after failure."
    )
    parser.add_argument(
        "--repeat-dataset",
        type=positive_int,
        default=1,
        help="Repeat the Brightkite dataset this many times, forcing unique user ids for each repeat.",
    )
    parser.add_argument("--sizes", type=positive_int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument(
        "--metrics",
        choices=[spec.name for spec in PREPROCESSING_METRICS],
        nargs="+",
        default=[spec.name for spec in PREPROCESSING_METRICS],
        help="Only run the selected preprocessing metrics.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge selected sizes/metrics into an existing output JSON instead of replacing the whole file.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = get_default_output_dir()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for input_order in concrete_input_orders(args):
        order_args = argparse.Namespace(**vars(args))
        order_args.input_order = input_order
        for backend in concrete_backends(order_args):
            payload = run_suite(order_args, backend=backend)
            output_path = build_output_path(
                Path(order_args.output_dir),
                order_args.library,
                order_args.timing_mode,
                backend,
                order_args.profile,
                order_args.input_order,
            )
            if order_args.append and output_path.exists():
                payload = merge_payload(load_catalog(output_path), payload)
            write_json(payload, output_path)
            print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
