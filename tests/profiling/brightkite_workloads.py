"""Brightkite-backed workloads used by profiling runners."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL

DEFAULT_ROWS = 4_000_000
IMPLEMENTATIONS = ("fastmob", "skmob")
JUMP_LENGTHS_ENTRYPOINTS = ("method", "function")
DEFAULT_JUMP_LENGTHS_ENTRYPOINT = "method"
WorkloadFunc = Callable[[Any], Any]


@dataclass(frozen=True)
class Workload:
    name: str
    dataset: str
    import_path: str
    kwargs: dict[str, Any]
    description: str


@dataclass(frozen=True)
class PreparedWorkload:
    workload: Workload
    implementation: str
    rows: int
    backend: str
    data: Any
    func: WorkloadFunc
    call_kwargs: dict[str, Any]
    jump_lengths_entrypoint: str | None = None


def load_brightkite(rows: int = DEFAULT_ROWS, *, backend: str = "pandas") -> Any:
    """Load and normalize Brightkite check-ins for trajectory APIs."""
    _BRIGHTKITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _BRIGHTKITE_PATH.exists():
        urllib.request.urlretrieve(_BRIGHTKITE_URL, _BRIGHTKITE_PATH)

    nrows = None if rows <= 0 else rows
    df = pd.read_csv(
        _BRIGHTKITE_PATH,
        sep="\t",
        header=0,
        nrows=nrows,
        names=["uid", "datetime", "lat", "lng", "location id"],
    )
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["location_id"] = df["location id"].astype("string")
    df = df.dropna(subset=["uid", "datetime", "lat", "lng"]).reset_index(drop=True)

    if backend == "pandas":
        return df
    if backend == "polars":
        import polars as pl

        return pl.from_pandas(df)
    raise ValueError(f"Unsupported backend: {backend!r}")


def trajectory_to_visits(traj: Any) -> pd.DataFrame:
    """Derive visit-like rows from Brightkite check-ins."""
    df = _to_pandas(traj).sort_values(["uid", "datetime"]).reset_index(drop=True)
    visits = pd.DataFrame(
        {
            "user_id": df["uid"],
            "location_id": df["location_id"].astype("string"),
            "area": df["location_id"].astype("string"),
            "start_timestamp": df["datetime"],
        }
    )
    next_time = df.groupby("uid", sort=False)["datetime"].shift(-1)
    fallback_end = visits["start_timestamp"] + pd.Timedelta(minutes=30)
    visits["end_timestamp"] = next_time.where(next_time.notna(), fallback_end)
    visits["end_timestamp"] = visits["end_timestamp"].where(
        visits["end_timestamp"] <= fallback_end,
        fallback_end,
    )
    bad_end = visits["end_timestamp"] <= visits["start_timestamp"]
    visits.loc[bad_end, "end_timestamp"] = visits.loc[bad_end, "start_timestamp"] + pd.Timedelta(minutes=30)
    visits["duration_minutes"] = (visits["end_timestamp"] - visits["start_timestamp"]).dt.total_seconds() / 60.0
    visits["day_of_week"] = visits["start_timestamp"].dt.day_name().str.lower()
    visits["purpose"] = _purpose_from_location(visits["location_id"])
    visits["location_type"] = visits["purpose"]
    return visits


def trajectory_to_od(traj: Any) -> pd.DataFrame:
    """Build origin-destination rows from consecutive check-ins per user."""
    df = _to_pandas(traj).sort_values(["uid", "datetime"]).reset_index(drop=True)
    previous = df.groupby("uid", sort=False)["location_id"].shift(1)
    trips = pd.DataFrame(
        {
            "origin_area": previous,
            "destination_area": df["location_id"],
        }
    )
    return trips.dropna().reset_index(drop=True)


def trajectory_to_stvd_distributions(traj: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two deterministic spatio-temporal distributions from Brightkite."""
    df = _to_pandas(traj).copy()
    df["time_bin"] = df["datetime"].dt.floor("10min").dt.strftime("%H:%M")
    df["lat_bucket"] = df["lat"].round(2)
    df["lng_bucket"] = df["lng"].round(2)
    grouped = df.groupby(["lat_bucket", "lng_bucket", "time_bin"], sort=False).size().reset_index(name="mean_volume")
    grouped["centroid"] = "POINT (" + grouped["lng_bucket"].astype(str) + " " + grouped["lat_bucket"].astype(str) + ")"
    dist = grouped[["centroid", "time_bin", "mean_volume"]]
    if len(dist) < 2:
        dist = pd.concat([dist, dist], ignore_index=True)
    midpoint = max(1, len(dist) // 2)
    return (
        dist.iloc[:midpoint].reset_index(drop=True),
        dist.iloc[midpoint:].reset_index(drop=True),
    )


def _purpose_from_location(location_ids: pd.Series) -> pd.Series:
    codes = pd.factorize(location_ids, sort=False)[0] % 4
    return pd.Series(codes, index=location_ids.index).map({0: "HOME", 1: "WORK", 2: "SHOP", 3: "OTHER"})


def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _materialize(result: Any) -> None:
    """Force lazy-ish result objects enough that profiling captures the work."""
    if isinstance(result, tuple):
        for item in result:
            _materialize(item)
        return
    if hasattr(result, "shape"):
        _ = result.shape
    elif hasattr(result, "__len__"):
        _ = len(result)


def _make_workload(name: str, dataset: str, import_path: str, description: str, **kwargs: Any) -> Workload:
    return Workload(name, dataset, import_path, kwargs, description)


def _fastmob_workloads() -> dict[str, Workload]:
    workloads = {
        "filter": _make_workload("filter", "trajectory", "fastmob.preprocessing.filter", "fastmob filter"),
        "compress": _make_workload("compress", "trajectory", "fastmob.preprocessing.compress", "fastmob compress"),
        "stay_locations": _make_workload(
            "stay_locations",
            "trajectory",
            "fastmob.preprocessing.stay_locations",
            "fastmob stay locations",
        ),
        "cluster": _make_workload("cluster", "trajectory", "fastmob.preprocessing.cluster", "fastmob cluster"),
        "jump_lengths": _make_workload(
            "jump_lengths",
            "trajectory",
            "fastmob.measures.individual.jump_lengths.jump_lengths",
            "fastmob jump lengths",
            merge=False,
        ),
        "radius_of_gyration": _make_workload(
            "radius_of_gyration",
            "trajectory",
            "fastmob.measures.individual.radius_of_gyration.radius_of_gyration",
            "fastmob radius of gyration",
        ),
        "k_radius_of_gyration": _make_workload(
            "k_radius_of_gyration",
            "trajectory",
            "fastmob.measures.individual.k_radius_of_gyration.k_radius_of_gyration",
            "fastmob k radius of gyration",
        ),
        "number_of_visits": _make_workload(
            "number_of_visits",
            "trajectory",
            "fastmob.measures.individual.number_of_visits.number_of_visits",
            "fastmob number of visits",
        ),
        "number_of_locations": _make_workload(
            "number_of_locations",
            "trajectory",
            "fastmob.measures.individual.number_of_locations.number_of_locations",
            "fastmob number of locations",
        ),
        "maximum_distance": _make_workload(
            "maximum_distance",
            "trajectory",
            "fastmob.measures.individual.maximum_distance.maximum_distance",
            "fastmob maximum distance",
        ),
        "distance_straight_line": _make_workload(
            "distance_straight_line",
            "trajectory",
            "fastmob.measures.individual.distance_straight_line.distance_straight_line",
            "fastmob distance straight line",
        ),
        "waiting_times": _make_workload(
            "waiting_times",
            "trajectory",
            "fastmob.measures.individual.waiting_times.waiting_times",
            "fastmob waiting times",
        ),
        "home_location": _make_workload(
            "home_location",
            "trajectory",
            "fastmob.measures.individual.home_location.home_location",
            "fastmob home location",
        ),
        "max_distance_from_home": _make_workload(
            "max_distance_from_home",
            "trajectory",
            "fastmob.measures.individual.max_distance_from_home.max_distance_from_home",
            "fastmob max distance from home",
        ),
        "visits_per_location": _make_workload(
            "visits_per_location",
            "trajectory",
            "fastmob.measures.collective.visits_per_location.visits_per_location",
            "fastmob visits per location",
        ),
        "homes_per_location": _make_workload(
            "homes_per_location",
            "trajectory",
            "fastmob.measures.collective.homes_per_location.homes_per_location",
            "fastmob homes per location",
        ),
        "visits_per_time_unit": _make_workload(
            "visits_per_time_unit",
            "trajectory",
            "fastmob.measures.collective.visits_per_time_unit.visits_per_time_unit",
            "fastmob visits per time unit",
        ),
        "mean_square_displacement": _make_workload(
            "mean_square_displacement",
            "trajectory",
            "fastmob.measures.collective.mean_square_displacement.mean_square_displacement",
            "fastmob mean square displacement",
        ),
        "random_location_entropy": _make_workload(
            "random_location_entropy",
            "trajectory",
            "fastmob.measures.collective.random_location_entropy.random_location_entropy",
            "fastmob random location entropy",
        ),
        "uncorrelated_location_entropy": _make_workload(
            "uncorrelated_location_entropy",
            "trajectory",
            "fastmob.measures.collective.uncorrelated_location_entropy.uncorrelated_location_entropy",
            "fastmob uncorrelated location entropy",
        ),
        "random_entropy": _make_workload(
            "random_entropy",
            "trajectory",
            "fastmob.measures.individual.random_entropy.random_entropy",
            "fastmob random entropy",
        ),
        "uncorrelated_entropy": _make_workload(
            "uncorrelated_entropy",
            "trajectory",
            "fastmob.measures.individual.uncorrelated_entropy.uncorrelated_entropy",
            "fastmob uncorrelated entropy",
        ),
        "real_entropy": _make_workload(
            "real_entropy",
            "trajectory",
            "fastmob.measures.individual.real_entropy.real_entropy",
            "fastmob real entropy",
        ),
        "frequency_rank": _make_workload(
            "frequency_rank",
            "trajectory",
            "fastmob.measures.individual.frequency_rank.frequency_rank",
            "fastmob frequency rank",
        ),
        "recency_rank": _make_workload(
            "recency_rank",
            "trajectory",
            "fastmob.measures.individual.recency_rank.recency_rank",
            "fastmob recency rank",
        ),
        "location_frequency": _make_workload(
            "location_frequency",
            "trajectory",
            "fastmob.measures.individual.location_frequency.location_frequency",
            "fastmob location frequency",
        ),
        "individual_mobility_network": _make_workload(
            "individual_mobility_network",
            "trajectory",
            "fastmob.measures.individual.individual_mobility_network.individual_mobility_network",
            "fastmob individual mobility network",
        ),
        "activity_transition_matrix": _make_workload(
            "activity_transition_matrix",
            "visits",
            "fastmob.measures.individual.activity.activity_transition_matrix",
            "fastmob activity transition matrix",
        ),
        "diversity": _make_workload(
            "diversity", "visits", "fastmob.measures.individual.diversity.diversity", "fastmob diversity"
        ),
        "regularity": _make_workload(
            "regularity", "visits", "fastmob.measures.individual.regularity.regularity", "fastmob regularity"
        ),
        "trajectory_entropy": _make_workload(
            "trajectory_entropy",
            "visits",
            "fastmob.measures.individual.entropy.trajectory_entropy",
            "fastmob trajectory entropy",
        ),
        "trajectory_predictability": _make_workload(
            "trajectory_predictability",
            "visits",
            "fastmob.measures.individual.entropy.trajectory_predictability",
            "fastmob trajectory predictability",
        ),
        "intermittance_and_degree_of_return": _make_workload(
            "intermittance_and_degree_of_return",
            "visits",
            "fastmob.measures.individual.mobility_profiling.intermittance_and_degree_of_return",
            "fastmob intermittance and degree of return",
        ),
        "exploration_profiling": _make_workload(
            "exploration_profiling",
            "visits",
            "fastmob.measures.individual.mobility_profiling.exploration_profiling",
            "fastmob exploration profiling",
            random_seed=0,
        ),
        "mean_area_volume": _make_workload(
            "mean_area_volume",
            "visits",
            "fastmob.measures.individual.mean_area_volume.mean_area_volume",
            "fastmob mean area volume",
        ),
        "discover_daily_motifs_from_agents": _make_workload(
            "discover_daily_motifs_from_agents",
            "visits",
            "fastmob.measures.individual.motifs.discover_daily_motifs_from_agents",
            "fastmob daily motifs",
        ),
        "od_matrix": _make_workload("od_matrix", "od", "fastmob.measures.collective.od.od_matrix", "fastmob OD matrix"),
        "od_metrics_per_area": _make_workload(
            "od_metrics_per_area",
            "od_metrics",
            "fastmob.measures.collective.od.od_metrics_per_area",
            "fastmob OD metrics per area",
        ),
        "stvd_emd": _make_workload(
            "stvd_emd", "stvd", "fastmob.measures.evaluation.spatial.stvd_emd", "fastmob STVD-EMD"
        ),
    }
    return dict(sorted(workloads.items()))


def _skmob_workload_candidates() -> dict[str, Workload]:
    def individual(name: str, skmob_name: str | None = None, **kwargs: Any) -> Workload:
        skmob_name = skmob_name or name
        return _make_workload(
            name,
            "trajectory",
            f"skmob.measures.individual.{skmob_name}",
            f"skmob {skmob_name}",
            **kwargs,
        )

    candidates = {
        "jump_lengths": individual("jump_lengths", show_progress=False, merge=False),
        "radius_of_gyration": individual("radius_of_gyration", show_progress=False),
        "k_radius_of_gyration": individual("k_radius_of_gyration", show_progress=False),
        "number_of_visits": individual("number_of_visits"),
        "number_of_locations": individual("number_of_locations"),
        "maximum_distance": individual("maximum_distance"),
        "distance_straight_line": individual("distance_straight_line"),
        "waiting_times": individual("waiting_times"),
        "home_location": individual("home_location"),
        "max_distance_from_home": individual("max_distance_from_home"),
        "random_entropy": individual("random_entropy"),
        "uncorrelated_entropy": individual("uncorrelated_entropy"),
        "real_entropy": individual("real_entropy"),
        "frequency_rank": individual("frequency_rank", show_progress=False),
        "recency_rank": individual("recency_rank", show_progress=False),
        "location_frequency": individual("location_frequency", show_progress=False),
        "individual_mobility_network": individual("individual_mobility_network"),
    }
    return dict(sorted(candidates.items()))


def _resolve_import(import_path: str) -> WorkloadFunc:
    module_name, func_name = import_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)


def _patch_skmob_shapely_compat() -> None:
    """Let scikit-mobility import with Shapely 2, which removed cascaded_union."""
    try:
        import shapely.ops as shapely_ops
    except ImportError:
        return
    if not hasattr(shapely_ops, "cascaded_union") and hasattr(shapely_ops, "unary_union"):
        shapely_ops.cascaded_union = shapely_ops.unary_union


def workload_registry(implementation: str = "fastmob") -> dict[str, Workload]:
    if implementation == "fastmob":
        return _fastmob_workloads()
    if implementation != "skmob":
        raise ValueError(f"Unsupported implementation: {implementation!r}")

    _patch_skmob_shapely_compat()
    available = {}
    for name, workload in _skmob_workload_candidates().items():
        try:
            _resolve_import(workload.import_path)
        except (ImportError, AttributeError):
            continue
        available[name] = workload
    return available


def available_workloads(implementation: str = "fastmob") -> list[str]:
    return list(workload_registry(implementation))


def build_dataset_for_workload(
    workload: Workload,
    rows: int,
    backend: str,
    implementation: str = "fastmob",
    *,
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> Any:
    load_backend = "pandas" if implementation == "skmob" else backend
    traj = load_brightkite(rows, backend=load_backend)

    if implementation == "skmob":
        if workload.dataset != "trajectory":
            raise ValueError(f"skmob profiling only supports trajectory workloads, got {workload.dataset!r}")
        _patch_skmob_shapely_compat()
        import skmob

        return skmob.TrajDataFrame(
            _to_pandas(traj),
            latitude="lat",
            longitude="lng",
            datetime="datetime",
            user_id="uid",
        )

    if _uses_fastmob_jump_lengths_tdf(workload, implementation):
        import fastmob

        _validate_jump_lengths_entrypoint(jump_lengths_entrypoint)
        return fastmob.TrajDataFrame(traj, sort=jump_lengths_entrypoint == "method")
    if workload.dataset == "trajectory":
        return traj
    if workload.dataset == "visits":
        return trajectory_to_visits(traj)
    if workload.dataset == "od":
        return trajectory_to_od(traj)
    if workload.dataset == "od_metrics":
        from fastmob.measures.collective.od import od_matrix

        return od_matrix(trajectory_to_od(traj))
    if workload.dataset == "stvd":
        return trajectory_to_stvd_distributions(traj)
    raise ValueError(f"Unknown workload dataset kind: {workload.dataset!r}")


def prepare_workload(
    name: str,
    *,
    rows: int = DEFAULT_ROWS,
    backend: str = "pandas",
    implementation: str = "fastmob",
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> PreparedWorkload:
    workloads = workload_registry(implementation)
    if name not in workloads:
        available = ", ".join(workloads) or "none"
        raise SystemExit(
            f"Unknown workload {name!r} for {implementation}. Available {implementation} workloads: {available}"
        )

    if implementation == "fastmob":
        try:
            import fastmob._core  # noqa: F401
        except ImportError as exc:
            raise SystemExit("fastmob._core is not importable. Run `maturin develop` first.") from exc

    workload = workloads[name]
    data = build_dataset_for_workload(
        workload,
        rows,
        backend,
        implementation,
        jump_lengths_entrypoint=jump_lengths_entrypoint,
    )
    if _uses_fastmob_jump_lengths_tdf(workload, implementation):
        _validate_jump_lengths_entrypoint(jump_lengths_entrypoint)
        if jump_lengths_entrypoint == "method":
            return PreparedWorkload(
                workload,
                implementation,
                rows,
                backend,
                data,
                _jump_lengths_method_entrypoint,
                {},
                jump_lengths_entrypoint,
            )
        return PreparedWorkload(
            workload,
            implementation,
            rows,
            backend,
            data,
            _jump_lengths_function_entrypoint,
            dict(workload.kwargs),
            jump_lengths_entrypoint,
        )

    func = _resolve_import(workload.import_path)
    return PreparedWorkload(workload, implementation, rows, backend, data, func, dict(workload.kwargs))


def execute_prepared_workload(prepared: PreparedWorkload) -> None:
    if prepared.workload.dataset == "stvd":
        left, right = prepared.data
        result = prepared.func(left, right, **prepared.call_kwargs)
    else:
        result = prepared.func(prepared.data, **prepared.call_kwargs)
    _materialize(result)


def _uses_fastmob_jump_lengths_tdf(workload: Workload, implementation: str) -> bool:
    return implementation == "fastmob" and workload.name == "jump_lengths"


def _validate_jump_lengths_entrypoint(entrypoint: str) -> None:
    if entrypoint not in JUMP_LENGTHS_ENTRYPOINTS:
        choices = ", ".join(JUMP_LENGTHS_ENTRYPOINTS)
        raise ValueError(f"Unsupported jump_lengths entrypoint: {entrypoint!r}. Expected one of: {choices}")


def _jump_lengths_method_entrypoint(tdf: Any) -> Any:
    return tdf.jump_lengths()


def _jump_lengths_function_entrypoint(tdf: Any, **kwargs: Any) -> Any:
    from fastmob.measures.individual.jump_lengths import jump_lengths

    return jump_lengths(
        tdf.df,
        datetime_col=tdf.datetime_col,
        lat_col=tdf.lat_col,
        lng_col=tdf.lng_col,
        uid_col=tdf.uid_col,
        presorted=tdf.sorted,
        **kwargs,
    )


def run_workload(
    name: str,
    *,
    rows: int = DEFAULT_ROWS,
    backend: str = "pandas",
    implementation: str = "fastmob",
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> dict[str, Any]:
    prepared = prepare_workload(
        name,
        rows=rows,
        backend=backend,
        implementation=implementation,
        jump_lengths_entrypoint=jump_lengths_entrypoint,
    )
    execute_prepared_workload(prepared)
    return {
        "workload": name,
        "rows": rows,
        "backend": backend,
        "implementation": implementation,
        "dataset": prepared.workload.dataset,
        "profiled_phase": "full",
        "jump_lengths_entrypoint": prepared.jump_lengths_entrypoint,
    }


def run_prepared_child(
    name: str,
    *,
    rows: int = DEFAULT_ROWS,
    backend: str = "pandas",
    implementation: str = "fastmob",
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> int:
    prepared = prepare_workload(
        name,
        rows=rows,
        backend=backend,
        implementation=implementation,
        jump_lengths_entrypoint=jump_lengths_entrypoint,
    )
    _allow_external_profiler_attach()
    ready = {
        "event": "ready",
        "pid": os.getpid(),
        "workload": name,
        "rows": rows,
        "backend": backend,
        "implementation": implementation,
        "dataset": prepared.workload.dataset,
        "jump_lengths_entrypoint": prepared.jump_lengths_entrypoint,
    }
    print(json.dumps(ready), flush=True)
    sys.stdin.readline()
    execute_prepared_workload(prepared)
    print(
        json.dumps(
            {
                "event": "done",
                "workload": name,
                "rows": rows,
                "backend": backend,
                "implementation": implementation,
                "dataset": prepared.workload.dataset,
                "jump_lengths_entrypoint": prepared.jump_lengths_entrypoint,
            }
        ),
        flush=True,
    )
    return 0


def _allow_external_profiler_attach() -> None:
    """Allow sibling external profilers to attach on Linux with ptrace_scope=1."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL(None)
        pr_set_ptracer = 0x59616D61
        pr_set_ptracer_any = ctypes.c_ulong(-1).value
        libc.prctl(pr_set_ptracer, pr_set_ptracer_any, 0, 0, 0)
    except Exception:  # noqa: BLE001
        return


def run_scalene_function_profile(
    name: str,
    *,
    rows: int = DEFAULT_ROWS,
    backend: str = "pandas",
    implementation: str = "fastmob",
    jump_lengths_entrypoint: str = DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
) -> dict[str, Any]:
    from scalene import scalene_profiler

    prepared = prepare_workload(
        name,
        rows=rows,
        backend=backend,
        implementation=implementation,
        jump_lengths_entrypoint=jump_lengths_entrypoint,
    )
    scalene_profiler.start()
    try:
        execute_prepared_workload(prepared)
    finally:
        scalene_profiler.stop()
    return {
        "workload": name,
        "rows": rows,
        "backend": backend,
        "implementation": implementation,
        "dataset": prepared.workload.dataset,
        "profiled_phase": "function",
        "jump_lengths_entrypoint": prepared.jump_lengths_entrypoint,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=False, help="Workload name to run.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--backend", choices=["pandas", "polars"], default="pandas")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="fastmob")
    parser.add_argument(
        "--jump-lengths-entrypoint",
        choices=JUMP_LENGTHS_ENTRYPOINTS,
        default=DEFAULT_JUMP_LENGTHS_ENTRYPOINT,
        help="fastmob jump_lengths TrajDataFrame entrypoint to profile.",
    )
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    parser.add_argument("--prepared-child", action="store_true", help="Prepare workload, wait on stdin, then execute.")
    parser.add_argument(
        "--scalene-function-profile", action="store_true", help="Profile only workload execution with Scalene."
    )
    args = parser.parse_args(argv)

    workloads = workload_registry(args.implementation)
    if args.list:
        for name, workload in workloads.items():
            print(f"{name}\t{workload.dataset}\t{args.implementation}\t{workload.description}")
        return 0
    if not args.workload:
        parser.error("--workload is required unless --list is used")

    if args.prepared_child:
        return run_prepared_child(
            args.workload,
            rows=args.rows,
            backend=args.backend,
            implementation=args.implementation,
            jump_lengths_entrypoint=args.jump_lengths_entrypoint,
        )
    if args.scalene_function_profile:
        result = run_scalene_function_profile(
            args.workload,
            rows=args.rows,
            backend=args.backend,
            implementation=args.implementation,
            jump_lengths_entrypoint=args.jump_lengths_entrypoint,
        )
    else:
        result = run_workload(
            args.workload,
            rows=args.rows,
            backend=args.backend,
            implementation=args.implementation,
            jump_lengths_entrypoint=args.jump_lengths_entrypoint,
        )

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
