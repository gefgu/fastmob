"""Brightkite-backed workloads used by the py-spy profiling runner."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from tests.shared.brightkite import _BRIGHTKITE_PATH, _BRIGHTKITE_URL


DEFAULT_ROWS = 4_000_000
WorkloadFunc = Callable[[Any], Any]


@dataclass(frozen=True)
class Workload:
    name: str
    dataset: str
    func: WorkloadFunc
    description: str


def load_brightkite(rows: int = DEFAULT_ROWS, *, backend: str = "pandas") -> Any:
    """Load and normalize Brightkite check-ins for skmob2 trajectory APIs."""
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
    visits.loc[bad_end, "end_timestamp"] = visits.loc[
        bad_end, "start_timestamp"
    ] + pd.Timedelta(minutes=30)
    visits["duration_minutes"] = (
        visits["end_timestamp"] - visits["start_timestamp"]
    ).dt.total_seconds() / 60.0
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
    grouped = (
        df.groupby(["lat_bucket", "lng_bucket", "time_bin"], sort=False)
        .size()
        .reset_index(name="mean_volume")
    )
    grouped["centroid"] = (
        "POINT ("
        + grouped["lng_bucket"].astype(str)
        + " "
        + grouped["lat_bucket"].astype(str)
        + ")"
    )
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
    return pd.Series(codes, index=location_ids.index).map(
        {0: "HOME", 1: "WORK", 2: "SHOP", 3: "OTHER"}
    )


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


def _trajectory_workloads() -> dict[str, Workload]:
    def make(name: str, import_path: str, kwargs: dict[str, Any] | None = None) -> Workload:
        kwargs = kwargs or {}

        def run(traj: Any) -> Any:
            module_name, func_name = import_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[func_name])
            return getattr(module, func_name)(traj, **kwargs)

        return Workload(name, "trajectory", run, f"{import_path} on Brightkite")

    return {
        "filter": make("filter", "skmob2.preprocessing.filter.filter"),
        "compress": make("compress", "skmob2.preprocessing.compress.compress"),
        "stay_locations": make(
            "stay_locations", "skmob2.preprocessing.stay_locations.stay_locations"
        ),
        "cluster": make("cluster", "skmob2.preprocessing.cluster.cluster"),
        "jump_lengths": make(
            "jump_lengths",
            "skmob2.measures.spatial.jump_lengths.jump_lengths",
            {"show_progress": False, "merge": False},
        ),
        "radius_of_gyration": make(
            "radius_of_gyration",
            "skmob2.measures.spatial.radius_of_gyration.radius_of_gyration",
            {"show_progress": False},
        ),
        "k_radius_of_gyration": make(
            "k_radius_of_gyration",
            "skmob2.measures.spatial.k_radius_of_gyration.k_radius_of_gyration",
            {"show_progress": False},
        ),
        "number_of_visits": make(
            "number_of_visits", "skmob2.measures.spatial.number_of_visits.number_of_visits"
        ),
        "number_of_locations": make(
            "number_of_locations",
            "skmob2.measures.spatial.number_of_locations.number_of_locations",
        ),
        "maximum_distance": make(
            "maximum_distance", "skmob2.measures.spatial.maximum_distance.maximum_distance"
        ),
        "distance_straight_line": make(
            "distance_straight_line",
            "skmob2.measures.spatial.distance_straight_line.distance_straight_line",
        ),
        "waiting_times": make(
            "waiting_times", "skmob2.measures.spatial.waiting_times.waiting_times"
        ),
        "home_location": make(
            "home_location", "skmob2.measures.spatial.home_location.home_location"
        ),
        "max_distance_from_home": make(
            "max_distance_from_home",
            "skmob2.measures.spatial.max_distance_from_home.max_distance_from_home",
        ),
        "visits_per_location": make(
            "visits_per_location",
            "skmob2.measures.flows.visits_per_location.visits_per_location",
        ),
        "homes_per_location": make(
            "homes_per_location",
            "skmob2.measures.flows.homes_per_location.homes_per_location",
        ),
        "visits_per_time_unit": make(
            "visits_per_time_unit",
            "skmob2.measures.flows.visits_per_time_unit.visits_per_time_unit",
        ),
        "mean_square_displacement": make(
            "mean_square_displacement",
            "skmob2.measures.flows.mean_square_displacement.mean_square_displacement",
        ),
        "random_location_entropy": make(
            "random_location_entropy",
            "skmob2.measures.flows.random_location_entropy.random_location_entropy",
        ),
        "uncorrelated_location_entropy": make(
            "uncorrelated_location_entropy",
            "skmob2.measures.flows.uncorrelated_location_entropy.uncorrelated_location_entropy",
        ),
        "random_entropy": make(
            "random_entropy", "skmob2.measures.visits.random_entropy.random_entropy"
        ),
        "uncorrelated_entropy": make(
            "uncorrelated_entropy",
            "skmob2.measures.visits.uncorrelated_entropy.uncorrelated_entropy",
        ),
        "real_entropy": make(
            "real_entropy", "skmob2.measures.visits.real_entropy.real_entropy"
        ),
        "frequency_rank": make(
            "frequency_rank", "skmob2.measures.visits.frequency_rank.frequency_rank"
        ),
        "recency_rank": make(
            "recency_rank", "skmob2.measures.visits.recency_rank.recency_rank"
        ),
        "location_frequency": make(
            "location_frequency",
            "skmob2.measures.visits.location_frequency.location_frequency",
        ),
        "individual_mobility_network": make(
            "individual_mobility_network",
            "skmob2.measures.visits.individual_mobility_network.individual_mobility_network",
        ),
    }


def _visit_workloads() -> dict[str, Workload]:
    def make(name: str, import_path: str, kwargs: dict[str, Any] | None = None) -> Workload:
        kwargs = kwargs or {}

        def run(visits: pd.DataFrame) -> Any:
            module_name, func_name = import_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[func_name])
            return getattr(module, func_name)(visits, **kwargs)

        return Workload(name, "visits", run, f"{import_path} on derived Brightkite visits")

    return {
        "activity_transition_matrix": make(
            "activity_transition_matrix",
            "skmob2.measures.visits.activity.activity_transition_matrix",
        ),
        "diversity": make("diversity", "skmob2.measures.visits.diversity.diversity"),
        "regularity": make(
            "regularity", "skmob2.measures.visits.regularity.regularity"
        ),
        "trajectory_entropy": make(
            "trajectory_entropy",
            "skmob2.measures.visits.entropy.trajectory_entropy",
        ),
        "trajectory_predictability": make(
            "trajectory_predictability",
            "skmob2.measures.visits.entropy.trajectory_predictability",
        ),
        "intermittance_and_degree_of_return": make(
            "intermittance_and_degree_of_return",
            "skmob2.measures.visits.mobility_profiling.intermittance_and_degree_of_return",
        ),
        "exploration_profiling": make(
            "exploration_profiling",
            "skmob2.measures.visits.mobility_profiling.exploration_profiling",
            {"random_seed": 0},
        ),
        "mean_area_volume": make(
            "mean_area_volume", "skmob2.measures.visits.mean_area_volume.mean_area_volume"
        ),
        "discover_daily_motifs_from_agents": make(
            "discover_daily_motifs_from_agents",
            "skmob2.measures.visits.motifs.discover_daily_motifs_from_agents",
        ),
    }


def _special_workloads() -> dict[str, Workload]:
    def od_matrix_run(trips: pd.DataFrame) -> Any:
        from skmob2.measures.flows.od import od_matrix

        return od_matrix(trips)

    def od_metrics_run(trips: pd.DataFrame) -> Any:
        from skmob2.measures.flows.od import od_matrix, od_metrics_per_area

        return od_metrics_per_area(od_matrix(trips))

    def stvd_run(distributions: tuple[pd.DataFrame, pd.DataFrame]) -> float:
        from skmob2.measures.stvd_emd import stvd_emd

        left, right = distributions
        return stvd_emd(left, right)

    return {
        "od_matrix": Workload("od_matrix", "od", od_matrix_run, "OD matrix from Brightkite transitions"),
        "od_metrics_per_area": Workload(
            "od_metrics_per_area",
            "od",
            od_metrics_run,
            "OD metrics from Brightkite transitions",
        ),
        "stvd_emd": Workload(
            "stvd_emd",
            "stvd",
            stvd_run,
            "STVD-EMD on derived Brightkite distributions",
        ),
    }


def workload_registry() -> dict[str, Workload]:
    workloads = {}
    workloads.update(_trajectory_workloads())
    workloads.update(_visit_workloads())
    workloads.update(_special_workloads())
    return dict(sorted(workloads.items()))


def build_dataset_for_workload(workload: Workload, rows: int, backend: str) -> Any:
    traj = load_brightkite(rows, backend=backend)
    if workload.dataset == "trajectory":
        return traj
    if workload.dataset == "visits":
        return trajectory_to_visits(traj)
    if workload.dataset == "od":
        return trajectory_to_od(traj)
    if workload.dataset == "stvd":
        return trajectory_to_stvd_distributions(traj)
    raise ValueError(f"Unknown workload dataset kind: {workload.dataset!r}")


def run_workload(name: str, *, rows: int = DEFAULT_ROWS, backend: str = "pandas") -> dict[str, Any]:
    workloads = workload_registry()
    if name not in workloads:
        available = ", ".join(workloads)
        raise SystemExit(f"Unknown workload {name!r}. Available workloads: {available}")

    try:
        import skmob2._core  # noqa: F401
    except ImportError as exc:
        raise SystemExit("skmob2._core is not importable. Run `maturin develop` first.") from exc

    workload = workloads[name]
    data = build_dataset_for_workload(workload, rows, backend)
    result = workload.func(data)
    _materialize(result)
    return {"workload": name, "rows": rows, "backend": backend, "dataset": workload.dataset}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=False, help="Workload name to run.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--backend", choices=["pandas", "polars"], default="pandas")
    parser.add_argument("--list", action="store_true", help="List workload names and exit.")
    args = parser.parse_args(argv)

    workloads = workload_registry()
    if args.list:
        for name, workload in workloads.items():
            print(f"{name}\t{workload.dataset}\t{workload.description}")
        return 0
    if not args.workload:
        parser.error("--workload is required unless --list is used")

    print(json.dumps(run_workload(args.workload, rows=args.rows, backend=args.backend)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
