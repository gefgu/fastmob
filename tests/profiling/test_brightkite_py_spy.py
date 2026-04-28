from __future__ import annotations

import json
from argparse import Namespace

import pandas as pd

from scripts.profile_brightkite_py_spy import build_profile_command, run_profiles
from tests.profiling.brightkite_workloads import (
    build_dataset_for_workload,
    execute_prepared_workload,
    prepare_workload,
    trajectory_to_od,
    trajectory_to_stvd_distributions,
    trajectory_to_visits,
    workload_registry,
)


def _tiny_brightkite() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["u1", "u1", "u1", "u2", "u2"],
            "datetime": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-02 00:00:00",
                    "2020-01-02 01:00:00",
                ]
            ),
            "lat": [0.0, 0.1, 0.2, 1.0, 1.1],
            "lng": [0.0, 0.1, 0.2, 1.0, 1.1],
            "location id": ["a", "b", "a", "c", "d"],
            "location_id": ["a", "b", "a", "c", "d"],
        }
    )


def test_workload_registry_contains_expected_public_workloads():
    registry = workload_registry()
    expected = {
        "radius_of_gyration",
        "jump_lengths",
        "stay_locations",
        "activity_transition_matrix",
        "od_matrix",
        "od_metrics_per_area",
        "stvd_emd",
    }
    assert expected.issubset(registry)


def test_derived_datasets_can_be_constructed_from_tiny_brightkite():
    traj = _tiny_brightkite()
    visits = trajectory_to_visits(traj)
    trips = trajectory_to_od(traj)
    dist_a, dist_b = trajectory_to_stvd_distributions(traj)

    assert {"user_id", "location_id", "start_timestamp", "end_timestamp"}.issubset(visits)
    assert {"origin_area", "destination_area"}.issubset(trips)
    assert {"centroid", "time_bin", "mean_volume"}.issubset(dist_a)
    assert {"centroid", "time_bin", "mean_volume"}.issubset(dist_b)


def test_registry_workload_dataset_kinds_are_buildable_without_brightkite(monkeypatch):
    import tests.profiling.brightkite_workloads as workloads

    monkeypatch.setattr(workloads, "load_brightkite", lambda rows, backend="pandas": _tiny_brightkite())

    for workload in workload_registry().values():
        dataset = build_dataset_for_workload(workload, rows=5, backend="pandas", implementation="skmob2")
        assert dataset is not None


def test_build_profile_command_uses_native_py_spy(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        rate=100,
    )
    assert profile.scope == "function"
    assert profile.implementation == "skmob2"
    assert profile.output_path == tmp_path / "skmob2" / "radius_of_gyration.svg"
    assert profile.speedscope_output_path == tmp_path / "skmob2" / "radius_of_gyration.speedscope.json"
    assert profile.command[:6] == [
        "py-spy",
        "record",
        "--native",
        "--format",
        "flamegraph",
        "--rate",
    ]
    assert "--pid" in profile.command
    assert profile.speedscope_command[:6] == [
        "py-spy",
        "record",
        "--native",
        "--format",
        "speedscope",
        "--rate",
    ]
    assert profile.child_command is not None
    assert "tests.profiling.brightkite_workloads" in profile.child_command
    assert "--prepared-child" in profile.child_command


def test_build_profile_command_preserves_full_scope_py_spy(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        rate=100,
        scope="full",
    )

    assert profile.scope == "full"
    assert profile.child_command is None
    assert "--" in profile.command
    assert "tests.profiling.brightkite_workloads" in profile.command
    assert profile.output_path.name == "radius_of_gyration.full.svg"


def test_runner_dry_run_writes_manifest(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        rate=100,
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scope="function",
        implementation="skmob2",
    )
    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["workload"] == "radius_of_gyration"
    assert manifest[0]["implementation"] == "skmob2"
    assert manifest[0]["scope"] == "function"
    assert manifest[0]["profiled_phase"] == "function"
    assert manifest[0]["status"] == "dry-run"
    assert manifest[0]["output_path"].endswith("skmob2/radius_of_gyration.svg")
    assert manifest[0]["speedscope_output_path"].endswith("skmob2/radius_of_gyration.speedscope.json")
    assert "--prepared-child" in manifest[0]["child_command"]


def test_runner_dry_run_both_uses_implementation_specific_paths(tmp_path, monkeypatch):
    import scripts.profile_brightkite_py_spy as runner

    monkeypatch.setattr(runner, "select_workloads", lambda requested, implementation="skmob2": ["radius_of_gyration"])
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        rate=100,
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scope="function",
        implementation="both",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["implementation"] for row in manifest] == ["skmob2", "skmob"]
    assert manifest[0]["output_path"].endswith("skmob2/radius_of_gyration.svg")
    assert manifest[1]["output_path"].endswith("skmob/radius_of_gyration.svg")


def test_preparation_resolves_callable_before_execution(monkeypatch):
    import sys
    import types

    import tests.profiling.brightkite_workloads as workloads

    events = []
    tiny = _tiny_brightkite()

    def fake_load(rows, backend="pandas"):
        events.append("load")
        return tiny

    def fake_resolve(import_path):
        events.append(f"resolve:{import_path}")

        def run(data, **kwargs):
            events.append("execute")
            return []

        return run

    monkeypatch.setattr(workloads, "load_brightkite", fake_load)
    monkeypatch.setattr(workloads, "_resolve_import", fake_resolve)
    monkeypatch.setattr(workloads, "_materialize", lambda result: events.append("materialize"))
    monkeypatch.setitem(sys.modules, "skmob2._core", types.ModuleType("skmob2._core"))

    prepared = prepare_workload("radius_of_gyration", rows=5, backend="pandas", implementation="skmob2")
    assert events == [
        "load",
        "resolve:skmob2.measures.spatial.radius_of_gyration.radius_of_gyration",
    ]
    execute_prepared_workload(prepared)
    assert events[-2:] == ["execute", "materialize"]
