from __future__ import annotations

import json
from argparse import Namespace

import pytest

from scripts.profile_brightkite_memray import build_profile_command, run_profiles, select_workloads
from tests.profiling.brightkite_workloads import run_workload


def test_memray_brightkite_workload(pytestconfig):
    workload = pytestconfig.getoption("--profile-workload")
    if not workload:
        pytest.skip("pass --profile-workload to run a Brightkite workload under memray")

    rows = int(pytestconfig.getoption("--profile-rows"))
    backend = pytestconfig.getoption("--profile-backend")
    implementation = pytestconfig.getoption("--profile-implementation")
    result = run_workload(workload, rows=rows, backend=backend, implementation=implementation)

    assert result["workload"] == workload
    assert result["rows"] == rows
    assert result["backend"] == backend
    assert result["implementation"] == implementation


def test_build_profile_command_uses_native_pytest_memray(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        pytest_bin="pytest",
        memray_bin="memray",
    )

    assert profile.scope == "function"
    assert profile.implementation == "skmob2"
    assert profile.bin_dir == tmp_path / "skmob2" / "bins"
    assert profile.bin_prefix == "radius_of_gyration"
    assert profile.flamegraph_path == tmp_path / "skmob2" / "flamegraphs" / "radius_of_gyration.html"
    assert profile.pytest_command[:4] == [
        "pytest",
        "tests/profiling/test_brightkite_memray.py::test_memray_brightkite_workload",
        "--profile-workload",
        "radius_of_gyration",
    ]
    assert "--memray" in profile.pytest_command
    assert "--native" in profile.pytest_command
    assert "--memray-bin-path" in profile.pytest_command
    assert "--memray-bin-prefix" in profile.pytest_command
    assert "--memray-bin-path" in profile.function_command
    assert profile.flamegraph_command == [
        "memray",
        "flamegraph",
        "-o",
        str(profile.flamegraph_path),
        str(profile.bin_path),
    ]


def test_build_profile_command_preserves_full_scope_pytest_memray(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        pytest_bin="pytest",
        memray_bin="memray",
        scope="full",
    )

    assert profile.scope == "full"
    assert profile.bin_prefix == "radius_of_gyration.full"
    assert "--memray" in profile.pytest_command
    assert profile.flamegraph_path.name == "radius_of_gyration.full.html"


def test_runner_dry_run_writes_manifest(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        pytest_bin="pytest",
        memray_bin="memray",
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
    assert manifest[0]["bin_path"].endswith("skmob2/bins/radius_of_gyration.bin")
    assert manifest[0]["flamegraph_path"].endswith("skmob2/flamegraphs/radius_of_gyration.html")
    assert "--memray-bin-path" in manifest[0]["function_command"]


def test_runner_dry_run_both_uses_implementation_specific_paths(tmp_path, monkeypatch):
    import scripts.profile_brightkite_memray as runner

    monkeypatch.setattr(runner, "select_workloads", lambda requested, implementation="skmob2": ["radius_of_gyration"])
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        pytest_bin="pytest",
        memray_bin="memray",
        scope="function",
        implementation="both",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["implementation"] for row in manifest] == ["skmob2", "skmob"]
    assert manifest[0]["bin_path"].endswith("skmob2/bins/radius_of_gyration.bin")
    assert manifest[1]["bin_path"].endswith("skmob/bins/radius_of_gyration.bin")


def test_select_workloads_rejects_unknown_memray_workload():
    with pytest.raises(SystemExit, match="Unknown workload"):
        select_workloads(["does_not_exist"])
