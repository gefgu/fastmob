from __future__ import annotations

import json
from argparse import Namespace

from scripts.profile_brightkite_samply import build_profile_command, run_profiles, select_workloads


def test_build_profile_command_function_scope_uses_pid_placeholder(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        rate=1000,
    )
    assert profile.scope == "function"
    assert profile.implementation == "fkmob"
    assert profile.output_path == tmp_path / "fkmob" / "radius_of_gyration.json.gz"
    assert profile.command[:7] == [
        "samply",
        "record",
        "--save-only",
        "--unstable-presymbolicate",
        "--no-open",
        "--rate",
        "1000",
    ]
    assert "-o" in profile.command
    assert str(profile.output_path) in profile.command
    assert "--pid" in profile.command
    assert "<prepared-child-pid>" in profile.command
    assert profile.child_command is not None
    assert "tests.profiling.brightkite_workloads" in profile.child_command
    assert "--prepared-child" in profile.child_command
    assert "--jump-lengths-entrypoint" in profile.child_command
    assert "method" in profile.child_command


def test_build_profile_command_can_select_jump_lengths_function_entrypoint(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        rate=1000,
        jump_lengths_entrypoint="function",
    )

    assert profile.child_command is not None
    entrypoint_index = profile.child_command.index("--jump-lengths-entrypoint") + 1
    assert profile.child_command[entrypoint_index] == "function"


def test_build_profile_command_full_scope_runs_child_directly(tmp_path):
    profile = build_profile_command(
        "radius_of_gyration",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        rate=1000,
        scope="full",
    )

    assert profile.scope == "full"
    assert profile.child_command is None
    assert "--pid" not in profile.command
    assert "--" in profile.command
    assert "tests.profiling.brightkite_workloads" in profile.command
    assert profile.output_path.name == "radius_of_gyration.full.json.gz"


def test_select_workloads_rejects_unknown():
    try:
        select_workloads(["does_not_exist"])
    except SystemExit as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("expected SystemExit for unknown workload")


def test_runner_dry_run_writes_manifest(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        rate=1000,
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scope="function",
        implementation="fkmob",
        samply_bin="samply",
        jump_lengths_entrypoint="method",
    )
    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["workload"] == "radius_of_gyration"
    assert manifest[0]["implementation"] == "fkmob"
    assert manifest[0]["scope"] == "function"
    assert manifest[0]["profiled_phase"] == "function"
    assert manifest[0]["jump_lengths_entrypoint"] == "method"
    assert manifest[0]["status"] == "dry-run"
    assert manifest[0]["output_path"].endswith("fkmob/radius_of_gyration.json.gz")
    assert "--prepared-child" in manifest[0]["child_command"]


def test_runner_dry_run_both_uses_implementation_specific_paths(tmp_path, monkeypatch):
    import scripts.profile_brightkite_samply as runner

    monkeypatch.setattr(runner, "select_workloads", lambda requested, implementation="fkmob": ["radius_of_gyration"])
    args = Namespace(
        rows=10_000,
        workload=["radius_of_gyration"],
        output_dir=str(tmp_path),
        rate=1000,
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scope="function",
        implementation="both",
        samply_bin="samply",
        jump_lengths_entrypoint="function",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["implementation"] for row in manifest] == ["fkmob", "skmob"]
    assert [row["jump_lengths_entrypoint"] for row in manifest] == ["function", "function"]
    assert manifest[0]["output_path"].endswith("fkmob/radius_of_gyration.json.gz")
    assert manifest[1]["output_path"].endswith("skmob/radius_of_gyration.json.gz")
