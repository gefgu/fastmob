from __future__ import annotations

import json
from argparse import Namespace

import pytest

from scripts.profile_brightkite_scalene import build_profile_command, run_profiles, select_workloads


def test_build_profile_command_uses_scalene_run_and_view(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
    )

    assert profile.implementation == "skmob2"
    assert profile.profile_dir == tmp_path / "skmob2"
    assert profile.json_path == tmp_path / "skmob2" / "jump_lengths.json"
    assert profile.html_path == tmp_path / "skmob2" / "jump_lengths.html"
    assert profile.run_command[:5] == [
        "scalene",
        "run",
        "--off",
        "-o",
        str(profile.json_path),
    ]
    assert "tests/profiling/brightkite_workloads.py" in profile.run_command
    assert "--scalene-function-profile" in profile.run_command
    assert profile.html_command == [
        "scalene",
        "view",
        "--standalone",
        "jump_lengths.json",
    ]


def test_build_profile_command_separates_skmob2_polars_outputs(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="polars",
        output_dir=tmp_path,
        scalene_bin="scalene",
        implementation="skmob2",
    )

    assert profile.profile_dir == tmp_path / "skmob2" / "polars"
    assert profile.json_path == tmp_path / "skmob2" / "polars" / "jump_lengths.json"
    assert profile.html_path == tmp_path / "skmob2" / "polars" / "jump_lengths.html"
    assert "--backend" in profile.run_command
    assert "polars" in profile.run_command


def test_runner_dry_run_writes_manifest(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["jump_lengths"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="skmob2",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["workload"] == "jump_lengths"
    assert manifest[0]["implementation"] == "skmob2"
    assert manifest[0]["status"] == "dry-run"
    assert manifest[0]["json_path"].endswith("skmob2/jump_lengths.json")
    assert manifest[0]["html_path"].endswith("skmob2/jump_lengths.html")
    assert "--scalene-function-profile" in manifest[0]["run_command"]
    assert manifest[0]["html_command"].endswith("jump_lengths.json")


def test_runner_dry_run_both_uses_implementation_specific_paths(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(
        runner,
        "select_workloads",
        lambda requested, implementation="skmob2", allow_unavailable=False: ["jump_lengths"],
    )
    args = Namespace(
        rows=10_000,
        workload=["jump_lengths"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="both",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["implementation"] for row in manifest] == ["skmob2", "skmob"]
    assert manifest[0]["json_path"].endswith("skmob2/jump_lengths.json")
    assert manifest[0]["html_path"].endswith("skmob2/jump_lengths.html")
    assert manifest[1]["json_path"].endswith("skmob/jump_lengths.json")
    assert manifest[1]["html_path"].endswith("skmob/jump_lengths.html")


def test_runner_defaults_to_jump_lengths_showcase(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    seen = []

    def fake_select_workloads(requested, implementation="skmob2", allow_unavailable=False):
        seen.append((requested, implementation))
        return ["jump_lengths"]

    monkeypatch.setattr(runner, "select_workloads", fake_select_workloads)
    args = Namespace(
        rows=10_000,
        workload=None,
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="both",
    )

    assert run_profiles(args) == 0
    assert seen == [(None, "skmob2"), (None, "skmob")]


def test_select_workloads_rejects_unknown_scalene_workload():
    with pytest.raises(SystemExit, match="Unknown workload"):
        select_workloads(["does_not_exist"])


def test_select_workloads_explains_missing_skmob(monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(runner, "workload_registry", lambda implementation="skmob2": {})

    with pytest.raises(SystemExit, match="scikit-mobility is not importable"):
        select_workloads(["jump_lengths"], implementation="skmob")
