from __future__ import annotations

import json
from argparse import Namespace

import pytest
from scripts.profile_brightkite_scalene import (
    DEFAULT_REDUCED_CPU_SAMPLING_RATE,
    DEFAULT_TIMEOUT_SECONDS,
    build_profile_command,
    run_profiles,
    select_workloads,
)


def test_build_profile_command_uses_scalene_run_and_view(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
    )

    assert profile.implementation == "fastmob"
    assert profile.profile_dir == tmp_path / "fastmob"
    assert profile.json_path == tmp_path / "fastmob" / "jump_lengths.json"
    assert profile.html_path == tmp_path / "fastmob" / "jump_lengths.html"
    assert profile.run_command[:5] == [
        "scalene",
        "run",
        "--profile-only",
        "fastmob",
        "--off",
    ]
    assert profile.run_command[5:7] == [
        "-o",
        str(profile.json_path),
    ]
    assert "tests/profiling/brightkite_workloads.py" in profile.run_command
    assert "--scalene-function-profile" in profile.run_command
    assert "--jump-lengths-entrypoint" in profile.run_command
    assert "method" in profile.run_command
    assert profile.html_command == [
        "scalene",
        "view",
        "--standalone",
        "jump_lengths.json",
    ]


def test_build_profile_command_uses_custom_profile_scope(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
        profile_scope="fastmob,narwhals,pandas",
    )

    scope_index = profile.run_command.index("--profile-only") + 1
    assert profile.run_command[scope_index] == "fastmob,narwhals,pandas"


def test_build_profile_command_can_select_jump_lengths_function_entrypoint(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
        jump_lengths_entrypoint="function",
    )

    entrypoint_index = profile.run_command.index("--jump-lengths-entrypoint") + 1
    assert profile.run_command[entrypoint_index] == "function"


def test_build_profile_command_can_enable_cpu_only(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
        cpu_only=True,
    )

    assert "--cpu-only" in profile.run_command


def test_build_profile_command_can_set_cpu_sampling_rate(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="pandas",
        output_dir=tmp_path,
        scalene_bin="scalene",
        cpu_sampling_rate=0.001,
    )

    rate_index = profile.run_command.index("--cpu-sampling-rate") + 1
    assert profile.run_command[rate_index] == "0.001"


def test_build_profile_command_separates_fastmob_polars_outputs(tmp_path):
    profile = build_profile_command(
        "jump_lengths",
        rows=10_000,
        backend="polars",
        output_dir=tmp_path,
        scalene_bin="scalene",
        implementation="fastmob",
    )

    assert profile.profile_dir == tmp_path / "fastmob" / "polars"
    assert profile.json_path == tmp_path / "fastmob" / "polars" / "jump_lengths.json"
    assert profile.html_path == tmp_path / "fastmob" / "polars" / "jump_lengths.html"
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
        implementation="fastmob",
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        profile_scope=None,
        reduced=False,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["workload"] == "jump_lengths"
    assert manifest[0]["implementation"] == "fastmob"
    assert manifest[0]["status"] == "dry-run"
    assert manifest[0]["timeout_seconds"] == DEFAULT_TIMEOUT_SECONDS
    assert manifest[0]["profile_scope"] == "fastmob"
    assert manifest[0]["json_path"].endswith("fastmob/jump_lengths.json")
    assert manifest[0]["html_path"].endswith("fastmob/jump_lengths.html")
    assert "--scalene-function-profile" in manifest[0]["run_command"]
    assert "--jump-lengths-entrypoint method" in manifest[0]["run_command"]
    assert "--profile-only fastmob" in manifest[0]["run_command"]
    assert manifest[0]["jump_lengths_entrypoint"] == "method"
    assert manifest[0]["html_command"].endswith("jump_lengths.json")


def test_runner_dry_run_both_uses_implementation_specific_paths(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(
        runner,
        "select_workloads",
        lambda requested, implementation="fastmob", allow_unavailable=False: ["jump_lengths"],
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
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        profile_scope=None,
        reduced=False,
        jump_lengths_entrypoint="function",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["implementation"] for row in manifest] == ["fastmob", "skmob"]
    assert manifest[0]["json_path"].endswith("fastmob/jump_lengths.json")
    assert manifest[0]["html_path"].endswith("fastmob/jump_lengths.html")
    assert manifest[0]["jump_lengths_entrypoint"] == "function"
    assert manifest[1]["json_path"].endswith("skmob/jump_lengths.json")
    assert manifest[1]["html_path"].endswith("skmob/jump_lengths.html")
    assert manifest[1]["jump_lengths_entrypoint"] == "function"


def test_runner_defaults_to_jump_lengths_showcase(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    seen = []

    def fake_select_workloads(requested, implementation="fastmob", allow_unavailable=False):
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
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        profile_scope=None,
        reduced=False,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 0
    assert seen == [(None, "fastmob"), (None, "skmob")]


def test_runner_timeout_marks_workload_and_continues(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(runner, "_require_executable", lambda executable, package_name: None)
    monkeypatch.setattr(
        runner,
        "select_workloads",
        lambda requested, implementation="fastmob", allow_unavailable=False: ["slow", "fast"],
    )
    monkeypatch.setattr(runner, "_render_html", lambda profile, *, timeout_seconds: (0, False, ""))

    results = iter([(124, True, "scalene run timed out after 30s"), (0, False, "")])
    seen_commands = []

    def fake_run_command(command, *, timeout_seconds):
        seen_commands.append((command, timeout_seconds))
        return next(results)

    monkeypatch.setattr(runner, "_run_command", fake_run_command)

    args = Namespace(
        rows=10_000,
        workload=["slow", "fast"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=False,
        reduced=False,
        reduced_top=20,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=30.0,
        profile_scope=None,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 124
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [row["workload"] for row in manifest] == ["slow", "fast"]
    assert manifest[0]["status"] == "timed_out"
    assert manifest[0]["returncode"] == 124
    assert manifest[0]["error"] == "scalene run timed out after 30s"
    assert manifest[0]["reduced_json_path"] == ""
    assert manifest[1]["status"] == "ok"
    assert [timeout for _, timeout in seen_commands] == [30.0, 30.0]


def test_runner_reduced_dry_run_skips_html_and_uses_temporary_json(tmp_path, capsys):
    args = Namespace(
        rows=10_000,
        workload=["jump_lengths"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        reduced=True,
        reduced_top=20,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        profile_scope=None,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 0
    output = capsys.readouterr().out
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert ".jump_lengths.scalene.tmp.json" in output
    assert "--cpu-only" in output
    assert "--cpu-sampling-rate 0.001" in output
    assert "reduce_scalene_json.py" in output
    assert "scalene view" not in output
    assert manifest[0]["json_path"] == ""
    assert manifest[0]["html_path"] == ""
    assert manifest[0]["html_command"] == ""
    assert manifest[0]["cpu_only"] is True
    assert manifest[0]["cpu_sampling_rate"] == DEFAULT_REDUCED_CPU_SAMPLING_RATE


def test_runner_timeout_skips_reducer_when_json_missing(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(runner, "_require_executable", lambda executable, package_name: None)
    monkeypatch.setattr(
        runner, "select_workloads", lambda requested, implementation="fastmob", allow_unavailable=False: ["slow"]
    )
    monkeypatch.setattr(
        runner,
        "_run_command",
        lambda command, *, timeout_seconds: (124, True, "scalene run timed out after 30s"),
    )

    def fail_if_reducer_runs(*args, **kwargs):
        raise AssertionError("reducer should not run after a timed-out Scalene profile")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_reducer_runs)

    args = Namespace(
        rows=10_000,
        workload=["slow"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=False,
        reduced=True,
        reduced_top=20,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=30.0,
        profile_scope=None,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 124
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["status"] == "timed_out"
    assert manifest[0]["reduced_json_path"] == ""


def test_runner_reduced_success_deletes_full_json_and_skips_html(tmp_path, monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(runner, "_require_executable", lambda executable, package_name: None)
    monkeypatch.setattr(
        runner, "select_workloads", lambda requested, implementation="fastmob", allow_unavailable=False: ["fast"]
    )
    monkeypatch.setattr(runner, "_run_command", lambda command, *, timeout_seconds, cwd=None: (0, False, ""))

    render_calls = []
    monkeypatch.setattr(runner, "_render_html", lambda profile, *, timeout_seconds: render_calls.append(profile))

    def fake_reduce(args, profile):
        profile.json_path.write_text("{}")
        reduced_path = profile.profile_dir / f"{profile.workload}.reduced.json"
        reduced_path.write_text("{}")
        return 0, "", str(reduced_path)

    monkeypatch.setattr(runner, "_reduce_profile", fake_reduce)

    args = Namespace(
        rows=10_000,
        workload=["fast"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=False,
        reduced=True,
        reduced_top=20,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=30.0,
        profile_scope=None,
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["status"] == "ok"
    assert manifest[0]["json_path"] == ""
    assert manifest[0]["html_path"] == ""
    assert manifest[0]["reduced_json_path"].endswith("fastmob/fast.reduced.json")
    assert manifest[0]["cpu_only"] is True
    assert manifest[0]["cpu_sampling_rate"] == DEFAULT_REDUCED_CPU_SAMPLING_RATE
    assert not (tmp_path / "fastmob" / ".fast.scalene.tmp.json").exists()
    assert render_calls == []


def test_runner_reduced_profile_memory_keeps_memory_profiling(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["jump_lengths"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        reduced=True,
        reduced_top=20,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        profile_scope=None,
        profile_memory=True,
        cpu_sampling_rate=None,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 0
    output = (tmp_path / "manifest.json").read_text()
    manifest = json.loads(output)
    assert manifest[0]["cpu_only"] is False
    assert manifest[0]["cpu_sampling_rate"] is None
    assert "--cpu-only" not in manifest[0]["run_command"]


def test_runner_passes_custom_scope_to_manifest_and_command(tmp_path):
    args = Namespace(
        rows=10_000,
        workload=["jump_lengths"],
        output_dir=str(tmp_path),
        backend="pandas",
        dry_run=True,
        continue_on_error=True,
        scalene_bin="scalene",
        implementation="fastmob",
        timeout_seconds=12.5,
        profile_scope="fastmob,narwhals,pandas",
        reduced=False,
        cpu_sampling_rate=None,
        jump_lengths_entrypoint="method",
    )

    assert run_profiles(args) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["timeout_seconds"] == 12.5
    assert manifest[0]["profile_scope"] == "fastmob,narwhals,pandas"
    assert "--profile-only fastmob,narwhals,pandas" in manifest[0]["run_command"]


def test_select_workloads_rejects_unknown_scalene_workload():
    with pytest.raises(SystemExit, match="Unknown workload"):
        select_workloads(["does_not_exist"])


def test_select_workloads_explains_missing_skmob(monkeypatch):
    import scripts.profile_brightkite_scalene as runner

    monkeypatch.setattr(runner, "workload_registry", lambda implementation="fastmob": {})

    with pytest.raises(SystemExit, match="scikit-mobility is not importable"):
        select_workloads(["jump_lengths"], implementation="skmob")
