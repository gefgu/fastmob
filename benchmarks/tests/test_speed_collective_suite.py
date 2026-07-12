from __future__ import annotations

from pathlib import Path

from benchmarks.collective import speed_suite as suite


def test_collective_registry_contains_only_collective_measures():
    names = [spec.name for spec in suite.COLLECTIVE_METRICS]

    assert "visits_per_location" in names
    assert "random_location_entropy" in names
    assert "random_entropy" not in names
    assert "real_entropy" not in names


def test_output_path_uses_collective_suite_name(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "fastmob", "prebuilt_tdf", "polars")
        == tmp_path / "fastmob_collective_speed_polars.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "workflow_tdf")
        == tmp_path / "skmob_collective_speed_workflow_tdf.json"
    )


def test_parse_args_defaults_to_both_fastmob_backends():
    args = suite.parse_args(["--library", "fastmob"])
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")
