from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.trajectory import speed_suite as suite


def test_parse_sizes_splits_on_comma():
    assert suite.parse_sizes("1000,10000,100000") == [1000, 10000, 100000]


def test_concrete_backends_expands_both():
    assert suite.concrete_backends("both") == ("pandas", "polars")
    assert suite.concrete_backends("pandas") == ("pandas",)


def test_parse_args_defaults():
    args = suite.parse_args([])
    assert args.backend == "both"
    assert args.sizes == suite.DEFAULT_SIZES
    assert args.iterations == 3


def test_make_synthetic_trajectory_pandas_has_expected_columns():
    pytest.importorskip("pandas")
    df = suite.make_synthetic_trajectory_pandas(200, num_users=5)
    assert set(df.columns) == {"uid", "datetime", "lat", "lng"}
    assert df["uid"].nunique() == 5


def test_run_size_smoke_with_tiny_pandas_dataframe():
    pytest.importorskip("pandas")
    result = suite.run_size(200, "pandas", iterations=1, sleep_seconds=0.0)

    assert result["backend"] == "pandas"
    for method in suite.INTERPOLATE_METHODS:
        assert result["interpolate"][method]["status"] == "ok"
    assert result["interpolate_at"]["status"] == "ok"
    for method in suite.DISTANCE_METHODS:
        assert result["trajectory_distance"][method]["status"] == "ok"


def test_run_suite_writes_expected_metadata(tmp_path: Path):
    pytest.importorskip("pandas")
    args = suite.parse_args(
        [
            "--backend",
            "pandas",
            "--sizes",
            "200",
            "--iterations",
            "1",
            "--sleep",
            "0",
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = suite.run_suite(args)

    assert payload["metadata"]["suite"] == "trajectory"
    assert payload["metadata"]["library"] == "fastmob"
    assert len(payload["results"]) == 1

    output_path = suite.write_json(payload, tmp_path / "fastmob_trajectory_speed.json")
    assert json.loads(output_path.read_text(encoding="utf-8"))["metadata"]["suite"] == "trajectory"
