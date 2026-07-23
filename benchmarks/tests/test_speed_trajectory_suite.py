from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.trajectory import speed_suite as suite


def test_parse_int_list_splits_on_comma():
    assert suite.parse_int_list("1000,10000,100000") == [1000, 10000, 100000]


def test_concrete_backends_expands_both_for_fastmob():
    args = suite.parse_args(["--library", "fastmob", "--backend", "both"])
    assert suite.concrete_backends(args) == ("pandas", "polars")

    args = suite.parse_args(["--library", "fastmob", "--backend", "pandas"])
    assert suite.concrete_backends(args) == ("pandas",)


def test_concrete_backends_ignores_backend_flag_for_other_libraries():
    args = suite.parse_args(["--library", "ptrail"])
    assert suite.concrete_backends(args) == (None,)


def test_parse_args_defaults():
    args = suite.parse_args([])
    assert args.library == "fastmob"
    assert args.sizes == suite.DEFAULT_SIZES
    assert args.distance_sizes == suite.DEFAULT_DISTANCE_SIZES
    assert args.iterations == 3


def _tiny_brightkite_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 2],
            "check-in_time": pd.to_datetime(
                [
                    "2010-01-01 00:00:00",
                    "2010-01-01 02:00:00",
                    "2010-01-01 03:00:00",
                    "2010-01-01 00:00:00",
                    "2010-01-01 01:00:00",
                    "2010-01-01 05:00:00",
                ]
            ),
            "latitude": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "longitude": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "location id": [1, 2, 3, 4, 5, 6],
        }
    )


def test_make_distance_sequences_pandas_picks_two_most_frequent_users():
    df = _tiny_brightkite_df()
    seq_a, seq_b = suite.make_distance_sequences_pandas(df, size=10)
    assert len(seq_a) == 3
    assert len(seq_b) == 3
    assert seq_a["user"].nunique() == 1
    assert seq_b["user"].nunique() == 1


def test_run_fastmob_size_smoke_with_tiny_dataset(tmp_path: Path, monkeypatch):
    df = _tiny_brightkite_df()
    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: df)

    result = suite.run_fastmob_size(
        tmp_path / "fake.gz", size=10, backend="pandas", distance_sizes=[10], iterations=1, sleep_seconds=0.0
    )

    assert result["backend"] == "pandas"
    for method in suite.INTERPOLATE_METHODS:
        assert result["interpolate"][method]["status"] == "ok"
    for method in suite.INTERPOLATE_AT_METHODS:
        assert result["interpolate_at"][method]["status"] == "ok"
    for method in suite.DISTANCE_METHODS:
        assert result["trajectory_distance"][10][method]["status"] == "ok"


def test_run_suite_writes_expected_metadata(tmp_path: Path, monkeypatch):
    df = _tiny_brightkite_df()
    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: df)
    data_path = tmp_path / "fake.gz"
    data_path.write_bytes(b"placeholder")

    args = suite.parse_args(
        [
            "--library",
            "fastmob",
            "--backend",
            "pandas",
            "--sizes",
            "10",
            "--distance-sizes",
            "10",
            "--iterations",
            "1",
            "--sleep",
            "0",
            "--data-path",
            str(data_path),
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
