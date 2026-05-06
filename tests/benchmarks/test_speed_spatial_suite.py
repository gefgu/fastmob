from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tests.benchmarks import speed_spatial_suite as suite


def test_metric_registry_contains_expected_spatial_measures():
    assert [spec.name for spec in suite.SPATIAL_METRICS] == [
        "distance_straight_line",
        "home_location",
        "jump_lengths",
        "k_radius_of_gyration",
        "max_distance_from_home",
        "maximum_distance",
        "number_of_locations",
        "number_of_visits",
        "radius_of_gyration",
        "waiting_times",
    ]

    kwargs = {spec.name: spec.kwargs for spec in suite.SPATIAL_METRICS}
    assert kwargs["jump_lengths"] == {"merge": False}
    assert kwargs["waiting_times"] == {"merge": False}
    assert kwargs["k_radius_of_gyration"] == {"k": 2}


def test_summarize_times_handles_values_and_empty_list():
    assert suite.summarize_times([0.3, 0.1, 0.2]) == {
        "average_seconds": pytest.approx(0.2),
        "minimum_seconds": 0.1,
    }
    assert suite.summarize_times([]) == {"average_seconds": None, "minimum_seconds": None}


def test_output_path_matches_library_and_timing_mode(tmp_path: Path):
    assert suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf") == tmp_path / "skmob2_spatial_speed.json"
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_spatial_speed_prebuilt_tdf.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "workflow_tdf")
        == tmp_path / "skmob_spatial_speed_workflow_tdf.json"
    )


def test_skmob_metric_kwargs_add_show_progress_when_supported():
    def fake_metric(_tdf, show_progress=True, merge=True):
        return None

    spec = suite.MetricSpec("jump_lengths", "unused", "jump_lengths", {"merge": False})

    assert suite.metric_kwargs_for_library(spec, "skmob", fake_metric) == {
        "merge": False,
        "show_progress": False,
    }
    assert suite.metric_kwargs_for_library(spec, "skmob2", fake_metric) == {"merge": False}


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"library": "skmob2"}, "results": []}, tmp_path / "result.json")

    assert output_path == tmp_path / "result.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"library": "skmob2"},
        "results": [],
    }


def test_benchmark_metric_records_import_skip(monkeypatch):
    spec = suite.MetricSpec("missing", "missing.module", "missing", {})

    def raise_skip(_spec, _library):
        raise suite.SkippedMetric("not installed")

    monkeypatch.setattr(suite, "import_metric", raise_skip)

    result = suite.benchmark_metric(spec, "skmob", lambda: object(), iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["reason"] == "not installed"
    assert result["times_seconds"] == []
    assert result["average_seconds"] is None
    assert result["minimum_seconds"] is None


def test_skmob_prebuilt_tdf_builds_once_outside_timing(monkeypatch):
    calls = _install_fake_skmob(monkeypatch)

    result = suite.benchmark_skmob_size(
        _tiny_pandas_df(),
        sys.modules["skmob"],
        3,
        timing_mode="prebuilt_tdf",
        iterations=2,
        sleep_seconds=0.0,
    )

    assert result["metrics"]["radius_of_gyration"]["status"] == "ok"
    assert calls["tdf"] == 1
    assert calls["metric"] == 3  # warmup + 2 timed runs


def test_skmob_workflow_tdf_builds_inside_each_timed_run(monkeypatch):
    calls = _install_fake_skmob(monkeypatch)

    result = suite.benchmark_skmob_size(
        _tiny_pandas_df(),
        sys.modules["skmob"],
        3,
        timing_mode="workflow_tdf",
        iterations=2,
        sleep_seconds=0.0,
    )

    assert result["metrics"]["radius_of_gyration"]["status"] == "ok"
    assert calls["tdf"] == 3  # warmup + 2 timed runs
    assert calls["metric"] == 3


def test_skmob2_smoke_with_tiny_dataframe_when_extension_is_available(monkeypatch, tmp_path: Path):
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first")
    pl = pytest.importorskip("polars", reason="Install polars to run the skmob2 smoke test")

    tiny = pl.DataFrame(
        {
            "user": [1, 1, 2, 2],
            "check-in_time": ["2020-01-01T00:00:00", "2020-01-01T01:00:00"] * 2,
            "latitude": [0.0, 0.0, 10.0, 10.0],
            "longitude": [0.0, 1.0, 10.0, 11.0],
            "location id": ["a", "b", "c", "d"],
        }
    ).with_columns(pl.col("check-in_time").str.to_datetime(format="%Y-%m-%dT%H:%M:%S"))
    data_path = tmp_path / "brightkite.tsv.gz"
    data_path.write_bytes(b"placeholder")

    monkeypatch.setattr(suite, "load_brightkite_polars", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "SPATIAL_METRICS",
        (
            suite.MetricSpec(
                "radius_of_gyration",
                "skmob2.measures.spatial.radius_of_gyration",
                "radius_of_gyration",
                {},
            ),
        ),
    )

    args = suite.parse_args(
        [
            "--library",
            "skmob2",
            "--sizes",
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

    assert payload["metadata"]["library"] == "skmob2"
    assert payload["results"][0]["metrics"]["radius_of_gyration"]["status"] == "ok"


def _install_fake_skmob(monkeypatch):
    calls = {"tdf": 0, "metric": 0}

    class FakeTrajDataFrame:
        def __init__(self, df, latitude=None, longitude=None, datetime=None, user_id=None):
            calls["tdf"] += 1
            self.df = df
            self.latitude = latitude
            self.longitude = longitude
            self.datetime = datetime
            self.user_id = user_id

    def fake_metric(tdf):
        calls["metric"] += 1
        return {"rows": len(tdf.df)}

    skmob_module = types.ModuleType("skmob")
    skmob_module.__path__ = []
    skmob_module.TrajDataFrame = FakeTrajDataFrame
    measures_module = types.ModuleType("skmob.measures")
    measures_module.__path__ = []
    individual_module = types.ModuleType("skmob.measures.individual")
    individual_module.radius_of_gyration = fake_metric

    monkeypatch.setitem(sys.modules, "skmob", skmob_module)
    monkeypatch.setitem(sys.modules, "skmob.measures", measures_module)
    monkeypatch.setitem(sys.modules, "skmob.measures.individual", individual_module)
    monkeypatch.setattr(
        suite,
        "SPATIAL_METRICS",
        (
            suite.MetricSpec(
                "radius_of_gyration",
                "skmob2.measures.spatial.radius_of_gyration",
                "radius_of_gyration",
                {},
            ),
        ),
    )
    return calls


def _tiny_pandas_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "user": [1, 1, 2],
            "check-in_time": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 00:00:00"]
            ),
            "latitude": [0.0, 0.0, 1.0],
            "longitude": [0.0, 1.0, 1.0],
            "location id": ["a", "b", "c"],
        }
    )
