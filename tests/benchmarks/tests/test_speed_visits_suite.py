from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from tests.benchmarks import speed_visits_suite as suite


def test_metric_registry_contains_expected_visits_and_collective_measures():
    assert [spec.name for spec in suite.VISITS_METRICS] == [
        "random_entropy",
        "uncorrelated_entropy",
        "real_entropy",
        "location_frequency",
        "individual_mobility_network",
        "recency_rank",
        "frequency_rank",
        "random_location_entropy",
        "uncorrelated_location_entropy",
        "mean_square_displacement",
        "visits_per_location",
        "homes_per_location",
        "visits_per_time_unit",
    ]

    kwargs = {spec.name: spec.kwargs for spec in suite.VISITS_METRICS}
    assert kwargs["frequency_rank"] == {}
    assert kwargs["location_frequency"] == {}
    assert kwargs["recency_rank"] == {}


def test_catalogs_include_expected_entries():
    skmob_catalog = suite.load_catalog(suite.SKMOB_CATALOG_PATH)
    movingpandas_catalog = suite.load_catalog(suite.MOVINGPANDAS_CATALOG_PATH)

    visits_names = {entry["name"] for entry in skmob_catalog["entries"] if entry["suite"] == "visits"}
    assert {"frequency_rank", "mean_square_displacement", "visits_per_time_unit"}.issubset(visits_names)
    assert any(entry["relationship"] == "no_close_analogue" for entry in movingpandas_catalog["entries"])


def test_summarize_times_handles_values_and_empty_list():
    assert suite.summarize_times([0.3, 0.1, 0.2]) == {
        "average_seconds": pytest.approx(0.2),
        "minimum_seconds": 0.1,
    }
    assert suite.summarize_times([]) == {"average_seconds": None, "minimum_seconds": None}


def test_summarize_memory_handles_values_and_empty_list():
    assert suite.summarize_memory([3.0, 1.0, 2.0]) == {
        "average_peak_memory_mb": pytest.approx(2.0),
        "minimum_peak_memory_mb": 1.0,
        "maximum_peak_memory_mb": 3.0,
    }
    assert suite.summarize_memory([]) == {
        "average_peak_memory_mb": None,
        "minimum_peak_memory_mb": None,
        "maximum_peak_memory_mb": None,
    }


def test_output_path_matches_library_backend_and_timing_mode(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "pandas")
        == tmp_path / "skmob2_visits_speed_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "polars")
        == tmp_path / "skmob2_visits_speed_polars.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_visits_speed_prebuilt_tdf.json"
    )
    assert suite.build_output_path(tmp_path, "movingpandas", "prebuilt_tdf") == tmp_path / "movingpandas_visits_speed.json"
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "pandas", "memory")
        == tmp_path / "skmob2_visits_memory_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf", profile="memory")
        == tmp_path / "skmob_visits_memory_prebuilt_tdf.json"
    )
    assert (
        suite.build_output_path(tmp_path, "movingpandas", "prebuilt_tdf", profile="memory")
        == tmp_path / "movingpandas_visits_memory.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "polars", input_order="sorted")
        == tmp_path / "skmob2_visits_speed_sorted_polars.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf", input_order="sorted")
        == tmp_path / "skmob_visits_speed_sorted_prebuilt_tdf.json"
    )


def test_parse_args_defaults_to_both_skmob2_backends():
    args = suite.parse_args(["--library", "skmob2"])
    assert args.backend == "both"
    assert args.profile == "speed"
    assert args.input_order == "raw"
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")
    assert tuple(suite.concrete_input_orders(args)) == ("raw",)


def test_parse_args_expands_both_input_orders():
    args = suite.parse_args(["--library", "skmob2", "--input-order", "both"])
    assert tuple(suite.concrete_input_orders(args)) == ("raw", "sorted")


def test_skmob_metric_kwargs_add_show_progress_when_supported():
    def fake_metric(_tdf, show_progress=True):
        return None

    spec = suite.BenchmarkSpec("location_frequency", "unused", "unused", "location_frequency", {})

    assert suite.metric_kwargs_for_library(spec, "skmob", fake_metric) == {"show_progress": False}
    assert suite.metric_kwargs_for_library(spec, "skmob2", fake_metric) == {}


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"library": "skmob2"}, "results": []}, tmp_path / "result.json")

    assert output_path == tmp_path / "result.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"library": "skmob2"},
        "results": [],
    }


def test_benchmark_metric_records_import_skip(monkeypatch):
    spec = suite.BenchmarkSpec("missing", "missing.module", "missing.module", "missing", {})

    def raise_skip(_spec, _library):
        raise suite.SkippedMetric("not installed")

    monkeypatch.setattr(suite, "import_metric", raise_skip)

    result = suite.benchmark_metric(spec, "skmob", lambda: object(), iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["reason"] == "not installed"
    assert result["times_seconds"] == []
    assert result["average_seconds"] is None
    assert result["minimum_seconds"] is None


def test_memory_skip_uses_memory_result_schema(monkeypatch):
    spec = suite.BenchmarkSpec("missing", "missing.module", "missing.module", "missing", {})

    def raise_skip(_spec, _library):
        raise suite.SkippedMetric("not installed")

    monkeypatch.setattr(suite, "import_metric", raise_skip)

    result = suite.benchmark_metric(
        spec,
        "skmob",
        lambda: object(),
        iterations=1,
        sleep_seconds=0.0,
        profile="memory",
    )

    assert result["status"] == "skipped"
    assert result["peak_memory_mb"] == []
    assert result["current_memory_mb"] == []
    assert result["average_peak_memory_mb"] is None
    assert "times_seconds" not in result


def test_run_memory_call_tracks_each_iteration_with_tracemalloc(monkeypatch):
    calls = {"func": 0, "start": 0, "get": 0, "stop": 0}

    def fake_func(input_value, scale=1):
        calls["func"] += 1
        return input_value * scale

    def fake_start():
        calls["start"] += 1

    def fake_get_traced_memory():
        calls["get"] += 1
        return calls["get"] * 1024 * 1024, calls["get"] * 2 * 1024 * 1024

    def fake_stop():
        calls["stop"] += 1

    monkeypatch.setattr(suite.tracemalloc, "start", fake_start)
    monkeypatch.setattr(suite.tracemalloc, "get_traced_memory", fake_get_traced_memory)
    monkeypatch.setattr(suite.tracemalloc, "stop", fake_stop)

    result = suite.run_memory_call(fake_func, lambda: 10, {"scale": 2}, iterations=2, sleep_seconds=0.0)

    assert calls == {"func": 3, "start": 2, "get": 2, "stop": 2}
    assert result["status"] == "ok"
    assert result["current_memory_mb"] == [1.0, 2.0]
    assert result["peak_memory_mb"] == [2.0, 4.0]
    assert result["average_peak_memory_mb"] == pytest.approx(3.0)
    assert "times_seconds" not in result


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

    assert result["metrics"]["location_frequency"]["status"] == "ok"
    assert calls["tdf"] == 1
    assert calls["metric"] == 3


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

    assert result["metrics"]["location_frequency"]["status"] == "ok"
    assert calls["tdf"] == 3
    assert calls["metric"] == 3


def test_skmob2_smoke_with_tiny_pandas_dataframe_when_extension_is_available(monkeypatch, tmp_path: Path):
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first")

    tiny = _tiny_pandas_df()
    data_path = tmp_path / "brightkite.tsv.gz"
    data_path.write_bytes(b"placeholder")

    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "VISITS_METRICS",
        (
            suite.BenchmarkSpec(
                "location_frequency",
                "skmob2.measures.visits.location_frequency",
                "skmob.measures.individual",
                "location_frequency",
                {},
            ),
        ),
    )

    args = suite.parse_args(
        [
            "--library",
            "skmob2",
            "--backend",
            "pandas",
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
    payload = suite.run_suite(args, backend="pandas")

    assert payload["metadata"]["library"] == "skmob2"
    assert payload["metadata"]["backend"] == "pandas"
    assert payload["results"][0]["metrics"]["location_frequency"]["status"] == "ok"


def test_skmob2_sorted_smoke_with_tiny_pandas_dataframe_when_extension_is_available(monkeypatch, tmp_path: Path):
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first")

    tiny = _tiny_pandas_df()
    data_path = tmp_path / "brightkite.tsv.gz"
    data_path.write_bytes(b"placeholder")

    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "VISITS_METRICS",
        (
            suite.BenchmarkSpec(
                "location_frequency",
                "skmob2.measures.visits.location_frequency",
                "skmob.measures.individual",
                "location_frequency",
                {},
            ),
        ),
    )

    args = suite.parse_args(
        [
            "--library",
            "skmob2",
            "--backend",
            "pandas",
            "--input-order",
            "sorted",
            "--input-cache-dir",
            str(tmp_path / "cache"),
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
    payload = suite.run_suite(args, backend="pandas")

    assert payload["metadata"]["input_order"] == "sorted"
    assert payload["metadata"]["input_cache_status"] == "created"
    assert payload["results"][0]["metrics"]["location_frequency"]["status"] == "ok"


def test_movingpandas_visits_suite_records_skips():
    result = suite.benchmark_movingpandas_size(10)

    assert result["metrics"]["frequency_rank"]["status"] == "skipped"
    assert "MovingPandas" in result["metrics"]["frequency_rank"]["reason"]


def test_isolated_metric_records_child_process_crash():
    def crash(_input):
        os._exit(17)

    result = suite.run_profiled_call_isolated(
        crash,
        lambda: None,
        {},
        profile="speed",
        iterations=1,
        sleep_seconds=0.0,
        retries=1,
    )

    assert result["status"] == "error"
    assert "17" in result["reason"]


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
    individual_module.location_frequency = fake_metric

    monkeypatch.setitem(sys.modules, "skmob", skmob_module)
    monkeypatch.setitem(sys.modules, "skmob.measures", measures_module)
    monkeypatch.setitem(sys.modules, "skmob.measures.individual", individual_module)
    monkeypatch.setattr(
        suite,
        "VISITS_METRICS",
        (
            suite.BenchmarkSpec(
                "location_frequency",
                "skmob2.measures.visits.location_frequency",
                "skmob.measures.individual",
                "location_frequency",
                {},
            ),
        ),
    )
    return calls


def _tiny_pandas_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 2],
            "check-in_time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-02 00:00:00",
                    "2020-01-02 01:00:00",
                    "2020-01-02 02:00:00",
                ]
            ),
            "latitude": [0.0, 1.0, 0.0, 10.0, 11.0, 10.0],
            "longitude": [0.0, 1.0, 0.0, 10.0, 11.0, 10.0],
            "location id": ["a", "b", "a", "c", "d", "c"],
        }
    )
