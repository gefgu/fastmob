from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.individual import real_entropy_speed_suite
from benchmarks.individual import speed_suite as suite


def test_default_individual_registry_excludes_real_entropy():
    names = [spec.name for spec in suite.INDIVIDUAL_METRICS]

    assert "radius_of_gyration" in names
    assert "random_entropy" in names
    assert "real_entropy" not in names
    assert [spec.name for spec in suite.EXPENSIVE_INDIVIDUAL_METRICS] == ["real_entropy"]


def test_old_flat_benchmark_entrypoints_are_removed():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "speed_spatial_suite.py").exists()
    assert not (root / "speed_visits_suite.py").exists()
    assert not (root / "speed_comparison_suite.py").exists()


def test_cataloged_default_individual_entries_have_benchmark_cases():
    catalog = suite.load_catalog(suite.SKMOB_CATALOG_PATH)
    catalog_names = {
        entry["name"]
        for entry in catalog["entries"]
        if entry.get("suite") == "individual"
    }
    benchmark_names = {spec.name for spec in suite.INDIVIDUAL_METRICS}

    assert catalog_names.issubset(benchmark_names)
    assert "real_entropy" not in catalog_names


def test_output_path_uses_individual_suite_name(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "skmob2", "prebuilt_tdf", "pandas")
        == tmp_path / "skmob2_individual_speed_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_individual_speed_prebuilt_tdf.json"
    )


def test_parse_args_defaults_to_both_skmob2_backends():
    args = suite.parse_args(["--library", "skmob2"])
    assert args.backend == "both"
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")


def test_json_writer_serializes_payload(tmp_path: Path):
    path = suite.write_json({"metadata": {"suite": "individual"}, "results": []}, tmp_path / "out.json")
    assert json.loads(path.read_text(encoding="utf-8"))["metadata"]["suite"] == "individual"


def test_benchmark_metric_records_import_skip():
    spec = suite.BenchmarkSpec("missing", "missing.module", "missing.module", "missing", {})
    result = suite.benchmark_metric(spec, "skmob2", object, iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["times_seconds"] == []


def test_sorted_skmob2_trajectory_benchmark_uses_presorted_keyword():
    spec = suite.BenchmarkSpec("maximum_distance", "unused", "unused", "maximum_distance", {})

    kwargs = suite.metric_kwargs_for_library(spec, "skmob2", lambda **_kwargs: None, input_order="sorted")

    assert kwargs == {"presorted": True}


def test_sorted_skmob2_visit_benchmark_does_not_add_presorted_keyword():
    spec = suite.BenchmarkSpec("random_entropy", "unused", "unused", "random_entropy", {}, input_kind="visits")

    kwargs = suite.metric_kwargs_for_library(spec, "skmob2", lambda **_kwargs: None, input_order="sorted")

    assert kwargs == {}


def test_clean_sorted_trajectory_input_pandas_drops_nulls_and_nans():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "user": [1, 1, None, 2, 2],
            "check-in_time": pd.to_datetime(["2020-01-01", None, "2020-01-03", "2020-01-04", "2020-01-05"]),
            "latitude": [0.0, 1.0, 2.0, float("nan"), 4.0],
            "longitude": [0.0, 1.0, 2.0, 3.0, float("nan")],
            "location id": ["a", "b", "c", "d", "e"],
        }
    )

    cleaned = suite.clean_sorted_trajectory_input(df, backend="pandas")

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["user"] == 1
    assert cleaned.index.tolist() == [0]


def test_clean_sorted_trajectory_input_polars_drops_nulls_and_nans():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "user": [1, 1, None, 2, 2],
            "check-in_time": [None, "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
            "latitude": [0.0, 1.0, 2.0, float("nan"), 4.0],
            "longitude": [0.0, 1.0, 2.0, 3.0, float("nan")],
            "location id": ["a", "b", "c", "d", "e"],
        }
    )

    cleaned = suite.clean_sorted_trajectory_input(df, backend="polars")

    assert cleaned.height == 1
    assert cleaned["user"].to_list() == [1]
    assert cleaned["latitude"].to_list() == [1.0]


def test_real_entropy_opt_in_suite_uses_expensive_registry(monkeypatch, tmp_path: Path):
    pd = pytest.importorskip("pandas")
    tiny = pd.DataFrame(
        {
            "user": [1, 1, 1],
            "check-in_time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "latitude": [0.0, 0.1, 0.2],
            "longitude": [0.0, 0.1, 0.2],
            "location id": ["a", "b", "a"],
        }
    )
    data_path = tmp_path / "brightkite.tsv"
    data_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: tiny)

    args = real_entropy_speed_suite.parse_args(
        [
            "--library",
            "skmob2",
            "--backend",
            "pandas",
            "--sizes",
            "3",
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
    payload = real_entropy_speed_suite.run_suite(args, backend="pandas")

    assert payload["metadata"]["suite"] == "individual_real_entropy"
    assert list(payload["results"][0]["metrics"]) == ["real_entropy"]
