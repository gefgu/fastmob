from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import speed_comparison_suite as suite


def test_metric_registry_contains_expected_comparison_metrics():
    assert [spec.name for spec in suite.COMPARISON_METRICS] == [
        "wasserstein_distance",
        "histogram_jensen_shannon_divergence",
        "column_distribution_wasserstein_distance",
        "column_distribution_jensen_shannon_divergence",
        "visits_per_user_wasserstein_distance",
        "visits_per_user_jensen_shannon_divergence",
    ]


def test_input_kinds_are_correct():
    kinds = {spec.name: spec.input_kind for spec in suite.COMPARISON_METRICS}
    assert kinds["wasserstein_distance"] == "array_pair"
    assert kinds["histogram_jensen_shannon_divergence"] == "array_pair"
    assert kinds["column_distribution_wasserstein_distance"] == "df_pair_column"
    assert kinds["column_distribution_jensen_shannon_divergence"] == "df_pair_column"
    assert kinds["visits_per_user_wasserstein_distance"] == "df_pair"
    assert kinds["visits_per_user_jensen_shannon_divergence"] == "df_pair"


def test_histogram_spec_has_bin_size_kwarg():
    spec = next(s for s in suite.COMPARISON_METRICS if s.name == "histogram_jensen_shannon_divergence")
    assert spec.kwargs == {"bin_size": 1.0}


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


def test_output_path_matches_backend_and_profile(tmp_path: Path):
    assert suite.build_output_path(tmp_path, "pandas") == tmp_path / "skmob2_comparison_speed_pandas.json"
    assert suite.build_output_path(tmp_path, "polars") == tmp_path / "skmob2_comparison_speed_polars.json"
    assert suite.build_output_path(tmp_path, "pandas", "memory") == tmp_path / "skmob2_comparison_memory_pandas.json"


def test_parse_args_defaults():
    args = suite.parse_args([])
    assert args.backend == "pandas"
    assert args.profile == "speed"
    assert tuple(suite.concrete_backends(args)) == ("pandas",)


def test_parse_args_both_backend():
    args = suite.parse_args(["--backend", "both"])
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")


def test_split_in_half_produces_two_halves():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": list(range(10))})
    first, second = suite.split_in_half(df)
    assert len(first) == 5
    assert len(second) == 5


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"suite": "comparison"}, "results": []}, tmp_path / "result.json")
    assert output_path == tmp_path / "result.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"suite": "comparison"},
        "results": [],
    }


def test_benchmark_metric_records_import_skip(monkeypatch):
    spec = suite.BenchmarkSpec("missing", "missing.module", "missing", "df_pair")

    def raise_skip(_spec):
        raise suite.SkippedMetric("not installed")

    monkeypatch.setattr(suite, "import_metric", raise_skip)

    result = suite.benchmark_metric(spec, lambda: (object(), object()), iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["reason"] == "not installed"
    assert result["times_seconds"] == []
    assert result["average_seconds"] is None


def test_skmob2_smoke_with_tiny_dataframe_when_extension_is_available(monkeypatch, tmp_path: Path):
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first")
    pd = pytest.importorskip("pandas")

    tiny = pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 2, 3, 3],
            "check-in_time": pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                ]
            ),
            "latitude": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1],
            "longitude": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1],
            "location id": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )

    data_path = tmp_path / "brightkite.tsv.gz"
    data_path.write_bytes(b"placeholder")

    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "COMPARISON_METRICS",
        (
            suite.BenchmarkSpec(
                "wasserstein_distance",
                "skmob2.measures.evaluation.metrics",
                "wasserstein_distance",
                "array_pair",
            ),
        ),
    )

    args = suite.parse_args(
        [
            "--backend",
            "pandas",
            "--sizes",
            "8",
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

    assert payload["metadata"]["suite"] == "comparison"
    assert payload["metadata"]["backend"] == "pandas"
    assert payload["results"][0]["metrics"]["wasserstein_distance"]["status"] == "ok"
