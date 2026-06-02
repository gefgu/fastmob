from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.evaluation import speed_suite as suite


def test_evaluation_registry_contains_expected_metrics():
    names = [spec.name for spec in suite.EVALUATION_METRICS]

    assert "wasserstein_distance" in names
    assert "histogram_jensen_shannon_divergence" in names
    assert "visits_per_user_wasserstein_distance" in names


def test_output_path_matches_backend_and_profile(tmp_path: Path):
    assert suite.build_output_path(tmp_path, "pandas") == tmp_path / "skmob2_evaluation_speed_pandas.json"
    assert suite.build_output_path(tmp_path, "polars", "memory") == tmp_path / "skmob2_evaluation_memory_polars.json"


def test_parse_args_both_backend():
    args = suite.parse_args(["--backend", "both"])
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"suite": "evaluation"}, "results": []}, tmp_path / "result.json")
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"suite": "evaluation"},
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
    assert result["average_seconds"] is None


def test_skmob2_smoke_with_tiny_dataframe_when_extension_is_available(monkeypatch, tmp_path: Path):
    pytest.importorskip("skmob2._core", reason="Build the skmob2 extension first")
    pd = pytest.importorskip("pandas")

    tiny = pd.DataFrame({"user": [1, 1, 2, 2], "latitude": [0.0, 1.0, 2.0, 3.0]})
    data_path = tmp_path / "brightkite.tsv"
    data_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(suite, "load_brightkite_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "EVALUATION_METRICS",
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
            "4",
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

    assert payload["metadata"]["suite"] == "evaluation"
    assert payload["results"][0]["metrics"]["wasserstein_distance"]["status"] == "ok"
