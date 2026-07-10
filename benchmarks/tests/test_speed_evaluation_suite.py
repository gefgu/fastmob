from __future__ import annotations

import json
from pathlib import Path

from benchmarks.evaluation import speed_suite as suite


def test_evaluation_registry_starts_with_legacy_metrics():
    names = [spec.name for spec in suite.EVALUATION_METRICS]

    assert names == [
        "common_part_of_commuters",
        "common_part_of_links",
        "common_part_of_commuters_distance",
        "r_squared",
        "rmse",
        "nrmse",
        "information_gain",
        "pearson_correlation",
        "spearman_correlation",
        "kullback_leibler_divergence",
        "max_error",
    ]
    assert "mse" not in names


def test_default_sizes_include_four_million():
    assert suite.DEFAULT_SIZES == [1_000, 10_000, 100_000, 1_000_000, 4_000_000]


def test_output_path_matches_library_profile_and_synthetic_source(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "fkmob")
        == tmp_path / "fkmob_evaluation_speed_synthetic.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "memory")
        == tmp_path / "skmob_evaluation_memory_synthetic.json"
    )


def test_parse_args_selects_library():
    assert suite.parse_args(["--library", "fkmob"]).library == "fkmob"
    assert suite.parse_args(["--library", "skmob"]).library == "skmob"


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"suite": "evaluation"}, "results": []}, tmp_path / "result.json")
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"suite": "evaluation"},
        "results": [],
    }


def test_synthetic_inputs_are_deterministic_and_positive():
    observed1, predicted1 = suite.make_synthetic_pair(16, seed=123, input_kind="array_pair")
    observed2, predicted2 = suite.make_synthetic_pair(16, seed=123, input_kind="array_pair")
    distances1, distances2 = suite.make_synthetic_pair(16, seed=123, input_kind="distance_pair")

    assert (observed1 == observed2).all()
    assert (predicted1 == predicted2).all()
    assert (observed1 > 0).all()
    assert (predicted1 > 0).all()
    assert (distances1 > 0).all()
    assert (distances2 > 0).all()


def test_benchmark_metric_records_import_skip(monkeypatch):
    spec = suite.BenchmarkSpec("missing", "missing.module", "missing.module", "missing")

    def raise_skip(_spec, _library):
        raise suite.SkippedMetric("not installed")

    monkeypatch.setattr(suite, "import_metric", raise_skip)
    result = suite.benchmark_metric(
        spec,
        "skmob",
        lambda: suite.make_synthetic_pair(4, seed=1),
        iterations=1,
        sleep_seconds=0.0,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "not installed"
    assert result["average_seconds"] is None


def test_benchmark_metric_warms_up_then_records_iterations(monkeypatch):
    calls: list[tuple[int, int]] = []
    spec = suite.BenchmarkSpec("fake", "fake.module", "fake.module", "fake")

    def fake_import(_spec, _library):
        def fake_metric(values1, values2):
            calls.append((len(values1), len(values2)))
            return 1.0

        return fake_metric

    monkeypatch.setattr(suite, "import_metric", fake_import)
    result = suite.benchmark_metric(
        spec,
        "fkmob",
        lambda: suite.make_synthetic_pair(8, seed=1),
        iterations=2,
        sleep_seconds=0.0,
    )

    assert result["status"] == "ok"
    assert result["iterations_completed"] == 2
    assert len(result["times_seconds"]) == 2
    assert calls == [(8, 8), (8, 8), (8, 8)]


def test_fkmob_tiny_synthetic_smoke():
    args = suite.parse_args(
        [
            "--library",
            "fkmob",
            "--sizes",
            "8",
            "--iterations",
            "1",
            "--sleep",
            "0",
        ]
    )
    payload = suite.run_suite(args)

    assert payload["metadata"]["suite"] == "evaluation"
    assert payload["metadata"]["library"] == "fkmob"
    assert payload["results"][0]["input_source"] == "synthetic"
    assert payload["results"][0]["metrics"]["common_part_of_commuters"]["status"] == "ok"
