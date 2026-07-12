from __future__ import annotations

from pathlib import Path

from benchmarks.preprocessing import speed_suite as suite


def test_preprocessing_registry_contains_expected_functions():
    names = [spec.name for spec in suite.PREPROCESSING_METRICS]

    assert names == ["filter", "compress", "stay_locations", "cluster"]


def test_output_path_uses_preprocessing_suite_name(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "fastmob", "prebuilt_tdf", "pandas")
        == tmp_path / "fastmob_preprocessing_speed_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_preprocessing_speed_prebuilt_tdf.json"
    )


def test_selected_specs_filters_metrics():
    args = suite.parse_args(["--library", "fastmob", "--metrics", "cluster", "filter"])

    assert [spec.name for spec in suite.selected_specs(args)] == ["filter", "cluster"]


def test_sorted_fastmob_trajectory_benchmark_uses_presorted_keyword():
    spec = suite.BenchmarkSpec("compress", "unused", "unused", "compress", {})

    kwargs = suite.metric_kwargs_for_library(spec, "fastmob", lambda **_kwargs: None, input_order="sorted")

    assert kwargs == {"presorted": True}


def test_sorted_fastmob_filter_benchmark_does_not_add_presorted_keyword():
    spec = suite.BenchmarkSpec("filter", "unused", "unused", "filter", {}, input_kind="filter")

    kwargs = suite.metric_kwargs_for_library(spec, "fastmob", lambda **_kwargs: None, input_order="sorted")

    assert kwargs == {}


def test_merge_payload_replaces_selected_metric_only():
    existing = {
        "metadata": {"iterations": 5, "sizes": [1000, 4000000]},
        "results": [
            {
                "size": 1000,
                "label": "1k",
                "rows": 1000,
                "metrics": {"filter": {"status": "ok"}},
            },
            {
                "size": 4000000,
                "label": "4M",
                "rows": 4000000,
                "metrics": {
                    "filter": {"status": "ok"},
                    "cluster": {"status": "error"},
                },
            },
        ],
    }
    partial = {
        "metadata": {"iterations": 3, "sizes": [4000000]},
        "results": [
            {
                "size": 4000000,
                "label": "4M",
                "rows": 4000000,
                "metrics": {"cluster": {"status": "ok"}},
            },
        ],
    }

    merged = suite.merge_payload(existing, partial)

    assert merged["metadata"]["iterations"] == 3
    assert merged["results"][0]["metrics"] == {"filter": {"status": "ok"}}
    assert merged["results"][1]["metrics"]["filter"] == {"status": "ok"}
    assert merged["results"][1]["metrics"]["cluster"] == {"status": "ok"}
