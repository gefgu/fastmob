from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks import plot_benchmark_comparisons as plot


def test_metric_rows_requires_expected_metric_timings():
    original_result = {
        "metrics": {
            "ok_metric": {"status": "ok", "average_seconds": 2.0},
            "errored_metric": {"status": "error", "reason": "boom", "average_seconds": None},
        }
    }
    optimized_result = {
        "metrics": {
            "ok_metric": {"status": "ok", "average_seconds": 1.0},
            "errored_metric": {"status": "ok", "average_seconds": 1.0},
        }
    }

    with pytest.raises(ValueError) as excinfo:
        plot.metric_rows(
            original_result,
            optimized_result,
            "speedup",
            ["ok_metric", "errored_metric", "missing_metric"],
        )

    message = str(excinfo.value)
    assert "errored_metric: original skmob status=error (boom)" in message
    assert "missing_metric: missing from original skmob JSON" in message


def test_metric_rows_uses_catalog_order_and_sorts_model_metrics():
    rows = plot.metric_rows(
        {
            "metrics": {
                "slow": {"status": "ok", "average_seconds": 4.0},
                "fast": {"status": "ok", "average_seconds": 2.0},
            }
        },
        {
            "metrics": {
                "slow": {"status": "ok", "average_seconds": 1.0},
                "fast": {"status": "ok", "average_seconds": 0.25},
            }
        },
        "speedup",
        ["slow", "fast"],
    )

    assert [row["metric"] for row in rows] == ["fast", "slow"]
    assert [row["speedup"] for row in rows] == [8.0, 4.0]


def test_generate_plots_models_location_only(tmp_path: Path, monkeypatch):
    original_json = tmp_path / "skmob_models_speed.json"
    optimized_json = tmp_path / "skmob2_models_speed.json"
    original_json.write_text(json.dumps(_location_model_payload("skmob", [12, 50])), encoding="utf-8")
    optimized_json.write_text(json.dumps(_location_model_payload("skmob2", [12, 50])), encoding="utf-8")

    drawn = []

    def fake_draw_plot(rows, **kwargs):
        drawn.append((rows, kwargs))

    monkeypatch.setattr(plot, "draw_plot", fake_draw_plot)

    status = plot.generate_plots(
        _model_args(original_json, optimized_json, tmp_path)
    )

    assert status == 0
    # One chart per location count (12 and 50)
    assert len(drawn) == 2
    assert [item[1]["backend"] for item in drawn] == [None, None]
    assert drawn[0][1]["output_path"].name == "model_location_only_12_locations.png"
    assert drawn[1][1]["output_path"].name == "model_location_only_50_locations.png"
    # Each chart has gravity_flows and radiation_flows rows
    assert {row["metric"] for row in drawn[0][0]} == {"gravity flows", "radiation flows"}


def test_generate_plots_groups_model_matrix_by_agents(tmp_path: Path, monkeypatch):
    original_json = tmp_path / "skmob_models_speed.json"
    optimized_json = tmp_path / "skmob2_models_speed.json"
    original_json.write_text(json.dumps(_model_matrix_payload("skmob")), encoding="utf-8")
    optimized_json.write_text(json.dumps(_model_matrix_payload("skmob2")), encoding="utf-8")

    drawn = []

    def fake_draw_plot(rows, **kwargs):
        drawn.append((rows, kwargs))

    monkeypatch.setattr(plot, "draw_plot", fake_draw_plot)

    status = plot.generate_plots(
        _model_args(original_json, optimized_json, tmp_path)
    )

    assert status == 0
    # One chart per agent count (2 and 10)
    assert len(drawn) == 2
    size_labels = [item[1]["size_label"] for item in drawn]
    assert size_labels == ["Agent-based / 2 agents", "Agent-based / 10 agents"]
    assert drawn[0][1]["output_path"].name == "model_agent_based_2_agents.png"
    # Each chart covers all locations (12, 50) × all trajectory models
    assert {row["metric"] for row in drawn[0][0]} == {
        "epr / 12 locs",
        "density epr / 12 locs",
        "spatial epr / 12 locs",
        "geosim / 12 locs",
        "sts epr / 12 locs",
        "epr / 50 locs",
        "density epr / 50 locs",
        "spatial epr / 50 locs",
        "geosim / 50 locs",
        "sts epr / 50 locs",
    }


def test_comparison_title_does_not_repeat_backend():
    assert plot.comparison_title("spatial", "polars") == "skmob2 vs skmob"
    assert plot.comparison_title("visits", "pandas") == "skmob2 vs skmob"


def _model_args(original_json: Path, optimized_json: Path, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        original_json=original_json,
        optimized_json=optimized_json,
        suite="models",
        backend=None,
        output_dir=tmp_path / "plots",
        sizes=None,
        sort="speedup",
        large_json_skmob2=None,
        large_json_skmob=None,
        large_loc_json_skmob2=None,
        large_loc_json_skmob=None,
    )


def _location_model_payload(library: str, location_counts: list[int]) -> dict:
    return {
        "metadata": {"suite": "models", "library": library, "iterations": 1},
        "results": [
            {
                "benchmark_group": "location_models",
                "label": f"{n_locs} locations",
                "n_agents": None,
                "n_locations": n_locs,
                "metrics": {
                    "gravity_flows": {"status": "ok", "average_seconds": 2.0},
                    "radiation_flows": {"status": "ok", "average_seconds": 1.0},
                },
            }
            for n_locs in location_counts
        ],
    }


def _model_matrix_payload(library: str) -> dict:
    return {
        "metadata": {"suite": "models", "library": library, "iterations": 1},
        "results": [
            {
                "benchmark_group": "trajectory_models",
                "label": f"{n_agents} agents / {n_locations} locations",
                "n_agents": n_agents,
                "n_locations": n_locations,
                "metrics": {
                    metric: {"status": "ok", "average_seconds": float(n_agents + n_locations)}
                    for metric in plot.MODEL_TRAJECTORY_METRICS
                },
            }
            for n_locations in (12, 50)
            for n_agents in (2, 10)
        ],
    }
