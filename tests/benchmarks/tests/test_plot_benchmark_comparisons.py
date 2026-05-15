from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tests.benchmarks import plot_benchmark_comparisons as plot


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


def test_generate_plots_supports_models_without_backend(tmp_path: Path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "entries": [
                    {"name": "gravity_flows", "suite": "models"},
                    {"name": "radiation_flows", "suite": "models"},
                ]
            }
        ),
        encoding="utf-8",
    )
    original_json = tmp_path / "skmob_models_speed.json"
    optimized_json = tmp_path / "skmob2_models_speed.json"
    original_json.write_text(json.dumps(_payload("skmob", ["1k", "2k"])), encoding="utf-8")
    optimized_json.write_text(json.dumps(_payload("skmob2", ["1k"])), encoding="utf-8")

    drawn = []

    def fake_draw_plot(rows, **kwargs):
        drawn.append((rows, kwargs))

    monkeypatch.setattr(plot, "draw_plot", fake_draw_plot)

    status = plot.generate_plots(
        argparse.Namespace(
            original_json=original_json,
            optimized_json=optimized_json,
            suite="models",
            backend=None,
            catalog=catalog,
            output_dir=tmp_path / "plots",
            sizes=None,
            sort="speedup",
        )
    )

    assert status == 0
    assert len(drawn) == 1
    assert drawn[0][1]["backend"] is None
    assert drawn[0][1]["output_path"].name == "skmob2_vs_skmob_models_1k.png"


def test_generate_plots_groups_model_matrix_by_locations(tmp_path: Path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": []}), encoding="utf-8")
    original_json = tmp_path / "skmob_models_speed.json"
    optimized_json = tmp_path / "skmob2_models_speed.json"
    original_json.write_text(json.dumps(_model_matrix_payload("skmob")), encoding="utf-8")
    optimized_json.write_text(json.dumps(_model_matrix_payload("skmob2")), encoding="utf-8")

    drawn = []

    def fake_draw_plot(rows, **kwargs):
        drawn.append((rows, kwargs))

    monkeypatch.setattr(plot, "draw_plot", fake_draw_plot)

    status = plot.generate_plots(
        argparse.Namespace(
            original_json=original_json,
            optimized_json=optimized_json,
            suite="models",
            backend=None,
            catalog=catalog,
            output_dir=tmp_path / "plots",
            sizes=None,
            sort="original-time",
        )
    )

    assert status == 0
    assert len(drawn) == 2
    assert [item[1]["size_label"] for item in drawn] == ["12 locations", "50 locations"]
    assert drawn[0][1]["output_path"].name == "skmob2_vs_skmob_models_12_locations.png"
    assert {row["metric"] for row in drawn[0][0]} == {
        "epr / 2 agents",
        "density_epr / 2 agents",
        "spatial_epr / 2 agents",
        "geosim / 2 agents",
        "sts_epr / 2 agents",
        "epr / 10 agents",
        "density_epr / 10 agents",
        "spatial_epr / 10 agents",
        "geosim / 10 agents",
        "sts_epr / 10 agents",
    }


def test_comparison_title_does_not_repeat_backend():
    assert plot.comparison_title("spatial", "polars") == "skmob2 vs skmob"
    assert plot.comparison_title("visits", "pandas") == "skmob2 vs skmob"


def _payload(library: str, labels: list[str]) -> dict:
    return {
        "metadata": {"suite": "models", "library": library, "iterations": 1},
        "results": [
            {
                "label": label,
                "metrics": {
                    "gravity_flows": {"status": "ok", "average_seconds": 2.0},
                    "radiation_flows": {"status": "ok", "average_seconds": 1.0},
                },
            }
            for label in labels
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
