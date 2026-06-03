from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.models import speed_suite as suite


def test_model_registry_contains_expected_generation_models():
    assert [spec.name for spec in suite.MODEL_BENCHMARKS] == [
        "gravity_flows",
        "gravity_probabilities",
        "radiation_flows",
        "radiation_probabilities",
        "markov_diary",
        "epr",
        "density_epr",
        "spatial_epr",
        "geosim",
        "sts_epr",
    ]


def test_output_path_matches_library_and_profile(tmp_path: Path):
    assert suite.build_output_path(tmp_path, "skmob2") == tmp_path / "skmob2_models_speed.json"
    assert suite.build_output_path(tmp_path, "skmob", "memory") == tmp_path / "skmob_models_memory.json"


def test_parse_args_defaults_to_speed_profile():
    args = suite.parse_args(["--library", "skmob2"])

    assert args.profile == "speed"
    assert args.n_agents == suite.DEFAULT_AGENT_COUNTS
    assert args.n_locations == suite.DEFAULT_LOCATION_COUNTS
    assert args.sizes == suite.DEFAULT_LOCATION_COUNTS


def test_parse_args_accepts_sizes_as_location_alias():
    args = suite.parse_args(["--library", "skmob2", "--sizes", "7", "11"])

    assert args.n_locations == [7, 11]
    assert args.sizes == [7, 11]


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"library": "skmob2"}, "results": []}, tmp_path / "result.json")

    assert output_path == tmp_path / "result.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"library": "skmob2"},
        "results": [],
    }


def test_summarize_times_handles_values_and_empty_list():
    assert suite.summarize_times([0.3, 0.1, 0.2]) == {
        "average_seconds": pytest.approx(0.2),
        "minimum_seconds": 0.1,
    }
    assert suite.summarize_times([]) == {"average_seconds": None, "minimum_seconds": None}


def test_memory_skip_uses_memory_result_schema():
    result = suite.skipped_result("missing dependency", profile="memory")

    assert result["status"] == "skipped"
    assert result["reason"] == "missing dependency"
    assert result["peak_memory_mb"] == []
    assert result["current_memory_mb"] == []
    assert result["average_peak_memory_mb"] is None
    assert "times_seconds" not in result


def test_expand_tessellation_repeats_with_unique_tile_ids():
    pd = pytest.importorskip("pandas")
    base = pd.DataFrame(
        {
            "tile_id": ["a", "b"],
            "lat": [10.0, 11.0],
            "lng": [20.0, 21.0],
            "population": [1, 2],
            "tot_outflow": [3, 4],
        }
    )

    expanded = suite.expand_tessellation(base, 5)

    assert len(expanded) == 5
    assert expanded["tile_id"].is_unique
    assert expanded["tile_id"].tolist() == ["a_0", "b_0", "a_1", "b_1", "a_2"]


def test_run_memory_call_tracks_each_iteration_with_tracemalloc(monkeypatch):
    calls = {"func": 0, "start": 0, "get": 0, "stop": 0}

    def fake_func():
        calls["func"] += 1

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

    result = suite.run_memory_call(fake_func, iterations=2, sleep_seconds=0.0)

    assert calls == {"func": 3, "start": 2, "get": 2, "stop": 2}
    assert result["status"] == "ok"
    assert result["peak_memory_mb"] == [2.0, 4.0]
    assert result["average_peak_memory_mb"] == pytest.approx(3.0)
    assert "times_seconds" not in result


def test_benchmark_model_records_import_errors():
    spec = suite.BenchmarkSpec("missing", "missing", {})

    result = suite.benchmark_model(spec, "skmob2", object(), object(), iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["times_seconds"] == []
    assert result["average_seconds"] is None


def test_starting_locations_match_agent_count_and_wrap_locations():
    assert suite.starting_locations(5, 3) == [0, 1, 2, 0, 1]


def test_social_graph_matches_agent_count():
    assert suite.social_graph(4) == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


def test_build_call_passes_agent_and_location_dimensions(monkeypatch):
    pd = pytest.importorskip("pandas")
    calls = {}

    class FakeEPR:
        def generate(self, start, end, tessellation, **kwargs):
            calls["epr"] = {"locations": len(tessellation), "kwargs": kwargs}
            return None

    class FakeGeoSim:
        def generate(self, start, end, tessellation, **kwargs):
            calls["geosim"] = {"locations": len(tessellation), "kwargs": kwargs}
            return None

    fake_models = types.ModuleType("skmob2.models")
    fake_models.Gravity = object
    fake_models.Radiation = object
    fake_models.MarkovDiaryGenerator = object
    fake_models.EPR = FakeEPR
    fake_models.DensityEPR = FakeEPR
    fake_models.SpatialEPR = FakeEPR
    fake_models.GeoSim = FakeGeoSim
    fake_models.STS_epr = object

    monkeypatch.setitem(sys.modules, "skmob2.models", fake_models)
    tessellation = pd.DataFrame({"tile_id": list("abc"), "lat": [1, 2, 3], "lng": [4, 5, 6]})

    suite.build_call(
        suite.BenchmarkSpec("epr", "epr", {"model": "EPR", "relevance_column": "population"}),
        "skmob2",
        tessellation,
        object(),
        n_agents=5,
    )()
    suite.build_call(
        suite.BenchmarkSpec("geosim", "geosim", {}),
        "skmob2",
        tessellation,
        object(),
        n_agents=4,
    )()

    assert calls["epr"]["locations"] == 3
    assert calls["epr"]["kwargs"]["n_agents"] == 5
    assert calls["epr"]["kwargs"]["starting_locations"] == [0, 1, 2, 0, 1]
    assert calls["geosim"]["kwargs"]["n_agents"] == 4
    assert calls["geosim"]["kwargs"]["social_graph"] == suite.social_graph(4)


def test_skmob2_smoke_with_fake_models(monkeypatch, tmp_path: Path):
    pd = pytest.importorskip("pandas")
    calls = {"gravity": 0, "epr": 0}

    class FakeGravity:
        def generate(self, tessellation, **kwargs):
            calls["gravity"] += 1
            return {"locations": len(tessellation), "kwargs": kwargs}

    class FakeEPR:
        def generate(self, start, end, tessellation, **kwargs):
            calls["epr"] += 1
            return {"locations": len(tessellation), "kwargs": kwargs}

    fake_models = types.ModuleType("skmob2.models")
    fake_models.Gravity = FakeGravity
    fake_models.Radiation = FakeGravity
    fake_models.MarkovDiaryGenerator = object
    fake_models.EPR = FakeEPR
    fake_models.DensityEPR = FakeEPR
    fake_models.SpatialEPR = FakeEPR
    fake_models.GeoSim = object
    fake_models.STS_epr = object

    monkeypatch.setitem(sys.modules, "skmob2.models", fake_models)
    monkeypatch.setattr(
        suite,
        "LOCATION_MODEL_BENCHMARKS",
        (suite.BenchmarkSpec("gravity_flows", "gravity", {"out_format": "flows"}),),
    )
    monkeypatch.setattr(suite, "DIARY_MODEL_BENCHMARKS", ())
    monkeypatch.setattr(
        suite,
        "TRAJECTORY_MODEL_BENCHMARKS",
        (suite.BenchmarkSpec("epr", "epr", {"model": "EPR", "relevance_column": "population"}),),
    )

    reference_dir = tmp_path / "models"
    reference_dir.mkdir()
    pd.DataFrame(
        {
            "tile_id": ["a", "b"],
            "lat": [10.0, 11.0],
            "lng": [20.0, 21.0],
            "population": [1, 2],
            "tot_outflow": [3, 4],
        }
    ).to_parquet(reference_dir / "input.parquet")
    pd.DataFrame({"uid": [1], "datetime": pd.to_datetime(["2020-01-01"]), "cluster": [0]}).to_parquet(
        reference_dir / "diary_training.parquet"
    )

    args = suite.parse_args(
        [
            "--library",
            "skmob2",
            "--n-locations",
            "2",
            "--n-agents",
            "3",
            "--iterations",
            "1",
            "--sleep",
            "0",
            "--reference-dir",
            str(reference_dir),
            "--output-dir",
            str(tmp_path),
        ]
    )
    payload = suite.run_suite(args)

    assert payload["metadata"]["suite"] == "models"
    assert payload["metadata"]["library"] == "skmob2"
    assert payload["results"][0]["metrics"]["gravity_flows"]["status"] == "ok"
    assert payload["results"][1]["benchmark_group"] == "trajectory_models"
    assert payload["results"][1]["label"] == "3 agents / 2 locations"
    assert payload["results"][1]["n_agents"] == 3
    assert payload["results"][1]["n_locations"] == 2
    assert payload["results"][1]["metrics"]["epr"]["status"] == "ok"
    assert calls["gravity"] == 2
    assert calls["epr"] == 2
