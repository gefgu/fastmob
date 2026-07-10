from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.privacy import speed_suite as suite


def test_attack_registry_contains_expected_privacy_attacks():
    assert [spec.name for spec in suite.PRIVACY_ATTACKS] == [
        "location_kl2",
        "location_sequence_kl2",
        "location_time_kl2",
        "unique_location_kl2",
        "location_frequency_kl2",
        "location_probability_kl2",
        "location_proportion_kl2",
        "home_work",
    ]
    assert [spec.class_name for spec in suite.PRIVACY_ATTACKS] == [
        "LocationAttack",
        "LocationSequenceAttack",
        "LocationTimeAttack",
        "UniqueLocationAttack",
        "LocationFrequencyAttack",
        "LocationProbabilityAttack",
        "LocationProportionAttack",
        "HomeWorkAttack",
    ]


def test_output_path_matches_library_backend_and_timing_mode(tmp_path: Path):
    assert (
        suite.build_output_path(tmp_path, "fkmob", "prebuilt_tdf", "pandas")
        == tmp_path / "fkmob_privacy_speed_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "fkmob", "prebuilt_tdf", "polars")
        == tmp_path / "fkmob_privacy_speed_polars.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf")
        == tmp_path / "skmob_privacy_speed_prebuilt_tdf.json"
    )
    assert (
        suite.build_output_path(tmp_path, "fkmob", "prebuilt_tdf", "pandas", input_order="sorted")
        == tmp_path / "fkmob_privacy_speed_sorted_pandas.json"
    )
    assert (
        suite.build_output_path(tmp_path, "skmob", "prebuilt_tdf", input_order="sorted")
        == tmp_path / "skmob_privacy_speed_sorted_prebuilt_tdf.json"
    )


def test_parse_args_defaults_to_both_fkmob_backends():
    args = suite.parse_args(["--library", "fkmob"])
    assert args.backend == "both"
    assert args.input_order == "raw"
    assert tuple(suite.concrete_backends(args)) == ("pandas", "polars")
    assert tuple(suite.concrete_input_orders(args)) == ("raw",)
    assert args.repeat_dataset == 1000


def test_parse_args_expands_both_input_orders():
    args = suite.parse_args(["--library", "fkmob", "--input-order", "both"])
    assert tuple(suite.concrete_input_orders(args)) == ("raw", "sorted")


def test_write_json_serializes_payload(tmp_path: Path):
    output_path = suite.write_json({"metadata": {"library": "fkmob"}, "results": []}, tmp_path / "result.json")

    assert output_path == tmp_path / "result.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "metadata": {"library": "fkmob"},
        "results": [],
    }


def test_repeat_privacy_toy_offsets_user_ids():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "lat": [1.0, 2.0],
            "lng": [3.0, 4.0],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "uid": [1, 2],
        }
    )

    repeated = suite.repeat_privacy_toy_pandas(df, 3)

    assert len(repeated) == 6
    assert repeated["uid"].tolist() == [1, 2, 3, 4, 5, 6]


def test_benchmark_attack_records_import_skip(monkeypatch):
    spec = suite.BenchmarkSpec("missing", "MissingAttack", {}, {})

    def raise_skip(_spec, _library):
        raise suite.SkippedAttack("not installed")

    monkeypatch.setattr(suite, "import_attack", raise_skip)

    result = suite.benchmark_attack(spec, "skmob", lambda: object(), iterations=1, sleep_seconds=0.0)

    assert result["status"] == "skipped"
    assert result["reason"] == "not installed"
    assert result["times_seconds"] == []
    assert result["average_seconds"] is None
    assert result["minimum_seconds"] is None


def test_benchmark_attack_warms_up_then_records_iterations(monkeypatch):
    calls = {"func": 0}
    spec = suite.BenchmarkSpec("fake", "FakeAttack", {}, {"targets": [1]})

    def fake_assess(input_value, show_progress=False, targets=None):
        calls["func"] += 1
        assert show_progress is False
        assert targets == [1]
        return input_value

    monkeypatch.setattr(suite, "import_attack", lambda _spec, _library: fake_assess)

    result = suite.benchmark_attack(spec, "fkmob", lambda: object(), iterations=2, sleep_seconds=0.0)

    assert result["status"] == "ok"
    assert result["iterations_completed"] == 2
    assert calls["func"] == 3


def test_skmob_prebuilt_tdf_builds_once_outside_timing(monkeypatch):
    calls = _install_fake_skmob(monkeypatch)

    result = suite.benchmark_skmob(
        _tiny_privacy_df(),
        sys.modules["skmob"],
        timing_mode="prebuilt_tdf",
        iterations=2,
        sleep_seconds=0.0,
    )

    assert result["metrics"]["location_kl2"]["status"] == "ok"
    assert calls["tdf"] == 1
    assert calls["metric"] == 3


def test_skmob_workflow_tdf_builds_inside_each_timed_run(monkeypatch):
    calls = _install_fake_skmob(monkeypatch)

    result = suite.benchmark_skmob(
        _tiny_privacy_df(),
        sys.modules["skmob"],
        timing_mode="workflow_tdf",
        iterations=2,
        sleep_seconds=0.0,
    )

    assert result["metrics"]["location_kl2"]["status"] == "ok"
    assert calls["tdf"] == 3
    assert calls["metric"] == 3


def test_fkmob_smoke_with_tiny_pandas_dataframe(monkeypatch, tmp_path: Path):
    tiny = _tiny_privacy_df()
    data_path = tmp_path / "privacy_toy.csv"
    data_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(suite, "load_privacy_toy_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "PRIVACY_ATTACKS",
        (suite.BenchmarkSpec("location_kl2", "LocationAttack", {"knowledge_length": 2}, {}),),
    )

    args = suite.parse_args(
        [
            "--library",
            "fkmob",
            "--backend",
            "pandas",
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

    assert payload["metadata"]["library"] == "fkmob"
    assert payload["metadata"]["backend"] == "pandas"
    assert payload["results"][0]["metrics"]["location_kl2"]["status"] == "ok"


def test_fkmob_sorted_smoke_with_tiny_pandas_dataframe(monkeypatch, tmp_path: Path):
    tiny = _tiny_privacy_df()
    data_path = tmp_path / "privacy_toy.csv"
    data_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(suite, "load_privacy_toy_pandas", lambda _path: tiny)
    monkeypatch.setattr(
        suite,
        "PRIVACY_ATTACKS",
        (suite.BenchmarkSpec("location_kl2", "LocationAttack", {"knowledge_length": 2}, {}),),
    )

    args = suite.parse_args(
        [
            "--library",
            "fkmob",
            "--backend",
            "pandas",
            "--input-order",
            "sorted",
            "--input-cache-dir",
            str(tmp_path / "cache"),
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
    assert payload["results"][0]["metrics"]["location_kl2"]["status"] == "ok"


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

    class FakeAttack:
        def __init__(self, knowledge_length=1):
            self.knowledge_length = knowledge_length

        def assess_risk(self, tdf, show_progress=False):
            calls["metric"] += 1
            return {"rows": len(tdf.df), "show_progress": show_progress}

    skmob_module = types.ModuleType("skmob")
    skmob_module.__path__ = []
    skmob_module.TrajDataFrame = FakeTrajDataFrame
    privacy_module = types.ModuleType("skmob.privacy")
    privacy_module.__path__ = []
    attacks_module = types.ModuleType("skmob.privacy.attacks")
    attacks_module.LocationAttack = FakeAttack

    monkeypatch.setitem(sys.modules, "skmob", skmob_module)
    monkeypatch.setitem(sys.modules, "skmob.privacy", privacy_module)
    monkeypatch.setitem(sys.modules, "skmob.privacy.attacks", attacks_module)
    monkeypatch.setattr(
        suite,
        "PRIVACY_ATTACKS",
        (suite.BenchmarkSpec("location_kl2", "LocationAttack", {"knowledge_length": 2}, {}),),
    )
    return calls


def _tiny_privacy_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "lat": [0.0, 0.0, 1.0, 1.0],
            "lng": [0.0, 1.0, 1.0, 2.0],
            "datetime": pd.to_datetime(
                ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 00:00:00", "2020-01-01 01:00:00"]
            ),
            "uid": [1, 1, 2, 2],
        }
    )
