from __future__ import annotations

import pandas as pd

from tests.profiling import brightkite_workloads as workloads


def _tiny_brightkite() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": [1, 1, 2],
            "datetime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05", "2020-01-01 00:10"]),
            "lat": [0.0, 0.1, 1.0],
            "lng": [0.0, 0.1, 1.0],
            "location_id": ["a", "b", "c"],
        }
    )


def test_prepare_jump_lengths_method_uses_fkmob_trajdataframe(monkeypatch):
    monkeypatch.setattr(workloads, "load_brightkite", lambda rows, *, backend: _tiny_brightkite())

    prepared = workloads.prepare_workload(
        "jump_lengths",
        rows=3,
        backend="pandas",
        implementation="fkmob",
        jump_lengths_entrypoint="method",
    )

    assert prepared.data.__class__.__name__ == "TrajDataFrame"
    assert prepared.data.sorted is True
    assert prepared.func is workloads._jump_lengths_method_entrypoint
    assert prepared.call_kwargs == {}
    assert prepared.jump_lengths_entrypoint == "method"


def test_prepare_jump_lengths_function_uses_fkmob_trajdataframe(monkeypatch):
    monkeypatch.setattr(workloads, "load_brightkite", lambda rows, *, backend: _tiny_brightkite())

    prepared = workloads.prepare_workload(
        "jump_lengths",
        rows=3,
        backend="pandas",
        implementation="fkmob",
        jump_lengths_entrypoint="function",
    )

    assert prepared.data.__class__.__name__ == "TrajDataFrame"
    assert prepared.data.sorted is False
    assert prepared.func is workloads._jump_lengths_function_entrypoint
    assert prepared.call_kwargs == {"merge": False}
    assert prepared.jump_lengths_entrypoint == "function"


def test_non_jump_lengths_trajectory_workload_keeps_plain_dataframe(monkeypatch):
    monkeypatch.setattr(workloads, "load_brightkite", lambda rows, *, backend: _tiny_brightkite())
    workload = workloads.workload_registry("fkmob")["radius_of_gyration"]

    data = workloads.build_dataset_for_workload(
        workload,
        rows=3,
        backend="pandas",
        implementation="fkmob",
        jump_lengths_entrypoint="method",
    )

    assert isinstance(data, pd.DataFrame)
    assert data.__class__.__name__ != "TrajDataFrame"
