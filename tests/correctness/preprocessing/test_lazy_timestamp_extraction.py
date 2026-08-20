"""Presorted paths must not materialize an Arrow timestamp copy."""

from __future__ import annotations

import importlib

import pandas as pd
import pytest


@pytest.fixture
def sorted_trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["u1", "u1"],
            "datetime": pd.date_range("2020-01-01", periods=2, freq="min"),
            "lat": [48.8566, 48.8567],
            "lng": [2.3522, 2.3523],
        }
    )


@pytest.mark.parametrize(
    ("module_name", "function_name", "kwargs"),
    [
            ("fastmob.preprocessing._stay_locations", "stay_locations", {}),
            ("fastmob.measures.individual.waiting_times", "waiting_times", {}),
            ("fastmob.preprocessing._compress", "compress", {}),
        ("fastmob.preprocessing._filter", "_filter_speed", {"is_sorted": True}),
        ("fastmob.preprocessing._segment", "segment", {"is_sorted": True, "method": "temporal", "mode": "hour"}),
        (
            "fastmob.preprocessing._simplify",
            "simplify",
            {"is_sorted": True, "method": "douglas_peucker", "epsilon_km": 0.01},
        ),
    ],
)
def test_automatic_paths_materialize_timestamps_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    sorted_trajectory: pd.DataFrame,
    module_name: str,
    function_name: str,
    kwargs: dict[str, object],
) -> None:
    module = importlib.import_module(module_name)

    def unexpected_conversion(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("presorted path materialized Arrow timestamps")

    getattr(module, function_name)(sorted_trajectory, **kwargs)
