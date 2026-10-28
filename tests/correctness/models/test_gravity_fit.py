"""Gravity.fit exposes confidence intervals and goodness-of-fit statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastmob.core import Locations
from fastmob.models import Gravity

pytest.importorskip("statsmodels")


def _locations(n=14, seed=0):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {
            "tile_id": list(range(n)),
            "lat": 45.0 + rng.random(n) * 1.5,
            "lng": 7.0 + rng.random(n) * 2.0,
            "relevance": rng.integers(1_000, 50_000, n),
            "tot_outflow": rng.integers(20_000, 60_000, n),
        }
    )
    return Locations.from_tessellation(table, tile_id_col="tile_id", lat_col="lat", lng_col="lng")


def _fitted(deterrence, args, seed=1):
    locations = _locations()
    truth = Gravity(
        gravity_type="singly constrained",
        deterrence_func_type=deterrence,
        deterrence_func_args=args,
        destination_exp=1.0,
    )
    flows = truth.generate(locations, out_format="flows_sample", random_state=seed)
    model = Gravity(gravity_type="singly constrained", deterrence_func_type=deterrence)
    model.fit(flows)
    return model


def test_power_law_distance_exponent_is_recovered_with_an_interval():
    model = _fitted("power_law", [-2.0])
    table = {row["parameter"]: row for row in model.fit_summary().to_pylist()}

    row = table["distance_coefficient"]
    assert row["ci_low"] <= -2.0 <= row["ci_high"]
    assert row["std_error"] > 0
    assert table["destination_exp"]["ci_low"] <= 1.0 <= table["destination_exp"]["ci_high"]
    assert row["estimate"] == pytest.approx(model.deterrence_func_args[0])


def test_exponential_decay_is_reported_as_a_positive_rate():
    model = _fitted("exponential", [-0.5])
    row = {r["parameter"]: r for r in model.fit_summary().to_pylist()}["distance_decay"]

    assert row["estimate"] > 0
    assert row["ci_low"] < row["estimate"] < row["ci_high"]


def test_statistics_and_the_error_before_fit():
    model = _fitted("power_law", [-2.0])
    stats = model.fit_statistics()

    assert stats["n_observations"] > 50
    assert 0.5 < stats["pseudo_r_squared"] <= 1.0
    assert stats["robust_errors"] is True
    with pytest.raises(RuntimeError, match="fit"):
        Gravity().fit_summary()
    with pytest.raises(RuntimeError, match="fit"):
        Gravity().fit_statistics()


def test_fit_tolerates_an_origin_without_observed_flows():
    from fastmob.core import FlowDataFrame

    locations = _locations()
    truth = Gravity(gravity_type="singly constrained", deterrence_func_args=[-2.0])
    flows = truth.generate(locations, out_format="flows_sample", random_state=5)
    table = flows.df[flows.df["origin"] != 0].reset_index(drop=True)
    observed = FlowDataFrame(table, locations=locations, tile_id="tile_id")

    model = Gravity(gravity_type="singly constrained")
    model.fit(observed)

    assert model.fit_summary().num_rows == 2
