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


def test_exponential_decay_is_reported_with_the_sign_it_was_generated_with():
    model = _fitted("exponential", [0.02])  # flows fall with distance: exp(-0.02 * km)
    row = {r["parameter"]: r for r in model.fit_summary().to_pylist()}["distance_decay"]

    assert row["ci_low"] <= 0.02 <= row["ci_high"]
    assert row["estimate"] == pytest.approx(0.02, abs=0.005)
    assert row["estimate"] == pytest.approx(model.deterrence_func_args[0])


def test_pairs_without_flow_are_kept_as_zeros():
    from fastmob.core import FlowDataFrame

    rng = np.random.default_rng(11)
    n = 30
    table = pd.DataFrame(
        {
            "tile_id": list(range(n)),
            "lat": 45.0 + rng.random(n) * 1.5,
            "lng": 7.0 + rng.random(n) * 2.0,
            "relevance": rng.integers(1_000, 50_000, n),
            "tot_outflow": rng.integers(30, 80, n),  # few trips per origin: many pairs have no flow
        }
    )
    locations = Locations.from_tessellation(table, tile_id_col="tile_id", lat_col="lat", lng_col="lng")
    truth = Gravity(gravity_type="singly constrained", deterrence_func_args=[-2.0])
    flows = truth.generate(locations, out_format="flows_sample", random_state=3)
    listed = flows.df[flows.df["flow"] > 0].reset_index(drop=True)  # what trajectory_to_od returns
    assert len(listed) < n * (n - 1) * 0.6
    observed = FlowDataFrame(listed, locations=locations, tile_id="tile_id")

    with_zeros = Gravity(gravity_type="singly constrained")
    with_zeros.fit(observed)
    without = Gravity(gravity_type="singly constrained")
    without.fit(observed, include_zeros=False)

    assert with_zeros.fit_statistics()["n_observations"] > without.fit_statistics()["n_observations"]
    # Leaving the zeros out flattens the distance decay.
    assert abs(with_zeros.deterrence_func_args[0] + 2.0) < abs(without.deterrence_func_args[0] + 2.0)
    assert without.deterrence_func_args[0] > with_zeros.deterrence_func_args[0]


def test_globally_constrained_fit_recovers_the_exponents():
    locations = _locations()
    truth = Gravity(
        gravity_type="globally constrained", deterrence_func_args=[-1.5], origin_exp=0.8, destination_exp=1.2
    )
    flows = truth.generate(locations, out_format="flows")
    model = Gravity(gravity_type="globally constrained")
    model.fit(flows)
    table = {r["parameter"]: r for r in model.fit_summary().to_pylist()}

    assert table["distance_coefficient"]["ci_low"] <= -1.5 <= table["distance_coefficient"]["ci_high"]
    assert table["origin_exp"]["estimate"] == pytest.approx(0.8, abs=0.1)
    assert table["destination_exp"]["estimate"] == pytest.approx(1.2, abs=0.1)


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


def test_concentrated_fit_equals_the_poisson_regression_with_origin_dummies():
    """The memory-light singly constrained fit must give what the textbook Poisson regression gives."""
    import statsmodels.api as sm
    from fastmob.models.gravity import _concentrated_poisson

    rng = np.random.default_rng(21)
    n_origins, n_destinations = 12, 15
    origin = np.repeat(np.arange(n_origins), n_destinations)
    log_size = rng.normal(8.0, 1.0, origin.size)
    distance = rng.uniform(1.0, 200.0, origin.size)
    design = np.column_stack([log_size, np.log(distance)])
    mean = np.exp(1.0 + 0.9 * log_size - 1.6 * np.log(distance) + rng.normal(0, 0.1, n_origins)[origin] - 2.0)
    response = rng.poisson(mean).astype(float)
    dummies = np.eye(n_origins)[origin]

    for robust in (True, False):
        ours = _concentrated_poisson(design, response, origin, robust)
        full = sm.GLM(response, np.hstack([design, dummies]), family=sm.families.Poisson()).fit(
            cov_type="HC0" if robust else "nonrobust"
        )
        np.testing.assert_allclose(ours.params, full.params[:2], rtol=1e-6)
        np.testing.assert_allclose(ours.bse, full.bse[:2], rtol=1e-5)
        assert ours.deviance == pytest.approx(full.deviance, rel=1e-8)
        assert ours.llf == pytest.approx(full.llf, rel=1e-8)
        assert ours.df_resid == full.df_resid
        assert ours.pearson_chi2 == pytest.approx(full.pearson_chi2, rel=1e-6)


def test_a_thousand_locations_fit_without_dense_dummies():
    import resource

    rng = np.random.default_rng(22)
    n = 600
    table = pd.DataFrame(
        {
            "tile_id": list(range(n)),
            "lat": 45.0 + rng.random(n) * 3.0,
            "lng": 7.0 + rng.random(n) * 4.0,
            "relevance": rng.integers(1_000, 50_000, n),
            "tot_outflow": rng.integers(50_000, 90_000, n),
        }
    )
    locations = Locations.from_tessellation(table, tile_id_col="tile_id", lat_col="lat", lng_col="lng")
    flows = Gravity(deterrence_func_args=[-2.0]).generate(locations, out_format="flows_sample", random_state=1)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = Gravity()
    model.fit(flows)

    grown_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grown_mb < 1500  # the dense one-hot design needed several gigabytes
    assert model.deterrence_func_args[0] == pytest.approx(-2.0, abs=0.05)
