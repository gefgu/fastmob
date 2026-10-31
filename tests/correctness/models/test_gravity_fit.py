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


@pytest.mark.parametrize("gravity_type", ["singly constrained", "globally constrained"])
def test_fit_custom_flow_columns_matches_canonical_columns(gravity_type):
    from fastmob.core import FlowDataFrame

    locations = _locations()
    flows = Gravity(gravity_type=gravity_type, deterrence_func_args=[-1.5]).generate(locations)
    custom = FlowDataFrame(
        flows.df.rename(columns={"origin": "Area_O", "destination": "Area_D", "flow": "Weight_Day"}),
        locations=locations,
    )
    expected, actual = Gravity(gravity_type=gravity_type), Gravity(gravity_type=gravity_type)
    expected.fit(flows)
    actual.fit(custom, origin_col="Area_O", destination_col="Area_D", flow_col="Weight_Day")
    assert actual.deterrence_func_args == pytest.approx(expected.deterrence_func_args)
    assert actual.destination_exp == pytest.approx(expected.destination_exp)
    assert actual.origin_exp == pytest.approx(expected.origin_exp)
    assert actual.fit_statistics()["n_observations"] == expected.fit_statistics()["n_observations"]


def test_fit_reports_missing_selected_columns():
    from fastmob.core import FlowDataFrame

    flows = FlowDataFrame(pd.DataFrame({"Area_O": [0], "Area_D": [1], "Weight_Day": [2]}), locations=_locations())
    with pytest.raises(KeyError, match="Gravity.fit missing flow columns.*origin.*Area_O"):
        Gravity().fit(flows)


@pytest.mark.parametrize("wrapped", [False, True])
def test_fit_explicit_locations_matches_attached_catalogue(wrapped):
    from fastmob.core import FlowDataFrame

    locations = _locations()
    flows = Gravity(deterrence_func_args=[-1.5]).generate(locations)
    observed = FlowDataFrame(flows.df) if wrapped else flows.df
    expected, actual = Gravity(), Gravity()
    expected.fit(flows)
    actual.fit(observed, locations=locations)
    assert actual.deterrence_func_args == pytest.approx(expected.deterrence_func_args)
    assert actual.destination_exp == pytest.approx(expected.destination_exp)


def test_fit_requires_locations_and_explicit_catalogue_takes_precedence():
    from fastmob.core import FlowDataFrame

    locations = _locations()
    flows = Gravity().generate(locations)
    with pytest.raises(AttributeError, match="pass locations="):
        Gravity().fit(FlowDataFrame(flows.df))
    with pytest.raises(TypeError, match="Locations catalogue"):
        Gravity().fit(flows.df, locations=object())
    flows.locations = _locations(seed=99)
    actual, expected = Gravity(), Gravity()
    actual.fit(flows, locations=locations)
    expected.fit(flows.df, locations=locations)
    assert actual.deterrence_func_args == pytest.approx(expected.deterrence_func_args)


def test_fit_accepts_duck_typed_flow_wrapper_and_dataframe_with_df_column():
    from types import SimpleNamespace

    locations = _locations()
    flows = Gravity().generate(locations)
    table = flows.df.assign(df=0)
    expected, wrapped, native = Gravity(), Gravity(), Gravity()
    expected.fit(flows)
    wrapped.fit(SimpleNamespace(df=table, locations=locations))
    native.fit(table, locations=locations)
    assert wrapped.deterrence_func_args == pytest.approx(expected.deterrence_func_args)
    assert native.deterrence_func_args == pytest.approx(expected.deterrence_func_args)


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


def _sized_locations(n=40, seed=3):
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {
            "tile_id": list(range(n)),
            "lat": 45.0 + rng.random(n) * 1.5,
            "lng": 7.0 + rng.random(n) * 2.0,
            "relevance": rng.integers(1_000, 50_000, n),
            "residents": rng.integers(1_000, 80_000, n),
            "jobs": rng.integers(500, 30_000, n),
            "tot_outflow": rng.integers(20_000, 60_000, n),
        }
    )
    return Locations.from_tessellation(table, tile_id_col="tile_id", lat_col="lat", lng_col="lng"), table


def _poisson_flows(locations, table, a, b, gamma, seed=0):
    from fastmob.core import FlowDataFrame

    rng = np.random.default_rng(seed)
    lat, lng = np.radians(table["lat"].to_numpy()), np.radians(table["lng"].to_numpy())
    h = (
        np.sin((lat[:, None] - lat[None]) / 2) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None] * np.sin((lng[:, None] - lng[None]) / 2) ** 2
    )
    distance = 2 * 6371.0088 * np.arcsin(np.sqrt(h))
    np.fill_diagonal(distance, 1.0)
    mean = 1e-3 * table["residents"].to_numpy()[:, None] ** a * table["jobs"].to_numpy()[None] ** b * distance**gamma
    np.fill_diagonal(mean, 0.0)
    origin, destination = np.nonzero(mean)
    flow = rng.poisson(mean[origin, destination])
    frame = pd.DataFrame({"origin": origin, "destination": destination, "flow": flow})
    return FlowDataFrame(frame[frame["flow"] > 0].reset_index(drop=True), locations=locations, tile_id="tile_id")


def test_separate_origin_and_destination_sizes_recover_both_exponents():
    locations, table = _sized_locations()
    observed = _poisson_flows(locations, table, a=0.8, b=1.2, gamma=-1.5)

    model = Gravity(gravity_type="globally constrained")
    model.fit(observed, origin_relevance_column="residents", destination_relevance_column="jobs")
    summary = model.fit_summary().to_pandas().set_index("parameter")

    for name, truth in (("origin_exp", 0.8), ("destination_exp", 1.2), ("distance_coefficient", -1.5)):
        assert summary.loc[name, "ci_low"] <= truth <= summary.loc[name, "ci_high"], name
    wrong = Gravity(gravity_type="globally constrained")
    wrong.fit(observed)  # one column for both sides: the default behaviour, a different model here
    assert wrong.origin_exp != pytest.approx(model.origin_exp, abs=0.05)


def test_one_relevance_column_is_the_default_for_both_sides():
    locations, table = _sized_locations()
    observed = _poisson_flows(locations, table, a=1.0, b=1.0, gamma=-1.5)

    default = Gravity(gravity_type="globally constrained")
    default.fit(observed, relevance_column="jobs")
    explicit = Gravity(gravity_type="globally constrained")
    explicit.fit(observed, origin_relevance_column="jobs", destination_relevance_column="jobs")

    assert default.fit_summary().equals(explicit.fit_summary())


def test_singly_constrained_rejects_an_origin_relevance_column_but_takes_a_destination_one():
    locations, table = _sized_locations()
    observed = _poisson_flows(locations, table, a=1.0, b=1.0, gamma=-1.5)

    model = Gravity(gravity_type="singly constrained")
    with pytest.raises(ValueError, match="origin_relevance_column"):
        model.fit(observed, origin_relevance_column="residents")
    model.fit(observed, destination_relevance_column="jobs")
    assert (
        model.fit_summary().to_pandas().set_index("parameter").loc["destination_exp", "ci_low"]
        < 1.0
        < (model.fit_summary().to_pandas().set_index("parameter").loc["destination_exp", "ci_high"])
    )


def test_cov_type_hc1_matches_statsmodels_and_only_changes_the_errors():
    import statsmodels.api as sm
    from fastmob.models.gravity import _concentrated_poisson

    rng = np.random.default_rng(21)
    n_origins, n_destinations = 12, 15
    origin = np.repeat(np.arange(n_origins), n_destinations)
    log_size = rng.normal(8.0, 1.0, origin.size)
    distance = rng.uniform(1.0, 200.0, origin.size)
    design = np.column_stack([log_size, np.log(distance)])
    mean = np.exp(1.0 + 0.9 * log_size - 1.6 * np.log(distance) - 2.0)
    response = rng.poisson(mean).astype(float)
    dummies = np.eye(n_origins)[origin]

    ours = _concentrated_poisson(design, response, origin, "HC1")
    full = sm.GLM(response, np.hstack([design, dummies]), family=sm.families.Poisson()).fit(cov_type="HC0")
    n, k = len(response), design.shape[1] + n_origins
    np.testing.assert_allclose(ours.bse, full.bse[:2] * np.sqrt(n / (n - k)), rtol=1e-5)  # HC1 = HC0 * n / (n - k)
    hc0 = _concentrated_poisson(design, response, origin, "HC0")
    np.testing.assert_allclose(ours.params, hc0.params)
    assert np.all(ours.bse > hc0.bse)

    locations, table = _sized_locations()
    observed = _poisson_flows(locations, table, a=0.8, b=1.2, gamma=-1.5)
    by_type = {}
    for cov_type in (None, "HC0", "HC1", "nonrobust"):
        for gravity_type in ("singly constrained", "globally constrained"):
            model = Gravity(gravity_type=gravity_type)
            model.fit(
                observed,
                destination_relevance_column="jobs",
                cov_type=cov_type,
                **({"origin_relevance_column": "residents"} if gravity_type.startswith("globally") else {}),
            )
            by_type[cov_type, gravity_type] = model
    for gravity_type in ("singly constrained", "globally constrained"):
        hc0, hc1 = by_type["HC0", gravity_type], by_type[("HC1", gravity_type)]
        assert by_type[None, gravity_type].fit_statistics()["cov_type"] == "HC0"
        assert by_type["nonrobust", gravity_type].fit_statistics()["robust_errors"] is False
        np.testing.assert_allclose(
            hc1.fit_summary().column("estimate").to_numpy(), hc0.fit_summary().column("estimate").to_numpy()
        )
        assert all(
            a > b
            for a, b in zip(
                hc1.fit_summary().column("std_error").to_pylist(), hc0.fit_summary().column("std_error").to_pylist()
            )
        )
    with pytest.raises(ValueError, match="cov_type"):
        Gravity().fit(observed, cov_type="HC3")
    legacy = Gravity()
    legacy.fit(observed, robust=False)
    assert legacy.fit_statistics()["cov_type"] == "nonrobust"
    overridden = Gravity()
    overridden.fit(observed, robust=False, cov_type="HC1")
    assert overridden.fit_statistics()["cov_type"] == "HC1"
