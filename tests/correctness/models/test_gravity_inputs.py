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
