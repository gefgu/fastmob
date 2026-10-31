"""Correctness tests for compare_flow_models."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob import FlowDataFrame
from fastmob.measures.evaluation import compare_flow_models


def _flows(origin, destination, flow):
    return FlowDataFrame(pd.DataFrame({"origin": origin, "destination": destination, "flow": flow}), tile_id="tile_id")


def test_identical_flows_score_perfectly():
    flows = _flows(["a", "a", "b"], ["b", "c", "c"], [10, 5, 7])
    row = compare_flow_models(flows, flows, pairs="union").to_pylist()[0]

    assert row["cpc"] == 1.0 and row["cpl"] == 1.0
    assert row["poisson_deviance"] == pytest.approx(0.0)
    assert row["pseudo_r_squared"] == pytest.approx(1.0)
    assert row["spearman"] == pytest.approx(1.0)


def test_a_worse_model_scores_lower_and_missing_pairs_count_as_zero():
    observed = _flows(["a", "a", "b"], ["b", "c", "c"], [10, 5, 7])
    close = _flows(["a", "a", "b"], ["b", "c", "c"], [9, 6, 7])
    far = _flows(["a", "b", "c"], ["b", "a", "a"], [3, 3, 16])

    table = compare_flow_models(observed, {"close": close, "far": far}, pairs="union").to_pylist()
    scores = {row["model"]: row for row in table}

    assert scores["close"]["cpc"] > scores["far"]["cpc"]
    assert scores["close"]["pseudo_r_squared"] > scores["far"]["pseudo_r_squared"]
    assert scores["far"]["n_pairs"] == scores["close"]["n_pairs"] == 5  # one common set: the union of all the pairs
    assert scores["close"]["observed_flow"] == scores["close"]["model_flow"] == 22.0


def test_self_loops_are_ignored_and_types_are_checked():
    observed = _flows(["a", "a"], ["a", "b"], [100, 4])
    model = _flows(["a", "a"], ["a", "b"], [1, 4])

    assert compare_flow_models(observed, model, pairs="union").to_pylist()[0]["cpc"] == 1.0
    with pytest.raises(TypeError):
        compare_flow_models(pd.DataFrame(), model)
    with pytest.raises(TypeError):
        compare_flow_models(observed, {"x": pd.DataFrame()})


def test_catalogue_pairs_make_a_models_score_independent_of_the_other_models():
    from fastmob.core import Locations

    places = pd.DataFrame(
        {
            "tile_id": list("abcd"),
            "lat": [45.0, 45.1, 45.2, 45.3],
            "lng": [7.0, 7.1, 7.0, 7.2],
            "relevance": [5, 6, 7, 8],
        }
    )
    locations = Locations.from_tessellation(places, tile_id_col="tile_id", lat_col="lat", lng_col="lng")

    def flows(origin, destination, flow):
        table = pd.DataFrame({"origin": origin, "destination": destination, "flow": flow})
        return FlowDataFrame(table, locations=locations, tile_id="tile_id")

    observed = flows(["a", "a", "b"], ["b", "c", "c"], [10, 5, 7])
    sparse = flows(["a", "b"], ["b", "c"], [9, 8])
    dense = flows(
        [o for o in "abcd" for d in "abcd" if o != d], [d for o in "abcd" for d in "abcd" if o != d], [1.0] * 12
    )

    alone = compare_flow_models(observed, {"sparse": sparse}).to_pylist()[0]
    together = {r["model"]: r for r in compare_flow_models(observed, {"sparse": sparse, "dense": dense}).to_pylist()}

    assert alone["n_pairs"] == 12
    assert together["sparse"] == alone
    union = compare_flow_models(observed, {"sparse": sparse, "dense": dense}, pairs="union").to_pylist()[0]
    assert union["n_pairs"] == 12  # the dense model lists every pair, so here both choices agree
    with pytest.raises(ValueError, match="pairs"):
        compare_flow_models(observed, sparse, pairs="all")
    with pytest.raises(ValueError, match="catalogue"):
        compare_flow_models(_flows(["a"], ["b"], [1]), sparse)
