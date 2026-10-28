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
    row = compare_flow_models(flows, flows).to_pylist()[0]

    assert row["cpc"] == 1.0 and row["cpl"] == 1.0
    assert row["poisson_deviance"] == pytest.approx(0.0)
    assert row["pseudo_r_squared"] == pytest.approx(1.0)
    assert row["spearman"] == pytest.approx(1.0)


def test_a_worse_model_scores_lower_and_missing_pairs_count_as_zero():
    observed = _flows(["a", "a", "b"], ["b", "c", "c"], [10, 5, 7])
    close = _flows(["a", "a", "b"], ["b", "c", "c"], [9, 6, 7])
    far = _flows(["a", "b", "c"], ["b", "a", "a"], [3, 3, 16])

    table = compare_flow_models(observed, {"close": close, "far": far}).to_pylist()
    scores = {row["model"]: row for row in table}

    assert scores["close"]["cpc"] > scores["far"]["cpc"]
    assert scores["close"]["pseudo_r_squared"] > scores["far"]["pseudo_r_squared"]
    assert scores["far"]["n_pairs"] == 5  # 3 observed and 3 model pairs, one shared
    assert scores["close"]["observed_flow"] == scores["close"]["model_flow"] == 22.0


def test_self_loops_are_ignored_and_types_are_checked():
    observed = _flows(["a", "a"], ["a", "b"], [100, 4])
    model = _flows(["a", "a"], ["a", "b"], [1, 4])

    assert compare_flow_models(observed, model).to_pylist()[0]["cpc"] == 1.0
    with pytest.raises(TypeError):
        compare_flow_models(pd.DataFrame(), model)
    with pytest.raises(TypeError):
        compare_flow_models(observed, {"x": pd.DataFrame()})
