from dataclasses import FrozenInstanceError

import numpy as np
import pyarrow as pa
import pytest
from fastmob_vis import Chart, bar, boxplot, ecdf, heatmap, scatter


def test_ecdf_accepts_list_valued_measure_columns():
    chart = ecdf(pa.table({"jump_lengths": [[2.0, 1.0], [3.0]]}), value_col="jump_lengths")

    assert isinstance(chart, Chart)
    assert chart.kind == "ecdf"
    assert chart.to_dict()["series"][0]["data"] == [[1.0, 1 / 3], [2.0, 2 / 3], [3.0, 0.98]]


def test_ecdf_comparison_uses_first_second_defaults():
    chart = ecdf(pa.table({"value": [1.0, 2.0]}), value_col="value", second=pa.table({"value": [3.0, 4.0]}))

    assert [series["name"] for series in chart.to_dict()["series"]] == ["first", "second"]


@pytest.mark.parametrize("factory, kwargs", [
    (bar, {"category_col": "category", "value_col": "value"}),
    (scatter, {"x_col": "x", "y_col": "y"}),
    (heatmap, {"x_col": "x", "y_col": "y", "value_col": "value"}),
    (boxplot, {"category_col": "category", "value_col": "value"}),
])
def test_generic_charts_return_immutable_specs(factory, kwargs):
    data = pa.table({"category": ["a", "b", "a"], "x": [1, 2, 1], "y": [2, 3, 2], "value": [1.0, 2.0, 3.0]})
    chart = factory(data, **kwargs)

    assert isinstance(chart, Chart)
    with pytest.raises(FrozenInstanceError):
        chart.kind = "other"


def test_ecdf_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        ecdf(pa.table({"value": [1.0, np.nan]}), value_col="value")
