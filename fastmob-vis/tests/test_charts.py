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


def test_ecdf_thins_long_curves_but_keeps_the_ends():
    values = np.linspace(0.0, 1.0, 50_000)
    data = pa.table({"value": values})
    points = ecdf(data, value_col="value", cdf_cutoff=1.0).to_dict()["series"][0]["data"]

    assert len(points) == 2000
    assert points[0][0] == 0.0 and points[-1] == [1.0, 1.0]


def test_svg_display_has_a_single_representation():
    chart = bar(pa.table({"c": ["a"], "v": [1.0]}), category_col="c", value_col="v", display="svg")

    assert chart._repr_html_() is None
    assert set(chart._repr_mimebundle_()) == {"image/svg+xml"}
