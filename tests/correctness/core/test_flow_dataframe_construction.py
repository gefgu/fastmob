"""Correctness tests for FlowDataFrame construction from dict/list/ndarray/pyarrow
inputs -- these paths must not require pandas (fastmob/core/flow_dataframe.py
routes them through pyarrow directly)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from fastmob import FlowDataFrame


def test_construct_from_dict_with_canonical_column_names():
    fdf = FlowDataFrame({"origin": ["A", "A", "B"], "destination": ["A", "B", "A"], "flow": [100, 50, 30]})
    assert isinstance(fdf.df, pa.Table)
    assert fdf.get_flow("A", "B") == 50
    assert fdf.get_flow("B", "C") == 0


def test_construct_from_dict_with_renamed_columns():
    fdf = FlowDataFrame({"src": ["A"], "dst": ["B"], "trips": [10]}, origin="src", destination="dst", flow="trips")
    assert isinstance(fdf.df, pa.Table)
    assert sorted(fdf.df.column_names) == ["destination", "flow", "origin"]
    assert fdf.get_flow("A", "B") == 10


def test_construct_from_list_of_lists():
    data = [["A", "B", 5], ["B", "A", 3]]
    fdf = FlowDataFrame(data, columns=["origin", "destination", "flow"])
    assert isinstance(fdf.df, pa.Table)
    assert fdf.get_flow("A", "B") == 5


def test_construct_from_numpy_ndarray():
    data = np.array([["A", "B", 5], ["B", "A", 3]], dtype=object)
    fdf = FlowDataFrame(data, columns=["origin", "destination", "flow"])
    assert isinstance(fdf.df, pa.Table)
    assert fdf.get_flow("B", "A") == 3


def test_construct_with_no_arguments_is_empty():
    fdf = FlowDataFrame()
    assert isinstance(fdf.df, pa.Table)
    assert fdf.df.num_rows == 0


def test_construct_from_raw_pyarrow_table():
    table = pa.table({"origin": ["A"], "destination": ["B"], "flow": [7]})
    fdf = FlowDataFrame(table)
    assert fdf.get_flow("A", "B") == 7


def test_to_matrix_works_from_pyarrow_backed_construction():
    fdf = FlowDataFrame({"origin": ["A", "A", "B"], "destination": ["A", "B", "A"], "flow": [100, 50, 30]})
    matrix = fdf.to_matrix()
    assert matrix.shape == (2, 2)
    assert matrix.sum() == pytest.approx(180.0)


def test_pandas_and_polars_inputs_keep_their_own_backend():
    pd = pytest.importorskip("pandas")
    pandas_df = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [1]})
    fdf_pd = FlowDataFrame(pandas_df)
    assert isinstance(fdf_pd.df, pd.DataFrame)

    pl = pytest.importorskip("polars", reason="Polars not installed")
    polars_df = pl.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [1]})
    fdf_pl = FlowDataFrame(polars_df)
    assert isinstance(fdf_pl.df, pl.DataFrame)
