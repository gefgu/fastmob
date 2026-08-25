"""Correctness tests for TrajDataFrame construction from dict/list/ndarray/pyarrow
inputs -- these paths must not require pandas (fastmob/core/trajectory_dataframe.py
routes them through pyarrow directly)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from fastmob import TrajDataFrame


def test_construct_from_dict_with_canonical_column_names():
    tdf = TrajDataFrame(
        {
            "uid": [1, 1, 2],
            "lat": [40.0, 40.1, 41.0],
            "lng": [-73.0, -73.1, -74.0],
            "datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00"],
        }
    )
    assert isinstance(tdf.df, pa.Table)
    assert tdf.df.num_rows == 3
    assert tdf.uid_col == "uid"
    assert tdf.lat_col == "lat"
    assert tdf.lng_col == "lng"


def test_construct_from_dict_with_renamed_columns():
    tdf = TrajDataFrame(
        {"user_id": [1], "latitude": [1.0], "longitude": [2.0], "datetime": ["2020-01-01 00:00:00"]},
        latitude="latitude",
        longitude="longitude",
        user_id="user_id",
    )
    assert isinstance(tdf.df, pa.Table)
    assert sorted(tdf.df.column_names) == ["datetime", "lat", "lng", "uid"]


def test_construct_from_list_of_lists_with_positional_columns():
    data = [
        [1, 39.984094, 116.319236, "2008-10-23 13:53:05"],
        [1, 39.984198, 116.319322, "2008-10-23 13:53:06"],
    ]
    tdf = TrajDataFrame(data, user_id=0, latitude=1, longitude=2, datetime=3)
    assert isinstance(tdf.df, pa.Table)
    assert tdf.df.num_rows == 2
    assert tdf.df["lat"].to_pylist() == pytest.approx([39.984094, 39.984198])


def test_construct_from_numpy_ndarray_with_positional_columns():
    data = np.array(
        [[1, 39.98, 116.3, "2008-10-23 13:53:05"], [1, 39.99, 116.4, "2008-10-23 13:53:06"]],
        dtype=object,
    )
    tdf = TrajDataFrame(data, user_id=0, latitude=1, longitude=2, datetime=3)
    assert isinstance(tdf.df, pa.Table)
    assert tdf.df.num_rows == 2


def test_construct_from_raw_pyarrow_table():
    table = pa.table(
        {
            "uid": [1, 1],
            "lat": [0.0, 1.0],
            "lng": [0.0, 0.0],
            "datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
        }
    )
    tdf = TrajDataFrame(table)
    assert tdf.df is table
    assert tdf.uid_col == "uid"


def test_construct_from_raw_pyarrow_table_with_renamed_columns():
    table = pa.table({"user_id": [1], "latitude": [1.0], "longitude": [2.0], "datetime": ["2020-01-01 00:00:00"]})
    tdf = TrajDataFrame(table, latitude="latitude", longitude="longitude", user_id="user_id")
    assert isinstance(tdf.df, pa.Table)
    assert sorted(tdf.df.column_names) == ["datetime", "lat", "lng", "uid"]


def test_pandas_and_polars_inputs_keep_their_own_backend():
    """Existing pandas/polars support must be unaffected by the pyarrow-first
    dict/list/ndarray construction paths."""
    pd = pytest.importorskip("pandas")
    pandas_df = pd.DataFrame({"uid": [1], "lat": [0.0], "lng": [0.0], "datetime": ["2020-01-01"]})
    tdf_pd = TrajDataFrame(pandas_df)
    assert isinstance(tdf_pd.df, pd.DataFrame)

    pl = pytest.importorskip("polars", reason="Polars not installed")
    polars_df = pl.DataFrame({"uid": [1], "lat": [0.0], "lng": [0.0], "datetime": ["2020-01-01"]})
    tdf_pl = TrajDataFrame(polars_df)
    assert isinstance(tdf_pl.df, pl.DataFrame)


def test_conversions_are_available_for_arrow_backed_trajectory():
    tdf = TrajDataFrame(
        {
            "uid": [1, 1],
            "lat": [40.0, 40.1],
            "lng": [-73.0, -73.1],
            "datetime": ["2020-01-01", "2020-01-02"],
        }
    )
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars", reason="Polars not installed")
    assert isinstance(tdf.to_pandas(), pd.DataFrame)
    assert isinstance(tdf.to_polars(), pl.DataFrame)
    assert tdf.to_pandas().columns.tolist() == tdf.df.column_names
    assert tdf.to_polars().height == tdf.df.num_rows
