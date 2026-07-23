"""Correctness tests for fastmob.preprocessing.latlng_to_h3."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fastmob.preprocessing import latlng_to_h3

# Cross-checked against h3-py: h3.latlng_to_cell(37.769377, -122.388519, 9) == '89283082e73ffff'.
KNOWN_LAT = 37.769377
KNOWN_LNG = -122.388519
KNOWN_CELL = 0x89283082E73FFFF


def test_known_cell_matches_h3_py_pandas():
    df = pd.DataFrame({"lat": [KNOWN_LAT], "lng": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9)
    assert int(result["h3_cell"].iloc[0]) == KNOWN_CELL


def test_known_cell_matches_h3_py_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pl.DataFrame({"lat": [KNOWN_LAT], "lng": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9)
    assert int(result["h3_cell"][0]) == KNOWN_CELL


def test_nan_row_maps_to_sentinel_pandas():
    df = pd.DataFrame({"lat": [KNOWN_LAT, math.nan], "lng": [KNOWN_LNG, 20.0]})
    result = latlng_to_h3(df, resolution=9)
    assert int(result["h3_cell"].iloc[0]) == KNOWN_CELL
    assert int(result["h3_cell"].iloc[1]) == 2**64 - 1


def test_null_row_maps_to_null_polars():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = pl.DataFrame({"lat": [KNOWN_LAT, None], "lng": [KNOWN_LNG, 20.0]})
    result = latlng_to_h3(df, resolution=9)
    assert int(result["h3_cell"][0]) == KNOWN_CELL
    assert result["h3_cell"][1] is None


def test_single_row():
    df = pd.DataFrame({"lat": [KNOWN_LAT], "lng": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9)
    assert len(result) == 1


def test_empty_input():
    df = pd.DataFrame({"lat": pd.Series([], dtype="float64"), "lng": pd.Series([], dtype="float64")})
    result = latlng_to_h3(df, resolution=9)
    assert len(result) == 0
    assert "h3_cell" in result.columns


def test_explicit_lat_lng_columns():
    df = pd.DataFrame({"latitude": [KNOWN_LAT], "longitude": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9, lat_col="latitude", lng_col="longitude")
    assert int(result["h3_cell"].iloc[0]) == KNOWN_CELL


def test_missing_lat_lng_columns_raises():
    df = pd.DataFrame({"foo": [1.0], "bar": [2.0]})
    with pytest.raises(ValueError, match="latitude/longitude"):
        latlng_to_h3(df, resolution=9)


def test_returns_same_backend_type():
    df = pd.DataFrame({"lat": [KNOWN_LAT], "lng": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9)
    assert isinstance(result, type(df))


def test_custom_output_column_name():
    df = pd.DataFrame({"lat": [KNOWN_LAT], "lng": [KNOWN_LNG]})
    result = latlng_to_h3(df, resolution=9, output_col="cell_id")
    assert "cell_id" in result.columns
    assert "h3_cell" not in result.columns


def test_batch_matches_row_by_row_pandas():
    lats = [KNOWN_LAT, -33.865143, 51.507351]
    lngs = [KNOWN_LNG, 151.209900, -0.127758]
    batch = latlng_to_h3(pd.DataFrame({"lat": lats, "lng": lngs}), resolution=9)
    for i in range(len(lats)):
        single = latlng_to_h3(pd.DataFrame({"lat": [lats[i]], "lng": [lngs[i]]}), resolution=9)
        assert int(batch["h3_cell"].iloc[i]) == int(single["h3_cell"].iloc[0])
