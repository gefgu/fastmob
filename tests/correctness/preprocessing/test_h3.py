"""Correctness tests for Fastmob's H3 conversions."""

from __future__ import annotations

import math

import pandas as pd
import pyarrow as pa
import pytest
from fastmob.preprocessing import h3_to_latlng, latlng_to_h3

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


def test_h3_to_latlng_matches_h3_py():
    h3 = pytest.importorskip("h3", reason="h3-py not installed")
    df = pd.DataFrame({"h3_cell": [KNOWN_CELL]})
    result = h3_to_latlng(df)
    expected_lat, expected_lng = h3.cell_to_latlng(h3.int_to_str(KNOWN_CELL))
    assert result["center_lat"].iloc[0] == pytest.approx(expected_lat)
    assert result["center_lng"].iloc[0] == pytest.approx(expected_lng)


def test_h3_to_latlng_preserves_backend_and_custom_columns():
    df = pd.DataFrame({"cell": [KNOWN_CELL]})
    result = h3_to_latlng(df, h3_col="cell", lat_col="lat0", lng_col="lng0")
    assert isinstance(result, type(df))
    assert result[["lat0", "lng0"]].notna().all().all()


def test_h3_to_latlng_null_and_invalid_cells_are_null():
    cells = pa.array([KNOWN_CELL, None, 2**64 - 1], type=pa.uint64())
    df = pd.DataFrame({"h3_cell": pd.array(cells.to_pylist(), dtype="UInt64")})
    result = h3_to_latlng(df)
    assert result["center_lat"].notna().tolist() == [True, False, False]
    assert result["center_lng"].notna().tolist() == [True, False, False]


def test_h3_to_latlng_polars_nulls():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    result = h3_to_latlng(pl.DataFrame({"h3_cell": [KNOWN_CELL, None]}))
    assert result["center_lat"][0] is not None
    assert result["center_lat"][1] is None


def test_h3_to_latlng_empty_input():
    df = pd.DataFrame({"h3_cell": pd.Series([], dtype="uint64")})
    result = h3_to_latlng(df)
    assert len(result) == 0
    assert {"center_lat", "center_lng"}.issubset(result.columns)


def test_h3_to_latlng_missing_column_raises():
    with pytest.raises(ValueError, match="H3 cell column"):
        h3_to_latlng(pd.DataFrame({"other": [KNOWN_CELL]}))
