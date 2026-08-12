"""Correctness tests for fastmob.data.datasets.* loaders (pyarrow-backed I/O)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from fastmob.core import FlowDataFrame, TrajDataFrame
from fastmob.data.datasets.foursquare_nyc.foursquare_nyc import foursquare_nyc
from fastmob.data.datasets.parking_san_francisco.parking_san_francisco import parking_san_francisco
from fastmob.data.datasets.taxi_san_francisco.taxi_san_francisco import taxi_san_francisco

_FOURSQUARE_LINE = "1\tvenue_b\tcat_b\tCoffee Shop\t40.7001\t-73.9001\t-240\tTue Apr 03 18:00:09 +0000 2012\n"


def test_foursquare_nyc_prepare_parses_tab_separated_checkins(tmp_path: Path):
    data_file = tmp_path / "TSMC2014_NYC.txt"
    data_file.write_text(_FOURSQUARE_LINE + _FOURSQUARE_LINE, encoding="ISO-8859-1")

    tdf = foursquare_nyc().prepare([str(data_file)])

    assert isinstance(tdf, TrajDataFrame)
    assert tdf.df.num_rows == 2
    assert tdf.df["lat"].to_pylist() == pytest.approx([40.7001, 40.7001])
    assert tdf.df["lng"].to_pylist() == pytest.approx([-73.9001, -73.9001])
    assert tdf.df["uid"].to_pylist() == [1, 1]
    dt = tdf.df["datetime"][0].as_py()
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2012, 4, 3, 18, 0, 9)


def test_flow_foursquare_nyc_prepare_builds_flowdataframe(tmp_path: Path, monkeypatch):
    """A GeoDataFrame `base_shape` skips tilers.py's Nominatim geocoding call
    (only a bare place-name string triggers it), so this stays offline."""
    gpd = pytest.importorskip("geopandas", reason="geopandas not installed (fastmob[geo])")
    shapely_geometry = pytest.importorskip("shapely.geometry", reason="shapely not installed (fastmob[geo])")
    import importlib

    module = importlib.import_module("fastmob.data.datasets.flow_foursquare_nyc.flow_foursquare_nyc")

    data_file = tmp_path / "TSMC2014_NYC.txt"
    lines = [
        "1\tvenue_b\tcat_b\tCoffee Shop\t40.7128\t-74.0060\t-240\tTue Apr 03 18:00:09 +0000 2012\n",
        "1\tvenue_a\tcat_a\tOffice\t40.7300\t-73.9950\t-240\tTue Apr 03 19:00:09 +0000 2012\n",
    ]
    data_file.write_text("".join(lines), encoding="ISO-8859-1")

    nyc_bbox = gpd.GeoDataFrame(geometry=[shapely_geometry.box(-74.05, 40.68, -73.90, 40.80)], crs="EPSG:4326")
    original_get = module.tilers.tiler.get
    monkeypatch.setattr(
        module.tilers.tiler,
        "get",
        lambda service_id, **kwargs: original_get(service_id, **{**kwargs, "base_shape": nyc_bbox}),
    )

    fdf = module.flow_foursquare_nyc().prepare([str(data_file)])
    assert isinstance(fdf, FlowDataFrame)


def test_parking_san_francisco_prepare_returns_pyarrow_table(tmp_path: Path):
    data_file = tmp_path / "meters.csv"
    data_file.write_text("POST_ID,LATITUDE,LONGITUDE\n1,37.77,-122.41\n2,37.78,-122.42\n", encoding="utf-8")

    result = parking_san_francisco().prepare([str(data_file)])

    assert isinstance(result, pa.Table)
    assert result.num_rows == 2
    assert result.column_names == ["POST_ID", "LATITUDE", "LONGITUDE"]


def test_taxi_san_francisco_prepare_applies_epoch_and_utc_offset(tmp_path: Path):
    data_file = tmp_path / "new_abboip.txt"
    # epoch 1213084687 == 2008-06-10 07:58:07 UTC; loader shifts by -7h.
    data_file.write_text("37.7 -122.4 0 1213084687\n", encoding="ISO-8859-1")

    tdf = taxi_san_francisco().prepare([str(data_file)])

    assert isinstance(tdf, TrajDataFrame)
    assert tdf.df.num_rows == 1
    assert tdf.df["uid"].to_pylist() == ["abboip"]
    dt = tdf.df["datetime"][0].as_py()
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (2008, 6, 10, 0, 58, 7)
