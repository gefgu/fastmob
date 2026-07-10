"""Correctness tests for TrajDataFrame methods added in the docs-parity refactor."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_tdf():
    """Small synthetic TrajDataFrame for testing."""
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [2, 1, 1, 2],
            "lat": [0.0, 1.0, 2.0, 3.0],
            "lng": [0.0, 0.0, 0.0, 0.0],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
        }
    )
    return TrajDataFrame(df)


# ------------------------------------------------------------------
# sort_by_uid_and_datetime
# ------------------------------------------------------------------


def test_sort_by_uid_and_datetime_order():
    tdf = _make_tdf()
    sorted_tdf = tdf.sort_by_uid_and_datetime()

    native = sorted_tdf.to_native()
    uid_vals = native["uid"].tolist()
    assert uid_vals == sorted(uid_vals), "UIDs must be non-decreasing after sort"
    assert sorted_tdf.sorted is True


def test_sort_by_uid_and_datetime_returns_new_object():
    tdf = _make_tdf()
    sorted_tdf = tdf.sort_by_uid_and_datetime()
    assert sorted_tdf is not tdf


def test_sort_no_uid_col():
    """Sort on a trajectory without a uid column."""
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "lat": [2.0, 1.0, 0.0],
            "lng": [0.0, 0.0, 0.0],
            "datetime": pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-01"]),
        }
    )
    tdf = TrajDataFrame(df)
    sorted_tdf = tdf.sort_by_uid_and_datetime()
    native = sorted_tdf.to_native()
    assert native["datetime"].iloc[0] < native["datetime"].iloc[-1]


# ------------------------------------------------------------------
# settings_from
# ------------------------------------------------------------------


def test_settings_from_copies_parameters():
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [1],
            "lat": [0.0],
            "lng": [0.0],
            "datetime": pd.date_range("2020-01-01", periods=1),
        }
    )
    tdf1 = TrajDataFrame(df.copy(), parameters={"source": "gps"})
    tdf2 = TrajDataFrame(df.copy(), parameters={})
    tdf2.settings_from(tdf1)
    assert tdf2.parameters == {"source": "gps"}


def test_settings_from_copies_crs():
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [1],
            "lat": [0.0],
            "lng": [0.0],
            "datetime": pd.date_range("2020-01-01", periods=1),
        }
    )
    tdf1 = TrajDataFrame(df.copy(), crs={"init": "epsg:32633"})
    tdf2 = TrajDataFrame(df.copy())
    tdf2.settings_from(tdf1)
    assert tdf2.crs == {"init": "epsg:32633"}


# ------------------------------------------------------------------
# timezone_conversion
# ------------------------------------------------------------------


def test_timezone_conversion_shifts_datetime():
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [1, 1],
            "lat": [39.984, 39.985],
            "lng": [116.319, 116.320],
            "datetime": pd.to_datetime(["2008-10-23 05:53:05", "2008-10-23 05:53:06"]),
        }
    )
    tdf = TrajDataFrame(df)
    tdf.timezone_conversion("GMT", "Asia/Shanghai")  # UTC+8

    native = tdf.to_native()
    shifted = native["datetime"].iloc[0]
    # GMT 05:53 + 8h = 13:53
    assert shifted.hour == 13
    assert shifted.tzinfo is None  # tz-naive result


def test_timezone_conversion_identity():
    """Converting to the same timezone should leave values unchanged."""
    from fkmob import TrajDataFrame

    dt = pd.to_datetime("2020-06-15 12:00:00")
    df = pd.DataFrame(
        {
            "uid": [1],
            "lat": [0.0],
            "lng": [0.0],
            "datetime": [dt],
        }
    )
    tdf = TrajDataFrame(df)
    tdf.timezone_conversion("UTC", "UTC")
    native = tdf.to_native()
    assert native["datetime"].iloc[0] == dt


# ------------------------------------------------------------------
# to_geodataframe  (requires geopandas)
# ------------------------------------------------------------------


def test_to_geodataframe_shape():
    gpd = pytest.importorskip("geopandas", reason="geopandas required")
    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [1, 1],
            "lat": [48.8566, 48.8578],
            "lng": [2.3522, 2.3530],
            "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
        }
    )
    tdf = TrajDataFrame(df)
    gdf = tdf.to_geodataframe()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 2
    assert gdf.geometry.geom_type.eq("Point").all()


# ------------------------------------------------------------------
# plot methods (require folium + geojson + matplotlib)
# ------------------------------------------------------------------


def test_plot_trajectory_missing_dep(monkeypatch):
    """plot_trajectory raises ImportError when visualization deps are absent."""
    import sys

    folium_backup = sys.modules.pop("folium", None)
    monkeypatch.setitem(sys.modules, "folium", None)
    monkeypatch.setitem(sys.modules, "fkmob.utils.plot", None)

    from fkmob import TrajDataFrame

    df = pd.DataFrame(
        {
            "uid": [1],
            "lat": [0.0],
            "lng": [0.0],
            "datetime": pd.date_range("2020-01-01", periods=1),
        }
    )
    tdf = TrajDataFrame(df)
    with pytest.raises(ImportError, match="visualization"):
        tdf.plot_trajectory()

    if folium_backup is not None:
        sys.modules["folium"] = folium_backup
