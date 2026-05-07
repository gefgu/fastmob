"""Correctness tests for skmob2.preprocessing.compress."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skmob2.preprocessing import compress


def test_compress_dense_cluster_becomes_one_point(compress_tdf):
    """Dense cluster of 5 points within 0.2 km → 1 output point."""
    dense = compress_tdf[compress_tdf["uid"] == "user_dense"].copy()
    result = compress(dense, spatial_radius_km=0.2)
    assert len(result) == 2  # 2 clusters of 5 → 2 output rows


def test_compress_sparse_trajectory_unchanged(compress_tdf):
    """Points already > 0.2 km apart → no compression, same count."""
    sparse = compress_tdf[compress_tdf["uid"] == "user_sparse"].copy()
    result = compress(sparse, spatial_radius_km=0.2)
    assert len(result) == 5


def test_compress_output_is_median_coordinates(compress_tdf):
    """Output coordinates should be median of group, not mean or first."""
    dense = compress_tdf[compress_tdf["uid"] == "user_dense"].copy()
    result = compress(dense, spatial_radius_km=0.2)

    group1_lats = [48.8600, 48.8601, 48.8599, 48.8602, 48.8598]
    group1_lngs = [2.3500, 2.3502, 2.3498, 2.3501, 2.3499]
    expected_lat1 = np.median(group1_lats)
    expected_lng1 = np.median(group1_lngs)

    row0 = result.iloc[0] if hasattr(result, "iloc") else result.row(0, named=True)
    if hasattr(row0, "__getitem__"):
        out_lat = float(row0["lat"])
        out_lng = float(row0["lng"])
    else:
        out_lat = float(row0.lat)
        out_lng = float(row0.lng)

    assert abs(out_lat - expected_lat1) < 1e-6
    assert abs(out_lng - expected_lng1) < 1e-6


def test_compress_output_datetime_is_first_of_group(compress_tdf):
    """Output datetime must be the first timestamp of each group."""
    dense = compress_tdf[compress_tdf["uid"] == "user_dense"].copy()
    result = compress(dense, spatial_radius_km=0.2)
    expected_first = pd.Timestamp("2020-01-01 00:00:00")

    if hasattr(result, "iloc"):
        out_dt = pd.Timestamp(result.iloc[0]["datetime"])
    else:
        out_dt = pd.Timestamp(result["datetime"][0])

    assert out_dt == expected_first


def test_compress_single_point_user():
    """Single-point user passes through unchanged."""
    df = pd.DataFrame(
        {
            "uid": ["u1"],
            "datetime": [pd.Timestamp("2020-01-01")],
            "lat": [48.8566],
            "lng": [2.3522],
        }
    )
    result = compress(df, spatial_radius_km=0.2)
    assert len(result) == 1


def test_compress_returns_same_backend_type(compress_tdf):
    """Pandas input → pandas output."""
    result = compress(compress_tdf, spatial_radius_km=0.2)
    assert isinstance(result, type(compress_tdf))


def test_compress_polars_backend(compress_tdf_polars):
    """Polars input → Polars output with correct row count."""
    import polars as pl

    result = compress(compress_tdf_polars, spatial_radius_km=0.2)
    assert isinstance(result, pl.DataFrame)
    dense = result.filter(pl.col("uid") == "user_dense")
    sparse = result.filter(pl.col("uid") == "user_sparse")
    assert len(dense) == 2
    assert len(sparse) == 5


def test_compress_multiuser_all_users_processed(compress_tdf):
    """Multi-user input: both users processed correctly."""
    result = compress(compress_tdf, spatial_radius_km=0.2)
    assert set(result["uid"].unique()) == {"user_dense", "user_sparse"}


@pytest.mark.skmob
def test_compress_matches_skmob(comparison_skmob):
    """Row count must match skmob on each comparison dataset."""
    from skmob.preprocessing import compression as skmob_compression
    import pandas as pd

    skmob_result = skmob_compression.compress(comparison_skmob, spatial_radius_km=0.2)
    our_result = compress(
        pd.DataFrame(comparison_skmob),
        spatial_radius_km=0.2,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )
    assert len(our_result) == len(skmob_result)
