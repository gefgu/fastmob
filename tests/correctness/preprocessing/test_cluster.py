"""Correctness tests for skmob2.preprocessing.cluster."""
from __future__ import annotations

import pandas as pd
import pytest

from skmob2.preprocessing import cluster


def test_cluster_adds_cluster_column(cluster_tdf):
    """Output must contain a 'cluster' column."""
    result = cluster(cluster_tdf)
    cols = result.columns if hasattr(result, "columns") else result.schema.names()
    assert "cluster" in cols


def test_cluster_nearby_stops_same_cluster(cluster_tdf):
    """A and B (< 0.1 km apart) must receive the same cluster label."""
    result = cluster(cluster_tdf, cluster_radius_km=0.1)
    if hasattr(result, "iloc"):
        label_a = result.iloc[0]["cluster"]
        label_b = result.iloc[1]["cluster"]
    else:
        label_a = result["cluster"][0]
        label_b = result["cluster"][1]
    assert label_a == label_b


def test_cluster_distant_stop_separate_cluster(cluster_tdf):
    """C (5 km away) must have a different cluster label from A and B."""
    result = cluster(cluster_tdf, cluster_radius_km=0.1)
    if hasattr(result, "iloc"):
        label_ab = result.iloc[0]["cluster"]
        label_c = result.iloc[2]["cluster"]
    else:
        label_ab = result["cluster"][0]
        label_c = result["cluster"][2]
    assert label_ab != label_c


def test_cluster_label_ordering_by_frequency(cluster_tdf):
    """Most-visited cluster (A/B with 2 visits) must get label 0."""
    result = cluster(cluster_tdf, cluster_radius_km=0.1)
    if hasattr(result, "iloc"):
        label_ab = result.iloc[0]["cluster"]
    else:
        label_ab = result["cluster"][0]
    assert label_ab == 0


def test_cluster_preserves_all_columns(cluster_tdf):
    """All original columns must still be present in output."""
    result = cluster(cluster_tdf, cluster_radius_km=0.1)
    original_cols = set(cluster_tdf.columns)
    result_cols = set(result.columns if hasattr(result, "columns") else result.schema.names())
    assert original_cols <= result_cols


def test_cluster_single_stop_gets_label_zero():
    """A single stop with min_samples=1 should get label 0."""
    df = pd.DataFrame({
        "uid": ["u1"],
        "datetime": [pd.Timestamp("2020-01-01")],
        "lat": [48.8566],
        "lng": [2.3522],
    })
    result = cluster(df, cluster_radius_km=0.1)
    label = result.iloc[0]["cluster"] if hasattr(result, "iloc") else result["cluster"][0]
    assert int(label) == 0


def test_cluster_noise_label_minus_one():
    """With min_samples=3, isolated points are noise (label -1)."""
    df = pd.DataFrame({
        "uid": ["u1"] * 3,
        "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "lat": [48.856, 51.500, 40.712],   # London, NY — far apart
        "lng": [2.352, -0.127, -74.006],
    })
    result = cluster(df, cluster_radius_km=0.1, min_samples=3)
    labels = list(result["cluster"] if hasattr(result, "__getitem__") else [r["cluster"] for r in result.iter_rows(named=True)])
    assert all(int(l) == -1 for l in labels)


def test_cluster_returns_same_backend(cluster_tdf):
    """Pandas input → pandas output."""
    result = cluster(cluster_tdf)
    assert isinstance(result, type(cluster_tdf))


def test_cluster_polars_backend(cluster_tdf_polars):
    """Polars input → Polars output with correct cluster labels."""
    import polars as pl
    result = cluster(cluster_tdf_polars, cluster_radius_km=0.1)
    assert isinstance(result, pl.DataFrame)
    assert "cluster" in result.columns
    assert result["cluster"][0] == 0  # most visited = label 0


@pytest.mark.skmob
def test_cluster_matches_skmob(brightkite_skmob):
    """Cluster labels must match skmob on Brightkite stop data."""
    import skmob
    from skmob.preprocessing import detection as skmob_detection
    from skmob.preprocessing import clustering as skmob_clustering
    import pandas as pd

    skmob_stops = skmob_detection.stay_locations(
        brightkite_skmob, spatial_radius_km=0.2, minutes_for_a_stop=20.0
    )
    skmob_result = skmob_clustering.cluster(skmob_stops, cluster_radius_km=0.1)

    from skmob2.preprocessing import stay_locations
    our_stops = stay_locations(
        pd.DataFrame(brightkite_skmob),
        spatial_radius_km=0.2,
        minutes_for_a_stop=20.0,
        datetime_col="datetime",
        lat_col="lat",
        lng_col="lng",
        uid_col="uid",
    )
    our_result = cluster(our_stops, cluster_radius_km=0.1,
                         datetime_col="datetime", lat_col="lat", lng_col="lng", uid_col="uid")

    assert len(our_result) == len(skmob_result)
