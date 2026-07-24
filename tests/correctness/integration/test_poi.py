"""Correctness tests for fastmob.integration.poi."""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.integration.poi import join_with_pois, join_with_pois_by_category

_TRAJ = pd.DataFrame(
    {
        "uid": ["u1", "u1"],
        "datetime": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:20:00"]),
        "lat": [0.0, 0.01],
        "lng": [0.0, 0.0],
    }
)

_POIS = pd.DataFrame(
    {
        "lat": [0.0001, 0.02],
        "lng": [0.0, 0.0],
        "id": [1, 2],
        "name_poi": ["near", "far"],
        "type_poi": ["shop", "park"],
    }
)


def test_join_with_pois_picks_single_nearest():
    result = join_with_pois(_TRAJ, _POIS)
    assert list(result["id_poi"]) == [1, 1]
    assert list(result["name_poi"]) == ["near", "near"]
    assert result["dist_poi"].iloc[0] == pytest.approx(11.12, abs=0.5)
    # traj point 1 (lat=0.01) is ~1100.8m from poi 1 (lat=0.0001) and
    # ~1111.9m from poi 2 (lat=0.02) -- poi 1 must still win.
    assert result["dist_poi"].iloc[1] == pytest.approx(1100.83, abs=0.5)


def test_join_with_pois_preserves_original_columns():
    result = join_with_pois(_TRAJ, _POIS)
    for col in _TRAJ.columns:
        assert col in result.columns


def test_join_with_pois_empty_pois_returns_none_and_inf():
    empty_pois = _POIS.iloc[:0]
    result = join_with_pois(_TRAJ, empty_pois)
    assert result["id_poi"].isna().all()
    assert (result["dist_poi"] == float("inf")).all()


def test_join_with_pois_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    pandas_result = join_with_pois(_TRAJ, _POIS)
    polars_result = join_with_pois(pl.from_pandas(_TRAJ), pl.from_pandas(_POIS)).to_pandas()
    assert list(pandas_result["id_poi"]) == list(polars_result["id_poi"])
    assert list(pandas_result["dist_poi"]) == pytest.approx(list(polars_result["dist_poi"]))


def test_join_with_pois_by_category_adds_one_pair_per_category():
    result = join_with_pois_by_category(_TRAJ, _POIS)
    assert "id_shop" in result.columns
    assert "dist_shop" in result.columns
    assert "id_park" in result.columns
    assert "dist_park" in result.columns
    # only one shop poi (id=1) and one park poi (id=2) exist
    assert list(result["id_shop"]) == [1, 1]
    assert list(result["id_park"]) == [2, 2]
    assert result["dist_shop"].iloc[0] < result["dist_park"].iloc[0]
