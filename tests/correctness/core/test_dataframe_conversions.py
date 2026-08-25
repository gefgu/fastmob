from __future__ import annotations

import pandas as pd
import pytest

from fastmob import Locations, Positionfixes, Staypoints, Tours, TrajDataFrame, Triplegs, Trips


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TrajDataFrame(
            pd.DataFrame(
                {
                    "uid": ["u1"],
                    "lat": [0.0],
                    "lng": [0.0],
                    "datetime": [pd.Timestamp("2020-01-01")],
                }
            )
        ),
        lambda: Positionfixes(
            pd.DataFrame(
                {
                    "uid": ["u1"],
                    "lat": [0.0],
                    "lng": [0.0],
                    "datetime": [pd.Timestamp("2020-01-01")],
                }
            )
        ),
        lambda: Staypoints(
            pd.DataFrame(
                {
                    "uid": ["u1"],
                    "lat": [0.0],
                    "lng": [0.0],
                    "started_at": [pd.Timestamp("2020-01-01")],
                    "finished_at": [pd.Timestamp("2020-01-01 01:00")],
                }
            )
        ),
        lambda: Locations(
            pd.DataFrame({"location_id": [1], "center_lat": [0.0], "center_lng": [0.0]})
        ),
        lambda: Triplegs(
            pd.DataFrame(
                {
                    "tripleg_id": [1],
                    "started_at": [pd.Timestamp("2020-01-01")],
                    "finished_at": [pd.Timestamp("2020-01-01 01:00")],
                    "length_km": [1.0],
                    "duration_s": [3600.0],
                }
            )
        ),
        lambda: Trips(
            pd.DataFrame(
                {
                    "trip_id": [1],
                    "started_at": [pd.Timestamp("2020-01-01")],
                    "finished_at": [pd.Timestamp("2020-01-01 01:00")],
                }
            )
        ),
        lambda: Tours(
            pd.DataFrame(
                {
                    "tour_id": [1],
                    "started_at": [pd.Timestamp("2020-01-01")],
                    "finished_at": [pd.Timestamp("2020-01-01 01:00")],
                }
            )
        ),
    ],
    ids=["traj", "positionfixes", "staypoints", "locations", "triplegs", "trips", "tours"],
)
def test_all_dataframe_wrappers_convert_to_pandas_and_polars(factory):
    wrapper = factory()
    polars = pytest.importorskip("polars", reason="Polars not installed")

    pandas_result = wrapper.to_pandas()
    polars_result = wrapper.to_polars()

    assert isinstance(pandas_result, pd.DataFrame)
    assert isinstance(polars_result, polars.DataFrame)
    assert pandas_result.columns.tolist() == list(wrapper.columns)
    assert len(pandas_result) == len(wrapper)
