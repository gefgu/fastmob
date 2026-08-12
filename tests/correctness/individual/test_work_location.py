from __future__ import annotations

import pandas as pd

from fastmob import TrajDataFrame
from fastmob.measures.individual import work_location


def _trajectory():
    return pd.DataFrame(
        {
            "uid": ["a", "a", "a", "b"],
            "datetime": pd.to_datetime(
                ["2024-01-01 09:00", "2024-01-02 10:00", "2024-01-01 23:00", "2024-01-06 10:00"]
            ),
            "lat": [1.0, 1.0, 9.0, 4.0],
            "lng": [2.0, 2.0, 9.0, 5.0],
        }
    )


def test_work_location_uses_weekday_daytime_records_and_omits_users_without_one():
    result = work_location(_trajectory())
    assert result.to_dict("records") == [{"uid": "a", "lat": 1.0, "lng": 2.0}]


def test_trajdataframe_work_location_uses_resolved_metadata():
    result = TrajDataFrame(_trajectory()).work_location()
    assert result.to_dict("records") == [{"uid": "a", "lat": 1.0, "lng": 2.0}]
