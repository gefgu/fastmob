"""Correctness tests for skmob2/measures/spatial/number_of_visits.py."""
from __future__ import annotations

import narwhals as nw
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Known-value assertions for the shared synthetic fixture (3 users, 5 points each).
# ---------------------------------------------------------------------------

EXPECTED_VISITS: dict[str, int] = {
    "user_a": 5,
    "user_b": 5,
    "user_c": 5,
}


def _to_dict(df) -> dict[str, int]:
    """Convert a number_of_visits result DataFrame to {uid: visit_count}.

    Uses Narwhals rows() to avoid the pyarrow dependency that polars .to_pandas()
    requires in this environment.
    """
    nw_df = nw.from_native(df, eager_only=True)
    columns = nw_df.columns
    uid_col = next((c for c in ("uid", "user", "user_id") if c in columns), None)
    if uid_col is None:
        raise AssertionError("Result lacks uid/user/user_id column")

    return {
        row[uid_col]: row["number_of_visits"]
        for row in nw_df.rows(named=True)
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_number_of_visits_known_values(synthetic_tdf):
    """Each user in the synthetic fixture has exactly 5 visit rows."""
    from skmob2.measures.spatial.number_of_visits import number_of_visits

    result = number_of_visits(synthetic_tdf)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_VISITS.keys())
    for uid, expected in EXPECTED_VISITS.items():
        assert mapping[uid] == expected, f"uid={uid!r}: got {mapping[uid]}, expected {expected}"


def test_number_of_visits_repeated_locations():
    """Repeated (lat, lng) pairs still count as separate visits."""
    from skmob2.measures.spatial.number_of_visits import number_of_visits

    df = pd.DataFrame(
        {
            "uid": ["a", "a", "a", "a"],
            "datetime": pd.date_range("2020-01-01", periods=4, freq="h"),
            "lat": [1.0, 2.0, 1.0, 2.0],
            "lng": [3.0, 4.0, 3.0, 4.0],
        }
    )
    result = number_of_visits(df)
    mapping = _to_dict(result)
    assert mapping["a"] == 4


def test_number_of_visits_no_uid():
    """Without a uid column the whole frame is treated as one individual."""
    from skmob2.measures.spatial.number_of_visits import number_of_visits

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
            "lat": [0.0, 1.0, 2.0],
            "lng": [0.0, 0.0, 0.0],
        }
    )
    result = number_of_visits(df)
    nw_result = nw.from_native(result, eager_only=True)
    assert "number_of_visits" in nw_result.columns
    assert len(nw_result) == 1
    row = nw_result.rows(named=True)[0]
    assert row["number_of_visits"] == 3


def test_number_of_visits_polars_known_values(synthetic_tdf_polars):
    """Polars input yields the same result as pandas."""
    from skmob2.measures.spatial.number_of_visits import number_of_visits

    result = number_of_visits(synthetic_tdf_polars)
    mapping = _to_dict(result)

    assert set(mapping.keys()) == set(EXPECTED_VISITS.keys())
    for uid, expected in EXPECTED_VISITS.items():
        assert mapping[uid] == expected, (
            f"uid={uid!r}: got {mapping[uid]}, expected {expected} (Polars)"
        )


@pytest.mark.skmob
def test_number_of_visits_matches_skmob(brightkite_skmob):
    """skmob2 result matches skmob on the Brightkite dataset."""
    import pandas as pd
    from skmob.measures.individual import number_of_visits as skmob_nov
    from skmob2.measures.spatial.number_of_visits import number_of_visits as skmob2_nov

    skmob_result = skmob_nov(brightkite_skmob)
    skmob2_input = pd.DataFrame(brightkite_skmob).copy()
    skmob2_result = skmob2_nov(skmob2_input)

    skmob_dict = dict(
        zip(skmob_result["uid"].tolist(), skmob_result["number_of_visits"].tolist())
    )
    skmob2_dict = _to_dict(skmob2_result)

    common = set(skmob_dict) & set(skmob2_dict)
    assert len(common) > 0
    for uid in common:
        assert skmob_dict[uid] == skmob2_dict[uid], (
            f"uid={uid}: skmob={skmob_dict[uid]}, skmob2={skmob2_dict[uid]}"
        )
