"""Correctness tests for compute_profiles in fastmob.measures.individual.profile_classification."""

from __future__ import annotations

import pandas as pd
import pytest

from fastmob.measures.individual.profile_classification import compute_profiles

VALID_PROFILES = {"routiners", "regulars", "scouters"}


def _skip_if_no_sklearn():
    pytest.importorskip("sklearn", reason="scikit-learn not installed (fastmob[ai])")


def _make_visits(n_each: int = 4, visits_per_user: int = 12) -> pd.DataFrame:
    """Synthetic stay-level visits with three clearly-separated mobility types.

    * routiners: always return to the same location (high degree of return).
    * scouters: always visit a new location (never revisit).
    * regulars: alternate between a home location and new places.

    Each stay is 30 minutes, one per hour, so `_stationarity` has a
    well-defined positive span/dwell for every user.
    """
    rows = []
    uid = 0
    base = pd.Timestamp("2020-01-01")

    def add_rows(name: str, locations: list[str]) -> None:
        for i, loc in enumerate(locations):
            start = base + pd.Timedelta(hours=i)
            rows.append(
                {
                    "uid": name,
                    "start_timestamp": start,
                    "end_timestamp": start + pd.Timedelta(minutes=30),
                    "location_id": loc,
                    "purpose": "HOME" if loc == "home" else "OTHER",
                }
            )

    for _ in range(n_each):
        add_rows(f"routiner_{uid}", ["home"] * visits_per_user)
        uid += 1
    for _ in range(n_each):
        add_rows(f"scouter_{uid}", [f"new_{uid}_{i}" for i in range(visits_per_user)])
        uid += 1
    for _ in range(n_each):
        locs = ["home" if i % 3 == 0 else f"explore_{uid}_{i}" for i in range(visits_per_user)]
        add_rows(f"regular_{uid}", locs)
        uid += 1

    return pd.DataFrame(rows)


def test_output_columns_and_profile_labels():
    _skip_if_no_sklearn()
    visits = _make_visits()
    result = compute_profiles(visits, n_clusters=3, random_state=0)
    expected_cols = {
        "uid",
        "intermittency",
        "degree_of_return",
        "regularity",
        "diversity",
        "entropy",
        "stationarity",
        "profile",
    }
    assert expected_cols.issubset(set(result.columns))
    assert set(result["profile"].unique()) <= VALID_PROFILES
    assert len(result) == 12


def test_routiners_and_scouters_correctly_separated():
    _skip_if_no_sklearn()
    visits = _make_visits()
    result = compute_profiles(visits, n_clusters=3, random_state=0)
    by_uid = result.set_index("uid")["profile"]
    for uid in by_uid.index:
        if uid.startswith("routiner"):
            assert by_uid[uid] == "routiners", uid
        elif uid.startswith("scouter"):
            assert by_uid[uid] == "scouters", uid


def test_pandas_polars_agree():
    _skip_if_no_sklearn()
    pl = pytest.importorskip("polars", reason="Polars not installed")
    visits = _make_visits()
    pandas_result = compute_profiles(visits, n_clusters=3, random_state=0).sort_values("uid").reset_index(drop=True)
    polars_result = (
        compute_profiles(pl.from_pandas(visits), n_clusters=3, random_state=0)
        .sort("uid")
        .to_pandas()
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(pandas_result, polars_result, check_dtype=False, atol=1e-9)


def test_works_without_purpose_column():
    _skip_if_no_sklearn()
    visits = _make_visits().drop(columns=["purpose"])
    result = compute_profiles(visits, n_clusters=3, random_state=0)
    assert len(result) == 12
    assert set(result["profile"].unique()) <= VALID_PROFILES


def test_explicit_column_overrides():
    _skip_if_no_sklearn()
    visits = _make_visits().rename(
        columns={"uid": "user", "start_timestamp": "start", "end_timestamp": "end", "location_id": "loc"}
    )
    result = compute_profiles(
        visits,
        user_id_col="user",
        location_id_col="loc",
        start_col="start",
        end_col="end",
        purpose_col="purpose",
        n_clusters=3,
        random_state=0,
    )
    assert len(result) == 12


def test_too_few_users_raises():
    _skip_if_no_sklearn()
    base = pd.Timestamp("2020-01-01")
    rows = [
        {
            "uid": "u1",
            "start_timestamp": base,
            "end_timestamp": base + pd.Timedelta(minutes=30),
            "location_id": "home",
        },
        {
            "uid": "u2",
            "start_timestamp": base,
            "end_timestamp": base + pd.Timedelta(minutes=30),
            "location_id": "home",
        },
    ]
    visits = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="need at least"):
        compute_profiles(visits, n_clusters=3, random_state=0)


def test_missing_required_column_raises():
    _skip_if_no_sklearn()
    df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError):
        compute_profiles(df, n_clusters=3)
