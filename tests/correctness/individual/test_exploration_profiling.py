"""Correctness tests for exploration_profiling in skmob2.measures.individual."""

from __future__ import annotations

import pandas as pd
import pytest

VALID_PROFILES = {"routiners", "regulars", "scouters"}


def _skip_if_no_core():
    pytest.importorskip(
        "skmob2._core",
        reason="Build the skmob2 extension first (maturin develop)",
    )


def _make_visits(n_each: int = 5, visits_per_user: int = 12) -> pd.DataFrame:
    """Synthetic visits with three clearly-separated mobility types.

    * routiners: always return to the same location.
    * scouters: always visit a new location (never revisit).
    * regulars: alternate between a home location and new places.
    """
    rows = []
    uid = 0
    for _ in range(n_each):
        name = f"routiner_{uid}"
        for _ in range(visits_per_user):
            rows.append({"agent_id": name, "location_id": "home"})
        uid += 1
    for _ in range(n_each):
        name = f"scouter_{uid}"
        for i in range(visits_per_user):
            rows.append({"agent_id": name, "location_id": f"new_{uid}_{i}"})
        uid += 1
    for _ in range(n_each):
        name = f"regular_{uid}"
        for i in range(visits_per_user):
            loc = "home" if i % 3 == 0 else f"explore_{uid}_{i}"
            rows.append({"agent_id": name, "location_id": loc})
        uid += 1
    return pd.DataFrame(rows)


def test_output_columns():
    """Result contains all expected columns including 'profile'."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none")
    for col in ["agent_id", "intermittency", "degree_of_return", "mean_return", "mean_exploration", "profile"]:
        assert col in result.columns


def test_profile_values_valid():
    """'profile' column only contains the three valid labels."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none")
    assert set(result["profile"]).issubset(VALID_PROFILES)


def test_three_profiles_all_present():
    """All three profile types appear when users are clearly separated."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none")
    assert set(result["profile"]) == VALID_PROFILES


def test_routiners_higher_dor_than_scouters():
    """Routiners have a higher mean degree_of_return than scouters."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none")
    routiners_dor = result.loc[result["profile"] == "routiners", "degree_of_return"].mean()
    scouters_dor = result.loc[result["profile"] == "scouters", "degree_of_return"].mean()
    assert routiners_dor > scouters_dor


def test_kmeans_method_row_count():
    """K-Means returns exactly one row per user."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none", clustering_method="kmeans")
    n_users = df["agent_id"].nunique()
    assert len(result) == n_users


def test_gmm_method_row_count():
    """GMM returns exactly one row per user."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    result = exploration_profiling(df, cold_start_strategy="none", clustering_method="gmm")
    n_users = df["agent_id"].nunique()
    assert len(result) == n_users


def test_impute_gaps_parameter_preserves_profile_output_shape():
    """impute_gaps is accepted and still returns one profiled row per user."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits(n_each=1, visits_per_user=6)
    base = pd.Timestamp("2020-01-01 02:00")
    df["start_timestamp"] = [base + pd.Timedelta(minutes=15 * i) for i in range(len(df))]
    df["end_timestamp"] = df["start_timestamp"]

    result = exploration_profiling(df, cold_start_strategy="none", impute_gaps=True)

    assert len(result) == df["agent_id"].nunique()
    assert "profile" in result.columns


def test_too_few_users_raises():
    """Fewer than 3 users triggers ValueError."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = pd.DataFrame(
        {
            "agent_id": ["u1", "u2"],
            "location_id": ["a", "b"],
        }
    )
    with pytest.raises(ValueError, match="at least 3"):
        exploration_profiling(df, cold_start_strategy="none")


def test_column_autodetection_uid():
    """Works with 'uid' as the user ID column name."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits().rename(columns={"agent_id": "uid"})
    result = exploration_profiling(df, cold_start_strategy="none")
    assert "uid" in result.columns
    assert "profile" in result.columns


def test_unknown_method_raises():
    """Passing an unknown clustering_method raises ValueError."""
    _skip_if_no_core()
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    df = _make_visits()
    with pytest.raises(ValueError, match="clustering_method"):
        exploration_profiling(df, cold_start_strategy="none", clustering_method="dbscan")


def test_polars_parity():
    """Pandas and Polars inputs produce identical profile assignments."""
    _skip_if_no_core()
    polars = pytest.importorskip("polars", reason="Install polars for this test")
    from skmob2.measures.individual.mobility_profiling import exploration_profiling

    # Build polars DataFrame directly (no pyarrow required)
    df_pd = _make_visits()
    df_pl = polars.DataFrame(
        {
            "agent_id": df_pd["agent_id"].tolist(),
            "location_id": df_pd["location_id"].tolist(),
        }
    )

    result_pd = exploration_profiling(df_pd, cold_start_strategy="none").sort_values("agent_id").reset_index(drop=True)
    result_pl_native = exploration_profiling(df_pl, cold_start_strategy="none")
    result_pl = (
        pd.DataFrame(
            {
                "agent_id": result_pl_native["agent_id"].to_list(),
                "profile": result_pl_native["profile"].to_list(),
            }
        )
        .sort_values("agent_id")
        .reset_index(drop=True)
    )
    result_pd_sorted = result_pd.sort_values("agent_id").reset_index(drop=True)
    pd.testing.assert_series_equal(
        result_pd_sorted["profile"].reset_index(drop=True),
        result_pl["profile"].reset_index(drop=True),
    )


def test_top_level_import():
    """exploration_profiling is accessible from the top-level skmob2 namespace."""
    import skmob2

    assert hasattr(skmob2, "exploration_profiling")
    assert callable(skmob2.exploration_profiling)
