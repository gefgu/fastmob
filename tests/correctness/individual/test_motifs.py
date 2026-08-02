"""Tests for daily home-anchored mobility motifs."""

import pandas as pd
import pytest


def _make_multi_day_df():
    rows = []
    for day_offset in range(2):
        base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_offset)
        rows += [
            {
                "agent_id": "u1",
                "location_id": "home",
                "purpose": "HOME",
                "start_timestamp": base,
                "end_timestamp": base + pd.Timedelta(hours=8),
                "duration_minutes": 480,
            },
            {
                "agent_id": "u1",
                "location_id": "work",
                "purpose": "WORK",
                "start_timestamp": base + pd.Timedelta(hours=9),
                "end_timestamp": base + pd.Timedelta(hours=17),
                "duration_minutes": 480,
            },
            {
                "agent_id": "u1",
                "location_id": "home",
                "purpose": "HOME",
                "start_timestamp": base + pd.Timedelta(hours=18),
                "end_timestamp": base + pd.Timedelta(hours=23),
                "duration_minutes": 300,
            },
        ]
    return pd.DataFrame(rows)


def test_daily_motifs_returns_primary_dataframe_only():
    from fastmob.measures.individual.motifs import daily_motifs

    result = daily_motifs(_make_multi_day_df())
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["agent_id", "date", "motif_id"]
    assert len(result) == 2
    assert ((result["motif_id"].to_numpy() >> 36) == 2).all()


@pytest.mark.parametrize(
    "drop_col,match",
    [
        ("agent_id", "Could not find a user-ID column"),
        ("location_id", "Could not find a location column"),
        ("purpose", "Could not find a purpose column"),
        ("start_timestamp", "Could not find a start-timestamp column"),
        ("end_timestamp", "Could not find an end-timestamp column"),
    ],
)
def test_daily_motifs_raises_when_required_column_missing(drop_col, match):
    from fastmob.measures.individual.motifs import daily_motifs

    df = _make_multi_day_df().drop(columns=[drop_col])
    with pytest.raises(ValueError, match=match):
        daily_motifs(df)


def test_canonical_motifs_keep_home_anchored_at_node_zero():
    from fastmob import _core

    left = _core.canonical_adjacency_form(3, [(0, 1), (1, 2)])
    right = _core.canonical_adjacency_form(3, [(1, 0), (0, 2)])
    assert left != right


def test_daily_motifs_uses_home_anchored_ids():
    from fastmob.measures.individual.motifs import daily_motifs

    base = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        [
            {
                "agent_id": "u1",
                "location_id": "home",
                "purpose": "HOME",
                "start_timestamp": base,
                "end_timestamp": base + pd.Timedelta(hours=8),
                "duration_minutes": 480,
            },
            {
                "agent_id": "u1",
                "location_id": "a",
                "purpose": "WORK",
                "start_timestamp": base + pd.Timedelta(hours=9),
                "end_timestamp": base + pd.Timedelta(hours=10),
                "duration_minutes": 60,
            },
            {
                "agent_id": "u1",
                "location_id": "b",
                "purpose": "SHOP",
                "start_timestamp": base + pd.Timedelta(hours=11),
                "end_timestamp": base + pd.Timedelta(hours=23),
                "duration_minutes": 720,
            },
        ]
    )
    result = daily_motifs(df)
    assert result["motif_id"].iloc[0] == (3 << 36) | 0b010001100


def test_daily_motifs_empty_result_schema_is_stable():
    from fastmob.measures.individual.motifs import daily_motifs, motif_distribution

    df = pd.DataFrame(
        {
            "agent_id": pd.Series([1], dtype="int64"),
            "location_id": ["work"],
            "purpose": ["WORK"],
            "start_timestamp": pd.to_datetime(["2020-01-01 09:00"]),
            "end_timestamp": pd.to_datetime(["2020-01-01 17:00"]),
        }
    )
    daily = daily_motifs(df)
    distribution = motif_distribution(daily)
    assert daily.empty
    assert pd.api.types.is_integer_dtype(daily["agent_id"])
    assert pd.api.types.is_datetime64_any_dtype(daily["date"])
    assert pd.api.types.is_integer_dtype(daily["motif_id"])
    assert pd.api.types.is_integer_dtype(distribution["motif_id"])
    assert pd.api.types.is_integer_dtype(distribution["count"])


def test_indexed_and_presorted_paths_match():
    from fastmob.measures.individual.motifs import daily_motifs

    df = _make_multi_day_df()
    df = pd.concat(
        [
            df,
            df.assign(
                agent_id="u2",
                start_timestamp=df["start_timestamp"] + pd.Timedelta(hours=1),
                end_timestamp=df["end_timestamp"] + pd.Timedelta(hours=1),
            ),
        ],
        ignore_index=True,
    ).sort_values(["agent_id", "start_timestamp"], kind="stable")
    expected = daily_motifs(df, presorted=True).sort_values(["agent_id", "date"]).reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    actual = daily_motifs(shuffled).sort_values(["agent_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_motif_distribution_sum_to_100():
    from fastmob.measures.individual.motifs import daily_motifs, motif_distribution

    distribution = motif_distribution(daily_motifs(_make_multi_day_df()))
    assert distribution["percentage"].sum() == pytest.approx(100.0, abs=1e-6)
    assert distribution["count"].sum() == 2


def test_motif_distribution_supports_custom_id_column():
    from fastmob.measures.individual.motifs import motif_distribution

    result = motif_distribution(pd.DataFrame({"kind": [7, 7, 9]}), motif_id_col="kind")
    assert result["kind"].tolist() == [7, 9]
    assert result["count"].tolist() == [2, 1]
    assert result["percentage"].tolist() == pytest.approx([200 / 3, 100 / 3])


def test_integer_and_string_locations_produce_the_same_motifs():
    from fastmob.measures.individual.motifs import daily_motifs

    strings = _make_multi_day_df()
    integers = strings.assign(location_id=strings["location_id"].map({"home": 10, "work": 20}))
    assert daily_motifs(strings)["motif_id"].tolist() == daily_motifs(integers)["motif_id"].tolist()


def test_daily_motifs_preserves_numeric_user_id_dtype():
    from fastmob.measures.individual.motifs import daily_motifs

    df = _make_multi_day_df()
    df["agent_id"] = df["agent_id"].map({"u1": 1}).astype("int64")
    result = daily_motifs(df)
    assert pd.api.types.is_integer_dtype(result["agent_id"])
    assert set(result["agent_id"].unique()) == {1}


def test_daily_motifs_preserves_numeric_user_id_dtype_polars():
    pl = pytest.importorskip("polars")
    from fastmob.measures.individual.motifs import daily_motifs

    df = _make_multi_day_df()
    df["agent_id"] = df["agent_id"].map({"u1": 1}).astype("int64")
    result = daily_motifs(pl.from_pandas(df))
    assert result.schema["agent_id"] == pl.Int64
    assert set(result["agent_id"].unique().to_list()) == {1}


def test_daily_motifs_handles_tz_aware_timestamps():
    from fastmob.measures.individual.motifs import daily_motifs

    aware = _make_multi_day_df()
    aware["start_timestamp"] = aware["start_timestamp"].dt.tz_localize("UTC")
    aware["end_timestamp"] = aware["end_timestamp"].dt.tz_localize("UTC")
    aware_result = daily_motifs(aware)
    naive_result = daily_motifs(_make_multi_day_df())
    assert aware_result["motif_id"].tolist() == naive_result["motif_id"].tolist()
    assert aware_result["date"].tolist() == naive_result["date"].tolist()


def test_old_agent_specific_api_is_removed():
    import fastmob

    assert not hasattr(fastmob, "discover_daily_motifs_from_agents")
