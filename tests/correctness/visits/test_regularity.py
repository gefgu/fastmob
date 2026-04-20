"""Tests for regularity measure."""
import pandas as pd
import numpy as np
import pytest
from skmob2.measures.visits.regularity import regularity


def _uniform_visits():
    """Three users with predictable regularity patterns.

    u1: always the same location → regularity = 1 - (1/3) = 0.667
    u2: all different locations → regularity = 1 - (3/3) = 0.0
    u3: two distinct locations, three visits → regularity = 1 - (2/3) = 0.333
    """
    return pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1",
                        "u2", "u2", "u2",
                        "u3", "u3", "u3"],
        "location_id": ["home", "home", "home",
                        "place_a", "place_b", "place_c",
                        "home", "work", "home"],
        "location_type": ["HOME", "HOME", "HOME",
                          "LEISURE", "SHOP", "WORK",
                          "HOME", "WORK", "HOME"],
    })


def test_regularity_all_same_location():
    """User who always visits the same location has high regularity."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1"],
        "location_id": ["home", "home", "home"],
        "location_type": ["HOME", "HOME", "HOME"],
    })
    result = regularity(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # unique (location_id, location_type) = 1; total = 3 → 1 - 1/3
    assert row["regularity"] == pytest.approx(1.0 - 1.0 / 3.0, rel=1e-9)


def test_regularity_all_different_locations():
    """User who visits all different locations has regularity = 0."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1"],
        "location_id": ["a", "b", "c"],
        "location_type": ["L1", "L2", "L3"],
    })
    result = regularity(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    assert row["regularity"] == pytest.approx(0.0, abs=1e-9)


def test_regularity_values_multi_user():
    """Verify exact regularity values for three users with known patterns."""
    df = _uniform_visits()
    result = regularity(df)
    result = result.set_index("agent_id")

    # u1: 1 unique / 3 total → 1 - 1/3
    assert result.loc["u1", "regularity"] == pytest.approx(1.0 - 1.0 / 3.0, rel=1e-9)
    # u2: 3 unique / 3 total → 1 - 3/3 = 0
    assert result.loc["u2", "regularity"] == pytest.approx(0.0, abs=1e-9)
    # u3: 2 unique ("home_HOME", "work_WORK") / 3 total → 1 - 2/3
    assert result.loc["u3", "regularity"] == pytest.approx(1.0 - 2.0 / 3.0, rel=1e-9)


def test_regularity_result_shape():
    """Result has exactly one row per user and columns [user_id_col, 'regularity']."""
    df = _uniform_visits()
    result = regularity(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert "agent_id" in result.columns
    assert "regularity" in result.columns


def test_regularity_values_in_range():
    """Regularity must be in [0, 1] for any input."""
    df = _uniform_visits()
    result = regularity(df)
    assert (result["regularity"] >= 0).all()
    assert (result["regularity"] <= 1.0 + 1e-9).all()


def test_regularity_single_visit():
    """A user with exactly one visit: unique=1, total=1 → regularity = 0."""
    df = pd.DataFrame({
        "agent_id":    ["u1"],
        "location_id": ["home"],
        "location_type": ["HOME"],
    })
    result = regularity(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # 1 - 1/1 = 0
    assert row["regularity"] == pytest.approx(0.0, abs=1e-9)


def test_regularity_without_location_type():
    """When location_type_col is absent, uniqueness is based on location_id only."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1", "u1"],
        "location_id": ["home", "home", "work", "home"],
    })
    result = regularity(df, location_type_col=None)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # 2 unique location_ids ("home", "work") / 4 total → 1 - 2/4 = 0.5
    assert row["regularity"] == pytest.approx(0.5, rel=1e-9)


def test_regularity_column_autodetection_user_id():
    """Auto-detects 'user_id' when 'agent_id' is absent."""
    df = pd.DataFrame({
        "user_id":     ["u1", "u1", "u2"],
        "location_id": ["home", "home", "other"],
        "location_type": ["HOME", "HOME", "WORK"],
    })
    result = regularity(df)
    assert "user_id" in result.columns
    assert len(result) == 2


def test_regularity_column_autodetection_location_type():
    """Auto-detects 'purpose' as location_type_col."""
    df = pd.DataFrame({
        "agent_id":  ["u1", "u1", "u1"],
        "location_id": ["home", "home", "work"],
        "purpose":   ["HOME", "HOME", "WORK"],
    })
    result = regularity(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # unique pairs: ("home", "HOME"), ("work", "WORK") → 2 unique / 3 total
    assert row["regularity"] == pytest.approx(1.0 - 2.0 / 3.0, rel=1e-9)


def test_regularity_same_location_different_type():
    """Same location_id with different types counts as different unique pairs."""
    df = pd.DataFrame({
        "agent_id":    ["u1", "u1", "u1"],
        "location_id": ["place", "place", "place"],
        "location_type": ["HOME", "WORK", "LEISURE"],
    })
    result = regularity(df)
    row = result[result["agent_id"] == "u1"].iloc[0]
    # 3 unique pairs / 3 total → 1 - 3/3 = 0
    assert row["regularity"] == pytest.approx(0.0, abs=1e-9)
