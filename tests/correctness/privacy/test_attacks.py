"""Correctness tests for fkmob/privacy/attacks.py."""

from __future__ import annotations

import pandas as pd
import pytest
import narwhals as nw
from pandas.testing import assert_frame_equal

from fkmob.privacy import attacks
from fkmob import privacy


LAT_LONS = [
    [43.8430139, 10.5079940],
    [43.5442700, 10.3261500],
    [43.7085300, 10.4036000],
    [43.7792500, 11.2462600],
    [43.8430139, 10.5079940],
    [43.7085300, 10.4036000],
    [43.8430139, 10.5079940],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.7085300, 10.4036000],
    [43.8430139, 10.5079940],
    [43.7792500, 11.2462600],
    [43.7085300, 10.4036000],
    [43.5442700, 10.3261500],
    [43.7792500, 11.2462600],
    [43.7085300, 10.4036000],
    [43.7792500, 11.2462600],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.5442700, 10.3261500],
]

LAT_LONS_FREQ = [
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.7085300, 10.4036000],
    [43.7085300, 10.4036000],
    [43.7792500, 11.2462600],
    [43.8430139, 10.5079940],
    [43.7792500, 11.2462600],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.5442700, 10.3261500],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.8430139, 10.5079940],
    [43.7085300, 10.4036000],
    [43.5442700, 10.3261500],
    [43.7085300, 10.4036000],
    [43.5442700, 10.3261500],
    [43.7085300, 10.4036000],
    [43.5442700, 10.3261500],
]


@pytest.fixture
def privacy_tdf():
    df = pd.DataFrame(LAT_LONS, columns=["lat", "lng"])
    df["datetime"] = pd.to_datetime(
        [
            "20110203 8:34:04",
            "20110203 9:34:04",
            "20110203 10:34:04",
            "20110204 10:34:04",
            "20110203 8:34:04",
            "20110203 9:34:04",
            "20110204 10:34:04",
            "20110204 11:34:04",
            "20110203 8:34:04",
            "20110203 9:34:04",
            "20110204 10:34:04",
            "20110204 11:34:04",
            "20110204 10:34:04",
            "20110204 11:34:04",
            "20110204 12:34:04",
            "20110204 10:34:04",
            "20110204 11:34:04",
            "20110205 12:34:04",
            "20110204 10:34:04",
            "20110204 11:34:04",
        ]
    )
    df["uid"] = [1] * 4 + [2] * 4 + [3] * 4 + [4] * 3 + [5] * 3 + [6] * 2
    return df.sort_values(["uid", "datetime"]).reset_index(drop=True)


@pytest.fixture
def privacy_frequency_tdf():
    df = pd.DataFrame(LAT_LONS_FREQ, columns=["lat", "lng"])
    df["datetime"] = pd.to_datetime(["20110203 8:34:04"] * len(df))
    df["uid"] = [1] * 11 + [2] * 4 + [3] * 9 + [4] * 12
    return df


def _risk_map(df):
    nw_df = nw.from_native(df, eager_only=True)
    return {row["uid"]: row["risk"] for row in nw_df.rows(named=True)}


def _native_to_pandas(df):
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df).copy()


def _normalized(df):
    out = _native_to_pandas(df)
    return out.sort_values(list(out.columns), kind="mergesort").reset_index(drop=True)


def test_privacy_attack_imports_are_compatible():
    assert attacks.LocationAttack is privacy.LocationAttack
    assert attacks.LocationSequenceAttack is privacy.LocationSequenceAttack
    assert attacks.LocationTimeAttack is privacy.LocationTimeAttack
    assert attacks.UniqueLocationAttack is privacy.UniqueLocationAttack
    assert attacks.LocationFrequencyAttack is privacy.LocationFrequencyAttack
    assert attacks.LocationProbabilityAttack is privacy.LocationProbabilityAttack
    assert attacks.LocationProportionAttack is privacy.LocationProportionAttack
    assert attacks.HomeWorkAttack is privacy.HomeWorkAttack


def test_assess_risk_schema(privacy_tdf):
    result = privacy.LocationAttack(knowledge_length=2).assess_risk(privacy_tdf)
    assert nw.from_native(result, eager_only=True).columns == ["uid", "risk"]


def test_location_multiset_matching_preserves_duplicates():
    df = pd.DataFrame(
        {
            "lat": [1.0, 1.0, 1.0, 2.0],
            "lng": [1.0, 1.0, 1.0, 2.0],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]),
            "uid": [1, 1, 2, 2],
        }
    )

    result = attacks.LocationAttack(knowledge_length=2).assess_risk(df, targets=[1])

    assert _risk_map(result) == {1: pytest.approx(1.0)}


def test_location_sequence_matching_preserves_order_with_repeats():
    df = pd.DataFrame(
        {
            "lat": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
            "lng": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
            "datetime": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "uid": [1, 1, 1, 2, 2, 2],
        }
    )

    result = attacks.LocationSequenceAttack(knowledge_length=3).assess_risk(df, targets=[1])

    assert _risk_map(result) == {1: pytest.approx(1.0)}


@pytest.mark.parametrize("precision,expected", [("day", 1.0), ("month", 1.0 / 3.0)])
def test_location_time_public_risk(privacy_tdf, precision, expected):
    result = attacks.LocationTimeAttack(knowledge_length=2, time_precision=precision).assess_risk(privacy_tdf, targets=[1])

    assert _risk_map(result) == {1: pytest.approx(expected)}


def test_date_time_precision_uses_hour_minute_second():
    dt = pd.Timestamp("2011-02-03 08:34:04")

    assert attacks._date_time_precision(dt, "Hour") == "2011238"
    assert attacks._date_time_precision(dt, "Minute") == "201123834"
    assert attacks._date_time_precision(dt, "Second") == "2011238344"


def test_location_time_hour_precision_distinguishes_hours():
    df = pd.DataFrame(
        {
            "lat": [1.0, 1.0],
            "lng": [2.0, 2.0],
            "datetime": pd.to_datetime(["2011-02-03 08:34:04", "2011-02-03 09:34:04"]),
            "uid": [1, 2],
        }
    )

    result = attacks.LocationTimeAttack(knowledge_length=1, time_precision="Hour").assess_risk(df, targets=[1])

    assert _risk_map(result) == {1: pytest.approx(1.0)}


def test_frequency_vector_matches_polars(privacy_frequency_tdf):
    pl = pytest.importorskip("polars")

    pandas_result = _normalized(attacks._frequency_vector(privacy_frequency_tdf))
    polars_result = _normalized(attacks._frequency_vector(pl.from_pandas(privacy_frequency_tdf)))

    assert_frame_equal(polars_result, pandas_result, check_dtype=False)


def test_probability_vector_matches_polars_and_sums_per_user(privacy_frequency_tdf):
    pl = pytest.importorskip("polars")

    pandas_result = _normalized(attacks._probability_vector(privacy_frequency_tdf))
    polars_result = _normalized(attacks._probability_vector(pl.from_pandas(privacy_frequency_tdf)))

    assert_frame_equal(polars_result, pandas_result, check_dtype=False)
    totals = pandas_result.groupby("uid")["prob"].sum()
    assert all(total == pytest.approx(1.0) for total in totals)


def test_unique_location_public_risk(privacy_frequency_tdf):
    result = attacks.UniqueLocationAttack(knowledge_length=2).assess_risk(privacy_frequency_tdf)

    assert _risk_map(result) == {
        1: pytest.approx(1.0 / 2.0),
        2: pytest.approx(1.0),
        3: pytest.approx(1.0 / 3.0),
        4: pytest.approx(1.0 / 2.0),
    }


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0), (0.5, 1.0 / 2.0)])
def test_location_frequency_public_risk(privacy_frequency_tdf, tolerance, expected):
    result = attacks.LocationFrequencyAttack(knowledge_length=2, tolerance=tolerance).assess_risk(
        privacy_frequency_tdf,
        targets=[1],
    )

    assert _risk_map(result) == {1: pytest.approx(expected)}


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0), (1.0, 1.0 / 2.0)])
def test_location_probability_public_risk(privacy_frequency_tdf, tolerance, expected):
    result = attacks.LocationProbabilityAttack(knowledge_length=2, tolerance=tolerance).assess_risk(
        privacy_frequency_tdf,
        targets=[1],
    )

    assert _risk_map(result) == {1: pytest.approx(expected)}


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0), (1.0, 1.0 / 2.0)])
def test_location_proportion_public_risk(privacy_frequency_tdf, tolerance, expected):
    result = attacks.LocationProportionAttack(knowledge_length=2, tolerance=tolerance).assess_risk(
        privacy_frequency_tdf,
        targets=[1],
    )

    assert _risk_map(result) == {1: pytest.approx(expected)}


def test_home_work_public_risk(privacy_frequency_tdf):
    result = attacks.HomeWorkAttack().assess_risk(privacy_frequency_tdf, targets=[1])

    assert _risk_map(result) == {1: pytest.approx(1.0 / 2.0)}


def test_assess_risk_all_users_and_targets(privacy_tdf):
    attack = attacks.LocationAttack(knowledge_length=2)
    all_risks = _risk_map(attack.assess_risk(privacy_tdf))
    target_risks = _risk_map(attack.assess_risk(privacy_tdf, targets=[1, 2]))

    assert all_risks == {
        1: pytest.approx(1.0 / 3.0),
        2: pytest.approx(1.0),
        3: pytest.approx(1.0 / 3.0),
        4: pytest.approx(1.0 / 3.0),
        5: pytest.approx(1.0 / 3.0),
        6: pytest.approx(1.0 / 4.0),
    }
    assert target_risks == {1: pytest.approx(1.0 / 3.0), 2: pytest.approx(1.0)}


def test_assess_risk_targets_dataframe(privacy_tdf):
    result = attacks.LocationSequenceAttack(knowledge_length=2).assess_risk(privacy_tdf, targets=privacy_tdf[:4])
    assert set(_risk_map(result)) == {1}


def test_force_instances_shape_and_probabilities(privacy_tdf):
    result = attacks.LocationSequenceAttack(knowledge_length=2).assess_risk(
        privacy_tdf,
        targets=[1],
        force_instances=True,
    )
    nw_result = nw.from_native(result, eager_only=True)

    assert nw_result.columns == ["lat", "lng", "datetime", "uid", "instance", "instance_elem", "prob"]
    assert len(nw_result) == 12
    assert set(nw_result.get_column("instance_elem").to_list()) == {1, 2}
    assert all(prob > 0 for prob in nw_result.get_column("prob").to_list())


def test_knowledge_length_larger_than_user_trajectory(privacy_tdf):
    result = attacks.LocationSequenceAttack(knowledge_length=99).assess_risk(privacy_tdf, targets=[6])

    assert _risk_map(result)[6] == pytest.approx(1.0 / 3.0)


def test_validation_errors():
    with pytest.raises(ValueError, match="knowledge_length"):
        attacks.LocationAttack(knowledge_length=0)
    with pytest.raises(ValueError, match="Tolerance"):
        attacks.LocationFrequencyAttack(knowledge_length=1, tolerance=-0.1)
    with pytest.raises(ValueError, match="Possible time precisions"):
        attacks.LocationTimeAttack(knowledge_length=1, time_precision="week")


def test_polars_input_matches_pandas(privacy_tdf):
    pl = pytest.importorskip("polars")
    attack = attacks.LocationAttack(knowledge_length=2)

    pandas_result = _risk_map(attack.assess_risk(privacy_tdf))
    polars_result = _risk_map(attack.assess_risk(pl.from_pandas(privacy_tdf)))

    assert polars_result == pandas_result


@pytest.mark.parametrize(
    "attack",
    [
        attacks.LocationAttack(knowledge_length=2),
        attacks.LocationTimeAttack(knowledge_length=2),
        attacks.UniqueLocationAttack(knowledge_length=2),
        attacks.LocationFrequencyAttack(knowledge_length=2),
        attacks.LocationProbabilityAttack(knowledge_length=2),
        attacks.LocationProportionAttack(knowledge_length=2),
        attacks.HomeWorkAttack(),
    ],
)
def test_order_independent_attacks_do_not_require_time_sorted_input(privacy_tdf, attack):
    shuffled = privacy_tdf.sample(frac=1, random_state=7).reset_index(drop=True)

    expected = _risk_map(attack.assess_risk(privacy_tdf))
    actual = _risk_map(attack.assess_risk(shuffled))

    assert actual == expected


@pytest.mark.parametrize(
    "attack",
    [
        attacks.LocationAttack(knowledge_length=2),
        attacks.LocationSequenceAttack(knowledge_length=2),
        attacks.LocationTimeAttack(knowledge_length=2),
        attacks.UniqueLocationAttack(knowledge_length=2),
        attacks.LocationFrequencyAttack(knowledge_length=2),
        attacks.LocationProbabilityAttack(knowledge_length=2),
        attacks.LocationProportionAttack(knowledge_length=2),
        attacks.HomeWorkAttack(),
    ],
)
def test_presorted_privacy_path_matches_default_on_sorted_input(privacy_tdf, attack):
    expected = _risk_map(attack.assess_risk(privacy_tdf))
    actual = _risk_map(attack.assess_risk(privacy_tdf, presorted=True))

    assert actual == expected


@pytest.mark.skmob
@pytest.mark.parametrize(
    ("fkmob_factory", "skmob_factory"),
    [
        (lambda: attacks.LocationAttack(2), lambda skmob_attacks: skmob_attacks.LocationAttack(2)),
        (lambda: attacks.LocationSequenceAttack(2), lambda skmob_attacks: skmob_attacks.LocationSequenceAttack(2)),
        (
            lambda: attacks.LocationTimeAttack(2, time_precision="Month"),
            lambda skmob_attacks: skmob_attacks.LocationTimeAttack(2, time_precision="Month"),
        ),
        (lambda: attacks.UniqueLocationAttack(2), lambda skmob_attacks: skmob_attacks.UniqueLocationAttack(2)),
        (
            lambda: attacks.LocationFrequencyAttack(2, tolerance=0.5),
            lambda skmob_attacks: skmob_attacks.LocationFrequencyAttack(2, tolerance=0.5),
        ),
        (
            lambda: attacks.LocationProbabilityAttack(2, tolerance=0.5),
            lambda skmob_attacks: skmob_attacks.LocationProbabilityAttack(2, tolerance=0.5),
        ),
        (
            lambda: attacks.LocationProportionAttack(2, tolerance=0.5),
            lambda skmob_attacks: skmob_attacks.LocationProportionAttack(2, tolerance=0.5),
        ),
        (lambda: attacks.HomeWorkAttack(), lambda skmob_attacks: skmob_attacks.HomeWorkAttack()),
    ],
)
def test_assess_risk_matches_skmob(privacy_tdf, fkmob_factory, skmob_factory):
    skmob_attacks = pytest.importorskip("skmob.privacy.attacks")
    skmob_core = pytest.importorskip("skmob.core.trajectorydataframe")

    skmob_tdf = skmob_core.TrajDataFrame(privacy_tdf, user_id="uid")
    expected = _risk_map(skmob_factory(skmob_attacks).assess_risk(skmob_tdf))
    actual = _risk_map(fkmob_factory().assess_risk(privacy_tdf))

    assert actual.keys() == expected.keys()
    for uid, risk in expected.items():
        assert actual[uid] == pytest.approx(risk)
