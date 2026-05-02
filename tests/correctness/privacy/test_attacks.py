"""Correctness tests for skmob2/privacy/attacks.py."""

from __future__ import annotations

import pandas as pd
import pytest
import narwhals as nw

from skmob2.privacy import attacks
from skmob2 import privacy


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


def _records(df):
    return nw.from_native(df, eager_only=True).rows(named=True)


def _risk_map(df):
    nw_df = nw.from_native(df, eager_only=True)
    return {row["uid"]: row["risk"] for row in nw_df.rows(named=True)}


def _single(df, uid):
    return df[df["uid"] == uid]


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


def test_location_match(privacy_tdf):
    first_instance = _records(privacy_tdf[:2])
    second_instance = _records(pd.concat([privacy_tdf[0:1], privacy_tdf[3:4]]))
    attack = attacks.LocationAttack(knowledge_length=1)

    first_matches = [attack._match(_single(privacy_tdf, uid), first_instance) for uid in range(1, 7)]
    second_matches = [attack._match(_single(privacy_tdf, uid), second_instance) for uid in range(1, 7)]

    assert 1.0 / sum(first_matches) == 1.0 / 4.0
    assert 1.0 / sum(second_matches) == 1.0 / 3.0


def test_location_sequence_match(privacy_tdf):
    first_instance = _records(privacy_tdf[:2])
    second_instance = _records(pd.concat([privacy_tdf[0:1], privacy_tdf[3:4]]))
    attack = attacks.LocationSequenceAttack(knowledge_length=1)

    first_matches = [attack._match(_single(privacy_tdf, uid), first_instance) for uid in range(1, 7)]
    second_matches = [attack._match(_single(privacy_tdf, uid), second_instance) for uid in range(1, 7)]

    assert 1.0 / sum(first_matches) == 1.0 / 3.0
    assert 1.0 / sum(second_matches) == 1.0 / 2.0


@pytest.mark.parametrize("precision,expected", [("day", 1.0), ("month", 1.0 / 4.0)])
def test_location_time_match(privacy_tdf, precision, expected):
    attack = attacks.LocationTimeAttack(knowledge_length=1, time_precision=precision)
    with_tmp = privacy_tdf.copy()
    with_tmp["tmp"] = with_tmp["datetime"].apply(lambda dt: attacks._date_time_precision(dt, precision))
    first_instance = _records(with_tmp[:2])
    matches = [attack._match(_single(with_tmp, uid), first_instance) for uid in range(1, 7)]

    assert 1.0 / sum(matches) == expected


def test_unique_location_match(privacy_frequency_tdf):
    freq = attacks._frequency_vector(privacy_frequency_tdf)
    instance = _records(freq)[1:3]
    attack = attacks.UniqueLocationAttack(knowledge_length=1)
    matches = [attack._match(_single(freq, uid), instance) for uid in range(1, 5)]

    assert 1.0 / sum(matches) == 1.0 / 3.0


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0 / 2.0), (0.5, 1.0 / 3.0)])
def test_location_frequency_match(privacy_frequency_tdf, tolerance, expected):
    freq = attacks._frequency_vector(privacy_frequency_tdf)
    instance = _records(freq)[1:3]
    attack = attacks.LocationFrequencyAttack(knowledge_length=1, tolerance=tolerance)
    matches = [attack._match(_single(freq, uid), instance) for uid in range(1, 5)]

    assert 1.0 / sum(matches) == expected


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0), (1.0, 1.0 / 3.0)])
def test_location_probability_match(privacy_frequency_tdf, tolerance, expected):
    prob = attacks._probability_vector(privacy_frequency_tdf)
    instance = _records(prob)[1:3]
    attack = attacks.LocationProbabilityAttack(knowledge_length=1, tolerance=tolerance)
    matches = [attack._match(_single(prob, uid), instance) for uid in range(1, 5)]

    assert 1.0 / sum(matches) == expected


@pytest.mark.parametrize("tolerance,expected", [(0.0, 1.0 / 2.0), (1.0, 1.0 / 3.0)])
def test_location_proportion_match(privacy_frequency_tdf, tolerance, expected):
    freq = attacks._frequency_vector(privacy_frequency_tdf)
    instance = _records(freq)[1:3]
    attack = attacks.LocationProportionAttack(knowledge_length=1, tolerance=tolerance)
    matches = [attack._match(_single(freq, uid), instance) for uid in range(1, 5)]

    assert 1.0 / sum(matches) == expected


def test_home_work_match(privacy_frequency_tdf):
    freq = attacks._frequency_vector(privacy_frequency_tdf)
    instance = _records(_single(freq, 1))[:2]
    attack = attacks.HomeWorkAttack()
    matches = [attack._match(_single(freq, uid), instance) for uid in range(1, 5)]

    assert 1.0 / sum(matches) == 1.0 / 2.0


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


@pytest.mark.skmob
@pytest.mark.parametrize(
    ("skmob2_factory", "skmob_factory"),
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
def test_assess_risk_matches_skmob(privacy_tdf, skmob2_factory, skmob_factory):
    skmob_attacks = pytest.importorskip("skmob.privacy.attacks")
    skmob_core = pytest.importorskip("skmob.core.trajectorydataframe")

    skmob_tdf = skmob_core.TrajDataFrame(privacy_tdf, user_id="uid")
    expected = _risk_map(skmob_factory(skmob_attacks).assess_risk(skmob_tdf))
    actual = _risk_map(skmob2_factory().assess_risk(privacy_tdf))

    assert actual.keys() == expected.keys()
    for uid, risk in expected.items():
        assert actual[uid] == pytest.approx(risk)
