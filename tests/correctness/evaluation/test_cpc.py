from __future__ import annotations

import fastmob
import numpy as np
import pandas as pd
import polars as pl
import pytest


def _trips(rows):
    return fastmob.Trips(
        pd.DataFrame(
            {
                "trip_id": range(len(rows)),
                "started_at": pd.Timestamp("2026-01-01"),
                "finished_at": pd.Timestamp("2026-01-01 01:00"),
                "origin_staypoint_id": [None] * len(rows),
                "destination_staypoint_id": [None] * len(rows),
                "tripleg_ids": [[] for _ in rows],
                "origin_location_id": [row[0] for row in rows],
                "destination_location_id": [row[1] for row in rows],
            }
        )
    )


def _numeric_trips(rows, dtype=np.uint64):
    return fastmob.Trips(
        pd.DataFrame(
            {
                "trip_id": range(len(rows)),
                "started_at": pd.Timestamp("2026-01-01"),
                "finished_at": pd.Timestamp("2026-01-01 01:00"),
                "origin_staypoint_id": [None] * len(rows),
                "destination_staypoint_id": [None] * len(rows),
                "tripleg_ids": [[] for _ in rows],
                "origin_location_id": np.array([row[0] for row in rows], dtype=dtype),
                "destination_location_id": np.array([row[1] for row in rows], dtype=dtype),
            }
        )
    )


def test_trip_cpc_uses_global_location_ids():
    left = _trips([("home", "work"), ("home", "work"), ("work", "home")])
    right = _trips([("home", "work"), ("park", "work")])
    assert left.common_part_of_commuters(right) == pytest.approx(0.4)
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)


def test_flow_cpc_aggregates_duplicate_edges_and_ignores_self_loops():
    left = fastmob.FlowDataFrame({"origin": [1, 1, 2], "destination": [2, 2, 2], "flow": [2, 3, 5]})
    right = fastmob.FlowDataFrame({"origin": [1], "destination": [2], "flow": [4]})
    assert left.common_part_of_commuters(right) == pytest.approx(8 / 9)


def test_trips_can_be_explicitly_materialized_as_flows():
    flows = _trips([(1, 2), (1, 2), (2, 2), (None, 1)]).to_flow_dataframe()
    assert flows.get_flow(1, 2) == 2


def test_trip_cpc_numeric_ids_skip_factorization_and_match_string_equivalent():
    # Same topology as test_trip_cpc_uses_global_location_ids (score 0.4), but
    # with H3-cell-scale uint64 ids -- past uint32::MAX -- to exercise the
    # numeric fast path that skips _joint_factorize_arrow_values entirely.
    big = 1 << 40
    home, work, park = big, big + 1, big + 2
    left = _numeric_trips([(home, work), (home, work), (work, home)])
    right = _numeric_trips([(home, work), (park, work)])
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)
    # left has 2 unique links {(home,work),(work,home)}, right has 2 unique
    # links {(home,work),(park,work)}; only (home,work) is shared -> 2*1/4.
    assert fastmob.common_part_of_links(left, right) == pytest.approx(0.5)


def test_trip_cpc_mixed_unsigned_width_uses_numeric_fast_path():
    # Differently-widthed unsigned columns (uint32 vs uint64) both qualify for
    # the numeric fast path once cast to a common uint64 type -- this used to
    # be exactly the case _joint_factorize_arrow_values rejects outright
    # ("jointly factorized columns must use the same Arrow data type").
    left = _numeric_trips([(1, 2), (1, 2), (2, 1)], dtype=np.uint32)
    right = _numeric_trips([(1, 2), (3, 2)], dtype=np.uint64)
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)


def test_trip_cpc_string_ids_still_use_factorization_fallback():
    # Non-numeric ids (e.g. "home"/"work") cannot skip factorization -- this
    # is the same scenario as test_trip_cpc_uses_global_location_ids, kept
    # here to make the fallback path explicit alongside the fast-path tests.
    left = _trips([("home", "work"), ("home", "work"), ("work", "home")])
    right = _trips([("home", "work"), ("park", "work")])
    assert fastmob.common_part_of_commuters(left, right) == pytest.approx(0.4)


def _distance_trips(distances):
    trips = _trips([(0, 1)] * len(distances)).to_pandas()
    trips["distance_km"] = distances
    return fastmob.Trips(trips)


def test_cpcd_is_one_for_identical_distance_distributions():
    distances = [0.5, 1.0, 3.0, 7.0, 7.5, 12.0]

    assert fastmob.measures.evaluation.common_part_of_commuters_distance(
        _distance_trips(distances), _distance_trips(distances)
    ) == pytest.approx(1.0)


def test_cpcd_counts_trips_per_two_kilometre_bin():
    observed = _distance_trips([1.0, 1.0, 3.0, 9.0])  # bins 0, 0, 1, 4
    predicted = _distance_trips([1.0, 3.0, 3.5, 3.9])  # bins 0, 1, 1, 1

    # min counts per bin: bin0 min(2, 1) + bin1 min(1, 3) + bin4 min(1, 0) = 2 of 4 observed trips
    assert fastmob.measures.evaluation.common_part_of_commuters_distance(observed, predicted) == pytest.approx(0.5)


# --- ordinary dataframes and detected column names (issues #16 and #17) ---

_OBSERVED = {"origin": ["a", "a", "b"], "destination": ["b", "c", "c"], "flow": [10, 5, 7]}
_PREDICTED = {"origin": ["a", "a", "b"], "destination": ["b", "c", "c"], "flow": [8, 5, 9]}


def _expected():
    return fastmob.common_part_of_commuters(fastmob.FlowDataFrame(_OBSERVED), fastmob.FlowDataFrame(_PREDICTED))


@pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
def test_cpc_accepts_plain_dataframes(backend):
    score = fastmob.common_part_of_commuters(backend(_OBSERVED), backend(_PREDICTED))
    assert score == pytest.approx(_expected())
    assert score == pytest.approx(2 * (8 + 5 + 7) / (22 + 22))
    assert fastmob.common_part_of_links(backend(_OBSERVED), backend(_PREDICTED)) == pytest.approx(1.0)


def test_cpc_mixes_flow_dataframe_and_plain_dataframe():
    score = fastmob.common_part_of_commuters(fastmob.FlowDataFrame(_OBSERVED), pd.DataFrame(_PREDICTED))
    assert score == pytest.approx(_expected())


def test_cpc_detects_od_matrix_style_names_independently_per_table():
    census = pd.DataFrame({"area_o": ["a", "a", "b"], "area_d": ["b", "c", "c"], "weight": [8, 5, 9]})
    wrapped = fastmob.FlowDataFrame(
        pd.DataFrame({"origin_area": ["a", "a", "b"], "destination_area": ["b", "c", "c"], "count": [10, 5, 7]})
    )
    assert fastmob.common_part_of_commuters(wrapped, census) == pytest.approx(_expected())
    assert fastmob.common_part_of_links(wrapped, census) == pytest.approx(1.0)


def test_cpc_explicit_column_names_resolve_ambiguity():
    table = pd.DataFrame({"origin": ["a"], "destination": ["b"], "count": [3], "weight": [100]})
    other = pd.DataFrame({"origin": ["a"], "destination": ["b"], "flow": [3]})
    with pytest.raises(ValueError, match="observed: ambiguous flow column"):
        fastmob.common_part_of_commuters(table, other)
    score = fastmob.common_part_of_commuters(table, other, observed_columns={"flow": "count"})
    assert score == pytest.approx(1.0)


def test_cpc_reports_missing_and_invalid_columns():
    good = pd.DataFrame(_OBSERVED)
    with pytest.raises(ValueError, match="predicted: could not find a flow column"):
        fastmob.common_part_of_commuters(good, good.drop(columns="flow"))
    with pytest.raises(ValueError, match="column 'nope' not found"):
        fastmob.common_part_of_commuters(good, good, predicted_columns={"origin": "nope"})
    with pytest.raises(ValueError, match="keys must be among"):
        fastmob.common_part_of_commuters(good, good, observed_columns={"count": "flow"})
    with pytest.raises(ValueError, match="must be numeric"):
        fastmob.common_part_of_commuters(good, good.assign(flow=["x", "y", "z"]))
    with pytest.raises(TypeError, match="eager dataframe"):
        fastmob.common_part_of_commuters(good, [[1, 2, 3]])
    with pytest.raises(TypeError, match="two Trips or two flow tables"):
        fastmob.common_part_of_commuters(_trips([("a", "b")]), good)


@pytest.mark.parametrize(
    "predicted_ids",
    [
        pd.Series(["a", "a", "b"], dtype="category"),
        pd.Series(["a", "a", "b"], dtype="string[pyarrow]"),
    ],
)
def test_cpc_unifies_string_like_id_types(predicted_ids):
    predicted = pd.DataFrame({"origin": predicted_ids, "destination": ["b", "c", "c"], "flow": [8, 5, 9]})
    assert fastmob.common_part_of_commuters(pd.DataFrame(_OBSERVED), predicted) == pytest.approx(_expected())


def test_cpc_unifies_integer_id_widths_and_empty_tables():
    wide = pd.DataFrame({"origin": np.array([1, 1], dtype=np.int64), "destination": [2, 3], "flow": [4, 4]})
    narrow = pd.DataFrame(
        {"origin": np.array([1, 1], dtype=np.int32), "destination": np.array([2, 3], dtype=np.uint8), "flow": [4, 2]}
    )
    assert fastmob.common_part_of_commuters(wide, narrow) == pytest.approx(2 * 6 / 14)
    empty = pd.DataFrame({"origin": [], "destination": [], "flow": []})
    assert fastmob.common_part_of_commuters(wide, empty) == 0.0


def test_cpc_rejects_ids_of_different_kinds_naming_the_types():
    numeric = pd.DataFrame({"origin": [1], "destination": [2], "flow": [1]})
    text = pd.DataFrame({"origin": ["1"], "destination": ["2"], "flow": [1]})
    with pytest.raises(ValueError, match="incompatible types.*observed origin is int64.*predicted origin is"):
        fastmob.common_part_of_commuters(numeric, text)


def test_cpc_input_checks_on_column_arguments():
    good = pd.DataFrame(_OBSERVED)
    with pytest.raises(TypeError, match="must be a mapping"):
        fastmob.common_part_of_commuters(good, good, observed_columns=["origin"])
    with pytest.raises(ValueError, match="three different columns"):
        fastmob.common_part_of_commuters(good, good, observed_columns={"destination": "origin"})
    odd = pd.DataFrame({"df": [1], "origin": ["a"], "destination": ["b"], "flow": [1]})
    assert fastmob.common_part_of_commuters(odd, odd) == pytest.approx(1.0)
