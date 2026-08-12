"""Comparison tests: fastmob.integration vs cached PyMove reference output.

Compares against ``pymove.utils.integration.join_with_pois`` /
``join_with_pois_by_category`` / ``join_with_events``, run once (see
``tests/populate_pymove_cache.py``) against the exact fixtures baked into
PyMove's own docstring examples. fastmob's test suite never imports pymove
itself at test time, only the committed parquet cache -- pymove needs a
dedicated, Python<=3.10, pandas<1.4 ``.venv-pymove`` incompatible with the
normal ``.venv`` (see ``benchmarks/environments/pymove.txt``).
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.integration.events import join_with_events
from fastmob.integration.poi import join_with_pois, join_with_pois_by_category

pytestmark = pytest.mark.pymove


def test_join_with_pois_matches_pymove(pymove_poi_reference):
    input_df = pymove_poi_reference.input_traj()
    pois = pymove_poi_reference.pois()
    reference = pymove_poi_reference.join_with_pois()

    result = join_with_pois(input_df, pois, poi_lng_col="lon")

    assert list(result["id_poi"]) == list(reference["id_poi"])
    assert list(result["name_poi"]) == list(reference["name_poi"])
    assert list(result["dist_poi"]) == pytest.approx(list(reference["dist_poi"]), abs=1.0)


def test_join_with_pois_by_category_matches_pymove(pymove_poi_reference):
    input_df = pymove_poi_reference.input_traj()
    pois = pymove_poi_reference.pois()
    reference = pymove_poi_reference.join_with_pois_by_category()

    result = join_with_pois_by_category(input_df, pois, poi_lng_col="lon")

    for category in ("policia", "comercio"):
        assert list(result[f"id_{category}"]) == list(reference[f"id_{category}"])
        assert list(result[f"dist_{category}"]) == pytest.approx(list(reference[f"dist_{category}"]), abs=1.0)


def test_join_with_events_matches_pymove(pymove_events_reference):
    input_df = pymove_events_reference.input_traj().assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
    events = pymove_events_reference.events()
    reference = pymove_events_reference.join_with_events()

    result = join_with_events(input_df, events, event_lng_col="lon", time_window_s=900)

    # PyMove leaves unmatched event_id/event_type as an empty string (its
    # column started as ``''``-filled and was only overwritten where an
    # event actually matched); fastmob reports a real null instead (see
    # CLAUDE.md's null-handling convention) -- both mean "no event in window".
    ref_event_id = [None if v == "" else v for v in reference["event_id"]]
    result_event_id = [None if pd.isna(v) else v for v in result["event_id"]]
    assert result_event_id == ref_event_id

    assert list(result["dist_event"]) == pytest.approx(list(reference["dist_event"]), abs=1.0)
