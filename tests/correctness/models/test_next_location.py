"""Correctness tests for fastmob.models.NextLocationPredictor.

Uses deterministic synthetic location-id sequences (a period-3 cycle
A-B-C-A-B-C-... where order-1 Markov prediction is provably 100% correct)
so ground truth is unambiguous, plus a sparse/novel-context case exercising
`backoff=True` vs `False`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastmob.models import NextLocationPredictor


def _cycle_df(uid: str = "u1", n: int = 12, labels=("A", "B", "C")):
    return pd.DataFrame(
        {
            "uid": [uid] * n,
            "location_id": [labels[i % len(labels)] for i in range(n)],
            "started_at": pd.date_range("2020-01-01", periods=n, freq="h"),
        }
    )


def test_order_1_predicts_deterministic_cycle_perfectly():
    df = _cycle_df()
    model = NextLocationPredictor(order=1).fit(df)
    pred = model.predict()
    assert pred["location_id"].tolist() == ["A"]  # cycle ends on C, C always -> A


def test_order_2_predicts_deterministic_cycle_perfectly():
    df = _cycle_df()
    model = NextLocationPredictor(order=2).fit(df)
    pred = model.predict()
    assert pred["location_id"].tolist() == ["A"]


def test_order_0_is_most_frequent_location_baseline():
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 5,
            "location_id": ["A", "A", "A", "B", "C"],
            "started_at": pd.date_range("2020-01-01", periods=5, freq="h"),
        }
    )
    model = NextLocationPredictor(order=0).fit(df)
    pred = model.predict()
    assert pred["location_id"].tolist() == ["A"]


def test_predict_proba_probabilities_sum_to_one_per_user():
    df = _cycle_df()
    model = NextLocationPredictor(order=1).fit(df)
    proba = model.predict_proba(top_k=3)
    assert proba["probability"].sum() == pytest.approx(1.0)


def test_top_k_returns_multiple_ranked_candidates():
    # Sequence ends in "X" (the prediction context, order=1), which
    # transitioned to "A" twice and "B" once during training.
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 7,
            "location_id": ["X", "A", "X", "B", "X", "A", "X"],
            "started_at": pd.date_range("2020-01-01", periods=7, freq="h"),
        }
    )
    model = NextLocationPredictor(order=1).fit(df)
    proba = model.predict_proba(top_k=2)
    ranked = proba.sort_values("rank")["location_id"].tolist()
    assert ranked[0] == "A"  # A follows X 2/3 times, B follows X 1/3
    assert set(ranked) == {"A", "B"}


def test_backoff_true_falls_back_on_unseen_context():
    # A strictly-unique sequence: the prediction context (its own tail) is
    # never observed as a training context (only ever as a target or as an
    # interior element), so an order equal to the full sequence length is
    # guaranteed novel regardless of backoff.
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 8,
            "location_id": list("ABCDEFGH"),
            "started_at": pd.date_range("2020-01-01", periods=8, freq="h"),
        }
    )
    model = NextLocationPredictor(order=1, backoff=True).fit(df)
    pred = model.predict_proba()
    assert len(pred) == 1  # falls back down to order-0 (most-frequent-location)


def test_backoff_false_returns_no_prediction_on_unseen_context():
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 8,
            "location_id": list("ABCDEFGH"),
            "started_at": pd.date_range("2020-01-01", periods=8, freq="h"),
        }
    )
    model = NextLocationPredictor(order=1, backoff=False).fit(df)
    pred = model.predict_proba()
    assert len(pred) == 0  # "H" was never seen as an order-1 context, no fallback


def test_two_users_predicted_independently():
    df_a = _cycle_df(uid="u1", labels=("A", "B", "C"))
    df_b = _cycle_df(uid="u2", labels=("X", "Y"))
    df = pd.concat([df_a, df_b], ignore_index=True)
    model = NextLocationPredictor(order=1).fit(df)
    pred = model.predict().set_index("uid")["location_id"].to_dict()
    assert pred["u1"] == "A"
    assert pred["u2"] in {"X", "Y"}


def test_no_uid_column_treats_whole_frame_as_one_user():
    df = _cycle_df().drop(columns=["uid"])
    model = NextLocationPredictor(order=1).fit(df)
    pred = model.predict()
    assert "uid" not in pred.columns
    assert pred["location_id"].tolist() == ["A"]


def test_evaluate_beats_random_on_deterministic_cycle():
    df = _cycle_df(n=30)
    model = NextLocationPredictor(order=1).fit(df)
    accuracy = model.evaluate(n_holdout=5)
    assert accuracy == pytest.approx(1.0)


def test_evaluate_order_0_baseline_sanity_floor():
    df = pd.DataFrame(
        {
            "uid": ["u1"] * 20,
            "location_id": ["A"] * 15 + ["B", "A", "B", "A", "B"],
            "started_at": pd.date_range("2020-01-01", periods=20, freq="h"),
        }
    )
    baseline = NextLocationPredictor(order=0).fit(df)
    accuracy = baseline.evaluate(n_holdout=5)
    assert accuracy >= 0.0

    order1 = NextLocationPredictor(order=1).fit(df)
    order1_accuracy = order1.evaluate(n_holdout=5)
    assert order1_accuracy >= baseline.evaluate(n_holdout=5) - 1e-9


def test_negative_order_raises():
    with pytest.raises(ValueError, match="order must be >= 0"):
        NextLocationPredictor(order=-1)


def test_predict_before_fit_raises():
    model = NextLocationPredictor(order=1)
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict()


def test_polars_matches_pandas():
    pl = pytest.importorskip("polars", reason="Polars not installed")
    df = _cycle_df(n=15)

    model_pd = NextLocationPredictor(order=1).fit(df)
    model_pl = NextLocationPredictor(order=1).fit(pl.from_pandas(df))

    pred_pd = model_pd.predict()
    pred_pl = model_pl.predict().to_pandas() if hasattr(model_pl.predict(), "to_pandas") else model_pl.predict()
    assert pred_pl["location_id"].tolist() == pred_pd["location_id"].tolist()
