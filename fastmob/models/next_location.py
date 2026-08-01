"""Simple (non-deep-learning) next-location prediction.

`NextLocationPredictor` is an order-k Markov chain fit directly over a
user's own location-id *sequence*, used to *predict* the next entry in that
sequence. Deliberately kept separate from `markov_diary_generator.py`'s
`MarkovDiaryGenerator`, a fixed-48-state hour-of-day x home/away chain that
*generates* synthetic mobility diaries -- a different concept despite both
being "Markov chains over mobility."

Ported from HuMobi's `predictors.markov.MarkovChain` (order-k transition
counts with backoff to lower orders on an unseen context) and
`predictors.wrapper.TopLoc` (the order-0, most-frequent-location baseline,
which `NextLocationPredictor(order=0)` gets "for free" as the order-0
table).
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fastmob._core import NextLocationModels
from fastmob.utils._common import (
    LOCATION_CANDIDATES,
    UID_CANDIDATES,
    _build_indexed_user_ranges,
    _build_presorted_user_ends,
    _extract_timestamps_s,
    _factorize_arrow_values,
    _pick_existing_column,
    _uint64_series,
)


class NextLocationPredictor:
    """Order-k Markov next-location predictor with optional backoff.

    Parameters
    ----------
    order : int, optional
        Markov chain order: number of most-recent locations used as
        context. ``order=0`` degenerates to a most-frequent-location
        baseline (HuMobi's ``TopLoc``). Default ``1``.
    backoff : bool, optional
        When an exact order-``order`` context was never observed, fall back
        to progressively lower orders (down to order 0) instead of
        returning no prediction. Default ``True``.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.models import NextLocationPredictor
    >>> df = pd.DataFrame({
    ...     "uid": ["u1"] * 6,
    ...     "location_id": ["A", "B", "C", "A", "B", "C"],
    ...     "started_at": pd.date_range("2020-01-01", periods=6, freq="h"),
    ... })
    >>> model = NextLocationPredictor(order=1).fit(df)
    >>> model.predict()["location_id"].tolist()
    ['A']
    """

    def __init__(self, order: int = 1, backoff: bool = True):
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = order
        self.backoff = backoff
        self._fitted = False

    def fit(
        self,
        traj: Any,
        location_col: str | None = None,
        uid_col: str | None = None,
        started_at_col: str | None = None,
    ) -> NextLocationPredictor:
        """Fit one Markov model per user from each user's location-id sequence.

        Parameters
        ----------
        traj : DataFrame-like or Staypoints
            Source data; any Narwhals-compatible eager backend, or a
            `fastmob.core.Staypoints` instance.
        location_col : str, optional
            Column holding each row's location identifier. Auto-detected
            when None.
        uid_col : str, optional
            User-ID column. Auto-detected when None; when absent, the whole
            frame is treated as one user.
        started_at_col : str, optional
            Datetime column to sort each user's sequence by. When None,
            rows are assumed already in chronological order per user
            (matching `Staypoints`' own ``started_at_col``, when ``traj``
            is a `Staypoints` instance).

        Returns
        -------
        NextLocationPredictor
            ``self``, fitted.
        """
        native_df = traj.df if hasattr(traj, "df") else traj
        if location_col is None:
            location_col = getattr(traj, "location_id_col", None)
        if started_at_col is None:
            started_at_col = getattr(traj, "started_at_col", None)
        if uid_col is None:
            uid_col = getattr(traj, "uid_col", None)

        df = nw.from_native(native_df, eager_only=True)
        if location_col is None:
            location_col = _pick_existing_column(df.columns, LOCATION_CANDIDATES)
        if location_col is None:
            raise ValueError(f"Could not find a location column; checked {LOCATION_CANDIDATES}")
        if uid_col is None:
            uid_col = _pick_existing_column(df.columns, UID_CANDIDATES)

        df = df.filter(~nw.col(location_col).is_null())
        location_codes, code_to_label = _factorize_column(df, location_col)
        df = df.with_columns(location_codes.alias("__location_code__"))

        if started_at_col is not None:
            timestamps_s = _extract_timestamps_s(df, started_at_col)
            uid_values, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps_s.to_arrow())
        else:
            uid_values, ends = _build_presorted_user_ends(df, uid_col)
            import pyarrow as pa

            indices = pa.array(range(len(df)), type=pa.uint64())

        codes_array = df.get_column("__location_code__").to_numpy().astype(np.uint64)
        index_values = indices.to_pylist()
        end_values = ends.to_pylist()

        self._models = NextLocationModels(codes_array, indices, ends, self.order, self.backoff)
        self._uid_values = uid_values.to_pylist() if uid_values is not None else [None]
        self._code_to_label = code_to_label
        self._backend = df.implementation
        self._sequences = {
            uid: codes_array[index_values[start:end]].tolist()
            for uid, (start, end) in zip(self._uid_values, zip([0, *end_values[:-1]], end_values), strict=True)
        }
        self.uid_col = uid_col
        self.location_col = location_col
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before predict()/predict_proba()/evaluate()")

    def predict_proba(self, top_k: int = 1) -> Any:
        """Predict each fitted user's next location(s) from their own training history's tail.

        Parameters
        ----------
        top_k : int, optional
            Number of ranked candidates to return per user. Default ``1``.

        Returns
        -------
        DataFrame
            One row per ``(uid, rank)``: columns ``[uid_col, "rank",
            location_col, "probability"]``, ``rank`` starting at 1, in the
            same backend :meth:`fit` was given. A user with fewer than
            ``top_k`` observed candidates contributes fewer rows.
        """
        self._require_fitted()
        context_flat: list[int] = []
        context_starts: list[int] = []
        context_ends: list[int] = []
        cursor = 0
        for uid in self._uid_values:
            seq = self._sequences[uid]
            ctx = seq[-self.order :] if self.order > 0 else []
            context_starts.append(cursor)
            context_flat.extend(ctx)
            cursor += len(ctx)
            context_ends.append(cursor)

        pred_codes, pred_probs, out_starts, out_ends = self._models.predict_batch(
            np.asarray(context_flat, dtype=np.uint64),
            pa.array(context_starts, type=pa.uint64()),
            pa.array(context_ends, type=pa.uint64()),
            top_k,
        )

        out_uid: list[Any] = []
        out_rank: list[int] = []
        out_label: list[Any] = []
        out_prob: list[float] = []
        for i, uid in enumerate(self._uid_values):
            start, end = out_starts[i], out_ends[i]
            for rank, (code, prob) in enumerate(zip(pred_codes[start:end], pred_probs[start:end]), start=1):
                out_rank.append(rank)
                out_label.append(self._code_to_label[code])
                out_prob.append(float(prob))
                if self.uid_col is not None:
                    out_uid.append(uid)

        result_dict: dict[str, list] = {}
        if self.uid_col is not None:
            result_dict[self.uid_col] = out_uid
        result_dict["rank"] = out_rank
        result_dict[self.location_col] = out_label
        result_dict["probability"] = out_prob
        return nw.from_dict(result_dict, backend=self._backend).to_native()

    def predict(self, top_k: int = 1) -> Any:
        """Predict each fitted user's single most likely next location.

        Returns
        -------
        DataFrame
            One row per user: columns ``[uid_col, location_col]`` holding
            the top-ranked prediction, in the same backend :meth:`fit` was
            given (use :meth:`predict_proba` for ranked
            candidates/probabilities).
        """
        proba = nw.from_native(self.predict_proba(top_k=max(top_k, 1)), eager_only=True)
        top1 = proba.filter(nw.col("rank") == 1).drop("rank", "probability")
        return top1.to_native()

    def evaluate(self, top_k: int = 1, n_holdout: int = 1) -> float:
        """Holdout-last-``n_holdout`` accuracy over the data given to :meth:`fit`.

        For each user with more than ``n_holdout`` observations, walks
        forward through the held-out tail one step at a time: at each step,
        fits a model on every prior observation (train portion plus any
        already-revealed holdout steps -- never the true value being
        predicted) and checks whether the true next location is among the
        top-``top_k`` predictions. Accuracy is averaged over every
        (user, step) pair across every user and step.

        This evaluates directly against :meth:`fit`'s own input rather than
        a separate held-out dataframe, since fastmob's `Staypoints`/
        `Locations` pipeline already produces one canonical per-user
        location sequence -- there is no second, independently-loaded test
        sequence to compare against in the common case.

        Parameters
        ----------
        top_k : int, optional
            Number of ranked candidates checked per prediction. Default ``1``.
        n_holdout : int, optional
            Number of most-recent observations per user to hold out and
            evaluate against. Users with ``len(sequence) <= n_holdout`` are
            skipped. Default ``1``.

        Returns
        -------
        float
            Fraction of (user, holdout step) pairs where the true next
            location was among the top-``top_k`` predictions. ``NaN`` if no
            user has enough observations to evaluate.
        """
        self._require_fitted()
        if n_holdout < 1:
            raise ValueError("n_holdout must be >= 1")

        eligible = [uid for uid in self._uid_values if len(self._sequences[uid]) > n_holdout]
        if not eligible:
            return float("nan")

        correct = 0
        total = 0
        for step in range(n_holdout):
            codes_flat: list[int] = []
            ends: list[int] = []
            context_flat: list[int] = []
            context_starts: list[int] = []
            context_ends: list[int] = []
            truths: list[int] = []
            cursor = 0
            ctx_cursor = 0
            for uid in eligible:
                seq = self._sequences[uid]
                train_end = len(seq) - n_holdout + step
                train_seq = seq[:train_end]
                codes_flat.extend(train_seq)
                cursor += len(train_seq)
                ends.append(cursor)

                ctx = train_seq[-self.order :] if self.order > 0 else []
                context_starts.append(ctx_cursor)
                context_flat.extend(ctx)
                ctx_cursor += len(ctx)
                context_ends.append(ctx_cursor)

                truths.append(seq[train_end])

            indices = pa.array(range(len(codes_flat)), type=pa.uint64())
            models = NextLocationModels(
                np.asarray(codes_flat, dtype=np.uint64),
                indices,
                pa.array(ends, type=pa.uint64()),
                self.order,
                self.backoff,
            )
            pred_codes, _pred_probs, out_starts, out_ends = models.predict_batch(
                np.asarray(context_flat, dtype=np.uint64),
                pa.array(context_starts, type=pa.uint64()),
                pa.array(context_ends, type=pa.uint64()),
                top_k,
            )

            for i in range(len(eligible)):
                preds = pred_codes[out_starts[i] : out_ends[i]]
                if truths[i] in preds:
                    correct += 1
                total += 1

        return correct / total if total else float("nan")


def _factorize_column(df: nw.DataFrame, col: str) -> tuple[nw.Series, dict[int, Any]]:
    """Dense uint64-code a column, returning (codes, {code: original_label})."""
    values = df.get_column(col).to_arrow()
    codes, representatives = _factorize_arrow_values(values, sort=False)
    labels = pc.take(values, representatives).to_pylist()
    return _uint64_series(df, codes), dict(enumerate(labels))
