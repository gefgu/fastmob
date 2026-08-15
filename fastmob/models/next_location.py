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
    _extract_timestamp_arrow,
    _factorize_arrow_values,
    _pick_existing_column,
    _uint32_series,
)


def _batched_tail_contexts(
    codes_array: pa.Array, indices: pa.Array, ends: pa.Array, order: int
) -> tuple[pa.Array, pa.Array, pa.Array]:
    """Vectorized equivalent of ``[seq[-order:] for seq in per_user_sequences]``.

    Extracts every user's trailing ``order`` location codes directly from
    the flat ``(codes_array, indices, ends)`` triple :meth:`NextLocationPredictor.fit`
    already builds, without ever materializing a full per-user Python list.
    The whole computation is `numpy` array arithmetic plus a single batched
    `pyarrow.compute.take`, so its cost is ``O(n_users * order)`` rather
    than ``O(total_rows)`` -- unlike building the full per-user sequence
    dict (see `NextLocationPredictor._sequences`), which used to be done
    eagerly in `fit()` purely to serve this one small-context need.
    """
    ends_np = np.asarray(ends.to_numpy(zero_copy_only=False), dtype=np.int64)
    n_users = len(ends_np)
    starts_np = np.empty(n_users, dtype=np.int64)
    if n_users:
        starts_np[0] = 0
        starts_np[1:] = ends_np[:-1]

    ctx_starts = np.maximum(starts_np, ends_np - order) if order > 0 else ends_np.copy()
    ctx_lengths = ends_np - ctx_starts

    context_ends_cum = np.cumsum(ctx_lengths)
    context_starts_cum = context_ends_cum - ctx_lengths
    total = int(ctx_lengths.sum())

    if total == 0:
        context_codes = pa.array([], type=pa.uint32())
    else:
        within_offset = np.arange(total, dtype=np.int64) - np.repeat(context_starts_cum, ctx_lengths)
        flat_positions = np.repeat(ctx_starts, ctx_lengths) + within_offset
        # Gather via pc.take (Arrow-native, O(total) in the *small* context
        # size) rather than `indices.to_numpy()`, which would materialize
        # the full O(total_rows)-sized `indices` array on every call --
        # exactly the eager-materialization cost this function exists to
        # avoid (see docstring). `indices`/`codes_array` stay Arrow arrays
        # throughout; only the tiny gathered result ever touches Python.
        flat_positions_arr = pa.array(flat_positions, type=pa.uint64())
        context_row_indices = pc.take(indices, flat_positions_arr)
        context_codes = pc.take(codes_array, context_row_indices)

    return (
        context_codes,
        pa.array(context_starts_cum, type=pa.uint64()),
        pa.array(context_ends_cum, type=pa.uint64()),
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
        location_values = df.get_column(location_col).to_arrow()
        location_codes_arrow, representatives = _factorize_arrow_values(location_values, sort=False)
        code_to_label = dict(enumerate(pc.take(location_values, representatives).to_pylist()))
        location_codes = _uint32_series(df, location_codes_arrow)
        df = df.with_columns(location_codes.alias("__location_code__"))

        if started_at_col is not None:
            timestamps = _extract_timestamp_arrow(df, started_at_col)
            uid_values, indices, ends = _build_indexed_user_ranges(df, uid_col, timestamps)
        else:
            uid_values, ends = _build_presorted_user_ends(df, uid_col)
            import pyarrow as pa

            indices = pa.array(range(len(df)), type=pa.uint64())

        codes_array = df.get_column("__location_code__").to_arrow()

        self._models = NextLocationModels(codes_array, indices, ends, self.order, self.backoff)
        self._uid_values = uid_values.to_pylist() if uid_values is not None else [None]
        self._code_to_label = code_to_label
        self._backend = df.implementation
        self._codes_array = codes_array
        self._indices = indices
        self._ends = ends
        self._sequences_cache: dict[Any, list[int]] | None = None
        self.uid_col = uid_col
        self.location_col = location_col
        self._fitted = True
        return self

    @property
    def _sequences(self) -> dict[Any, list[int]]:
        """Full per-user location-code sequences, lazily materialized.

        Only :meth:`evaluate` needs each user's *entire* history (to slice
        progressively longer training windows across holdout steps);
        :meth:`predict_proba` only needs each user's short context tail and
        gets it from :func:`_batched_tail_contexts` instead, without ever
        touching this property. Building this dict eagerly in :meth:`fit`
        used to dominate its wall time -- one `pyarrow.compute.take(...).
        to_pylist()` call per user, proportional to total row count -- for
        the common case where `evaluate()` is never called.
        """
        if self._sequences_cache is None:
            end_values = self._ends.to_pylist()
            starts = [0, *end_values[:-1]]
            self._sequences_cache = {
                uid: pc.take(self._codes_array, self._indices.slice(start, end - start)).to_pylist()
                for uid, start, end in zip(self._uid_values, starts, end_values, strict=True)
            }
        return self._sequences_cache

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
        context_codes, context_starts, context_ends = _batched_tail_contexts(
            self._codes_array, self._indices, self._ends, self.order
        )

        pred_codes, pred_probs, out_starts, out_ends = self._models.predict_batch(
            context_codes, context_starts, context_ends, top_k
        )

        out_uid: list[Any] = []
        out_rank: list[int] = []
        out_label: list[Any] = []
        out_prob: list[float] = []
        pred_code_values = pred_codes.to_pylist()
        pred_prob_values = pred_probs.to_pylist()
        output_starts = out_starts.to_pylist()
        output_ends = out_ends.to_pylist()
        for i, uid in enumerate(self._uid_values):
            start, end = output_starts[i], output_ends[i]
            for rank, (code, prob) in enumerate(zip(pred_code_values[start:end], pred_prob_values[start:end]), start=1):
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
                pa.array(codes_flat, type=pa.uint32()),
                indices,
                pa.array(ends, type=pa.uint64()),
                self.order,
                self.backoff,
            )
            pred_codes, _pred_probs, out_starts, out_ends = models.predict_batch(
                pa.array(context_flat, type=pa.uint32()),
                pa.array(context_starts, type=pa.uint64()),
                pa.array(context_ends, type=pa.uint64()),
                top_k,
            )

            prediction_values = pred_codes.to_pylist()
            output_starts = out_starts.to_pylist()
            output_ends = out_ends.to_pylist()
            for i in range(len(eligible)):
                preds = prediction_values[output_starts[i] : output_ends[i]]
                if truths[i] in preds:
                    correct += 1
                total += 1

        return correct / total if total else float("nan")
