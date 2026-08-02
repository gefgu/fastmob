"""Activity distribution, transition matrix, and motif comparison."""

from __future__ import annotations

from collections import Counter
from typing import Any

import narwhals as nw

from fastmob.utils._common import ACTIVITY_CANDIDATES, _pick_existing_column

from ._utils import _is_null, _series_values
from .metrics import jensen_shannon_divergence, matrix_jensen_shannon_divergence


def activity_distribution_jensen_shannon_divergence(df1: Any, df2: Any, activity_col: str | None = None) -> float:
    """Compare categorical activity distributions with Jensen-Shannon divergence."""
    n1 = nw.from_native(df1, eager_only=True)
    n2 = nw.from_native(df2, eager_only=True)
    activity_col = activity_col or _pick_existing_column(n1.columns, ACTIVITY_CANDIDATES)
    if activity_col is None or activity_col not in n1.columns or activity_col not in n2.columns:
        raise ValueError(f"Could not detect activity column in both dataframes. Tried: {ACTIVITY_CANDIDATES}.")
    vals1 = [v for v in _series_values(n1, activity_col) if not _is_null(v)]
    vals2 = [v for v in _series_values(n2, activity_col) if not _is_null(v)]
    c1 = Counter(vals1)
    c2 = Counter(vals2)
    labels = sorted(set(c1) | set(c2), key=lambda value: str(value))
    return jensen_shannon_divergence([c1[label] for label in labels], [c2[label] for label in labels])


def activity_transition_matrix_jensen_shannon_divergence(
    matrix1: Any,
    matrix2: Any,
    categories1: list[Any] | None = None,
    categories2: list[Any] | None = None,
) -> float:
    """Compare activity transition matrices with Jensen-Shannon divergence."""
    return matrix_jensen_shannon_divergence(matrix1, matrix2, categories1, categories2)


def motif_distribution_jensen_shannon_divergence(
    daily1: Any,
    daily2: Any,
    *,
    motif_id_col: str = "motif_id",
) -> float:
    """Compare two daily-motif results (as returned by ``daily_motifs`` or
    ``daily_motifs_from_staypoints``) with Jensen-Shannon divergence.
    """
    n1 = nw.from_native(daily1, eager_only=True)
    n2 = nw.from_native(daily2, eager_only=True)
    if motif_id_col not in n1.columns or motif_id_col not in n2.columns:
        raise ValueError(f"Motif ID column {motif_id_col!r} does not exist in both dataframes.")
    motifs1 = _series_values(n1, motif_id_col)
    motifs2 = _series_values(n2, motif_id_col)
    c1 = Counter(motifs1)
    c2 = Counter(motifs2)
    labels = sorted(set(c1) | set(c2), key=lambda value: str(value))
    return jensen_shannon_divergence([c1[label] for label in labels], [c2[label] for label in labels])
