"""Generic grouped-comparison engine backing ``BaseDataFrame.compare_to()``.

Replaces the earlier per-grouping-semantic wrapper functions
(``column_distribution_wasserstein_distance``, ``visits_per_user_wasserstein_distance``,
``dwell_time_wasserstein_distance``, ...): grouping is just "an existing column",
and the metric is a pluggable callable. Fine for a handful of groups
(``day_period``, ``purpose``, weekday); not intended for high-cardinality grouping
(e.g. ``group_col="uid"`` with thousands of users) -- pre-aggregate first in that
case, then compare the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import narwhals as nw

from fastmob.utils._common import _finite_arrow_array

from .metrics import wasserstein_distance


@dataclass(frozen=True)
class ComparisonResult:
    """Result of :func:`compare_to`: an overall value and, if grouped, a per-group breakdown."""

    overall: float
    groups: list[tuple[Any, float]] | None = None


def _require_column(df: nw.DataFrame, column: str, other: nw.DataFrame) -> None:
    if column not in df.columns or column not in other.columns:
        raise ValueError(f"Column {column!r} must be present in both dataframes.")


def compare_to(
    df1: Any,
    df2: Any,
    value_col: str,
    *,
    group_col: str | None = None,
    metric: Callable[[Any, Any], float] | None = None,
) -> ComparisonResult:
    """Compare ``value_col`` between two dataframes, optionally grouped by ``group_col``.

    ``metric`` defaults to :func:`wasserstein_distance` and accepts any
    ``(array, array) -> float`` callable, e.g. :func:`jensen_shannon_divergence`.
    When ``group_col`` is given, the comparison runs once per label present in
    both dataframes (sorted for determinism) and ``overall`` is the mean across
    groups.
    """
    metric = metric or wasserstein_distance
    n1 = nw.from_native(df1, eager_only=True)
    n2 = nw.from_native(df2, eager_only=True)
    _require_column(n1, value_col, n2)

    if group_col is None:
        v1 = _finite_arrow_array(n1.get_column(value_col))
        v2 = _finite_arrow_array(n2.get_column(value_col))
        return ComparisonResult(overall=float(metric(v1, v2)))

    _require_column(n1, group_col, n2)
    labels1 = set(n1.get_column(group_col).unique().to_list())
    labels2 = set(n2.get_column(group_col).unique().to_list())
    labels = sorted((labels1 & labels2) - {None}, key=str)

    groups: list[tuple[Any, float]] = []
    for label in labels:
        v1 = _finite_arrow_array(n1.filter(nw.col(group_col) == label).get_column(value_col))
        v2 = _finite_arrow_array(n2.filter(nw.col(group_col) == label).get_column(value_col))
        if len(v1) and len(v2):
            groups.append((label, float(metric(v1, v2))))

    if not groups:
        return ComparisonResult(overall=float("nan"), groups=None)
    overall = sum(value for _, value in groups) / len(groups)
    return ComparisonResult(overall=overall, groups=groups)
