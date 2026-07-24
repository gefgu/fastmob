"""Transport-mode classification and modal-split aggregation for Triplegs.

Ported from trackintel's `predict_transport_mode`/`calculate_modal_split`
(`analysis/labelling.py`, `analysis/modal_split.py`). Operates on the small,
already-aggregated per-tripleg summary table (one row per tripleg, never per
raw GPS fix), so -- like `fastmob.preprocessing.cluster`/`stay_locations`/
`segment` before it -- this is plain Narwhals, no new Rust kernel needed;
CLAUDE.md's mandatory Rust-kernel workflow applies to `fastmob/measures/*`,
not every public function.
"""

from __future__ import annotations

import math
from typing import Any

import narwhals as nw

# trackintel's own default thresholds are expressed in m/s (15/3.6,
# 100/3.6); Triplegs' speed column is already km/h, so the equivalent
# thresholds are expressed directly in km/h here (15 km/h ~ upper bound of
# walking/biking, 100 km/h ~ upper bound of typical car travel).
DEFAULT_MODE_CATEGORIES: dict[float, str] = {
    15.0: "slow_mobility",
    100.0: "motorized_mobility",
    math.inf: "fast_mobility",
}

_FREQ_TRUNCATE_UNITS: dict[str, str] = {
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
}


def _as_native_and_wrapper(triplegs: Any) -> tuple[Any, Any]:
    """Return ``(native_df, triplegs_or_None)`` -- ``triplegs`` is the original
    wrapper object when one was passed (so callers can rebuild the same type),
    or ``None`` when a plain dataframe was passed directly.
    """
    if hasattr(triplegs, "df"):
        return triplegs.df, triplegs
    return triplegs, None


def predict_transport_mode(
    triplegs: Any,
    method: str = "simple-coarse",
    categories: dict[float, str] | None = None,
    *,
    speed_col: str = "mean_speed_kmh",
) -> Any:
    """Classify each tripleg's transport mode from its average speed.

    Parameters
    ----------
    triplegs : Triplegs or DataFrame-like
        A `fastmob.core.Triplegs` instance, or any Narwhals-compatible
        dataframe with a ``speed_col`` column.
    method : str, optional
        Only ``"simple-coarse"`` (speed-threshold binning) is implemented.
    categories : dict[float, str], optional
        Mapping of upper-bound speed threshold (km/h) to mode label,
        evaluated in ascending order (the last key should be ``math.inf``
        to catch every remaining row). Defaults to
        :data:`DEFAULT_MODE_CATEGORIES`.
    speed_col : str, optional
        Column holding each row's average speed, km/h. Default
        ``"mean_speed_kmh"`` (Triplegs' own column).

    Returns
    -------
    Triplegs or DataFrame-like
        Input with an added ``mode`` column, in the same type/backend as
        input.

    References
    ----------
    - Martin, F. et al. (2023) trackintel: ``predict_transport_mode``.
    """
    if method != "simple-coarse":
        raise ValueError(f"unknown transport-mode method: {method!r}; choose from ['simple-coarse']")

    cats = DEFAULT_MODE_CATEGORIES if categories is None else categories
    sorted_items = sorted(cats.items(), key=lambda kv: kv[0])

    native_df, wrapper = _as_native_and_wrapper(triplegs)
    nw_df = nw.from_native(native_df, eager_only=True)

    mode_expr = None
    for threshold, label in reversed(sorted_items):
        if mode_expr is None:
            mode_expr = nw.lit(label)
        else:
            mode_expr = nw.when(nw.col(speed_col) <= threshold).then(nw.lit(label)).otherwise(mode_expr)

    result_native = nw_df.with_columns(mode_expr.alias("mode")).to_native()

    if wrapper is not None:
        from ..core.triplegs_dataframe import Triplegs

        return Triplegs(result_native, uid_col=wrapper.uid_col, validate=False)
    return result_native


predict_transport_mode.__module__ = "fastmob.preprocessing"


def calculate_modal_split(
    triplegs: Any,
    freq: str | None = None,
    metric: str = "count",
    per_user: bool = False,
    normalize: bool = False,
) -> Any:
    """Aggregate a ``mode``-labeled Triplegs table into a modal-split table.

    Parameters
    ----------
    triplegs : Triplegs or DataFrame-like
        Must already have a ``mode`` column (see :func:`predict_transport_mode`).
    freq : str, optional
        Time-bucket the split by ``started_at``: one of ``"hour"``,
        ``"day"``, ``"week"``, ``"month"``. ``None`` (default): no time
        bucketing.
    metric : str, optional
        ``"count"`` (default, number of triplegs), ``"distance"`` (sum of
        ``length_km``), or ``"duration"`` (sum of ``duration_s``).
    per_user : bool, optional
        When True, split per user as well (requires a ``uid`` column).
        Default False.
    normalize : bool, optional
        When True, normalize each row-group (excluding ``mode`` itself) to
        sum to 1. Default False.

    Returns
    -------
    DataFrame-like
        One row per ``(mode`` [, ``uid``] [, time bucket]``)``, with a
        ``value`` column.
    """
    if metric not in ("count", "distance", "duration"):
        raise ValueError(f"unknown modal-split metric: {metric!r}; choose from ['count', 'distance', 'duration']")
    if freq is not None and freq not in _FREQ_TRUNCATE_UNITS:
        raise ValueError(f"unknown modal-split freq: {freq!r}; choose from {sorted(_FREQ_TRUNCATE_UNITS)}")

    native_df, wrapper = _as_native_and_wrapper(triplegs)
    nw_df = nw.from_native(native_df, eager_only=True)
    if "mode" not in nw_df.columns:
        raise ValueError("calculate_modal_split requires a 'mode' column; run predict_transport_mode first")

    uid_col = getattr(wrapper, "uid_col", None)
    group_cols: list[str] = ["mode"]
    if per_user:
        if uid_col is None:
            raise ValueError("per_user=True requires triplegs to have a uid column")
        group_cols = [uid_col, *group_cols]
    if freq is not None:
        unit = _FREQ_TRUNCATE_UNITS[freq]
        nw_df = nw_df.with_columns(nw.col("started_at").dt.truncate(unit).alias("__freq_bucket__"))
        group_cols = ["__freq_bucket__", *group_cols]

    if metric == "count":
        agg_expr = nw.len().alias("value")
    elif metric == "distance":
        agg_expr = nw.col("length_km").sum().alias("value")
    else:
        agg_expr = nw.col("duration_s").sum().alias("value")

    result = nw_df.group_by(group_cols).agg(agg_expr)

    if normalize:
        norm_group_cols = [col for col in group_cols if col != "mode"]
        if norm_group_cols:
            totals = result.group_by(norm_group_cols).agg(nw.col("value").sum().alias("__total__"))
            result = (
                result.join(totals, on=norm_group_cols, how="left")
                .with_columns((nw.col("value") / nw.col("__total__")).alias("value"))
                .drop("__total__")
            )
        else:
            total = float(result.get_column("value").to_numpy().sum())
            result = result.with_columns((nw.col("value") / total).alias("value"))

    return result.to_native()


calculate_modal_split.__module__ = "fastmob.preprocessing"
