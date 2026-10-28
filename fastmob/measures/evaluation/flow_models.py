"""Compare flow models (gravity, radiation, ...) with observed origin-destination flows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fastmob.measures.evaluation.cpc import common_part_of_commuters, common_part_of_links


def _pairs(flows: Any) -> pa.Table:
    """Origin-destination pairs and their flow, without self-loops and with one row per pair."""
    from fastmob.core.flow_dataframe import DESTINATION, FLOW, ORIGIN

    frame = nw.from_native(flows.df, eager_only=True)
    table = pa.table(
        {
            "origin": frame.get_column(ORIGIN).to_arrow(),
            "destination": frame.get_column(DESTINATION).to_arrow(),
            "flow": pc.cast(frame.get_column(FLOW).to_arrow(), pa.float64()),
        }
    )
    table = table.filter(pc.not_equal(table["origin"], table["destination"]))
    return (
        table.group_by(["origin", "destination"])
        .aggregate([("flow", "sum")])
        .rename_columns(["origin", "destination", "flow"])
    )


def _deviance(y: np.ndarray, fitted: np.ndarray) -> float:
    fitted = np.clip(fitted, 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / fitted), 0.0)
    return float(2.0 * np.sum(term - (y - fitted)))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    sorted_values = values[order]
    start = 0
    for end in np.flatnonzero(np.diff(sorted_values)) + 1:  # average the ranks of ties
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    ranks[order[start:]] = (start + len(values) - 1) / 2.0
    return ranks


def compare_flow_models(observed: Any, models: Mapping[str, Any] | Any) -> pa.Table:
    """Score one or several flow models against observed origin-destination flows.

    Every model is scored on the same set of pairs: the union of the origin-destination pairs of the observed
    table and of all the models, leaving out self-loops. A pair that is missing from a table has flow 0 there,
    so the scores of different models can be compared with each other. The Poisson deviance and pseudo R-squared
    treat the flows as counts, so compare models whose total flow is on the scale of the observed one (for
    example generated with the observed outflows).

    Two cautions. The common part of links (CPL) compares which pairs have flow. Expected flows
    (``out_format="flows"``) are positive on every pair, so their CPL is fixed by the number of observed pairs and
    says nothing about the model; use sampled (integer) flows for CPL. And a model that was fitted to the observed
    flows is scored in-sample: compare it with a parameter-free model, or score it on flows that were held out,
    before concluding that it is better.

    Parameters
    ----------
    observed : FlowDataFrame
        Observed flows with ``origin``, ``destination`` and ``flow`` columns.
    models : FlowDataFrame or mapping of str to FlowDataFrame
        The model flows to score, by name. A single FlowDataFrame is named ``"model"``.

    Returns
    -------
    pyarrow.Table
        One row per model with ``model``, ``n_pairs`` (pairs in the union), ``observed_flow`` and
        ``model_flow`` (totals), ``cpc`` (common part of commuters, 0 to 1), ``cpl`` (common part of links),
        ``r_squared`` (squared Pearson correlation of the pair flows), ``spearman`` (rank correlation),
        ``poisson_deviance`` and ``pseudo_r_squared`` (one minus the deviance over the deviance of a model that
        predicts the mean observed flow for every pair).

    Raises
    ------
    TypeError
        If ``observed`` or a model is not a FlowDataFrame.

    See Also
    --------
    fastmob.models.Gravity.fit : Fit a gravity model and get confidence intervals.
    common_part_of_commuters : The CPC on its own.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob import FlowDataFrame
    >>> from fastmob.measures.evaluation import compare_flow_models
    >>> table = pd.DataFrame({"origin": ["a", "a", "b"], "destination": ["b", "c", "c"], "flow": [10, 5, 7]})
    >>> flows = FlowDataFrame(table, tile_id="tile_id")
    >>> compare_flow_models(flows, flows).column("cpc").to_pylist()
    [1.0]
    """
    from fastmob.core.flow_dataframe import FlowDataFrame

    if not isinstance(observed, FlowDataFrame):
        raise TypeError("observed must be a FlowDataFrame")
    named = {"model": models} if isinstance(models, FlowDataFrame) else dict(models)
    if not all(isinstance(m, FlowDataFrame) for m in named.values()):
        raise TypeError("every model must be a FlowDataFrame")
    obs_pairs = _pairs(observed)
    model_pairs = {name: _pairs(model) for name, model in named.items()}
    universe = (
        pa.concat_tables(
            [obs_pairs.select(["origin", "destination"])]
            + [t.select(["origin", "destination"]) for t in model_pairs.values()]
        )
        .group_by(["origin", "destination"])
        .aggregate([])
    )
    observed_full = universe.join(
        obs_pairs.rename_columns(["origin", "destination", "observed"]),
        keys=["origin", "destination"],
        join_type="left outer",
    )
    out: dict[str, list] = {
        k: []
        for k in (
            "model",
            "n_pairs",
            "observed_flow",
            "model_flow",
            "cpc",
            "cpl",
            "r_squared",
            "spearman",
            "poisson_deviance",
            "pseudo_r_squared",
        )
    }
    for name, pred_pairs in model_pairs.items():
        joined = observed_full.join(
            pred_pairs.rename_columns(["origin", "destination", "predicted"]),
            keys=["origin", "destination"],
            join_type="left outer",
        )
        y = np.nan_to_num(joined["observed"].to_numpy(zero_copy_only=False).astype(float))
        mu = np.nan_to_num(joined["predicted"].to_numpy(zero_copy_only=False).astype(float))

        dev, null = _deviance(y, mu), _deviance(y, np.full_like(y, y.mean()))
        r = np.corrcoef(y, mu)[0, 1] if y.std() > 0 and mu.std() > 0 else float("nan")
        rho = np.corrcoef(_ranks(y), _ranks(mu))[0, 1] if y.std() > 0 and mu.std() > 0 else float("nan")
        out["model"].append(name)
        out["n_pairs"].append(len(y))
        out["observed_flow"].append(float(y.sum()))
        out["model_flow"].append(float(mu.sum()))
        out["cpc"].append(float(common_part_of_commuters(_as_flow(obs_pairs), _as_flow(pred_pairs))))
        out["cpl"].append(float(common_part_of_links(_as_flow(obs_pairs), _as_flow(pred_pairs))))
        out["r_squared"].append(float(r * r))
        out["spearman"].append(float(rho))
        out["poisson_deviance"].append(dev)
        out["pseudo_r_squared"].append(1.0 - dev / null if null > 0 else float("nan"))
    return pa.table(out)


def _as_flow(pairs: pa.Table) -> Any:
    from fastmob.core.flow_dataframe import FlowDataFrame

    return FlowDataFrame(pairs, tile_id="tile_id")
