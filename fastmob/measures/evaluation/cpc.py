"""Sparse common-part-of-commuters comparisons for Trips and flow data."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob.utils._common import _all_unsigned_integer_arrow, _as_arrow, _joint_factorize_arrow_values


def common_part_of_commuters(observed: Any, predicted: Any) -> float:
    """Return the common part of commuters (CPC) between observed and predicted flows.

    CPC is ``2 * sum(min(observed, predicted)) / (sum(observed) + sum(predicted))`` over origin-destination
    pairs: 1 when the two flow tables are identical, 0 when they share no trips. Self-loops and missing
    endpoints are ignored.

    Parameters
    ----------
    observed : Trips or FlowDataFrame
        Two ``Trips`` tables (each trip counts 1) or two sparse ``FlowDataFrame`` tables (weighted by ``flow``).
        Both must be the same type.
    predicted : Trips or FlowDataFrame
        Two ``Trips`` tables (each trip counts 1) or two sparse ``FlowDataFrame`` tables (weighted by ``flow``).
        Both must be the same type.

    Returns
    -------
    float
        CPC between 0 and 1 (higher is more similar). It depends on the zone size: finer zones give lower values.
    """
    from fastmob._core import common_part_of_commuters as _cpc
    from fastmob._core import common_part_of_commuters_u64 as _cpc_u64
    from fastmob.core.flow_dataframe import DESTINATION, FLOW, ORIGIN, FlowDataFrame
    from fastmob.core.trips_dataframe import Trips

    if isinstance(observed, Trips) and isinstance(predicted, Trips):
        observed_df = nw.from_native(observed.df, eager_only=True)
        predicted_df = nw.from_native(predicted.df, eager_only=True)
        columns = ("origin_location_id", "destination_location_id")
        if any(column not in df.columns for df in (observed_df, predicted_df) for column in columns):
            raise ValueError("CPC requires Trips with origin_location_id and destination_location_id columns")
        origin_a, destination_a = (observed_df.get_column(column) for column in columns)
        origin_b, destination_b = (predicted_df.get_column(column) for column in columns)
        weights_a = weights_b = None
    elif isinstance(observed, FlowDataFrame) and isinstance(predicted, FlowDataFrame):
        observed_df = nw.from_native(observed.df, eager_only=True)
        predicted_df = nw.from_native(predicted.df, eager_only=True)
        columns = (ORIGIN, DESTINATION, FLOW)
        if any(column not in df.columns for df in (observed_df, predicted_df) for column in columns):
            raise ValueError("CPC requires FlowDataFrames with origin, destination, and flow columns")
        origin_a, destination_a = (observed_df.get_column(column) for column in (ORIGIN, DESTINATION))
        origin_b, destination_b = (predicted_df.get_column(column) for column in (ORIGIN, DESTINATION))
        weights_a = pc.cast(_as_arrow(observed_df.get_column(FLOW)), pa.float64())
        weights_b = pc.cast(_as_arrow(predicted_df.get_column(FLOW)), pa.float64())
    else:
        raise TypeError(
            "CPC compares two Trips or two FlowDataFrame instances; convert Trips explicitly to FlowDataFrame to mix them."
        )

    if _all_unsigned_integer_arrow(origin_a, destination_a, origin_b, destination_b):
        origin_a, destination_a, origin_b, destination_b = (
            pc.cast(_as_arrow(value), pa.uint64()) for value in (origin_a, destination_a, origin_b, destination_b)
        )
        return float(_cpc_u64(origin_a, destination_a, weights_a, origin_b, destination_b, weights_b))

    origin_a, destination_a, origin_b, destination_b = _joint_factorize_arrow_values(
        origin_a, destination_a, origin_b, destination_b
    )
    return float(_cpc(origin_a, destination_a, weights_a, origin_b, destination_b, weights_b))


def common_part_of_links(observed: Any, predicted: Any) -> float:
    """Return the common part of links (CPL) between observed and predicted flows.

    CPL is ``2 * |links in both| / (|links observed| + |links predicted|)``: the share of origin-destination
    links that exist in both tables, whatever their weight. It is 1 when both have exactly the same links.

    Parameters
    ----------
    observed : Trips or FlowDataFrame
        Two ``Trips`` tables or two ``FlowDataFrame`` tables of the same type.
    predicted : Trips or FlowDataFrame
        Two ``Trips`` tables or two ``FlowDataFrame`` tables of the same type.

    Returns
    -------
    float
        CPL between 0 and 1 (higher is more similar).
    """
    from fastmob._core import common_part_of_links as _cpl
    from fastmob._core import common_part_of_links_u64 as _cpl_u64

    # Reuse CPC's model validation and joint endpoint factorization by using
    # its prepared inputs through this intentionally parallel dispatch.
    from fastmob.core.flow_dataframe import DESTINATION, FLOW, ORIGIN, FlowDataFrame
    from fastmob.core.trips_dataframe import Trips

    if isinstance(observed, Trips) and isinstance(predicted, Trips):
        a, b = nw.from_native(observed.df, eager_only=True), nw.from_native(predicted.df, eager_only=True)
        origin_a, destination_a = a.get_column("origin_location_id"), a.get_column("destination_location_id")
        origin_b, destination_b = b.get_column("origin_location_id"), b.get_column("destination_location_id")
        weights_a = weights_b = None
    elif isinstance(observed, FlowDataFrame) and isinstance(predicted, FlowDataFrame):
        a, b = nw.from_native(observed.df, eager_only=True), nw.from_native(predicted.df, eager_only=True)
        origin_a, destination_a = a.get_column(ORIGIN), a.get_column(DESTINATION)
        origin_b, destination_b = b.get_column(ORIGIN), b.get_column(DESTINATION)
        weights_a, weights_b = (
            pc.cast(_as_arrow(a.get_column(FLOW)), pa.float64()),
            pc.cast(_as_arrow(b.get_column(FLOW)), pa.float64()),
        )
    else:
        raise TypeError("CPL compares two Trips or two FlowDataFrame instances")

    if _all_unsigned_integer_arrow(origin_a, destination_a, origin_b, destination_b):
        origin_a, destination_a, origin_b, destination_b = (
            pc.cast(_as_arrow(value), pa.uint64()) for value in (origin_a, destination_a, origin_b, destination_b)
        )
        return float(_cpl_u64(origin_a, destination_a, weights_a, origin_b, destination_b, weights_b))

    origin_a, destination_a, origin_b, destination_b = _joint_factorize_arrow_values(
        origin_a, destination_a, origin_b, destination_b
    )
    return float(_cpl(origin_a, destination_a, weights_a, origin_b, destination_b, weights_b))


def common_part_of_commuters_distance(observed: Any, predicted: Any) -> float:
    """Compare the trip-distance distributions of two sets of trips (CPCD).

    Distances are grouped in 2 km bins and the result is ``sum(min(count_observed, count_predicted)) /
    number of observed trips``: 1 for identical distance distributions, 0 for none in common.

    Parameters
    ----------
    observed : Trips
        Two ``Trips`` tables, each with a ``distance_km`` column (kilometres).
    predicted : Trips
        Two ``Trips`` tables, each with a ``distance_km`` column (kilometres).

    Returns
    -------
    float
        CPCD between 0 and 1 (higher is more similar when the two sets have similar numbers of trips).
    """
    from fastmob._core import common_part_of_commuters_distance as _cpcd
    from fastmob.core.trips_dataframe import Trips

    if not isinstance(observed, Trips) or not isinstance(predicted, Trips):
        raise TypeError("CPCD compares two Trips with distance_km columns")
    values = []
    for trips in (observed, predicted):
        df = nw.from_native(trips.df, eager_only=True)
        if "distance_km" not in df.columns:
            raise ValueError("CPCD requires Trips with a distance_km column")
        values.append(pc.cast(_as_arrow(df.get_column("distance_km")), pa.float64()))
    return float(_cpcd(*values))
