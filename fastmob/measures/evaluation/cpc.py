"""Sparse common-part-of-commuters comparisons for Trips and flow data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob.utils._common import _all_unsigned_integer_arrow, _as_arrow, _joint_factorize_arrow_values

_FLOW_ROLES = ("origin", "destination", "flow")


def _resolve_flow_column(
    columns: list[str], which: str, role: str, requested: str | None, canonical: str, candidates: list[str]
):
    """Pick the column that plays ``role`` in a flow table.

    An explicit name must exist. Otherwise the canonical name wins when present, then exactly one of the shared
    candidate names; several candidates are ambiguous and none is missing.
    """
    if requested is not None:
        if requested not in columns:
            raise ValueError(f"{which}: {role} column {requested!r} not found; available columns: {columns}")
        return requested
    if canonical in columns:
        return canonical
    found = [name for name in candidates if name in columns]
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError(
            f"{which}: ambiguous {role} column, {found} all look like it. Name the one to use with "
            f"{which}_columns= ({role!r} key)."
        )
    raise ValueError(
        f"{which}: could not find a {role} column (looked for {[canonical, *candidates]}); available columns: "
        f"{columns}. Name it with {which}_columns= ({role!r} key)."
    )


def _flow_arrays(table: Any, which: str, names: Mapping[str, str] | None):
    """Origin, destination and flow columns of a flow table: a ``FlowDataFrame`` or any eager dataframe."""
    from fastmob.core.flow_dataframe import DESTINATION, FLOW, ORIGIN, FlowDataFrame
    from fastmob.utils._common import DEST_CANDIDATES, FLOW_CANDIDATES, ORIGIN_CANDIDATES

    if names is not None and not isinstance(names, Mapping):
        raise TypeError(f"{which}_columns must be a mapping such as {{'flow': 'count'}}, not {type(names).__name__}")
    unknown = set(names or {}) - set(_FLOW_ROLES)
    if unknown:
        raise ValueError(f"{which}_columns keys must be among {_FLOW_ROLES}, got {sorted(unknown)}")
    native = table.df if isinstance(table, FlowDataFrame) else table
    try:
        frame = nw.from_native(native, eager_only=True)
    except TypeError as error:
        raise TypeError(
            f"{which} must be a FlowDataFrame or an eager dataframe with origin, destination and flow columns, "
            f"not {type(table).__name__}"
        ) from error
    columns = list(frame.columns)
    names = names or {}
    origin = _resolve_flow_column(columns, which, "origin", names.get("origin"), ORIGIN, ORIGIN_CANDIDATES)
    destination = _resolve_flow_column(
        columns, which, "destination", names.get("destination"), DESTINATION, DEST_CANDIDATES
    )
    flow = _resolve_flow_column(columns, which, "flow", names.get("flow"), FLOW, FLOW_CANDIDATES)
    if len({origin, destination, flow}) < 3:
        raise ValueError(
            f"{which}: origin, destination and flow must be three different columns, got {[origin, destination, flow]}"
        )
    if not frame.schema[flow].is_numeric():
        raise ValueError(f"{which} flow column {flow!r} must be numeric, not {frame.schema[flow]}")
    return (
        frame.get_column(origin),
        frame.get_column(destination),
        pc.cast(_as_arrow(frame.get_column(flow)), pa.float64()),
    )


def _harmonised_endpoints(*columns: Any) -> list[Any]:
    """Give the four endpoint columns one Arrow type, so ids from different sources can be compared.

    Dictionary (categorical) columns are decoded, string and large_string are unified, integers of different widths
    or signs become int64 (uint64 when all are unsigned) and an empty or null-typed column takes the others' type.
    Ids of really different kinds, such as integers and strings, raise a ``ValueError`` naming the types.
    """
    arrays = [_as_arrow(column) for column in columns]
    arrays = [a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a for a in arrays]
    arrays = [a.dictionary_decode() if pa.types.is_dictionary(a.type) else a for a in arrays]
    kinds = [a.type for a in arrays if len(a) > 0 and not pa.types.is_null(a.type)]
    if not kinds:  # empty tables have no meaningful id type
        return [pc.cast(a, pa.int64()) for a in arrays]
    if all(pa.types.is_string(t) or pa.types.is_large_string(t) for t in kinds):
        target = pa.large_string() if any(pa.types.is_large_string(t) for t in kinds) else pa.string()
    elif all(pa.types.is_integer(t) for t in kinds):
        target = pa.uint64() if all(pa.types.is_unsigned_integer(t) for t in kinds) else pa.int64()
    elif all(t == kinds[0] for t in kinds):
        target = kinds[0]
    else:
        names = ["observed origin", "observed destination", "predicted origin", "predicted destination"]
        raise ValueError(
            "the location ids of the two tables have incompatible types: "
            + ", ".join(f"{n} is {a.type}" for n, a in zip(names, arrays, strict=True))
            + ". Convert them to the same kind (for example both to str) first."
        )
    return [a if a.type == target else pc.cast(a, target) for a in arrays]


def _prepared_pairs(
    observed: Any,
    predicted: Any,
    caller: str,
    observed_columns: Mapping[str, str] | None,
    predicted_columns: Mapping[str, str] | None,
):
    """The four endpoint columns and the two weight arrays shared by CPC and CPL."""
    from fastmob.core.trips_dataframe import Trips

    trips = (isinstance(observed, Trips), isinstance(predicted, Trips))
    if all(trips):
        if observed_columns or predicted_columns:
            raise ValueError(f"{caller}: observed_columns and predicted_columns apply to flow tables, not Trips")
        pairs = []
        for table in (observed, predicted):
            frame = nw.from_native(table.df, eager_only=True)
            columns = ("origin_location_id", "destination_location_id")
            if any(column not in frame.columns for column in columns):
                raise ValueError(f"{caller} requires Trips with origin_location_id and destination_location_id columns")
            pairs.append((frame.get_column(columns[0]), frame.get_column(columns[1])))
        (origin_a, destination_a), (origin_b, destination_b) = pairs
        origin_a, destination_a, origin_b, destination_b = _harmonised_endpoints(
            origin_a, destination_a, origin_b, destination_b
        )
        return origin_a, destination_a, None, origin_b, destination_b, None
    if any(trips):
        raise TypeError(
            f"{caller} compares two Trips or two flow tables (FlowDataFrame or dataframes with origin, destination "
            "and flow columns); convert Trips explicitly to a flow table with od_matrix to mix them."
        )
    origin_a, destination_a, weights_a = _flow_arrays(observed, "observed", observed_columns)
    origin_b, destination_b, weights_b = _flow_arrays(predicted, "predicted", predicted_columns)
    origin_a, destination_a, origin_b, destination_b = _harmonised_endpoints(
        origin_a, destination_a, origin_b, destination_b
    )
    return origin_a, destination_a, weights_a, origin_b, destination_b, weights_b


def common_part_of_commuters(
    observed: Any,
    predicted: Any,
    *,
    observed_columns: Mapping[str, str] | None = None,
    predicted_columns: Mapping[str, str] | None = None,
) -> float:
    """Return the common part of commuters (CPC) between observed and predicted flows.

    CPC is ``2 * sum(min(observed, predicted)) / (sum(observed) + sum(predicted))`` over origin-destination
    pairs: 1 when the two flow tables are identical, 0 when they share no trips. Self-loops and missing
    endpoints are ignored.

    Flow tables can be ``FlowDataFrame`` objects or ordinary eager dataframes (pandas, Polars, PyArrow).
    The origin, destination and flow columns of each table are found by name: ``origin``, ``destination`` and
    ``flow`` first, then ``origin_area``/``area_o``/... for the endpoints (the names ``od_matrix`` produces) and
    ``count``, ``flows``, ``weight`` or ``trips`` for the flow. The two tables may use different names. When
    several names match (for example both ``count`` and ``weight``) a ``ValueError`` asks you to choose with
    ``observed_columns`` or ``predicted_columns``.

    Parameters
    ----------
    observed : Trips, FlowDataFrame or DataFrame
        Two ``Trips`` tables (each trip counts 1) or two flow tables (weighted by their flow column). Both must be
        of the same kind; a ``FlowDataFrame`` can be compared with an ordinary dataframe.
    predicted : Trips, FlowDataFrame or DataFrame
        The second table, as above.
    observed_columns : mapping of str to str, optional
        Column names of ``observed``, with any of the keys ``"origin"``, ``"destination"`` and ``"flow"``.
        Columns that are not named are detected.
    predicted_columns : mapping of str to str, optional
        The same for ``predicted``.

    Returns
    -------
    float
        CPC between 0 and 1 (higher is more similar). It depends on the zone size: finer zones give lower values.

    Raises
    ------
    TypeError
        If the inputs are not both Trips or both flow tables.
    ValueError
        If a column cannot be found, several columns match one role, or the flow column is not numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob import common_part_of_commuters
    >>> observed = pd.DataFrame({"origin": ["a", "a"], "destination": ["b", "c"], "flow": [10, 5]})
    >>> predicted = pd.DataFrame({"origin_area": ["a", "a"], "destination_area": ["b", "c"], "count": [10, 10]})
    >>> round(common_part_of_commuters(observed, predicted), 3)
    0.8
    """
    from fastmob._core import common_part_of_commuters as _cpc
    from fastmob._core import common_part_of_commuters_u64 as _cpc_u64

    origin_a, destination_a, weights_a, origin_b, destination_b, weights_b = _prepared_pairs(
        observed, predicted, "CPC", observed_columns, predicted_columns
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


def common_part_of_links(
    observed: Any,
    predicted: Any,
    *,
    observed_columns: Mapping[str, str] | None = None,
    predicted_columns: Mapping[str, str] | None = None,
) -> float:
    """Return the common part of links (CPL) between observed and predicted flows.

    CPL is ``2 * |links in both| / (|links observed| + |links predicted|)``: the share of origin-destination
    links that exist in both tables, whatever their weight. It is 1 when both have exactly the same links.
    Inputs and column detection are the same as for `common_part_of_commuters`.

    Parameters
    ----------
    observed : Trips, FlowDataFrame or DataFrame
        Two ``Trips`` tables or two flow tables of the same kind.
    predicted : Trips, FlowDataFrame or DataFrame
        The second table.
    observed_columns : mapping of str to str, optional
        Column names of ``observed`` (keys ``"origin"``, ``"destination"``, ``"flow"``).
    predicted_columns : mapping of str to str, optional
        The same for ``predicted``.

    Returns
    -------
    float
        CPL between 0 and 1 (higher is more similar).
    """
    from fastmob._core import common_part_of_links as _cpl
    from fastmob._core import common_part_of_links_u64 as _cpl_u64

    origin_a, destination_a, weights_a, origin_b, destination_b, weights_b = _prepared_pairs(
        observed, predicted, "CPL", observed_columns, predicted_columns
    )
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
