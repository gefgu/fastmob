"""Collective location interest networks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from fastmob._core import factorize_arrow
from fastmob._core import interest_network as _interest_network
from fastmob.utils._common import _as_arrow

if TYPE_CHECKING:
    from fastmob.core.locations_dataframe import Locations
    from fastmob.core.staypoints_dataframe import Staypoints


def interest_network(staypoints: Staypoints | Any, locations: Locations) -> Any:
    """Return pairs of global locations visited by the same people.

    Each output row is an undirected edge. ``n_people`` is the number of
    distinct users that visited both endpoint locations at least once;
    repeated staypoints do not increase an edge's weight. Endpoint orientation
    and output rows follow the supplied global Locations catalogue order.

    Parameters
    ----------
    staypoints : Staypoints or DataFrame-like
        Staypoints assigned to ``locations``. Raw dataframe-like input is
        validated through :meth:`Staypoints.coerce`.
    locations : Locations
        Global location catalogue linked to the staypoint ``location_id``
        column.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        Same backend as ``staypoints`` with columns ``location_id_a``,
        ``location_id_b``, and ``n_people``.
    """
    from fastmob.core.locations_dataframe import Locations
    from fastmob.core.staypoints_dataframe import Staypoints

    if not isinstance(locations, Locations):
        raise TypeError("interest_network requires a Locations instance")
    locations.require_global()
    sp = Staypoints.coerce(staypoints)
    if sp.uid_col is None:
        raise ValueError("interest_network requires Staypoints with a user-ID column")

    assigned = sp.associate_global_locations(locations, location_id_col=locations.location_id_col)
    frame = nw.from_native(assigned.df, eager_only=True)
    catalogue = nw.from_native(locations.df, eager_only=True)
    location_id_col = locations.location_id_col
    if location_id_col != "location_id":
        catalogue = catalogue.rename({location_id_col: "location_id"})

    memberships = frame.select([assigned.uid_col, "location_id"]).drop_nulls().unique()
    catalogue_ids = pa.array(catalogue.get_column("location_id").to_arrow())
    if len(memberships) == 0:
        return nw.from_arrow(
            pa.table(
                {
                    "location_id_a": catalogue_ids.slice(0, 0),
                    "location_id_b": catalogue_ids.slice(0, 0),
                    "n_people": pa.array([], type=pa.uint64()),
                }
            ),
            backend=frame.implementation,
        ).to_native()

    user_codes, _ = factorize_arrow(memberships.get_column(assigned.uid_col).to_arrow(), sort=False)
    ranks = pc.index_in(pa.array(memberships.get_column("location_id").to_arrow()), value_set=catalogue_ids)
    if ranks.null_count:
        raise ValueError("Staypoints contains location IDs absent from the global Locations catalogue")
    rank_values = pc.cast(ranks, pa.uint64())
    raw_a, raw_b, raw_counts = _interest_network(user_codes, rank_values)
    rank_a = _as_arrow(raw_a)
    rank_b = _as_arrow(raw_b)
    result = pa.table(
        {
            "location_id_a": pc.take(catalogue_ids, rank_a),
            "location_id_b": pc.take(catalogue_ids, rank_b),
            "n_people": _as_arrow(raw_counts),
        }
    )
    return nw.from_arrow(result, backend=frame.implementation).to_native()
