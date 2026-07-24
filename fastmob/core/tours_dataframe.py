"""Tours — one row per tour (a maximal run of consecutive trips that starts
and ends at the same location).

Requires each trip's ``origin_staypoint_id``/``destination_staypoint_id`` to
resolve to a ``location_id`` via an already-clustered `Staypoints` table
(see `Staypoints.generate_locations`). A per-user sequential scan over
trips (cheap at this row count -- far fewer trips than raw GPS fixes or even
triplegs -- so a small NumPy-array-based loop is used here instead of a
vectorized expression, no Rust kernel justified); `import pandas`/
`import polars` stay forbidden inside `fastmob/` per CLAUDE.md.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np

from .base import BaseDataFrame

_REQUIRED_COLUMNS = ["tour_id", "started_at", "finished_at"]


class Tours(BaseDataFrame):
    """One row per tour (a sequence of trips returning to its starting location).

    ``.df`` columns: ``tour_id``, ``started_at``, ``finished_at``,
    ``location_id`` (the anchor location the tour starts and ends at),
    ``journey`` (list of the trip ids making up the tour), plus ``uid_col``
    when present.

    Parameters
    ----------
    df : DataFrame-like
        Source data; any Narwhals-compatible eager backend.
    uid_col : str, optional
        User-ID column name.
    validate : bool, optional
        When True (default), check that required columns are present.
    """

    def __init__(self, df: Any, uid_col: str | None = None, validate: bool = True):
        super().__init__(df)
        self.uid_col = uid_col
        if validate:
            nw_df = nw.from_native(df, eager_only=True)
            missing = [col for col in _REQUIRED_COLUMNS if col not in nw_df.columns]
            if missing:
                raise ValueError(f"Tours is missing required columns: {missing}")

    @staticmethod
    def from_trips(trips: Any, staypoints_with_location: Any) -> Tours:
        """Group consecutive trips into tours.

        See :meth:`fastmob.core.trips_dataframe.Trips.generate_tours`.
        """
        sp_nw = nw.from_native(staypoints_with_location.df, eager_only=True)
        if "location_id" not in sp_nw.columns:
            raise ValueError(
                "Tours.from_trips requires staypoints_with_location to have a 'location_id' column; "
                "run Staypoints.generate_locations() first"
            )
        if "staypoint_id" not in sp_nw.columns:
            raise ValueError("Tours.from_trips requires staypoints_with_location to have a 'staypoint_id' column")

        uid_col = trips.uid_col
        group_key = uid_col if uid_col else "__uid__"

        tl_nw = nw.from_native(trips.df, eager_only=True)
        if not uid_col:
            tl_nw = tl_nw.with_columns(nw.lit(0, dtype=nw.Int64).alias(group_key))

        staypoint_ids = sp_nw.get_column("staypoint_id").to_numpy()
        location_ids = sp_nw.get_column("location_id").to_numpy()
        loc_lookup = dict(zip(staypoint_ids.tolist(), location_ids.tolist()))

        tl_nw = tl_nw.sort([group_key, "started_at"])
        n = len(tl_nw)
        uid_values = tl_nw.get_column(group_key).to_numpy()
        trip_id = tl_nw.get_column("trip_id").to_numpy()
        started_at = tl_nw.get_column("started_at").to_numpy()
        finished_at = tl_nw.get_column("finished_at").to_numpy()
        origin_staypoint_id = tl_nw.get_column("origin_staypoint_id").to_numpy()
        destination_staypoint_id = tl_nw.get_column("destination_staypoint_id").to_numpy()

        def _location_of(staypoint_id: Any) -> Any:
            if staypoint_id is None or (isinstance(staypoint_id, float) and np.isnan(staypoint_id)):
                return None
            return loc_lookup.get(staypoint_id)

        origin_location = [_location_of(sid) for sid in origin_staypoint_id]
        destination_location = [_location_of(sid) for sid in destination_staypoint_id]

        first_of_user = np.zeros(n, dtype=bool)
        if n > 0:
            first_of_user[0] = True
        if n > 1:
            first_of_user[1:] = uid_values[1:] != uid_values[:-1]

        tour_uid: list[Any] = []
        tour_started_at: list[Any] = []
        tour_finished_at: list[Any] = []
        tour_location_id: list[Any] = []
        tour_journey: list[list[Any]] = []
        tour_id_counter = 0

        i = 0
        while i < n:
            if not first_of_user[i]:
                i += 1
                continue
            user_end = i + 1
            while user_end < n and not first_of_user[user_end]:
                user_end += 1

            j = i
            while j < user_end:
                origin_loc = origin_location[j]
                if origin_loc is None:
                    j += 1
                    continue
                closing = None
                for k in range(j, user_end):
                    if destination_location[k] == origin_loc:
                        closing = k
                        break
                if closing is None:
                    j += 1
                    continue
                tour_uid.append(uid_values[j])
                tour_started_at.append(started_at[j])
                tour_finished_at.append(finished_at[closing])
                tour_location_id.append(origin_loc)
                tour_journey.append([trip_id[m] for m in range(j, closing + 1)])
                tour_id_counter += 1
                j = closing + 1
            i = user_end

        # Fixed-dtype NumPy arrays rather than plain Python lists of NumPy
        # scalars -- some backends (e.g. Polars) can't infer a schema from a
        # list of individual `numpy.datetime64` objects.
        out_dict: dict[str, Any] = {
            "tour_id": np.arange(tour_id_counter, dtype=np.int64),
            "started_at": np.array(tour_started_at, dtype="datetime64[us]"),
            "finished_at": np.array(tour_finished_at, dtype="datetime64[us]"),
            "location_id": tour_location_id,
            "journey": tour_journey,
        }
        if uid_col:
            out_dict[uid_col] = tour_uid

        return Tours(nw.from_dict(out_dict, backend=tl_nw.implementation).to_native(), uid_col=uid_col)
