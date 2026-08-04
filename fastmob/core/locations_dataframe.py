"""Locations — user-scoped recurring clusters or globally shared places."""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .base import BaseDataFrame


class Locations(BaseDataFrame):
    """One row per location in either a user or global scope.

    Parameters
    ----------
    df : DataFrame-like
        Source data; any Narwhals-compatible eager backend.
    uid_col : str, optional
        User-ID column name for user-scoped locations. Must be ``None`` for
        global locations.
    scope : {"user", "global"}, optional
        Location identity scope. When omitted it is inferred from ``uid_col``.
    scheme : {"cluster", "h3", "external"}, optional
        Location-ID scheme. User locations default to ``"cluster"`` and
        global locations default to ``"external"``.
    location_id_col, center_lat_col, center_lng_col : str, optional
        Column name overrides.
    validate : bool, optional
        When True (default), check that required columns are present.
    """

    def __init__(
        self,
        df: Any,
        uid_col: str | None = None,
        location_id_col: str = "location_id",
        center_lat_col: str = "center_lat",
        center_lng_col: str = "center_lng",
        scope: str | None = None,
        scheme: str | None = None,
        validate: bool = True,
    ):
        super().__init__(df)
        self.uid_col = uid_col
        self.location_id_col = location_id_col
        self.center_lat_col = center_lat_col
        self.center_lng_col = center_lng_col
        inferred_scope = "user" if uid_col is not None else "global"
        self.scope = inferred_scope if scope is None else scope
        if self.scope not in {"user", "global"}:
            raise ValueError("Locations scope must be 'user' or 'global'")
        if self.scope == "global" and uid_col is not None:
            raise ValueError("Locations scope='global' requires uid_col=None")
        self.scheme = ("cluster" if self.scope == "user" else "external") if scheme is None else scheme
        if self.scheme not in {"cluster", "h3", "external"}:
            raise ValueError("Locations scheme must be 'cluster', 'h3', or 'external'")

        if validate:
            nw_df = nw.from_native(df, eager_only=True)
            required = [location_id_col, center_lat_col, center_lng_col]
            missing = [col for col in required if col not in nw_df.columns]
            if missing:
                raise ValueError(f"Locations is missing required columns: {missing}")
            key_cols = [location_id_col] if self.scope == "global" or uid_col is None else [uid_col, location_id_col]
            if len(nw_df.select(key_cols).unique()) != len(nw_df):
                raise ValueError(f"Locations requires unique rows by {key_cols}")

    def identify(self, staypoints: Any, method: str = "freq", **kwargs: Any) -> Locations:
        """Label each location as ``"home"``, ``"work"``, or ``"other"``.

        See :func:`fastmob.preprocessing.identify_locations`.
        """
        if self.scope != "user":
            raise ValueError("identify is only defined for user-scoped Locations; home/work purposes are user-specific")
        from ..preprocessing import identify_locations

        return identify_locations(self, staypoints, method=method, **kwargs)

    def validate_staypoint_assignments(
        self,
        staypoints: Any,
        *,
        user_id_col: str,
        location_id_col: str,
    ) -> None:
        """Validate non-null staypoint location identities against this catalogue.

        The validation runs as backend-native distinct/anti joins instead of
        materializing Python sets. Global catalogues match ``location_id``;
        user-scoped catalogues match ``(user_id, location_id)``.
        """
        visits = nw.from_native(staypoints, eager_only=True)
        if location_id_col not in visits.columns:
            raise ValueError(f"Staypoints is missing location-ID column {location_id_col!r}")
        catalogue = nw.from_native(self.df, eager_only=True)
        if self.scope == "global":
            visit_keys = visits.select(location_id_col).drop_nulls().unique()
            catalogue_keys = catalogue.select(self.location_id_col).rename({self.location_id_col: location_id_col})
            unknown = visit_keys.join(catalogue_keys, on=location_id_col, how="anti")
            if len(unknown) > 0:
                raise ValueError("Staypoints contains location IDs absent from the global Locations catalogue")
            return

        if self.uid_col is None or self.uid_col not in catalogue.columns:
            raise ValueError("User-scoped Locations requires its user-ID column in the catalogue")
        if user_id_col not in visits.columns:
            raise ValueError(f"Staypoints is missing user-ID column {user_id_col!r}")
        visit_keys = visits.select([user_id_col, location_id_col]).drop_nulls().unique()
        catalogue_keys = catalogue.select([self.uid_col, self.location_id_col]).rename(
            {self.uid_col: user_id_col, self.location_id_col: location_id_col}
        )
        unknown = visit_keys.join(catalogue_keys, on=[user_id_col, location_id_col], how="anti")
        if len(unknown) > 0:
            raise ValueError("Staypoints contains (user, location) IDs absent from the user-scoped Locations catalogue")
