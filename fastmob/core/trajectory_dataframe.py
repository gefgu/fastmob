"""TrajDataFrame — Narwhals-backed trajectory wrapper."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa

from fastmob.core.base import BaseDataFrame
from fastmob.measures.individual import jump_lengths
from fastmob.utils._common import _detect_trajectory_columns, _prepare_trajectory, require_optional

LATITUDE = "lat"
LONGITUDE = "lng"
DATETIME = "datetime"
UID = "uid"
TID = "tid"
TILE_ID = "tile_id"
DEFAULT_CRS = {"init": "epsg:4326"}


def _nearest_tessellation_values(origin, tessellation, column: str):
    """Return the value from the closest tessellation point for each origin row."""

    def nearest_index(row):
        point = row.geometry
        distances = tessellation.geometry.apply(
            lambda candidate: (point.y - candidate.y) ** 2 + (point.x - candidate.x) ** 2
        )
        return distances.idxmin()

    indices = origin.apply(nearest_index, axis=1)
    return tessellation.loc[indices, column]


class TrajDataFrame(BaseDataFrame):
    """Narwhals-backed wrapper for trajectory data.

    Accepts any eager DataFrame backend (pandas, polars, …) and exposes a
    unified mobility-analysis API.  Column names are auto-detected from a
    priority list; custom names can be supplied explicitly.

    Parameters
    ----------
    df : DataFrame-like
        Source data.  Accepted types: ``pandas.DataFrame``, ``polars.DataFrame``,
        any Narwhals-compatible eager frame, plain ``list``, ``numpy.ndarray``, or
        ``dict``.
    sort : bool, optional
        If ``True``, sort the underlying data by ``(uid, datetime)`` on
        construction. Default ``False``.
    timestamp : bool, optional
        If ``True``, parse the datetime column from Unix timestamps. Default ``False``.
    datetime_col : str, optional
        Name of the datetime column in *df* (overrides auto-detection).
    lat_col : str, optional
        Name of the latitude column (overrides auto-detection).
    lng_col : str, optional
        Name of the longitude column (overrides auto-detection).
    uid_col : str, optional
        Name of the user-ID column (overrides auto-detection).
    latitude : str, optional
        Source column to rename to ``'lat'``.
    longitude : str, optional
        Source column to rename to ``'lng'``.
    datetime : str, optional
        Source column to rename to ``'datetime'``.
    user_id : str, optional
        Source column to rename to ``'uid'``.
    trajectory_id : str, optional
        Source column to rename to ``'tid'``.
    crs : dict, optional
        Coordinate reference system. Default ``{"init": "epsg:4326"}``.
    parameters : dict, optional
        Arbitrary metadata dictionary. Default ``{}``.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> data = [
    ...     [1, 39.984094, 116.319236, "2008-10-23 13:53:05"],
    ...     [1, 39.984198, 116.319322, "2008-10-23 13:53:06"],
    ...     [1, 39.984224, 116.319402, "2008-10-23 13:53:11"],
    ... ]
    >>> df = pd.DataFrame(data, columns=["uid", "lat", "lng", "datetime"])
    >>> tdf = fastmob.TrajDataFrame(df)
    >>> tdf.uid_col
    'uid'
    """

    def __init__(
        self,
        df,
        sort: bool = False,
        timestamp: bool = False,
        datetime_col: str | None = None,
        lat_col: str | None = None,
        lng_col: str | None = None,
        uid_col: str | None = None,
        latitude: str | None = None,
        longitude: str | None = None,
        datetime: str | None = None,
        user_id: str | None = None,
        trajectory_id: str | None = None,
        crs: dict | None = None,
        parameters: dict | None = None,
        **kwargs,
    ):
        if isinstance(df, TrajDataFrame):
            super().__init__(df.df)
            self.sorted = df.sorted
            self.datetime_col = df.datetime_col
            self.lat_col = df.lat_col
            self.lng_col = df.lng_col
            self.uid_col = df.uid_col
            self.trajectory_id_col = getattr(df, "trajectory_id_col", TID)
            self.crs = getattr(df, "crs", DEFAULT_CRS)
            self.parameters = getattr(df, "parameters", {})
            self._info = getattr(df, "_info", None)
            return

        latitude = LATITUDE if latitude is None else latitude
        longitude = LONGITUDE if longitude is None else longitude
        datetime = DATETIME if datetime is None else datetime
        user_id = UID if user_id is None else user_id
        trajectory_id = TID if trajectory_id is None else trajectory_id
        self.crs = DEFAULT_CRS if crs is None else crs
        self.parameters = {} if parameters is None else parameters
        self._info = None

        # Always routed through _rename_columns, even when no renaming is
        # needed: dict/list/ndarray inputs must still be coerced into a real
        # table (Narwhals can't wrap a bare dict/list), while an
        # already-Narwhals-native `df` with nothing to rename short-circuits
        # inside _rename_columns without any extra work.
        df = self._rename_columns(
            df,
            {
                latitude: LATITUDE,
                longitude: LONGITUDE,
                datetime: DATETIME,
                user_id: UID,
                trajectory_id: TID,
            },
        )

        super().__init__(df)

        self.sorted = False
        # `in`/`.columns` on the raw native object is only reliable for
        # pandas/polars (whose `.columns` happen to be name strings); a raw
        # pyarrow.Table's `.columns` is a list of ChunkedArrays and `in`
        # doesn't check column names at all. Go through Narwhals so column
        # detection works uniformly across every accepted backend.
        nw_columns = nw.from_native(self.df, eager_only=True).columns
        datetime_col = DATETIME if datetime_col is None and DATETIME in nw_columns else datetime_col
        lat_col = LATITUDE if lat_col is None and LATITUDE in nw_columns else lat_col
        lng_col = LONGITUDE if lng_col is None and LONGITUDE in nw_columns else lng_col
        uid_col = UID if uid_col is None and UID in nw_columns else uid_col
        self.trajectory_id_col = TID if TID in nw_columns else None
        self.datetime_col, self.lat_col, self.lng_col, self.uid_col = _detect_trajectory_columns(
            nw.from_native(self.df, eager_only=True), datetime_col, lat_col, lng_col, uid_col
        )

        if timestamp and self.datetime_col is not None:
            nw_df = nw.from_native(self.df)
            col_dtype = nw_df.schema[self.datetime_col]
            if not isinstance(col_dtype, nw.Datetime):
                if "polars" in str(type(self.df)).lower():
                    import polars as pl

                    native_pl_df = nw_df.to_native()
                    self.df = native_pl_df.with_columns(pl.col(self.datetime_col).str.to_datetime(time_zone="UTC"))
                else:
                    self.df = nw_df.with_columns(nw.col(self.datetime_col).str.to_datetime()).to_native()

        if sort:
            nw_df = nw.from_native(self.df, eager_only=True)
            self.df = _prepare_trajectory(
                nw_df,
                datetime_col=self.datetime_col,
                lat_col=self.lat_col,
                lng_col=self.lng_col,
                uid_col=self.uid_col,
            ).to_native()
            self.sorted = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_columns(df, mapping):
        mapping = {source: target for source, target in mapping.items() if source != target}
        # dict/list/ndarray inputs are coerced into a pyarrow.Table
        # unconditionally, even with an empty `mapping` -- Narwhals can't
        # wrap a bare dict/list, so this coercion must not be skipped just
        # because there's nothing to rename.
        if isinstance(df, dict):
            table = pa.table(df)
            filtered = {source: target for source, target in mapping.items() if source in table.column_names}
            return table.rename_columns(filtered) if filtered else table
        if isinstance(df, (list, np.ndarray)) and len(df) > 0:
            n_cols = len(df[0])
            columns = [str(mapping.get(idx, idx)) for idx in range(n_cols)]
            arrays = [pa.array([row[i] for row in df]) for i in range(n_cols)]
            return pa.table(dict(zip(columns, arrays)))
        if not mapping:
            return df
        try:
            return nw.from_native(df, eager_only=True).rename(mapping).to_native()
        except Exception:  # noqa: BLE001
            return df

    def _to_pandas(self):
        """Return a pandas DataFrame regardless of the backing store.

        Only used by the geopandas-bridging methods below (geopandas
        subclasses pandas.DataFrame, so real pandas is unavoidable there) --
        pandas is imported locally since every caller already sits behind
        an optional extra that transitively requires it (`data`/
        `tessellation` via geopandas, or `vis` via fastmob-vis[legacy]).
        """
        return nw.from_native(self.df, eager_only=True).to_pandas()

    # ------------------------------------------------------------------
    # Measure methods
    # ------------------------------------------------------------------

    def jump_lengths(self, merge: bool = False):
        """Compute the jump lengths (km) between consecutive GPS points.

        Parameters
        ----------
        merge : bool, optional
            If ``True``, merge the result back onto the original DataFrame.
            Default ``False``.

        Returns
        -------
        DataFrame
            A DataFrame with columns ``uid`` (when a user column is present)
            and ``jump_lengths`` containing a list of distances per user.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.jump_lengths()  # doctest: +SKIP
        """
        return jump_lengths(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            presorted=self.sorted,
            merge=merge,
        )

    def radius_of_gyration(self):
        """Compute the radius of gyration (km) for each user.

        Returns
        -------
        DataFrame
            A DataFrame with columns ``uid`` (when present) and
            ``radius_of_gyration``.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.radius_of_gyration()  # doctest: +SKIP
        """
        from fastmob.measures.individual import radius_of_gyration

        return radius_of_gyration(
            self.df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def interpolate(self, method: str = "linear", sampling_rate_s: float = 3600.0, **method_kwargs) -> TrajDataFrame:
        """Fill gaps in the trajectory using a named interpolation algorithm.

        Parameters
        ----------
        method : str, optional
            One of ``"linear"``, ``"cubic_spline"``, ``"kinematic"``, or
            ``"random_walk"``. Default ``"linear"``.
        sampling_rate_s : float, optional
            Maximum time gap, in seconds, allowed between consecutive points
            before an interpolated point is inserted. Default ``3600.0``.
        **method_kwargs
            Method-specific parameters; see
            :func:`fastmob.trajectory.interpolate`.

        Returns
        -------
        TrajDataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.to_datetime(
        ...         ["2020-01-01 00:00", "2020-01-01 02:00", "2020-01-01 03:00"]
        ...     ),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.interpolate(sampling_rate_s=3600.0)  # doctest: +SKIP
        """
        from fastmob.trajectory import interpolate as _interpolate

        result = _interpolate(
            self.df,
            method=method,
            sampling_rate_s=sampling_rate_s,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            presorted=self.sorted,
            **method_kwargs,
        )
        return TrajDataFrame(
            result,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def smooth(self, method: str = "kalman_cv", **method_kwargs) -> TrajDataFrame:
        """Smooth the trajectory's positions using a named algorithm.

        Unlike :meth:`interpolate`, row count and row order are unchanged --
        every point's ``(lat, lng)`` is replaced with a denoised estimate at
        its original timestamp.

        Parameters
        ----------
        method : str, optional
            Only ``"kalman_cv"`` is shipped currently. Default ``"kalman_cv"``.
        **method_kwargs
            Method-specific parameters; see
            :func:`fastmob.trajectory.smooth`.

        Returns
        -------
        TrajDataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.smooth()  # doctest: +SKIP
        """
        from fastmob.trajectory import smooth as _smooth

        result = _smooth(
            self.df,
            method=method,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            presorted=self.sorted,
            **method_kwargs,
        )
        return TrajDataFrame(
            result,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def interpolate_at(self, at, method: str = "linear"):
        """Query the interpolated position of each user at one or more timestamps.

        Parameters
        ----------
        at :
            A single timestamp-like, or a sequence of timestamp-likes.
        method : str, optional
            ``"linear"`` (default) or ``"nearest"``.

        Returns
        -------
        DataFrame
            One row per ``(uid, query_time)`` pair with ``lat``, ``lng``,
            and a ``valid`` column.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.to_datetime(
        ...         ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
        ...     ),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.interpolate_at("2020-01-01 00:30")  # doctest: +SKIP
        """
        from fastmob.trajectory import interpolate_at as _interpolate_at

        return _interpolate_at(
            self.df,
            at,
            method=method,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            presorted=self.sorted,
        )

    def trajectory_distance(self, other, method: str = "dtw", **method_kwargs) -> float:
        """Compute a similarity/distance metric against another trajectory's point-sequence.

        Parameters
        ----------
        other :
            Another trajectory (any Narwhals-compatible eager backend, or a
            ``TrajDataFrame``). Must represent a single user's point-sequence.
        method : str, optional
            One of ``"dtw"``, ``"frechet"``, ``"hausdorff"``, or ``"lcss"``.
            Default ``"dtw"``.
        **method_kwargs
            Method-specific parameters; see
            :func:`fastmob.trajectory.trajectory_distance`.

        Returns
        -------
        float

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "lat": [0.0, 1.0, 2.0], "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.trajectory_distance(df, method="dtw")
        0.0
        """
        from fastmob.trajectory import trajectory_distance as _trajectory_distance

        return _trajectory_distance(self, other, method=method, **method_kwargs)

    # ------------------------------------------------------------------
    # Preprocessing methods
    # ------------------------------------------------------------------

    def compress(self, spatial_radius_km: float = 0.2, inplace: bool = False) -> TrajDataFrame:
        """Compress the trajectory by collapsing nearby consecutive points.

        Parameters
        ----------
        spatial_radius_km : float, optional
            Spatial radius (km) used to decide whether two consecutive points
            belong to the same stop. Default 0.2.
        inplace : bool, optional
            If ``True``, modify this object and return ``self``.
            If ``False`` (default), return a new ``TrajDataFrame``.

        Returns
        -------
        TrajDataFrame

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1],
        ...     "lat": [0.0, 0.0001, 1.0],
        ...     "lng": [0.0, 0.0001, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.compress()  # doctest: +SKIP
        """
        from fastmob.preprocessing import compress

        kwargs = {
            "spatial_radius_km": spatial_radius_km,
            "datetime_col": self.datetime_col,
            "lat_col": self.lat_col,
            "lng_col": self.lng_col,
            "uid_col": self.uid_col,
        }
        if inplace:
            self.df = compress(self.df, **kwargs)
            return self

        compressed_df = compress(self.df, **kwargs)
        return TrajDataFrame(
            compressed_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    def stay_locations(self, inplace: bool = False, **kwargs) -> TrajDataFrame:
        """Detect stay locations (stops) in the trajectory.

        Parameters
        ----------
        inplace : bool, optional
            If ``True``, modify this object and return ``self``.
            If ``False`` (default), return a new ``TrajDataFrame``.
        **kwargs
            Extra keyword arguments forwarded to
            ``fastmob.preprocessing.stay_locations``.

        Returns
        -------
        TrajDataFrame
            A DataFrame whose rows are detected stops, with an extra
            ``leaving_datetime`` column.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1, 1, 1],
        ...     "lat": [0.0, 0.0001, 0.0, 1.0],
        ...     "lng": [0.0, 0.0, 0.0001, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=4, freq="30min"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.stay_locations(minutes_for_a_stop=20)  # doctest: +SKIP
        """
        from fastmob.preprocessing import stay_locations

        col_kwargs = {
            "datetime_col": self.datetime_col,
            "lat_col": self.lat_col,
            "lng_col": self.lng_col,
            "uid_col": self.uid_col,
        }
        if inplace:
            self.df = stay_locations(self.df, **col_kwargs, **kwargs)
            return self

        stay_df = stay_locations(self.df, **col_kwargs, **kwargs)
        return TrajDataFrame(
            stay_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
        )

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def to_flowdataframe(self, tessellation, self_loops: bool = True):
        """Aggregate the trajectory into a FlowDataFrame using a tessellation.

        Points outside the tessellation are silently dropped.

        Parameters
        ----------
        tessellation : geopandas.GeoDataFrame
            Spatial tessellation with a ``tile_id`` column and polygon geometries.
        self_loops : bool, optional
            If ``True`` (default), include movements that start and end in the
            same tile.

        Returns
        -------
        FlowDataFrame

        Notes
        -----
        Requires ``fastmob[data]``::

            pip install "fastmob[data]"

        Examples
        --------
        >>> import fastmob
        >>> tdf = fastmob.data.load_dataset("foursquare_nyc")  # doctest: +SKIP
        >>> from fastmob.tessellation.tilers import tiler  # doctest: +SKIP
        >>> tess = tiler.get("squared", base_shape="New York City", meters=2000)  # doctest: +SKIP
        >>> fdf = tdf.to_flowdataframe(tess)  # doctest: +SKIP
        """
        from fastmob.core import FlowDataFrame

        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("geopandas is required for flow datasets: pip install fastmob[data]") from exc

        frame = self._to_pandas()
        frame = frame.sort_values([self.uid_col, self.datetime_col], kind="mergesort").reset_index(drop=True)
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(frame[self.lng_col], frame[self.lat_col]),
            crs="EPSG:4326",
        )
        tess = tessellation
        if getattr(tess, "crs", None) is None:
            tess = tess.set_crs("EPSG:4326")
        if points.crs != tess.crs:
            points = points.to_crs(tess.crs)

        joined = gpd.sjoin(points, tess[[TILE_ID, "geometry"]], how="left", predicate="within")
        joined["destination"] = joined[TILE_ID].shift(-1)
        joined["next_uid"] = joined[self.uid_col].shift(-1)
        flow = joined[joined[self.uid_col] == joined["next_uid"]].dropna(subset=[TILE_ID, "destination"])
        flow = flow.groupby([TILE_ID, "destination"], dropna=True).size().reset_index(name="flow")
        flow = flow.rename(columns={TILE_ID: "origin"})
        if not self_loops:
            flow = flow[flow["origin"] != flow["destination"]]
        return FlowDataFrame(flow, tessellation=tessellation)

    def to_geodataframe(self):
        """Convert to a ``geopandas.GeoDataFrame`` with Point geometry.

        Returns
        -------
        geopandas.GeoDataFrame
            Same rows as the trajectory, with an additional ``geometry``
            column containing ``shapely.geometry.Point`` objects built from
            the latitude and longitude columns.

        Notes
        -----
        Requires ``fastmob[data]``::

            pip install "fastmob[data]"

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1],
        ...     "lat": [48.8566, 48.8578],
        ...     "lng": [2.3522, 2.3530],
        ...     "datetime": pd.date_range("2020-01-01", periods=2, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> gdf = tdf.to_geodataframe()  # doctest: +SKIP
        """
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError('geopandas is required: pip install "fastmob[data]"') from exc

        native_df = self._to_pandas()
        return gpd.GeoDataFrame(
            native_df,
            geometry=gpd.points_from_xy(native_df[self.lng_col], native_df[self.lat_col]),
            crs="EPSG:4326",
        )

    def mapping(self, tessellation, remove_na: bool = False) -> TrajDataFrame:
        """Assign each trajectory point to a tile in a spatial tessellation.

        Adds a ``tile_id`` column to the result.

        Parameters
        ----------
        tessellation : geopandas.GeoDataFrame
            Spatial tessellation with Polygon or Point geometries and a
            ``tile_id`` column.
        remove_na : bool, optional
            If ``True``, remove points that fall outside the tessellation.
            Default ``False`` (keep them with ``NaN`` tile_id).

        Returns
        -------
        TrajDataFrame
            Original trajectory with an extra ``tile_id`` column.

        Notes
        -----
        Requires ``fastmob[data]``::

            pip install "fastmob[data]"

        Examples
        --------
        >>> import fastmob
        >>> tdf = fastmob.data.load_dataset("foursquare_nyc")  # doctest: +SKIP
        >>> from fastmob.tessellation.tilers import tiler  # doctest: +SKIP
        >>> tess = tiler.get("squared", base_shape="New York City", meters=2000)  # doctest: +SKIP
        >>> mapped = tdf.mapping(tess)  # doctest: +SKIP
        """
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import Point, Polygon
        except ImportError as exc:
            raise ImportError('geopandas and shapely are required: pip install "fastmob[data]"') from exc

        gdf = self.to_geodataframe()

        tess = tessellation
        if getattr(tess, "crs", None) is None:
            tess = tess.set_crs("EPSG:4326")
        if gdf.crs != tess.crs:
            gdf = gdf.to_crs(tess.crs)

        tile_id_col = TILE_ID
        # Ensure tessellation has the tile_id column
        if tile_id_col not in tess.columns:
            raise ValueError(f"Tessellation must have a '{tile_id_col}' column.")

        if all(isinstance(x, Polygon) for x in tess.geometry):
            how = "inner" if remove_na else "left"
            joined = gpd.sjoin(gdf, tess[[tile_id_col, "geometry"]], how=how, predicate="within")
            tile_ids = joined[[tile_id_col]]
        elif all(isinstance(x, Point) for x in tess.geometry):
            tile_series = _nearest_tessellation_values(gdf, tess, tile_id_col)
            tile_ids = pd.DataFrame({tile_id_col: tile_series.values}, index=tile_series.index)
        else:
            raise ValueError("Tessellation geometry must be all Polygon or all Point.")

        native_df = self._to_pandas()
        result_df = native_df.merge(tile_ids, left_index=True, right_index=True)
        return TrajDataFrame(
            result_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            crs=self.crs,
            parameters=self.parameters.copy(),
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def sort_by_uid_and_datetime(self) -> TrajDataFrame:
        """Return a copy sorted by user ID then datetime.

        Returns
        -------
        TrajDataFrame
            New TrajDataFrame with rows sorted ascending by
            ``(uid, datetime)``.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [2, 1, 1],
        ...     "lat": [0.0, 1.0, 2.0],
        ...     "lng": [0.0, 0.0, 0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> sorted_tdf = tdf.sort_by_uid_and_datetime()
        """
        sort_cols = []
        if self.uid_col:
            sort_cols.append(self.uid_col)
        sort_cols.append(self.datetime_col)

        nw_df = nw.from_native(self.df, eager_only=True)
        sorted_df = nw_df.sort(sort_cols).to_native()

        result = TrajDataFrame(
            sorted_df,
            datetime_col=self.datetime_col,
            lat_col=self.lat_col,
            lng_col=self.lng_col,
            uid_col=self.uid_col,
            crs=self.crs,
            parameters=self.parameters.copy(),
        )
        result.sorted = True
        return result

    def settings_from(self, other: TrajDataFrame) -> None:
        """Copy metadata attributes from another TrajDataFrame.

        Parameters
        ----------
        other : TrajDataFrame
            Source TrajDataFrame to copy attributes from.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1], "lat": [0.0], "lng": [0.0],
        ...     "datetime": pd.date_range("2020-01-01", periods=1),
        ... })
        >>> tdf1 = fastmob.TrajDataFrame(df.copy())
        >>> tdf2 = fastmob.TrajDataFrame(df.copy(), parameters={"source": "gps"})
        >>> tdf1.settings_from(tdf2)
        >>> tdf1.parameters
        {'source': 'gps'}
        """
        self.crs = getattr(other, "crs", self.crs)
        self.parameters = dict(getattr(other, "parameters", self.parameters))
        self.sorted = getattr(other, "sorted", self.sorted)
        self.uid_col = getattr(other, "uid_col", self.uid_col)
        self.datetime_col = getattr(other, "datetime_col", self.datetime_col)
        self.lat_col = getattr(other, "lat_col", self.lat_col)
        self.lng_col = getattr(other, "lng_col", self.lng_col)
        self.trajectory_id_col = getattr(other, "trajectory_id_col", self.trajectory_id_col)

    def timezone_conversion(self, from_timezone: str, to_timezone: str) -> None:
        """Convert the datetime column from one timezone to another, in place.

        The result has timezone information stripped (tz-naive), matching the
        behaviour of the original scikit-mobility implementation.

        Parameters
        ----------
        from_timezone : str
            Current timezone of the datetime column, e.g. ``'GMT'``.
        to_timezone : str
            Target timezone, e.g. ``'Asia/Shanghai'``.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> df = pd.DataFrame({
        ...     "uid": [1, 1],
        ...     "lat": [39.984, 39.985],
        ...     "lng": [116.319, 116.320],
        ...     "datetime": pd.to_datetime(["2008-10-23 05:53:05", "2008-10-23 05:53:06"]),
        ... })
        >>> tdf = fastmob.TrajDataFrame(df)
        >>> tdf.timezone_conversion("GMT", "Asia/Shanghai")
        >>> tdf[tdf.datetime_col].iloc[0]  # doctest: +SKIP
        Timestamp('2008-10-23 13:53:05')
        """
        nw_df = nw.from_native(self.df, eager_only=True)
        self.df = nw_df.with_columns(
            nw.col(self.datetime_col)
            .dt.replace_time_zone(from_timezone)
            .dt.convert_time_zone(to_timezone)
            .dt.replace_time_zone(None)
        ).to_native()

    # ------------------------------------------------------------------
    # Visualization methods  (require fastmob[vis])
    # ------------------------------------------------------------------

    def plot_trajectory(
        self,
        map_f=None,
        max_users=None,
        max_points: int = 1000,
        style_function=None,
        tiles: str = "cartodbpositron",
        zoom: int = 12,
        hex_color=None,
        weight: float = 2,
        opacity: float = 0.75,
        dashArray: str = "0, 0",
        start_end_markers: bool = True,
        control_scale: bool = True,
    ):
        """Plot trajectories on an interactive Folium map.

        Parameters
        ----------
        map_f : folium.Map, optional
            Existing map to draw on. Creates a new map if ``None``.
        max_users : int, optional
            Maximum number of users to plot. Defaults to 10 with a warning.
        max_points : int, optional
            Maximum GPS points per user. Trajectories are down-sampled if
            longer. Default 1000.
        style_function : callable, optional
            GeoJson style factory ``(weight, color, opacity, dashArray) → fn``.
            Defaults to ``fastmob_vis.plot.traj_style_function``.
        tiles : str, optional
            Folium tile layer name. Default ``'cartodbpositron'``.
        zoom : int, optional
            Initial zoom level. Default 12.
        hex_color : str, optional
            Fixed hex color for all lines. Random color per user if ``None``.
        weight : float, optional
            Line thickness. Default 2.
        opacity : float, optional
            Line opacity. Default 0.75.
        dashArray : str, optional
            SVG dash pattern, e.g. ``'5, 5'``. Default ``'0, 0'`` (solid).
        start_end_markers : bool, optional
            Add green/red markers at start and end. Default ``True``.
        control_scale : bool, optional
            Add a map scale bar. Default ``True``.

        Returns
        -------
        folium.Map

        Notes
        -----
        Requires ``fastmob[vis]``::

            pip install "fastmob[vis]"

        Examples
        --------
        >>> import fastmob
        >>> tdf = fastmob.data.load_dataset("foursquare_nyc")  # doctest: +SKIP
        >>> m = tdf.plot_trajectory(zoom=12)  # doctest: +SKIP
        """
        try:
            plot = require_optional("fastmob_vis.plot", "vis")
        except ImportError as exc:
            raise ImportError('Visualization requires fastmob-vis: pip install "fastmob[vis]"') from exc

        kwargs: dict[str, Any] = {
            "map_f": map_f,
            "max_users": max_users,
            "max_points": max_points,
            "tiles": tiles,
            "zoom": zoom,
            "hex_color": hex_color,
            "weight": weight,
            "opacity": opacity,
            "dashArray": dashArray,
            "start_end_markers": start_end_markers,
            "control_scale": control_scale,
        }
        if style_function is not None:
            kwargs["style_function"] = style_function
        return plot.plot_trajectory(self._to_pandas(), **kwargs)

    def plot_stops(
        self,
        map_f=None,
        max_users=None,
        tiles: str = "cartodbpositron",
        zoom: int = 12,
        hex_color=None,
        opacity: float = 0.3,
        radius: float = 12,
        number_of_sides: int = 4,
        popup: bool = True,
        control_scale: bool = True,
    ):
        """Plot detected stop locations on an interactive Folium map.

        Requires a TrajDataFrame with a ``leaving_datetime`` column
        (output of :func:`fastmob.preprocessing.stay_locations`).

        Parameters
        ----------
        map_f : folium.Map, optional
            Existing map. Creates a new map if ``None``.
        max_users : int, optional
            Maximum number of users to plot. Defaults to 10 with a warning.
        tiles : str, optional
            Folium tile layer. Default ``'cartodbpositron'``.
        zoom : int, optional
            Initial zoom. Default 12.
        hex_color : str, optional
            Fixed hex color. Random color per user if ``None``.
        opacity : float, optional
            Marker fill opacity. Default 0.3.
        radius : float, optional
            Marker radius. Default 12.
        number_of_sides : int, optional
            Number of polygon sides for each marker. Default 4.
        popup : bool, optional
            Show an info popup on click. Default ``True``.
        control_scale : bool, optional
            Add a scale bar. Default ``True``.

        Returns
        -------
        folium.Map

        Notes
        -----
        Requires ``fastmob[vis]``::

            pip install "fastmob[vis]"

        Examples
        --------
        >>> import fastmob
        >>> tdf = fastmob.data.load_dataset("foursquare_nyc")  # doctest: +SKIP
        >>> stdf = tdf.stay_locations(minutes_for_a_stop=20)  # doctest: +SKIP
        >>> m = stdf.plot_stops(zoom=12)  # doctest: +SKIP
        """
        try:
            plot = require_optional("fastmob_vis.plot", "vis")
        except ImportError as exc:
            raise ImportError('Visualization requires fastmob-vis: pip install "fastmob[vis]"') from exc

        return plot.plot_stops(
            self._to_pandas(),
            map_f=map_f,
            max_users=max_users,
            tiles=tiles,
            zoom=zoom,
            hex_color=hex_color,
            opacity=opacity,
            radius=radius,
            number_of_sides=number_of_sides,
            popup=popup,
            control_scale=control_scale,
        )

    def plot_diary(
        self,
        user,
        start_datetime=None,
        end_datetime=None,
        ax=None,
        legend: bool = False,
    ):
        """Plot a mobility diary for a single user as a coloured time-span chart.

        Requires a clustered stop DataFrame (output of
        ``fastmob.preprocessing.cluster``), with ``cluster`` and
        ``leaving_datetime`` columns.

        Parameters
        ----------
        user : str or int
            Identifier of the user to plot.
        start_datetime : datetime, optional
            Only stops after this datetime are included. Defaults to the
            earliest stop.
        end_datetime : datetime, optional
            Only stops before this datetime are included. Defaults to the
            latest departure.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if ``None``.
        legend : bool, optional
            Show a cluster-ID legend. Default ``False``.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        Requires ``fastmob[vis]``::

            pip install "fastmob[vis]"

        Examples
        --------
        >>> import fastmob
        >>> tdf = fastmob.data.load_dataset("foursquare_nyc")  # doctest: +SKIP
        >>> stdf = tdf.stay_locations(minutes_for_a_stop=20)  # doctest: +SKIP
        >>> cstdf = fastmob.preprocessing.cluster(stdf)  # doctest: +SKIP
        >>> ax = cstdf.plot_diary(user=1)  # doctest: +SKIP
        """
        try:
            plot = require_optional("fastmob_vis.plot", "vis")
        except ImportError as exc:
            raise ImportError('Visualization requires fastmob-vis: pip install "fastmob[vis]"') from exc

        return plot.plot_diary(
            self._to_pandas(),
            user=user,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            ax=ax,
            legend=legend,
        )
