"""FlowDataFrame — Narwhals-backed origin-destination flow wrapper."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa

from fastmob.core.base import BaseDataFrame
from fastmob.utils._common import require_optional

TILE_ID = "tile_id"
ORIGIN = "origin"
DESTINATION = "destination"
FLOW = "flow"


class FlowDataFrame(BaseDataFrame):
    """Narwhals-backed wrapper for origin-destination flow data.

    Stores a flow table with columns ``origin``, ``destination``, and
    ``flow``, optionally coupled with a spatial tessellation
    (``geopandas.GeoDataFrame``).

    Parameters
    ----------
    df : DataFrame-like, optional
        Source data.  Accepted types: ``pandas.DataFrame``,
        ``polars.DataFrame``, any Narwhals-compatible eager frame, ``list``,
        ``numpy.ndarray``, or ``dict``.
    origin : str, optional
        Column name for origin tile IDs. Default ``'origin'``.
    destination : str, optional
        Column name for destination tile IDs. Default ``'destination'``.
    flow : str, optional
        Column name for flow values. Default ``'flow'``.
    tile_id : str, optional
        Column name for tile IDs in the tessellation. Default ``'tile_id'``.
    tessellation : geopandas.GeoDataFrame, optional
        Spatial tessellation associated with the flow data.
    parameters : dict, optional
        Arbitrary metadata dictionary. Default ``{}``.

    Examples
    --------
    >>> import pandas as pd
    >>> import fastmob
    >>> flows = pd.DataFrame({
    ...     "origin": ["A", "A", "B"],
    ...     "destination": ["A", "B", "A"],
    ...     "flow": [100, 50, 30],
    ... })
    >>> fdf = fastmob.FlowDataFrame(flows)
    >>> fdf.get_flow("A", "B")
    50
    """

    def __init__(
        self,
        df=None,
        origin: str = ORIGIN,
        destination: str = DESTINATION,
        flow: str = FLOW,
        tile_id: str = TILE_ID,
        tessellation: Any | None = None,
        locations: Any | None = None,
        parameters: dict | None = None,
        **kwargs,
    ):
        if isinstance(df, FlowDataFrame):
            super().__init__(df.df)
            self.tessellation = getattr(df, "tessellation", tessellation)
            self.locations = getattr(df, "locations", locations)
            self.tile_id = getattr(df, "tile_id", tile_id)
            self.parameters = getattr(df, "parameters", {})
            self._info = getattr(df, "_info", None)
            return

        self.locations = locations
        self.tessellation = locations if locations is not None else tessellation
        self.tile_id = tile_id
        self.parameters = {} if parameters is None else parameters
        self._info = None

        frame = self._coerce_frame(df, kwargs)
        frame = self._rename_columns(frame, {origin: ORIGIN, destination: DESTINATION, flow: FLOW})
        super().__init__(frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_frame(df, kwargs):
        if df is None:
            return pa.table({})
        if isinstance(df, dict):
            return pa.table(df)
        if isinstance(df, (list, np.ndarray)):
            if len(df) == 0:
                return pa.table({})
            n_cols = len(df[0])
            names = kwargs.get("columns") or [str(i) for i in range(n_cols)]
            arrays = [pa.array([row[i] for row in df]) for i in range(n_cols)]
            return pa.table(dict(zip(names, arrays)))
        return df

    @staticmethod
    def _rename_columns(df, mapping):
        mapping = {source: target for source, target in mapping.items() if source != target}
        if not mapping:
            return df
        try:
            return nw.from_native(df, eager_only=True).rename(mapping).to_native()
        except Exception:  # noqa: BLE001
            return df

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_flow(self, origin_id: str, destination_id: str) -> int | float:
        """Return the flow between two tile IDs (0 if no such pair exists).

        Parameters
        ----------
        origin_id : str
            Origin tile identifier.
        destination_id : str
            Destination tile identifier.

        Returns
        -------
        int or float
            Flow value, or 0 if the pair is not present.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> flows = pd.DataFrame({
        ...     "origin": ["A", "A", "B"],
        ...     "destination": ["A", "B", "A"],
        ...     "flow": [100, 50, 30],
        ... })
        >>> fdf = fastmob.FlowDataFrame(flows)
        >>> fdf.get_flow("A", "B")
        50
        >>> fdf.get_flow("B", "C")
        0
        """
        nw_df = nw.from_native(self.df, eager_only=True)
        matches = nw_df.filter((nw.col(ORIGIN) == origin_id) & (nw.col(DESTINATION) == destination_id))
        if len(matches) == 0:
            return 0
        return matches.get_column(FLOW).to_list()[0]

    def common_part_of_commuters(self, other: FlowDataFrame) -> float:
        """Compare sparse OD flows with another FlowDataFrame using Rust CPC.

        Parameters
        ----------
        other : FlowDataFrame
            Reference sparse OD flows with the same location identity scheme.

        Returns
        -------
        float
            Common part of commuters score.

        Examples
        --------
        >>> score = flows.common_part_of_commuters(reference_flows)  # doctest: +SKIP
        """
        from fastmob.measures.evaluation.cpc import common_part_of_commuters

        return common_part_of_commuters(self, other)

    def common_part_of_links(self, other: FlowDataFrame) -> float:
        """Return the common part of links score against another flow table.

        Parameters
        ----------
        other : FlowDataFrame
            Reference sparse OD flows.

        Returns
        -------
        float
            Common part of links score.

        Examples
        --------
        >>> score = flows.common_part_of_links(reference_flows)  # doctest: +SKIP
        """
        from fastmob.measures.evaluation.cpc import common_part_of_links

        return common_part_of_links(self, other)

    def settings_from(self, other: FlowDataFrame) -> None:
        """Copy metadata attributes from another FlowDataFrame.

        Parameters
        ----------
        other : FlowDataFrame
            Source FlowDataFrame to copy attributes from.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> flows = pd.DataFrame({"origin": ["A"], "destination": ["B"], "flow": [10]})
        >>> fdf1 = fastmob.FlowDataFrame(flows.copy())
        >>> fdf2 = fastmob.FlowDataFrame(flows.copy(), parameters={"year": 2020})
        >>> fdf1.settings_from(fdf2)
        >>> fdf1.parameters
        {'year': 2020}
        """
        self.tessellation = getattr(other, "tessellation", self.tessellation)
        self.tile_id = getattr(other, "tile_id", self.tile_id)
        self.parameters = dict(getattr(other, "parameters", self.parameters))

    def get_geometry(self, tile_id: str):
        """Return the geometry of a tessellation tile.

        Parameters
        ----------
        tile_id : str
            Identifier of the tile to look up.

        Returns
        -------
        shapely geometry
            The geometry associated with *tile_id* in the tessellation.

        Raises
        ------
        ValueError
            If no tessellation is attached or the tile ID is not found.

        Examples
        --------
        >>> import fastmob  # doctest: +SKIP
        >>> fdf = fastmob.data.load_dataset("flow_foursquare_nyc")  # doctest: +SKIP
        >>> geom = fdf.get_geometry("36005")  # doctest: +SKIP
        """
        if self.tessellation is None:
            raise ValueError("No tessellation attached to this FlowDataFrame.")

        tile_id_str = str(tile_id)
        col = self.tile_id  # column name in the tessellation GeoDataFrame
        matches = self.tessellation[self.tessellation[col].astype(str) == tile_id_str]
        if len(matches) == 0:
            raise ValueError(f'tile_id "{tile_id}" is not present in the tessellation.')
        return matches.geometry.iloc[0]

    # ------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------

    def to_matrix(self) -> np.ndarray:
        """Convert the flow table to a numpy matrix.

        The rows and columns are ordered by the tile IDs in the tessellation
        (if present) or by the sorted union of all origin and destination IDs.

        Returns
        -------
        numpy.ndarray
            Square flow matrix of shape ``(n_tiles, n_tiles)``.

        Examples
        --------
        >>> import pandas as pd
        >>> import fastmob
        >>> flows = pd.DataFrame({
        ...     "origin": ["A", "A", "B"],
        ...     "destination": ["A", "B", "A"],
        ...     "flow": [100, 50, 30],
        ... })
        >>> fdf = fastmob.FlowDataFrame(flows)
        >>> fdf.to_matrix()
        array([[100.,  50.],
               [ 30.,   0.]])
        """
        if len(self.df) == 0:
            return np.zeros((0, 0))

        nw_df = nw.from_native(self.df, eager_only=True)
        origins = nw_df.get_column(ORIGIN).to_list()
        destinations = nw_df.get_column(DESTINATION).to_list()
        if self.tessellation is not None and self.tile_id in self.tessellation:
            tile_ids = [self._key(value) for value in self.tessellation[self.tile_id].values]
        else:
            tile_ids = sorted({self._key(value) for value in origins} | {self._key(value) for value in destinations})

        index = {tile_id: i for i, tile_id in enumerate(tile_ids)}
        matrix = np.zeros((len(tile_ids), len(tile_ids)), dtype=float)
        flows = nw_df.get_column(FLOW).to_list()
        for origin, destination, flow in zip(origins, destinations, flows):
            matrix[index[self._key(origin)], index[self._key(destination)]] = flow
        return matrix

    @staticmethod
    def _key(value) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    # ------------------------------------------------------------------
    # Visualization methods  (require fastmob[vis])
    # ------------------------------------------------------------------

    def _legacy_plot_flows(
        self,
        map_f=None,
        min_flow: float = 0,
        tiles: str = "cartodbpositron",
        zoom: int = 6,
        flow_color: str = "red",
        opacity: float = 0.5,
        flow_weight: float = 5,
        flow_exp: float = 0.5,
        style_function=None,
        flow_popup: bool = False,
        num_od_popup: int = 5,
        tile_popup: bool = True,
        radius_origin_point: float = 5,
        color_origin_point: str = "#3186cc",
        control_scale: bool = True,
    ):
        """Plot origin-destination flows on an interactive Folium map.

        Requires a tessellation to be attached (see :meth:`to_flowdataframe`).

        Parameters
        ----------
        map_f : folium.Map, optional
            Existing map. Creates a new map if ``None``.
        min_flow : float, optional
            Only flows larger than this value are drawn. Default 0.
        tiles : str, optional
            Folium tile layer. Default ``'cartodbpositron'``.
        zoom : int, optional
            Initial zoom. Default 6.
        flow_color : str, optional
            Color for flow edges. Default ``'red'``.
        opacity : float, optional
            Edge opacity. Default 0.5.
        flow_weight : float, optional
            Weight factor for edge thickness. Default 5.
        flow_exp : float, optional
            Exponent for edge thickness scaling. Default 0.5.
        style_function : callable, optional
            Custom GeoJson style factory. Defaults to
            ``fastmob_vis.plot.flow_style_function``.
        flow_popup : bool, optional
            Show a popup on edge click. Default ``False``.
        num_od_popup : int, optional
            Number of OD pairs in origin popup. Default 5.
        tile_popup : bool, optional
            Show origin-tile popup. Default ``True``.
        radius_origin_point : float, optional
            Radius of origin circle markers. Default 5.
        color_origin_point : str, optional
            Color of origin markers. Default ``'#3186cc'``.
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
        >>> fdf = fastmob.data.load_dataset("flow_foursquare_nyc")  # doctest: +SKIP
        >>> m = fdf.plot_flows(flow_color="red", zoom=10)  # doctest: +SKIP
        """
        try:
            plot = require_optional("fastmob_vis.plot", "vis")
        except ImportError as exc:
            raise ImportError('Visualization requires fastmob-vis: pip install "fastmob[vis]"') from exc

        kwargs: dict[str, Any] = {
            "map_f": map_f,
            "min_flow": min_flow,
            "tiles": tiles,
            "zoom": zoom,
            "flow_color": flow_color,
            "opacity": opacity,
            "flow_weight": flow_weight,
            "flow_exp": flow_exp,
            "flow_popup": flow_popup,
            "num_od_popup": num_od_popup,
            "tile_popup": tile_popup,
            "radius_origin_point": radius_origin_point,
            "color_origin_point": color_origin_point,
            "control_scale": control_scale,
        }
        if style_function is not None:
            kwargs["style_function"] = style_function
        return plot.plot_flows(self, **kwargs)

    def _legacy_plot_tessellation(
        self,
        map_f=None,
        maxitems: int = -1,
        style_func_args: dict | None = None,
        popup_features: list | None = None,
        tiles: str = "cartodbpositron",
        zoom: int = 6,
        geom_col: str = "geometry",
        control_scale: bool = True,
    ):
        """Plot the spatial tessellation on an interactive Folium map.

        Parameters
        ----------
        map_f : folium.Map, optional
            Existing map. Creates a new map if ``None``.
        maxitems : int, optional
            Maximum number of tiles to render. ``-1`` renders all. Default -1.
        style_func_args : dict, optional
            Style overrides: ``weight``, ``color``, ``opacity``,
            ``fillColor``, ``fillOpacity``, ``radius``.
        popup_features : list, optional
            Tessellation columns to show in click popups.
        tiles : str, optional
            Folium tile layer. Default ``'cartodbpositron'``.
        zoom : int, optional
            Initial zoom. Default 6.
        geom_col : str, optional
            Name of the geometry column in the tessellation. Default ``'geometry'``.
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
        >>> fdf = fastmob.data.load_dataset("flow_foursquare_nyc")  # doctest: +SKIP
        >>> m = fdf.plot_tessellation(popup_features=["tile_id", "population"])  # doctest: +SKIP
        """
        if popup_features is None:
            popup_features = []
        if style_func_args is None:
            style_func_args = {}
        try:
            plot = require_optional("fastmob_vis.plot", "vis")
        except ImportError as exc:
            raise ImportError('Visualization requires fastmob-vis: pip install "fastmob[vis]"') from exc

        if self.tessellation is None:
            raise ValueError("No tessellation attached to this FlowDataFrame.")

        return plot.plot_gdf(
            self.tessellation,
            map_f=map_f,
            maxitems=maxitems,
            style_func_args=style_func_args,
            popup_features=popup_features,
            tiles=tiles,
            zoom=zoom,
            geom_col=geom_col,
            control_scale=control_scale,
        )
