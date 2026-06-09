"""Visualization utilities for skmob2 TrajDataFrame and FlowDataFrame.

Requires ``skmob2[visualization]``::

    pip install "skmob2[visualization]"

All functions accept plain ``pandas.DataFrame`` objects (for trajectory/stop
plots) or skmob2 ``FlowDataFrame`` objects (for flow and tessellation plots).
"""

from __future__ import annotations

import json
import warnings

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from folium.plugins import HeatMap
from geojson import LineString

from skmob2.utils.utils import get_geom_centroid

# Column name constants (mirror skmob2 core defaults)
_UID = "uid"
_LAT = "lat"
_LNG = "lng"
_DATETIME = "datetime"
_LEAVING_DATETIME = "leaving_datetime"
_CLUSTER = "cluster"
_ORIGIN = "origin"
_DESTINATION = "destination"
_FLOW = "flow"
_TILE_ID = "tile_id"

STACKLEVEL = 2

# Color-blind-friendly palette from <http://mkweb.bcgsc.ca/colorblind/palettes.mhtml>
COLOR = {
    0: "#6A0213",
    1: "#008607",
    2: "#F60239",
    3: "#00E307",
    4: "#FFDC3D",
    5: "#003C86",
    6: "#9400E6",
    7: "#009FFA",
    8: "#FF71FD",
    9: "#7CFFFA",
    10: "#68023F",
    11: "#008169",
    12: "#EF0096",
    13: "#00DCB5",
    14: "#FFCFE2",
}


def get_color(k: int = -2, color_dict: dict = COLOR) -> str:
    """Return a color from the palette (gray if k == -1, random if k < -1)."""
    if k < -1:
        return np.random.choice(list(color_dict.values()))
    if k == -1:
        return "#808080"
    return color_dict[k % len(color_dict)]


def random_hex() -> str:
    """Return a random hex color string."""

    def _rand():
        return np.random.randint(0, 255)

    return "#%02X%02X%02X" % (_rand(), _rand(), _rand())


def traj_style_function(weight, color, opacity, dashArray):
    """Return a Folium GeoJson style function for trajectory lines."""
    return lambda feature: dict(color=color, weight=weight, opacity=opacity, dashArray=dashArray)


def flow_style_function(weight, color, opacity, weight_factor, flow_exp):
    """Return a Folium GeoJson style function for flow edges."""
    return lambda feature: dict(color=color, weight=weight_factor * weight**flow_exp, opacity=opacity)


def plot_trajectory(
    tdf: pd.DataFrame,
    map_f=None,
    max_users=None,
    max_points: int = 1000,
    style_function=traj_style_function,
    tiles: str = "cartodbpositron",
    zoom: int = 12,
    hex_color=None,
    weight: float = 2,
    opacity: float = 0.75,
    dashArray: str = "0, 0",
    start_end_markers: bool = True,
    control_scale: bool = True,
):
    """Plot trajectories on a Folium map.

    Parameters
    ----------
    tdf : pandas.DataFrame
        DataFrame with at least ``lng``, ``lat``, and ``datetime`` columns.
    map_f : folium.Map, optional
        Existing map to draw on. If ``None``, a new map is created.
    max_users : int, optional
        Maximum number of users to plot. Defaults to 10 with a warning.
    max_points : int, optional
        Maximum points per user; trajectory is down-sampled if longer. Default 1000.
    style_function : callable, optional
        GeoJson style factory ``(weight, color, opacity, dashArray) -> style_fn``.
    tiles : str, optional
        Folium tile layer. Default ``'cartodbpositron'``.
    zoom : int, optional
        Initial zoom level. Default 12.
    hex_color : str, optional
        Fixed hex color. If ``None``, a random color is chosen per user.
    weight : float, optional
        Line thickness. Default 2.
    opacity : float, optional
        Line opacity. Default 0.75.
    dashArray : str, optional
        Dash pattern, e.g. ``'5, 5'`` for dashed lines. Default ``'0, 0'``.
    start_end_markers : bool, optional
        Whether to add green/red markers at the start and end. Default ``True``.
    control_scale : bool, optional
        Add a scale bar. Default ``True``.

    Returns
    -------
    folium.Map
    """
    if max_users is None:
        max_users = 10
        warnings.warn(
            "Only the trajectories of the first 10 users will be plotted. "
            "Use `max_users` to change this.",
            stacklevel=STACKLEVEL,
        )

    nu = 0
    try:
        groups = tdf.groupby(_UID)
    except KeyError:
        groups = [[None, tdf]]

    warned = False
    for user, df in groups:
        if nu >= max_users:
            break
        nu += 1

        traj = df[[_LNG, _LAT]]

        if max_points is None:
            di = 1
        else:
            if not warned:
                warnings.warn(
                    "Trajectories will be down-sampled to at most `max_points` points. "
                    "Pass `max_points=None` to disable.",
                    stacklevel=STACKLEVEL,
                )
                warned = True
            di = max(1, len(traj) // max_points)
        traj = traj[::di]

        if nu == 1 and map_f is None:
            center = list(np.median(traj, axis=0)[::-1])
            map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=control_scale)

        trajlist = traj.values.tolist()
        line = LineString(trajlist)

        color = get_color(-2) if hex_color is None else hex_color

        tgeojson = folium.GeoJson(
            line,
            name="tgeojson",
            style_function=style_function(weight, color, opacity, dashArray),
        )
        tgeojson.add_to(map_f)

        if start_end_markers:
            dtime, la, lo = df.loc[df[_DATETIME].idxmin()][[_DATETIME, _LAT, _LNG]].values
            dtime = dtime.strftime("%Y/%m/%d %H:%M")
            mker = folium.Marker(trajlist[0][::-1], icon=folium.Icon(color="green"))
            popup = folium.Popup(
                "<i>Start</i><BR>{}<BR>Coord: <a href='https://www.google.co.uk/maps/place/{},{}'"
                " target='_blank'>{}, {}</a>".format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)),
                max_width=300,
            )
            mker = mker.add_child(popup)
            mker.add_to(map_f)

            dtime, la, lo = df.loc[df[_DATETIME].idxmax()][[_DATETIME, _LAT, _LNG]].values
            dtime = dtime.strftime("%Y/%m/%d %H:%M")
            mker = folium.Marker(trajlist[-1][::-1], icon=folium.Icon(color="red"))
            popup = folium.Popup(
                "<i>End</i><BR>{}<BR>Coord: <a href='https://www.google.co.uk/maps/place/{},{}'"
                " target='_blank'>{}, {}</a>".format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)),
                max_width=300,
            )
            mker = mker.add_child(popup)
            mker.add_to(map_f)

    return map_f


def plot_points_heatmap(
    tdf: pd.DataFrame,
    map_f=None,
    max_points: int = 1000,
    tiles: str = "cartodbpositron",
    zoom: int = 2,
    min_opacity: float = 0.5,
    radius: int = 25,
    blur: int = 15,
    gradient=None,
):
    """Plot a heatmap of trajectory points on a Folium map.

    Parameters
    ----------
    tdf : pandas.DataFrame
        DataFrame with ``lat`` and ``lng`` columns.
    map_f : folium.Map, optional
        Existing map. If ``None``, a new map is created.
    max_points : int, optional
        Maximum total points to plot. Default 1000.
    tiles : str, optional
        Folium tile layer. Default ``'cartodbpositron'``.
    zoom : int, optional
        Initial zoom. Default 2.
    min_opacity : float, optional
        Minimum heatmap opacity. Default 0.5.
    radius : int, optional
        Radius of each point. Default 25.
    blur : int, optional
        Blur amount. Default 15.
    gradient : dict, optional
        Color gradient, e.g. ``{0.4: 'blue', 0.65: 'lime', 1: 'red'}``.

    Returns
    -------
    folium.Map
    """
    di = 1 if max_points is None else max(1, len(tdf) // max_points)
    traj = tdf[::di][[_LAT, _LNG]]

    if map_f is None:
        center = list(np.median(tdf[[_LNG, _LAT]], axis=0)[::-1])
        map_f = folium.Map(zoom_start=zoom, tiles=tiles, control_scale=True, location=center)

    HeatMap(traj.values, min_opacity=min_opacity, radius=radius, blur=blur, gradient=gradient).add_to(map_f)
    return map_f


def plot_stops(
    stdf: pd.DataFrame,
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
    """Plot stop locations on a Folium map.

    Expects a DataFrame with a ``leaving_datetime`` column (output of
    ``skmob2.preprocessing.stay_locations``).

    Parameters
    ----------
    stdf : pandas.DataFrame
        Stop DataFrame with ``lat``, ``lng``, ``datetime``, and
        ``leaving_datetime`` columns.
    map_f : folium.Map, optional
        Existing map. If ``None``, a new map is created.
    max_users : int, optional
        Maximum number of users to plot. Defaults to 10 with a warning.
    tiles : str, optional
        Folium tile layer. Default ``'cartodbpositron'``.
    zoom : int, optional
        Initial zoom. Default 12.
    hex_color : str, optional
        Fixed hex color for markers. If ``None``, random colors are chosen.
    opacity : float, optional
        Marker fill opacity. Default 0.3.
    radius : float, optional
        Marker size. Default 12.
    number_of_sides : int, optional
        Number of sides for the polygon marker. Default 4.
    popup : bool, optional
        Show a popup on click. Default ``True``.
    control_scale : bool, optional
        Add a scale bar. Default ``True``.

    Returns
    -------
    folium.Map
    """
    if max_users is None:
        max_users = 10
        warnings.warn(
            "Only the stops of the first 10 users will be plotted. "
            "Use `max_users` to change this.",
            stacklevel=STACKLEVEL,
        )

    if map_f is None:
        lo_la = stdf[[_LNG, _LAT]].values
        center = list(np.median(lo_la, axis=0)[::-1])
        map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=control_scale)

    nu = 0
    try:
        groups = stdf.groupby(_UID)
    except KeyError:
        groups = [[None, stdf]]

    for user, df in groups:
        if nu >= max_users:
            break
        nu += 1

        color = get_color(-2) if hex_color is None else hex_color

        for idx, row in df.iterrows():
            la = row[_LAT]
            lo = row[_LNG]
            t0 = row[_DATETIME]
            try:
                t1 = row[_LEAVING_DATETIME]
                _sides = number_of_sides
                marker_radius = radius
            except KeyError:
                t1 = t0
                _sides = number_of_sides
                marker_radius = radius // 2

            u = user
            cl = ""
            try:
                ncluster = row[_CLUSTER]
                cl = "<BR>Cluster: {}".format(ncluster)
                color = get_color(ncluster)
            except (KeyError, NameError):
                pass

            fpoly = folium.RegularPolygonMarker(
                [la, lo],
                radius=marker_radius,
                color=color,
                fill_color=color,
                fill_opacity=opacity,
                number_of_sides=_sides,
            )
            if popup:
                popup_html = folium.Popup(
                    "User: {}<BR>Coord: <a href='https://www.google.co.uk/maps/place/{},{}'"
                    " target='_blank'>{}, {}</a><BR>Arr: {}<BR>Dep: {}{}".format(
                        u,
                        la,
                        lo,
                        np.round(la, 4),
                        np.round(lo, 4),
                        t0.strftime("%Y/%m/%d %H:%M"),
                        t1.strftime("%Y/%m/%d %H:%M"),
                        cl,
                    ),
                    max_width=300,
                )
                fpoly = fpoly.add_child(popup_html)

            fpoly.add_to(map_f)

    return map_f


def plot_diary(
    cstdf: pd.DataFrame,
    user,
    start_datetime=None,
    end_datetime=None,
    ax=None,
    legend: bool = False,
):
    """Plot a mobility diary for one user as a coloured time-span chart.

    Requires a clustered stop DataFrame (output of ``skmob2.preprocessing.cluster``),
    with ``cluster`` and ``leaving_datetime`` columns.

    Parameters
    ----------
    cstdf : pandas.DataFrame
        Clustered stops DataFrame.
    user : str or int
        User identifier to plot.
    start_datetime : datetime, optional
        Start of the plot window. Defaults to the earliest stop.
    end_datetime : datetime, optional
        End of the plot window. Defaults to the latest departure.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.
    legend : bool, optional
        Show a cluster-ID legend. Default ``False``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(20, 2))

    df = cstdf if user is None else cstdf[cstdf[_UID] == user]

    if len(df) == 0:
        raise KeyError("User id is not in the input DataFrame.")

    if start_datetime is None:
        start_datetime = df[_DATETIME].min()
    elif isinstance(start_datetime, str):
        start_datetime = pd.to_datetime(start_datetime)
    if end_datetime is None:
        end_datetime = df[_LEAVING_DATETIME].max()
    elif isinstance(end_datetime, str):
        end_datetime = pd.to_datetime(end_datetime)

    current_labels: list = []
    for _idx, row in df.iterrows():
        t0 = row[_DATETIME]
        t1 = row[_LEAVING_DATETIME]
        cl = row[_CLUSTER]
        color = get_color(cl)
        if start_datetime <= t0 <= end_datetime:
            if cl in current_labels:
                ax.axvspan(t0.to_pydatetime(), t1.to_pydatetime(), lw=0.0, alpha=0.75, color=color)
            else:
                current_labels.append(cl)
                ax.axvspan(t0.to_pydatetime(), t1.to_pydatetime(), lw=0.0, alpha=0.75, color=color, label=cl)

    plt.xlim(start_datetime, end_datetime)

    if legend:
        import operator

        handles, labels_str = ax.get_legend_handles_labels()
        labels = list(map(int, labels_str))
        hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
        handles2, labels2 = zip(*hl)
        ax.legend(handles2, labels2, ncol=15, bbox_to_anchor=(1.0, -0.2), frameon=0)

    ax.set_title("user %s" % user)
    return ax


def plot_flows(
    fdf,
    map_f=None,
    min_flow: float = 0,
    tiles: str = "cartodbpositron",
    zoom: int = 6,
    flow_color: str = "red",
    opacity: float = 0.5,
    flow_weight: float = 5,
    flow_exp: float = 0.5,
    style_function=flow_style_function,
    flow_popup: bool = False,
    num_od_popup: int = 5,
    tile_popup: bool = True,
    radius_origin_point: float = 5,
    color_origin_point: str = "#3186cc",
    control_scale: bool = True,
):
    """Plot origin-destination flows on a Folium map.

    Parameters
    ----------
    fdf : FlowDataFrame
        skmob2 FlowDataFrame with a tessellation attached.
    map_f : folium.Map, optional
        Existing map. If ``None``, a new map is created.
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
        GeoJson style factory.
    flow_popup : bool, optional
        Show a popup on edge click. Default ``False``.
    num_od_popup : int, optional
        Number of OD pairs shown in each origin popup. Default 5.
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
    """
    if map_f is None:
        lon, lat = np.mean(
            np.array(list(fdf.tessellation.geometry.apply(get_geom_centroid).values)),
            axis=0,
        )
        map_f = folium.Map(location=[lat, lon], tiles=tiles, zoom_start=zoom, control_scale=control_scale)

    mean_flows = fdf[_FLOW].mean()

    orig_groups = fdf.groupby(by=_ORIGIN)
    for orig, od in orig_groups:
        geom = fdf.get_geometry(orig)
        lon_o, lat_o = get_geom_centroid(geom)

        for dest, flow_val in od[[_DESTINATION, _FLOW]].values:
            if orig == dest:
                continue
            if flow_val < min_flow:
                continue

            geom = fdf.get_geometry(dest)
            lon_d, lat_d = get_geom_centroid(geom)

            gjc = LineString([(lon_o, lat_o), (lon_d, lat_d)])
            fgeojson = folium.GeoJson(
                gjc,
                name="geojson",
                style_function=style_function(flow_val / mean_flows, flow_color, opacity, flow_weight, flow_exp),
            )
            if flow_popup:
                popup = folium.Popup("flow from %s to %s: %s" % (orig, dest, int(flow_val)), max_width=300)
                fgeojson = fgeojson.add_child(popup)
            fgeojson.add_to(map_f)

    if radius_origin_point > 0:
        for orig, od in orig_groups:
            name = "origin: %s" % str(orig).replace("'", "_")
            t_d = [[fv, d] for d, fv in od[[_DESTINATION, _FLOW]].values]
            trips_info = "<br/>".join(
                ["flow to %s: %s" % (str(dd).replace("'", "_"), int(tt)) for tt, dd in sorted(t_d, reverse=True)[:num_od_popup]]
            )

            geom = fdf.get_geometry(orig)
            lon_o, lat_o = get_geom_centroid(geom)
            fmarker = folium.CircleMarker(
                [lat_o, lon_o],
                radius=radius_origin_point,
                weight=2,
                color=color_origin_point,
                fill=True,
                fill_color=color_origin_point,
            )
            if tile_popup:
                popup = folium.Popup(name + "<br/>" + trips_info, max_width=300)
                fmarker = fmarker.add_child(popup)
            fmarker.add_to(map_f)

    return map_f


# --- GeoDataFrame polygon/point plotting ---

default_style_func_args = {
    "weight": 1,
    "color": "random",
    "opacity": 0.5,
    "fillColor": "random",
    "fillOpacity": 0.25,
    "radius": 5,
}

def geojson_style_function(weight, color, opacity, fillColor, fillOpacity):
    """Return a Folium GeoJson style function for polygon/line features."""
    return lambda feature: dict(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity)


def manage_colors(color: str, fillColor: str):
    if color == "random":
        if fillColor == "random":
            color = random_hex()
            fillColor = color
        else:
            color = random_hex()
    elif fillColor == "random":
        fillColor = random_hex()
    return color, fillColor


def add_to_map(gway, g, map_f, style_func_args: dict, popup_features: list = []):
    """Add a single shapely geometry to a Folium map."""
    styles = []
    for k in ["weight", "color", "opacity", "fillColor", "fillOpacity", "radius"]:
        if k in style_func_args:
            v = style_func_args[k]
            styles.append(v(g) if callable(v) else v)
        else:
            styles.append(default_style_func_args[k])
    weight, color, opacity, fillColor, fillOpacity, radius = styles
    color, fillColor = manage_colors(color, fillColor)

    if isinstance(gway, shapely.geometry.multipolygon.MultiPolygon):
        vertices = [list(zip(*p.exterior.xy)) for p in gway.geoms]
        gj = folium.GeoJson(
            {"type": "MultiPolygon", "coordinates": [vertices]},
            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity),
        )
    elif isinstance(gway, shapely.geometry.polygon.Polygon):
        vertices = list(zip(*gway.exterior.xy))
        gj = folium.GeoJson(
            {"type": "Polygon", "coordinates": [vertices]},
            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity),
        )
    elif isinstance(gway, shapely.geometry.multilinestring.MultiLineString):
        vertices = [list(zip(*seg.xy)) for seg in gway.geoms]
        gj = folium.GeoJson(
            {"type": "MultiLineString", "coordinates": vertices},
            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity),
        )
    elif isinstance(gway, shapely.geometry.linestring.LineString):
        vertices = list(zip(*gway.xy))
        gj = folium.GeoJson(
            {"type": "LineString", "coordinates": vertices},
            style_function=geojson_style_function(weight=weight, color=color, opacity=opacity, fillColor=fillColor, fillOpacity=fillOpacity),
        )
    else:
        # Point
        point = list(zip(*gway.xy))[0]
        gj = folium.Circle(location=point[::-1], radius=radius, color=color, fill=True, fill_color=fillColor)

    popup_parts = []
    for pf in popup_features:
        try:
            popup_parts.append("%s: %s" % (pf, g[pf]))
        except KeyError:
            pass

    try:
        popup_str = "<br>".join(popup_parts)
        popup_str += json.dumps(g.tags)
        popup_str = popup_str.replace("'", "_")
    except AttributeError:
        popup_str = "<br>".join(popup_parts)

    if popup_str:
        gj.add_child(folium.Popup(popup_str, max_width=300))

    gj.add_to(map_f)
    return map_f


def plot_gdf(
    gdf,
    map_f=None,
    maxitems: int = -1,
    style_func_args: dict = {},
    popup_features: list = [],
    tiles: str = "cartodbpositron",
    zoom: int = 6,
    geom_col: str = "geometry",
    control_scale: bool = True,
):
    """Plot a GeoDataFrame on a Folium map.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to visualize.
    map_f : folium.Map, optional
        Existing map. If ``None``, a new map is created.
    maxitems : int, optional
        Maximum tiles to plot. ``-1`` means all. Default -1.
    style_func_args : dict, optional
        Style overrides for weight, color, opacity, fillColor, fillOpacity, radius.
    popup_features : list, optional
        Column names to include in click popups.
    tiles : str, optional
        Folium tile layer. Default ``'cartodbpositron'``.
    zoom : int, optional
        Initial zoom. Default 6.
    geom_col : str, optional
        Name of the geometry column. Default ``'geometry'``.
    control_scale : bool, optional
        Add a scale bar. Default ``True``.

    Returns
    -------
    folium.Map
    """
    if map_f is None:
        lon, lat = np.mean(
            np.array(list(gdf[geom_col].apply(get_geom_centroid).values)),
            axis=0,
        )
        map_f = folium.Map(location=[lat, lon], tiles=tiles, zoom_start=zoom, control_scale=control_scale)

    count = 0
    for k in gdf.index:
        g = gdf.loc[k]
        if isinstance(g[geom_col], gpd.geoseries.GeoSeries):
            for i in range(len(g[geom_col])):
                map_f = add_to_map(g[geom_col].iloc[i], g.iloc[i], map_f, popup_features=popup_features, style_func_args=style_func_args)
        else:
            map_f = add_to_map(g[geom_col], g, map_f, popup_features=popup_features, style_func_args=style_func_args)

        count += 1
        if count == maxitems:
            break

    return map_f
