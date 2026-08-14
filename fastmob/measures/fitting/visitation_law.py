"""H3-first data preparation, binning, and fitting for the universal visitation law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fastmob._core import (
    aggregate_visitation_cells_arrow,
    bin_visitation_law_arrow,
    latlng_to_h3_centered_arrow,
    visitation_distances,
)
from fastmob.core.staypoints_dataframe import Staypoints
from fastmob.measures.individual.home_location import home_location
from fastmob.utils._common import (
    USER_ID_CANDIDATES,
    _as_arrow,
    _detect_required_column,
    _extract_timestamps,
    _factorize_arrow_values,
    _strip_time_zone,
)

_VISITATION_YLABEL = r"$\rho_i(r,f)$ (visitors km$^{-2}$)"
_H3_CELL_COL = "h3_cell"
# Internal-only: the dense UInt32 user codes `_prepare_visitation_law_data`
# already factorizes once, carried through so `_bin_visitation_law` can reuse
# them instead of factorizing `user_id_col` a second time. Dropped before
# `data` is returned to callers, so it never appears in the public schema.
_USER_CODE_COL = "__user_code__"


@dataclass(frozen=True)
class VisitationLawFit:
    """Fitted universal visitation law and the data used to estimate it.

    ``data`` contains per-user, per-H3-cell observations. ``spectrum`` holds
    the aggregate ``rf`` and ``rho`` values that were fitted.
    """

    data: Any
    spectrum: Any
    eta: float
    mu: float
    r2: float


def _empty_visitation_law_data(df: nw.DataFrame, user_id_col: str) -> Any:
    return nw.from_dict(
        {user_id_col: [], _H3_CELL_COL: [], "r_km": [], "f": [], "rf": [], "n_staypoints": []},
        backend=df.implementation,
    ).to_native()


def _arrow_series(df: nw.DataFrame, name: str, values: Any) -> nw.Series:
    """Build a backend-matching series from an Arrow-compatible result."""
    return nw.from_arrow(pa.table({name: pa.array(values)}), backend=df.implementation).get_column(name)


def _h3_centered_staypoints(
    df: nw.DataFrame,
    *,
    lat_col: str,
    lng_col: str,
    h3_resolution: int,
) -> nw.DataFrame:
    """Assign H3 cells and replace coordinates with each cell's fixed center."""
    cells, center_lats, center_lngs = latlng_to_h3_centered_arrow(
        df.get_column(lat_col).to_arrow(),
        df.get_column(lng_col).to_arrow(),
        h3_resolution,
    )
    return df.with_columns(
        _arrow_series(df, _H3_CELL_COL, cells).alias(_H3_CELL_COL),
        _arrow_series(df, lat_col, center_lats).alias(lat_col),
        _arrow_series(df, lng_col, center_lngs).alias(lng_col),
    ).drop_nulls(subset=[_H3_CELL_COL, lat_col, lng_col])


def _prepare_visitation_law_data(
    staypoints: Any | Staypoints,
    *,
    locations: Any | None = None,
    h3_resolution: int = 9,
    user_id_col: str | None = None,
    timestamp_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    start_night: int = 22,
    end_night: int = 7,
    presorted: bool = False,
) -> Any:
    """Prepare per-user H3-cell observations for the universal visitation law.

    A dataframe or :class:`~fastmob.core.Staypoints` is tessellated into
    shared H3 cells. Timezone-aware timestamps are converted to local naive
    time, matching the daily-motif convention. Home and destination positions
    are H3 cell centers, so one cell has one coordinate for every user.

    Returns rows ``[user_id, h3_cell, r_km, f, rf, n_staypoints]`` where ``f`` is
    the number of distinct local calendar days and ``rf = r_km * f``.

    Parameters
    ----------
    staypoints:
        Eager dataframe with user, timestamp, latitude, and longitude columns,
        or a :class:`~fastmob.core.Staypoints` object.
    h3_resolution:
        Shared H3 destination and home resolution, from 0 through 15.
    user_id_col, timestamp_col, lat_col, lng_col:
        Optional dataframe column overrides. Staypoints metadata supplies the
        defaults for a Staypoints input.
    start_night, end_night:
        Local hour bounds passed to :func:`fastmob.home_location`.
    presorted:
        Trust that staypoints are grouped by user for the home-detection fast path.
    """
    if isinstance(h3_resolution, bool) or not isinstance(h3_resolution, int) or not 0 <= h3_resolution <= 15:
        raise ValueError("h3_resolution must be an integer between 0 and 15")

    if locations is not None:
        if locations.scope != "global":
            raise ValueError("fit_visitation_law requires global Locations when locations is supplied")
        if isinstance(staypoints, Staypoints):
            staypoints = staypoints.associate_global_locations(locations)

    df, user_id_col, timestamp_col, lat_col, lng_col = Staypoints.resolve_dataframe(
        staypoints,
        user_id_col=user_id_col,
        timestamp_col=timestamp_col,
        lat_col=lat_col,
        lng_col=lng_col,
        require_coordinates=True,
    )
    df = _strip_time_zone(df, timestamp_col)
    if locations is None:
        centered = _h3_centered_staypoints(df, lat_col=lat_col, lng_col=lng_col, h3_resolution=h3_resolution)
    else:
        location_id_col = locations.location_id_col
        if location_id_col not in df.columns:
            raise ValueError(f"Staypoints is missing global location-ID column {location_id_col!r}")
        catalogue = (
            nw.from_native(locations.df, eager_only=True)
            .select([location_id_col, locations.center_lat_col, locations.center_lng_col])
            .rename(
                {
                    location_id_col: _H3_CELL_COL,
                    locations.center_lat_col: lat_col,
                    locations.center_lng_col: lng_col,
                }
            )
        )
        centered = df.drop(lat_col, lng_col).join(
            catalogue, left_on=location_id_col, right_on=_H3_CELL_COL, how="inner"
        )
    centered = centered.drop_nulls(subset=[user_id_col, timestamp_col, lat_col, lng_col])
    if len(centered) == 0:
        return _empty_visitation_law_data(df, user_id_col)

    homes = nw.from_native(
        home_location(
            centered.to_native(),
            start_night=start_night,
            end_night=end_night,
            datetime_col=timestamp_col,
            lat_col=lat_col,
            lng_col=lng_col,
            uid_col=user_id_col,
            presorted=presorted,
        ),
        eager_only=True,
    ).rename({lat_col: "home_lat", lng_col: "home_lng"})

    staypoints_by_cell = centered.with_columns(nw.col(_H3_CELL_COL).cast(nw.UInt64))
    if len(staypoints_by_cell) == 0:
        return _empty_visitation_law_data(df, user_id_col)

    # A per-(user, cell) group-by mixing `n_unique()` with `len()`/`first()`/
    # `first()` in one call cannot be expressed as a single vectorized
    # pandas groupby through Narwhals, so it silently falls back to a slow
    # per-group Python path at this dataframe's scale (see
    # narwhals-dev.github.io/narwhals/concepts/improve_group_by_operation).
    # Aggregating in Rust sidesteps that backend-dependent fallback entirely.
    user_id_values = pa.array(staypoints_by_cell.get_column(user_id_col).to_arrow())
    user_codes, user_representatives = _factorize_arrow_values(user_id_values, sort=False)
    user_labels_by_code = pc.take(user_id_values, user_representatives)
    timestamps_ms = _extract_timestamps(staypoints_by_cell, timestamp_col)

    group_user_codes, group_cells, n_staypoints, f_values, loc_lat, loc_lng = (
        _as_arrow(values)
        for values in aggregate_visitation_cells_arrow(
            user_codes,
            staypoints_by_cell.get_column(_H3_CELL_COL).to_arrow(),
            timestamps_ms.to_arrow(),
            staypoints_by_cell.get_column(lat_col).to_arrow(),
            staypoints_by_cell.get_column(lng_col).to_arrow(),
        )
    )
    counts = nw.from_arrow(
        pa.table(
            {
                user_id_col: pc.take(user_labels_by_code, group_user_codes),
                _USER_CODE_COL: group_user_codes,
                _H3_CELL_COL: group_cells,
                "n_staypoints": n_staypoints,
                "f": f_values,
                "loc_lat": loc_lat,
                "loc_lng": loc_lng,
            }
        ),
        backend=df.implementation,
    )
    if len(counts) == 0:
        return _empty_visitation_law_data(df, user_id_col)

    merged = counts.join(homes, on=user_id_col, how="inner").with_columns(
        nw.col("n_staypoints").cast(nw.Int64),
        nw.col("home_lat").cast(nw.Float64),
        nw.col("home_lng").cast(nw.Float64),
        nw.col("loc_lat").cast(nw.Float64),
        nw.col("loc_lng").cast(nw.Float64),
    )
    if len(merged) == 0:
        return _empty_visitation_law_data(df, user_id_col)

    r_km = visitation_distances(
        merged.get_column("home_lat").to_arrow(),
        merged.get_column("home_lng").to_arrow(),
        merged.get_column("loc_lat").to_arrow(),
        merged.get_column("loc_lng").to_arrow(),
    )
    return (
        merged.with_columns(nw.new_series("r_km", r_km, backend=df.implementation))
        .with_columns((nw.col("r_km") * nw.col("f")).alias("rf"))
        .select([user_id_col, _USER_CODE_COL, _H3_CELL_COL, "r_km", "f", "rf", "n_staypoints"])
        .sort([user_id_col, _H3_CELL_COL])
        .to_native()
    )


def _bin_visitation_law(
    vl_df: Any,
    *,
    user_id_col: str | None = None,
    h3_cell_col: str = _H3_CELL_COL,
    n_bins: int = 30,
    distance_bin_width_km: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Bin prepared visitation rows into the ``(rf, rho)`` fit inputs.

    Counts distinct users per H3 cell, radial-distance bin, and visitation
    frequency, then averages their annular densities in logarithmic ``rf``
    bins.

    Parameters
    ----------
    vl_df:
        Dataframe returned by :func:`_prepare_visitation_law_data`.
    user_id_col, h3_cell_col:
        Column names in ``vl_df``. ``h3_cell_col`` defaults to ``"h3_cell"``.
    n_bins:
        Number of logarithmic ``rf`` bins.
    distance_bin_width_km:
        Width of radial-distance annuli in kilometres.
    """
    if distance_bin_width_km <= 0:
        raise ValueError("distance_bin_width_km must be positive.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    df = nw.from_native(vl_df, eager_only=True)
    user_id_col = _detect_required_column(df, user_id_col, USER_ID_CANDIDATES)
    required = [h3_cell_col, "r_km", "f", "rf"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Columns are required: {missing}.")

    # Reuse `_prepare_visitation_law_data`'s user-id factorization when it's
    # present: `bin_visitation_law_arrow` skips its own factorization for a
    # already-factorized UInt32 array, so passing these codes instead of raw
    # `user_id_col` values avoids factorizing the same column a second time.
    user_id_values = (
        df.get_column(_USER_CODE_COL).to_arrow()
        if _USER_CODE_COL in df.columns
        else df.get_column(user_id_col).to_arrow()
    )
    rf_values, rho_values = bin_visitation_law_arrow(
        user_id_values,
        df.get_column(h3_cell_col).to_arrow(),
        df.get_column("r_km").to_arrow(),
        df.get_column("f").to_arrow(),
        n_bins,
        distance_bin_width_km,
    )
    return np.asarray(rf_values, dtype=float), np.asarray(rho_values, dtype=float), _VISITATION_YLABEL


def _visitation_law_curve(
    rf_values: np.ndarray,
    eta: float,
    mu: float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth curve for ``rho(r, f) = mu * (r*f)^(-eta)``.

    Parameters
    ----------
    rf_values:
        Positive ``r*f`` values defining the curve range.
    eta:
        Scaling exponent.
    mu:
        Attractiveness prefactor.
    n_points:
        Number of curve points. Must be at least 2 unless all ``rf`` values are
        equal, in which case one point is returned.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``rf`` and ``rho`` arrays for plotting.
    """
    rf_values = np.asarray(rf_values, dtype=float)
    mask = np.isfinite(rf_values) & (rf_values > 0)
    if not mask.any():
        raise ValueError("rf_values must contain at least one positive finite value.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    rf_min = float(rf_values[mask].min())
    rf_max = float(rf_values[mask].max())
    if rf_min == rf_max:
        rf_fit = np.array([rf_min], dtype=float)
    else:
        rf_fit = np.logspace(np.log10(rf_min), np.log10(rf_max), n_points)
    rho_fit = float(mu) * np.power(rf_fit, -float(eta))
    return rf_fit, rho_fit


def _fit_visitation_law_spectrum(
    rf_values: np.ndarray,
    rho_values: np.ndarray,
    *,
    min_rf: float | None = None,
    max_rf: float | None = None,
    return_curve: bool = False,
    n_curve_points: int = 200,
) -> tuple[float, float, float] | tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit the universal visitation law to binned ``(rf, rho)`` values.

    The fitted model is ``rho(r, f) = mu * (r*f)^(-eta)``. Fitting uses
    ordinary least squares in log-log space.

    Parameters
    ----------
    rf_values, rho_values:
        Positive binned values, typically returned by
        :func:`_bin_visitation_law`.
    min_rf, max_rf:
        Optional positive inclusive range for fitting.
    return_curve:
        When ``True``, also return smooth fitted curve arrays.
    n_curve_points:
        Number of points in the fitted curve.

    Returns
    -------
    tuple
        ``(eta, mu, r2)`` or ``(eta, mu, r2, rf_fit, rho_fit)`` when
        ``return_curve=True``. ``r2`` is computed in log-log space.
    """
    rf_values = np.asarray(rf_values, dtype=float)
    rho_values = np.asarray(rho_values, dtype=float)
    if rf_values.shape != rho_values.shape:
        raise ValueError("rf_values and rho_values must have the same shape.")

    mask = np.isfinite(rf_values) & np.isfinite(rho_values) & (rf_values > 0) & (rho_values > 0)
    if min_rf is not None:
        if min_rf <= 0:
            raise ValueError("min_rf must be positive when provided.")
        mask &= rf_values >= min_rf
    if max_rf is not None:
        if max_rf <= 0:
            raise ValueError("max_rf must be positive when provided.")
        mask &= rf_values <= max_rf

    x = np.log(rf_values[mask])
    y = np.log(rho_values[mask])
    if len(x) < 2:
        raise ValueError("At least two positive finite data points are required to fit.")

    slope, intercept = np.polyfit(x, y, 1)
    eta = float(-slope)
    mu = float(np.exp(intercept))

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    if return_curve:
        rf_fit, rho_fit = _visitation_law_curve(
            rf_values[mask],
            eta=eta,
            mu=mu,
            n_points=n_curve_points,
        )
        return eta, mu, r2, rf_fit, rho_fit
    return eta, mu, r2


def fit_visitation_law(
    staypoints: Any | Staypoints,
    *,
    locations: Any | None = None,
    h3_resolution: int = 9,
    user_id_col: str | None = None,
    timestamp_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    start_night: int = 22,
    end_night: int = 7,
    presorted: bool = False,
    min_rf: float | None = None,
    max_rf: float | None = None,
    n_bins: int = 30,
    distance_bin_width_km: float = 1.0,
) -> VisitationLawFit:
    """Fit the universal visitation law from staypoints.

    Uses supplied global locations directly, or assigns shared H3 cells when
    no catalogue is supplied; it then fits ``rho = mu * rf**(-eta)``.

    Returns
    -------
    VisitationLawFit
        ``data`` has one row per user and H3 cell with ``r_km``, distinct-day
        frequency ``f``, ``rf``, and ``n_staypoints``. ``spectrum`` contains
        the fitted ``rf``/``rho`` points; ``eta``, ``mu``, and ``r2`` are the
        fitted parameters.
    """
    data = _prepare_visitation_law_data(
        staypoints,
        locations=locations,
        h3_resolution=h3_resolution,
        user_id_col=user_id_col,
        timestamp_col=timestamp_col,
        lat_col=lat_col,
        lng_col=lng_col,
        start_night=start_night,
        end_night=end_night,
        presorted=presorted,
    )
    rf_values, rho_values, _label = _bin_visitation_law(
        data,
        user_id_col=user_id_col,
        n_bins=n_bins,
        distance_bin_width_km=distance_bin_width_km,
    )
    eta, mu, r2 = _fit_visitation_law_spectrum(
        rf_values,
        rho_values,
        min_rf=min_rf,
        max_rf=max_rf,
    )
    data_df = nw.from_native(data, eager_only=True)
    spectrum = nw.from_dict(
        {"rf": rf_values, "rho": rho_values},
        backend=data_df.implementation,
    ).to_native()
    if _USER_CODE_COL in data_df.columns:
        data = data_df.drop(_USER_CODE_COL).to_native()
    return VisitationLawFit(data=data, spectrum=spectrum, eta=eta, mu=mu, r2=r2)
