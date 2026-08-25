from __future__ import annotations

from typing import Any

import narwhals as nw

from fastmob.core.base import unwrap_native
from fastmob._core import latlng_to_h3_arrow as _latlng_to_h3_arrow
from fastmob._core import latlng_to_h3_numpy as _latlng_to_h3_numpy
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob.utils._common import LAT_CANDIDATES, LNG_CANDIDATES, _as_arrow, _pick_existing_column

H3_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={"convert": _latlng_to_h3_arrow},
    numpy_ops={"convert": _latlng_to_h3_numpy},
)


def latlng_to_h3(
    traj: Any,
    resolution: int = 9,
    *,
    lat_col: str | None = None,
    lng_col: str | None = None,
    output_col: str = "h3_cell",
) -> Any:
    """Convert each row's (lat, lng) to an H3 cell index, Rust-accelerated.

    Rows with a non-finite or out-of-range coordinate (or a real Arrow null,
    for Polars/PyArrow input) get a null cell rather than failing the whole
    batch: `NaN`/inf coordinates in a NumPy-backed frame map to the reserved
    sentinel ``2**64 - 1`` (not a valid H3 index), while an Arrow-backed frame
    gets a genuine null slot in the output column.

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend.
    resolution:
        H3 resolution, 0-15.
    lat_col, lng_col:
        Explicit column name overrides; auto-detected when None.
    output_col:
        Name of the new column holding the H3 cell index.

    Returns
    -------
    DataFrame
        `traj` with `output_col` added, in the same backend as input.

    Examples
    --------
    >>> import pandas as pd
    >>> from fastmob.preprocessing import latlng_to_h3
    >>> df = pd.DataFrame({"lat": [37.769377], "lng": [-122.388519]})
    >>> result = latlng_to_h3(df, resolution=9)
    >>> hex(result["h3_cell"].iloc[0])
    '0x89283082e73ffff'
    """
    df = nw.from_native(unwrap_native(traj), eager_only=True)
    ops = H3_DISPATCHER.get_ops(df)

    if lat_col is None:
        lat_col = _pick_existing_column(df.columns, LAT_CANDIDATES)
    if lng_col is None:
        lng_col = _pick_existing_column(df.columns, LNG_CANDIDATES)
    if lat_col is None or lng_col is None:
        raise ValueError(
            "Could not find latitude/longitude column(s). "
            f"Available columns: {df.columns}. Pass lat_col/lng_col explicitly."
        )

    lats_data = ops["extract_data"](df.get_column(lat_col))
    lngs_data = ops["extract_data"](df.get_column(lng_col))

    raw_cells = ops["convert"](lats_data, lngs_data, resolution)
    cells = _as_arrow(raw_cells)

    # `.alias(...)` is required, not just `new_series(output_col, ...)`: when
    # `cells` is an Arrow-native array (Polars/PyArrow backends), the backend
    # series constructor can pick up that array's own (empty) field name
    # instead of the `name` argument -- `.alias` forces the name at the
    # Narwhals level regardless of what the underlying object thinks its name is.
    new_col = nw.new_series(output_col, cells, backend=df.implementation).alias(output_col)
    result = df.with_columns(new_col)
    return result.to_native()


latlng_to_h3.__module__ = "fastmob.preprocessing"
