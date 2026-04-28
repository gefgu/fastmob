from __future__ import annotations

from typing import Any


from .._common import _prepare_trajectory


def visits_per_time_unit(
    traj: Any,
    time_unit: str | None = None,
    *,
    freq: str = "1h",
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the number of trajectory points per time unit across all users.

    Groups all trajectory records by time bins of width ``freq`` and counts
    the number of records (visits) in each bin.  The result covers only bins
    that contain at least one record.

    Narwhals does not expose a time-frequency groupby operation, so this
    function converts to pandas internally and returns a pandas DataFrame.
    The ``freq`` parameter follows pandas offset alias syntax (e.g. ``"1h"``,
    ``"1D"``, ``"15min"``).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
    time_unit:
        Alias for ``freq`` (skmob compatibility).  When provided, overrides
        ``freq``.
    freq:
        Pandas-compatible offset alias for the time bin width.  Default: ``"1h"``.
    datetime_col:
        Explicit datetime column name.  Auto-detected when None.
    lat_col:
        Explicit latitude column name.  Auto-detected when None.
    lng_col:
        Explicit longitude column name.  Auto-detected when None.
    uid_col:
        Explicit user-ID column name.  Auto-detected when None.

    Returns
    -------
    pandas.DataFrame
        One row per non-empty time bin with columns
        ``[datetime_col, "n_visits"]``, sorted chronologically.

    @usedBy
        skmob2.measures.flows.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)

    Side effects
        Converts the input to a pandas DataFrame internally regardless of the
        original backend, because Narwhals does not provide time-resampling.
        The return type is always a pandas DataFrame.
    """
    if time_unit is not None:
        freq = time_unit
    import pandas as pd  # noqa: PLC0415 — import inside function because pandas conversion is intentional

    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    # Convert to pandas for resample support.
    pandas_df = pd.DataFrame(df.to_native())

    # Ensure the datetime column is a proper pandas datetime type.
    pandas_df[datetime_col] = pd.to_datetime(pandas_df[datetime_col])

    counts = pandas_df.set_index(datetime_col).resample(freq).size().rename("n_visits").reset_index()
    # Keep only bins that have at least one visit.
    counts = counts[counts["n_visits"] > 0].reset_index(drop=True)

    return counts
