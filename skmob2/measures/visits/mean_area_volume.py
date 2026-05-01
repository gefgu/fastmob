"""Mean area volume measure for visit data."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import narwhals as nw

from .._common import (
    LOCATION_CANDIDATES,
    TIMESTAMP_CANDIDATES,
    USER_ID_CANDIDATES,
    _pick_existing_column,
)

_END_TIMESTAMP_CANDIDATES: list[str] = ["end_timestamp", "end_time"]
_10MIN = timedelta(minutes=10)


def _floor_10min(dt: Any) -> Any:
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def mean_area_volume(
    visits: Any,
    *,
    area_col: str | None = None,
    user_id_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
) -> Any:
    """Mean user volume per area and 10-minute time bin, averaged across days of the week.

    For each area the function:

    1. Expands each stay into 10-minute presence slots.
    2. Counts unique users per slot per calendar date.
    3. Averages counts over dates for each day-of-week.
    4. Averages the 7 day-of-week means (missing days contribute 0).

    Parameters
    ----------
    visits:
        Any Narwhals-compatible eager dataframe with one row per stay event.
    area_col:
        Area column name. Auto-detected from ``LOCATION_CANDIDATES`` when None.
    user_id_col:
        User ID column. Auto-detected from ``USER_ID_CANDIDATES`` when None.
    start_col:
        Stay start timestamp. Auto-detected from ``TIMESTAMP_CANDIDATES`` when None.
    end_col:
        Stay end timestamp. Auto-detected from ``["end_timestamp", "end_time"]`` when None.

    Returns
    -------
    DataFrame
        Same backend as input. Columns: ``area``, ``time_bin`` (``"HH:MM"``),
        ``mean_volume``. Rows where ``mean_volume == 0`` are excluded.
        Sorted by ``(area, time_bin)``.

    Raises
    ------
    ValueError
        If any required column cannot be detected.

    Notes
    -----
    - A stay where ``start == end`` produces exactly one bin (the floored start).
    - A stay where ``start > end`` produces no bins and is silently skipped.
    - Stays spanning midnight are handled correctly: bins are attributed to the
      date on which each bin falls.

    Examples
    --------
    >>> import pandas as pd
    >>> from skmob2.measures.visits import mean_area_volume
    >>> visits = pd.DataFrame(
    ...     {
    ...         "user_id": ["u1", "u2", "u1"],
    ...         "area": ["home", "home", "work"],
    ...         "start_timestamp": pd.to_datetime(
    ...             ["2020-01-01 08:00", "2020-01-01 08:10", "2020-01-01 09:00"]
    ...         ),
    ...         "end_timestamp": pd.to_datetime(
    ...             ["2020-01-01 08:20", "2020-01-01 08:20", "2020-01-01 09:10"]
    ...         ),
    ...     }
    ... )
    >>> result = mean_area_volume(visits)
    >>> print(result.round({"mean_volume": 3}).to_string(index=False))
    area time_bin  mean_volume
    home    08:00        0.143
    home    08:10        0.286
    home    08:20        0.286
    work    09:00        0.143
    work    09:10        0.143
    """
    nw_df = nw.from_native(visits, eager_only=True)
    cols = nw_df.columns

    if area_col is None:
        area_col = _pick_existing_column(cols, LOCATION_CANDIDATES)
    if user_id_col is None:
        user_id_col = _pick_existing_column(cols, USER_ID_CANDIDATES)
    if start_col is None:
        start_col = _pick_existing_column(cols, TIMESTAMP_CANDIDATES)
    if end_col is None:
        end_col = _pick_existing_column(cols, _END_TIMESTAMP_CANDIDATES)

    missing = [
        name
        for name, val in [
            ("area", area_col),
            ("user_id", user_id_col),
            ("start_timestamp", start_col),
            ("end_timestamp", end_col),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(f"Could not detect required column(s): {missing}. Available columns: {cols}")

    if len(nw_df) == 0:
        return nw.from_dict(
            {"area": [], "time_bin": [], "mean_volume": []},
            backend=nw_df.implementation,
        ).to_native()

    areas_list = nw_df.get_column(area_col).to_list()
    uids_list = nw_df.get_column(user_id_col).to_list()
    starts_list = nw_df.get_column(start_col).to_list()
    ends_list = nw_df.get_column(end_col).to_list()

    # presence[(area, date, dow, time_bin)] = set of user ids
    presence: defaultdict = defaultdict(set)

    for area, uid, start, end in zip(areas_list, uids_list, starts_list, ends_list):
        if start is None or end is None:
            continue
        s = _floor_10min(start)
        e = _floor_10min(end)
        t = s
        while t <= e:
            key = (area, t.date(), t.weekday(), t.strftime("%H:%M"))
            presence[key].add(uid)
            t = t + _10MIN

    # user_count per slot + track unique dates per (area, dow)
    dates_per_area_dow: defaultdict = defaultdict(set)
    dow_sum: defaultdict = defaultdict(float)

    for (area, date, dow, tb), users_set in presence.items():
        dates_per_area_dow[(area, dow)].add(date)
        dow_sum[(area, dow, tb)] += len(users_set)

    # dow mean = total / n_unique_dates
    area_bin_sum: defaultdict = defaultdict(float)
    for (area, dow, tb), total in dow_sum.items():
        n_dates = len(dates_per_area_dow[(area, dow)])
        area_bin_sum[(area, tb)] += total / n_dates

    # final mean = sum_of_dow_means / 7; filter zeros; sort
    rows = []
    for (area, tb), val in area_bin_sum.items():
        mv = val / 7.0
        if mv > 0:
            rows.append((area, tb, mv))

    rows.sort(key=lambda r: (str(r[0]), r[1]))

    return nw.from_dict(
        {
            "area": [r[0] for r in rows],
            "time_bin": [r[1] for r in rows],
            "mean_volume": [r[2] for r in rows],
        },
        backend=nw_df.implementation,
    ).to_native()
