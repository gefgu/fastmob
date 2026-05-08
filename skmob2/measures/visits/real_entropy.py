"""Real entropy of individual mobility trajectories (Kontoyiannis estimator)."""

from __future__ import annotations

from typing import Any

import numpy as np
import narwhals as nw
import pandas as pd
from skmob2._core import real_entropy_batch as _real_entropy_batch_rust

from .._common import _build_user_ranges, _prepare_trajectory


def _skmob_true_entropy(sequence: list) -> float:
    """Match scikit-mobility's private _true_entropy estimator (LZ77 scan).

    Kept as a public function for test compatibility and as a reference
    implementation.  The measure itself uses the Rust batch kernel which
    implements the same algorithm.
    """
    n = len(sequence)
    if n <= 1:
        return 0.0

    sum_lambda = 3.0

    def in_seq(prefix: list, candidate: list) -> bool:
        for i in range(len(prefix) - len(candidate) + 1):
            if prefix[i : i + len(candidate)] == candidate:
                return True
        return False

    for i in range(1, n - 1):
        j = i + 1
        while j < n and in_seq(sequence[:i], sequence[i:j]):
            j += 1
        if j == n:
            j += 1
        sum_lambda += j - i

    return float(n * np.log2(n) / sum_lambda)


def real_entropy(
    traj: Any,
    *,
    datetime_col: str | None = None,
    lat_col: str | None = None,
    lng_col: str | None = None,
    uid_col: str | None = None,
) -> Any:
    """Return the real (true) entropy of mobility for each user.

    Real entropy is estimated using the Kontoyiannis (1998) Lempel-Ziv
    entropy rate estimator applied to the sequence of visited locations.
    Each location is encoded as the string ``"<lat>_<lng>"`` using exact
    float equality — matching the skmob convention (no spatial clustering).

    The estimator captures both the frequency and the order of visits,
    unlike random entropy (which ignores order) and uncorrelated entropy
    (which ignores temporal correlations).

    Parameters
    ----------
    traj:
        Trajectory dataframe; any Narwhals-compatible eager backend (pandas,
        polars, …).  Must have datetime, latitude, and longitude columns.
        A user-ID column is optional; when absent the whole frame is treated
        as a single individual.
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
    DataFrame
        One row per user with columns ``[uid_col, "real_entropy"]``.
        The returned backend matches the input backend.



    Examples
    --------
    >>> import pandas as pd
    >>> import skmob2
    >>> url = skmob2.utils.constants.BRIGHTKITE_SAMPLE
    >>> df = pd.read_csv(
    ...     url,
    ...     sep="\\t",
    ...     header=0,
    ...     nrows=5000,
    ...     names=["uid", "datetime", "lat", "lng", "location id"],
    ... )
    >>> df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    >>> df["location_id"] = df["location id"].astype("string")
    >>> df = df.dropna(subset=["uid", "datetime", "lat", "lng"])[
    ...     ["uid", "datetime", "lat", "lng", "location_id"]
    ... ]
    >>> print(df.head().to_string(index=False))
     uid                  datetime       lat         lng                              location_id
       0 2010-10-16 06:02:04+00:00 39.891383 -105.070814         7a0f88982aa015062b95e3b4843f9ca2
       0 2010-10-16 03:48:54+00:00 39.891077 -105.068532         dd7cd3d264c2d063832db506fba8bf79
       0 2010-10-14 18:25:51+00:00 39.750469 -104.999073 9848afcc62e500a01cf6fbf24b797732f8963683
       0 2010-10-14 00:21:47+00:00 39.752713 -104.996337         2ef143e12038c870038df53e0478cefc
       0 2010-10-13 23:31:51+00:00 39.752508 -104.996637         424eb3dd143292f9e013efa00486c907
    >>> from skmob2 import real_entropy
    >>> result = real_entropy(df)
    >>> print(result.round({"real_entropy": 3}).head().to_string(index=False))
     uid  real_entropy
       0         4.905
       1         2.200
       2         4.683

    References
    ----------
    - [SQBB2010] Song, C., Qu, Z., Blumm, N. & Barabasi, A. L. (2010) Limits of Predictability in Human Mobility. Science 327(5968), 1018-1021, https://science.sciencemag.org/content/327/5968/1018

    @usedBy
        skmob2.measures.visits.__init__, skmob2.measures.__init__,
        skmob2.__init__ (re-exported as public API)
    """
    df, datetime_col, lat_col, lng_col, uid_col = _prepare_trajectory(
        traj,
        datetime_col=datetime_col,
        lat_col=lat_col,
        lng_col=lng_col,
        uid_col=uid_col,
    )

    native = df.to_native()
    if isinstance(native, pd.DataFrame):
        tokens = (native[lat_col].astype(str) + "_" + native[lng_col].astype(str)).tolist()
    else:
        loc_key_col = "__skmob2_loc_key__"
        df = df.with_columns(
            (nw.col(lat_col).cast(nw.String) + nw.lit("_") + nw.col(lng_col).cast(nw.String)).alias(loc_key_col)
        )
        tokens = df.get_column(loc_key_col).to_list()

    if uid_col is None:
        entropies = _real_entropy_batch_rust(tokens, [(0, len(tokens))])
        return nw.from_dict(
            {"real_entropy": entropies},
            backend=df.implementation,
        ).to_native()

    uid_values, ranges = _build_user_ranges(df, uid_col)
    entropies = _real_entropy_batch_rust(tokens, ranges)

    return nw.from_dict(
        {uid_col: uid_values, "real_entropy": entropies},
        backend=df.implementation,
    ).to_native()
