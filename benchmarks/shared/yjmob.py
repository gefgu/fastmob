"""Loader for the YJMob100K (HuMob Challenge) big-scale benchmark fixture.

Every big-scale benchmark in this repo (H3 conversion, OD-matrix building,
profile clustering, contact-network construction, road-network distances,
...) reads the *same* real trajectory dataset instead of a synthetic one:
100,000 users, ~111.5M rows, real WGS84 coordinates, 75 days.

The parquet file itself (~572 MB) is not committed to this repo -- it comes
from the HuMob Challenge / YJMob100K dataset, which has its own
redistribution terms. Point ``FASTMOB_YJMOB_DATA_PATH`` at a local copy to
run these benchmarks; when it's unset (the default, e.g. in CI) callers
should skip cleanly rather than fail -- see :func:`yjmob_data_path`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_VAR = "FASTMOB_YJMOB_DATA_PATH"


def yjmob_data_path() -> Path | None:
    """Return the configured YJMob100K parquet path, or None if unavailable."""
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def skip_reason() -> str:
    return (
        f"skipped: set {ENV_VAR} to a local YJMob100K/HuMob Challenge parquet "
        "(columns: uid, timestamp, lat, lon, d, t) to run this benchmark"
    )


def load_yjmob(data_path: Path, *, n_users: int | None = None) -> Any:
    """Load the YJMob100K parquet as a Polars DataFrame.

    Parameters
    ----------
    data_path:
        Path to the YJMob100K parquet (see :func:`yjmob_data_path`).
    n_users:
        When given, subsample to the first ``n_users`` distinct ``uid``
        values (by first appearance) rather than loading all 100,000 -- used
        for the suite's smaller size points; omit for the full-scale point.
    """
    import polars as pl

    df = pl.read_parquet(data_path)
    if n_users is None:
        return df

    kept_uids = df.get_column("uid").unique(maintain_order=True).head(n_users)
    return df.filter(pl.col("uid").is_in(kept_uids))


def load_yjmob_pandas(data_path: Path, *, n_users: int | None = None) -> Any:
    """Same as :func:`load_yjmob`, converted to pandas for pandas-backend timings."""
    return load_yjmob(data_path, n_users=n_users).to_pandas()
