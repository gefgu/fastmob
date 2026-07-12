"""Profile-friendly wrappers for comparison functions.

Each wrapper accepts a single DataFrame (from the Brightkite input pipeline),
splits it in half, and calls the underlying comparison function. This lets the
existing profile_function.py / profile_importable_samply.py infrastructure
work without modification.
"""

from __future__ import annotations

from typing import Any


def _split(df: Any) -> tuple[Any, Any]:
    mid = len(df) // 2
    try:
        return df.head(mid), df.tail(len(df) - mid)
    except TypeError:
        return df[:mid], df[mid:]


def column_distribution_wasserstein(df: Any) -> Any:
    from fastmob.comparison import column_distribution_wasserstein_distance

    df1, df2 = _split(df)
    return column_distribution_wasserstein_distance(df1, df2, "latitude")


def column_distribution_jsd(df: Any) -> Any:
    from fastmob.comparison import column_distribution_jensen_shannon_divergence

    df1, df2 = _split(df)
    return column_distribution_jensen_shannon_divergence(df1, df2, "latitude")


def visits_per_user_wasserstein(df: Any) -> Any:
    from fastmob.comparison import visits_per_user_wasserstein_distance

    df1, df2 = _split(df)
    return visits_per_user_wasserstein_distance(df1, df2)


def visits_per_user_jsd(df: Any) -> Any:
    from fastmob.comparison import visits_per_user_jensen_shannon_divergence

    df1, df2 = _split(df)
    return visits_per_user_jensen_shannon_divergence(df1, df2)


def wasserstein_arrays(df: Any) -> Any:
    """Profile the flat Rust kernel on two latitude arrays."""

    from fastmob.comparison import wasserstein_distance

    arr = df["latitude"].to_numpy() if hasattr(df["latitude"], "to_numpy") else df["latitude"].values
    mid = len(arr) // 2
    return wasserstein_distance(arr[:mid], arr[mid:])
