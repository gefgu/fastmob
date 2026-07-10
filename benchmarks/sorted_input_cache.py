"""Helpers for cached sorted benchmark inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_INPUT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_ROW_ORDER_COL = "__fkmob_benchmark_row_order__"


@dataclass(frozen=True)
class SortedInput:
    data: Any
    path: Path
    status: str


def sorted_cache_path(
    cache_dir: Path,
    *,
    suite: str,
    backend: str,
    data_path: Path,
    repeat_factor: int | None = None,
) -> Path:
    suffix = f"_repeat{repeat_factor}" if repeat_factor is not None else ""
    source_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", data_path.name)
    return cache_dir / f"{suite}_{source_name}_{backend}{suffix}_sorted.parquet"


def stable_sort_pandas(df: Any, *, uid_col: str, datetime_col: str) -> Any:
    sorted_df = df.copy()
    sorted_df[_ROW_ORDER_COL] = range(len(sorted_df))
    sorted_df = sorted_df.sort_values([uid_col, datetime_col, _ROW_ORDER_COL], kind="mergesort")
    return sorted_df.drop(columns=[_ROW_ORDER_COL]).reset_index(drop=True)


def stable_sort_polars(df: Any, *, uid_col: str, datetime_col: str) -> Any:
    return df.with_row_index(_ROW_ORDER_COL).sort(uid_col, datetime_col, _ROW_ORDER_COL).drop(_ROW_ORDER_COL)


def load_or_create_sorted_input(
    *,
    cache_dir: Path,
    suite: str,
    backend: str,
    data_path: Path,
    load_raw: Callable[[], Any],
    uid_col: str,
    datetime_col: str,
    repeat_factor: int | None = None,
) -> SortedInput:
    path = sorted_cache_path(
        cache_dir,
        suite=suite,
        backend=backend,
        data_path=data_path,
        repeat_factor=repeat_factor,
    )
    if path.exists():
        if backend == "polars":
            import polars as pl

            return SortedInput(pl.read_parquet(path), path, "reused")

        import pandas as pd

        return SortedInput(pd.read_parquet(path), path, "reused")

    df = load_raw()
    if backend == "polars":
        sorted_df = stable_sort_polars(df, uid_col=uid_col, datetime_col=datetime_col)
    else:
        sorted_df = stable_sort_pandas(df, uid_col=uid_col, datetime_col=datetime_col)

    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_df.write_parquet(path) if backend == "polars" else sorted_df.to_parquet(path, index=False)
    return SortedInput(sorted_df, path, "created")
