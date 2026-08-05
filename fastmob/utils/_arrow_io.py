"""pyarrow-based CSV/parquet I/O helpers.

fastmob's file I/O (``fastmob/io/file.py`` and the bundled dataset loaders
under ``fastmob/data/datasets/``) reads/writes through pyarrow rather than
pandas, since pyarrow is a guaranteed hard dependency and pandas is not.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq


def read_delimited(
    path,
    *,
    delimiter: str = ",",
    encoding: str = "utf8",
    header: bool = True,
    column_names: list[str] | None = None,
    **convert_options_kwargs: Any,
) -> pa.Table:
    """Read a delimited text file into a pyarrow.Table.

    When ``header`` is False and ``column_names`` is None, columns are
    auto-named (``"f0"``, ``"f1"``, ...); pass explicit ``column_names`` for
    headerless files with known columns instead.
    """
    read_options = pa_csv.ReadOptions(
        encoding=encoding,
        column_names=column_names,
        autogenerate_column_names=not header and column_names is None,
    )
    parse_options = pa_csv.ParseOptions(delimiter=delimiter)
    convert_options = pa_csv.ConvertOptions(**convert_options_kwargs) if convert_options_kwargs else None
    return pa_csv.read_csv(
        str(path), read_options=read_options, parse_options=parse_options, convert_options=convert_options
    )


def read_parquet(path, **kwargs: Any) -> pa.Table:
    """Read a parquet file into a pyarrow.Table."""
    return pq.read_table(str(path), **kwargs)


def write_delimited(table: pa.Table, path, **kwargs: Any) -> None:
    """Write a pyarrow.Table to a delimited text file."""
    write_options = pa_csv.WriteOptions(**kwargs) if kwargs else None
    pa_csv.write_csv(table, str(path), write_options=write_options)


def write_parquet(table: pa.Table, path, **kwargs: Any) -> None:
    """Write a pyarrow.Table to a parquet file."""
    pq.write_table(table, str(path), **kwargs)


def table_from_pylist(rows: list[dict]) -> pa.Table:
    """Build a pyarrow.Table from a list of row-dicts."""
    return pa.Table.from_pylist(rows)
