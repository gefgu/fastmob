from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv

from fastmob.core import TrajDataFrame
from fastmob.utils import _arrow_io

_GEOLIFE_PLT_HEADER_ROWS = 6
_GEOLIFE_PLT_COLUMN_NAMES = ["lat", "lng", "_reserved", "_altitude", "_days", "date", "time"]
_GEOLIFE_PLT_COLUMN_TYPES = {"lat": pa.float64(), "lng": pa.float64(), "date": pa.string(), "time": pa.string()}
_GEOLIFE_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _skip_invalid_plt_row(_row):
    return "skip"


def _parse_geolife_plt(plt_path: Path, user_id: str, encoding: str) -> pa.Table:
    table = pa_csv.read_csv(
        str(plt_path),
        read_options=pa_csv.ReadOptions(
            skip_rows=_GEOLIFE_PLT_HEADER_ROWS,
            column_names=_GEOLIFE_PLT_COLUMN_NAMES,
            encoding=encoding,
            use_threads=False,
        ),
        parse_options=pa_csv.ParseOptions(delimiter=",", invalid_row_handler=_skip_invalid_plt_row),
        convert_options=pa_csv.ConvertOptions(
            include_columns=["lat", "lng", "date", "time"], column_types=_GEOLIFE_PLT_COLUMN_TYPES
        ),
    )
    datetime_str = pc.binary_join_element_wise(table["date"], table["time"], " ")
    datetime_col = pc.strptime(datetime_str, format=_GEOLIFE_DATETIME_FORMAT, unit="us")
    uid_col = pa.array([user_id] * table.num_rows, type=pa.string())
    return pa.table({"uid": uid_col, "lat": table["lat"], "lng": table["lng"], "datetime": datetime_col})


def read(filename, **kwargs):
    """Read a trajectory table from disk and return a `TrajDataFrame`.

    CSV and delimited text files are read with `pyarrow.csv.read_csv`; parquet
    files are read with `pyarrow.parquet.read_table`.

    Parameters
    ----------
    filename : str
        path and name of the file to read.
    **kwargs : dict
        For parquet files, passed to `pyarrow.parquet.read_table`. For CSV/
        delimited files, passed to `fastmob.utils._arrow_io.read_delimited`
        (accepts ``delimiter``, ``encoding``, ``header``, ``column_names``,
        plus any `pyarrow.csv.ConvertOptions` keyword).

    Returns
    -------
    TrajDataFrame
        object loaded from file.
    """
    path = Path(filename)
    if path.suffix.lower() == ".parquet":
        table = _arrow_io.read_parquet(path, **kwargs)
    else:
        table = _arrow_io.read_delimited(path, **kwargs)
    return TrajDataFrame(table)


def write(tdf, filename, **kwargs):
    """Write a trajectory dataframe to disk.

    CSV and delimited text files are written with `pyarrow.csv.write_csv`;
    parquet files are written with `pyarrow.parquet.write_table`. The input
    is materialized as a pyarrow.Table via Narwhals first, so any
    Narwhals-compatible backend (pandas, polars, pyarrow, ...) is accepted
    regardless of `tdf`'s own backend.

    Parameters
    ----------
    tdf : TrajDataFrame or DataFrame-like
        TrajDataFrame object (or any Narwhals-compatible dataframe) that
        will be saved.
    filename : str
        path and name of the output file.
    **kwargs : dict
        For parquet files, passed to `pyarrow.parquet.write_table`. For CSV/
        delimited files, passed to `pyarrow.csv.WriteOptions`.

    Returns
    -------
    None
    """
    path = Path(filename)
    native = tdf.df if isinstance(tdf, TrajDataFrame) else tdf
    table = nw.from_native(native, eager_only=True).to_arrow()
    if path.suffix.lower() == ".parquet":
        return _arrow_io.write_parquet(table, path, **kwargs)
    return _arrow_io.write_delimited(table, path, **kwargs)


def load_geolife_trajectories(path, user_ids=None, **kwargs):
    """Load Microsoft GeoLife `.plt` trajectory files into a `TrajDataFrame`.

    Parameters
    ----------
    path : str
        local path of the directory 'Geolife Trajectories 1.3/'
    user_ids : list, optional
        list of user IDs to load. If empty or None, all users are loaded.
    **kwargs : dict
        Additional keyword arguments passed to the `TrajDataFrame` constructor.
        ``encoding`` (default ``"utf-8"``) is applied when reading each `.plt`
        file and is not forwarded to `TrajDataFrame`.

    Returns
    -------
    TrajDataFrame
        a TrajDataFrame with ``uid`` (string), ``lat``/``lng`` (float64), and
        ``datetime`` (parsed as ``timestamp[us]``) columns.
    """
    root = Path(path)
    encoding = kwargs.pop("encoding", "utf-8")
    user_filter = None if user_ids is None else {str(user_id) for user_id in user_ids}
    jobs = []
    for plt_path in sorted(root.rglob("*.plt")):
        parts = plt_path.parts
        user_id = next((part for part in reversed(parts) if part.isdigit() and len(part) == 3), plt_path.parent.name)
        if user_filter is not None and user_id not in user_filter:
            continue
        jobs.append((plt_path, user_id))

    if not jobs:
        return TrajDataFrame(_arrow_io.table_from_pylist([]), **kwargs)

    max_workers = min(64, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tables = list(executor.map(lambda job: _parse_geolife_plt(job[0], job[1], encoding), jobs))
    return TrajDataFrame(pa.concat_tables(tables), **kwargs)
