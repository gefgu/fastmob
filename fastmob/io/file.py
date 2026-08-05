from __future__ import annotations

from pathlib import Path

import narwhals as nw

from fastmob.core import TrajDataFrame
from fastmob.utils import _arrow_io


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

    Returns
    -------
    TrajDataFrame
        a TrajDataFrame containing all trajectories
    """
    root = Path(path)
    rows = []
    user_filter = None if user_ids is None else {str(user_id) for user_id in user_ids}
    for plt_path in sorted(root.rglob("*.plt")):
        parts = plt_path.parts
        user_id = next((part for part in reversed(parts) if part.isdigit() and len(part) == 3), plt_path.parent.name)
        if user_filter is not None and user_id not in user_filter:
            continue
        with plt_path.open(encoding=kwargs.pop("encoding", "utf-8")) as handle:
            for line_no, line in enumerate(handle):
                if line_no < 6:
                    continue
                fields = line.strip().split(",")
                if len(fields) < 7:
                    continue
                rows.append(
                    {
                        "uid": user_id,
                        "lat": float(fields[0]),
                        "lng": float(fields[1]),
                        "datetime": f"{fields[5]} {fields[6]}",
                    }
                )
    return TrajDataFrame(_arrow_io.table_from_pylist(rows), **kwargs)
