from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastmob.core import TrajDataFrame


def read(filename, **kwargs):
    """Read a trajectory table from disk and return a `TrajDataFrame`.

    CSV and delimited text files are read with `pandas.read_csv`; parquet files
    are read with `pandas.read_parquet`.

    Parameters
    ----------
    filename : str
        path and name of the file to read.
    **kwargs : dict
        Additional keyword arguments passed to `pandas.read_csv` or `pandas.read_parquet`.

    Returns
    -------
    TrajDataFrame
        object loaded from file.
    """
    path = Path(filename)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, **kwargs)
    else:
        frame = pd.read_csv(path, **kwargs)
    return TrajDataFrame(frame)


def write(tdf, filename, **kwargs):
    """Write a trajectory dataframe to disk.

    CSV and delimited text files are written with `DataFrame.to_csv`; parquet
    files are written with `DataFrame.to_parquet`.

    Parameters
    ----------
    tdf : TrajDataFrame or pandas.DataFrame
        TrajDataFrame object that will be saved.
    filename : str
        path and name of the output file.
    **kwargs : dict
        Additional keyword arguments passed to `DataFrame.to_csv` or `DataFrame.to_parquet`.

    Returns
    -------
    None
    """
    path = Path(filename)
    frame = tdf.df if isinstance(tdf, TrajDataFrame) else tdf
    if path.suffix.lower() == ".parquet":
        return frame.to_parquet(path, **kwargs)
    return frame.to_csv(path, index=kwargs.pop("index", False), **kwargs)


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
    return TrajDataFrame(pd.DataFrame(rows), **kwargs)