"""Download the first rows of the Brightkite check-in dataset."""

from __future__ import annotations

import gzip
import os
import urllib.request

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv

from fastmob.data._frames import TrajDataFrame
from fastmob.data._remote import cache_dir

BRIGHTKITE_URL = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"


def load_brightkite(max_rows: int | None = 300_000, cache: str | os.PathLike | None = None) -> TrajDataFrame:
    """Load Brightkite location check-ins, downloading only the rows requested.

    Brightkite (Stanford SNAP) holds about 4.7 million check-ins by 58 thousand users of
    a location-based social network, 2008 to 2010, worldwide. The compressed file is about
    60 MB; the file is streamed and the download stops after ``max_rows`` rows, so the
    default 300,000 rows cost about 4 MB. Rows without coordinates are dropped. The first
    rows are not a random sample of users: they hold the earliest user ids' complete
    histories only up to the cut-off.

    Parameters
    ----------
    max_rows : int, optional
        Number of raw rows to read before dropping rows without coordinates. ``None`` reads
        the whole file (about 60 MB).
    cache : str or path, optional
        Directory to cache the rows in. The default is a ``fastmob_data`` folder inside the
        system temporary directory.

    Returns
    -------
    TrajDataFrame
        Columns ``uid`` (string), ``datetime`` (UTC), ``lat`` and ``lng`` (degrees).

    Raises
    ------
    OSError
        If the file cannot be downloaded. No substitute data is generated.

    Examples
    --------
    >>> from fastmob.data import load_brightkite
    >>> tdf = load_brightkite(max_rows=50_000)
    """
    path = cache_dir("brightkite", cache) / f"checkins_{'all' if max_rows is None else max_rows}.tsv"
    if not path.exists():
        part = path.with_suffix(".part")
        with (
            urllib.request.urlopen(BRIGHTKITE_URL) as response,
            gzip.GzipFile(fileobj=response) as lines,
            part.open("wb") as out,
        ):
            for i, line in enumerate(lines):
                if max_rows is not None and i >= max_rows:
                    break
                out.write(line)
        part.rename(path)
    table = pa_csv.read_csv(
        str(path),
        read_options=pa_csv.ReadOptions(column_names=["uid", "datetime", "lat", "lng", "_venue"]),
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        convert_options=pa_csv.ConvertOptions(
            column_types={"uid": pa.string(), "datetime": pa.string(), "lat": pa.float64(), "lng": pa.float64()},
            include_columns=["uid", "datetime", "lat", "lng"],
        ),
    )
    table = table.filter(pc.and_(pc.is_valid(table["lat"]), pc.is_valid(table["lng"])))
    stamps = pc.strptime(table["datetime"], format="%Y-%m-%dT%H:%M:%SZ", unit="us")
    return TrajDataFrame(table.set_column(1, "datetime", stamps))
