"""Download a subset of the Microsoft GeoLife GPS trajectories."""

from __future__ import annotations

import os
import zipfile
from collections.abc import Iterable

from fastmob.data._frames import TrajDataFrame
from fastmob.data._remote import HttpRangeFile, cache_dir

GEOLIFE_URL = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"


def load_geolife(
    users: Iterable[str | int] | None = ("000", "001", "002"),
    max_trips_per_user: int | None = 25,
    cache: str | os.PathLike | None = None,
) -> TrajDataFrame:
    """Load GPS trajectories of selected GeoLife users, downloading only what is needed.

    GeoLife is a public GPS dataset collected by Microsoft Research Asia (182 users,
    about 25 million fixes, mostly in Beijing, sampled every 1 to 5 seconds). The full
    archive is about 313 MB; this loader reads the requested users' `.plt` files straight
    from the archive with HTTP range requests, so a few users cost a few megabytes.
    Files are cached per combination of arguments, so repeated calls with the same arguments do not download again and a call never returns files cached for a different selection.

    Parameters
    ----------
    users : iterable of str or int, optional
        GeoLife user ids (three-digit codes such as ``"000"``). ``None`` loads every user,
        which downloads the whole archive. The default is the first three users.
    max_trips_per_user : int, optional
        Keep only each user's first N trip files, in file-name order. ``None`` keeps all.
    cache : str or path, optional
        Directory to cache downloads in. The default is a ``fastmob_data`` folder inside the
        system temporary directory.

    Returns
    -------
    TrajDataFrame
        Columns ``uid`` (string), ``lat``/``lng`` (degrees, WGS 84) and ``datetime``
        (UTC, as recorded by GeoLife).

    Raises
    ------
    OSError
        If the archive cannot be downloaded or does not support partial downloads. No
        substitute data is generated.

    Examples
    --------
    >>> from fastmob.data import load_geolife
    >>> tdf = load_geolife(users=["000"], max_trips_per_user=5)
    """
    from fastmob.io import load_geolife_trajectories

    wanted = None if users is None else {f"{int(u):03d}" if str(u).isdigit() else str(u) for u in users}
    selection = "all" if wanted is None else "-".join(sorted(wanted))
    root = cache_dir("geolife", cache) / f"{selection}_{max_trips_per_user}"
    marker = root / ".complete"
    if not marker.exists():
        archive = zipfile.ZipFile(HttpRangeFile(GEOLIFE_URL))
        members: dict[str, list[str]] = {}
        for name in archive.namelist():
            parts = name.split("/")
            if name.endswith(".plt") and len(parts) >= 3 and (wanted is None or parts[-3] in wanted):
                members.setdefault(parts[-3], []).append(name)
        for names in members.values():
            for name in sorted(names)[:max_trips_per_user]:
                if not (root / name).exists():
                    archive.extract(name, root)
        marker.touch()
    return load_geolife_trajectories(root, user_ids=None if wanted is None else sorted(wanted))
