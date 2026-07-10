from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
GEOLIFE_PAGE_URL = "https://www.microsoft.com/en-my/download/details.aspx?id=52367"
GEOLIFE_DOWNLOAD_URL = (
    "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/"
    "Geolife%20Trajectories%201.3.zip"
)
GEOLIFE_ZIP_PATH = DATA_DIR / "Geolife Trajectories 1.3.zip"
GEOLIFE_EXTRACTED_DIR = DATA_DIR / "Geolife Trajectories 1.3"
GEOLIFE_DEFAULT_ROWS = 10_000

PLT_COLUMNS = [
    "latitude",
    "longitude",
    "unused",
    "altitude",
    "days_since_1899",
    "date",
    "time",
]


def ensure_geolife_dataset() -> Path:
    """Return the extracted GeoLife dataset path, downloading it if needed."""
    data_root = _find_data_root(GEOLIFE_EXTRACTED_DIR)
    if data_root is not None:
        return data_root

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GEOLIFE_ZIP_PATH.exists():
        urllib.request.urlretrieve(GEOLIFE_DOWNLOAD_URL, GEOLIFE_ZIP_PATH)

    with zipfile.ZipFile(GEOLIFE_ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    data_root = _find_data_root(GEOLIFE_EXTRACTED_DIR) or _find_data_root(DATA_DIR)
    if data_root is None:
        raise FileNotFoundError(
            "Could not find GeoLife Data directory after extracting "
            f"{GEOLIFE_ZIP_PATH}. Source page: {GEOLIFE_PAGE_URL}"
        )
    return data_root


def load_geolife_pandas(
    *,
    mode: str = "slice",
    rows: int = GEOLIFE_DEFAULT_ROWS,
    dataset_root: Path | None = None,
) -> pd.DataFrame:
    """Load GeoLife PLT files into normalized trajectory columns."""
    if mode not in {"slice", "full"}:
        raise ValueError("GeoLife mode must be 'slice' or 'full'")
    if rows <= 0:
        raise ValueError("GeoLife rows must be greater than zero")

    root = dataset_root or ensure_geolife_dataset()
    frames: list[pd.DataFrame] = []
    remaining = rows if mode == "slice" else None

    for plt_path in iter_geolife_plt_files(root):
        if remaining is not None and remaining <= 0:
            break
        frame = read_geolife_plt(plt_path, root)
        if frame.empty:
            continue
        if remaining is not None and len(frame) > remaining:
            frame = frame.iloc[:remaining].copy()
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)

    if not frames:
        raise FileNotFoundError(f"No GeoLife .plt trajectory files found under {root}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["user", "check-in_time", "trajectory_id", "latitude", "longitude"], kind="mergesort")
    df = df.dropna(subset=["user", "check-in_time", "latitude", "longitude"]).reset_index(drop=True)
    return df


def iter_geolife_plt_files(root: Path) -> Iterable[Path]:
    """Yield GeoLife trajectory files in deterministic user/path order."""
    data_root = _find_data_root(root)
    if data_root is None:
        return iter(())
    return iter(sorted(data_root.glob("*/Trajectory/*.plt"), key=lambda path: path.as_posix()))


def read_geolife_plt(path: Path, data_root: Path) -> pd.DataFrame:
    """Parse a GeoLife .plt file and normalize columns for skmob/fkmob tests."""
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=PLT_COLUMNS,
        usecols=["latitude", "longitude", "date", "time"],
    )
    if df.empty:
        return pd.DataFrame(columns=["user", "check-in_time", "latitude", "longitude", "trajectory_id"])

    user = path.parent.parent.name
    trajectory_id = path.stem
    timestamp_text = df["date"].astype("string") + " " + df["time"].astype("string")
    out = pd.DataFrame(
        {
            "user": user,
            "check-in_time": pd.to_datetime(timestamp_text, errors="coerce"),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "trajectory_id": f"{user}/{trajectory_id}",
        }
    )
    return out.dropna(subset=["check-in_time", "latitude", "longitude"])


def read_geolife_plt_text(text: str, *, user: str = "000", trajectory_id: str = "sample") -> pd.DataFrame:
    """Parse GeoLife .plt text for focused parser tests."""
    df = pd.read_csv(
        io.StringIO(text),
        skiprows=6,
        header=None,
        names=PLT_COLUMNS,
        usecols=["latitude", "longitude", "date", "time"],
    )
    timestamp_text = df["date"].astype("string") + " " + df["time"].astype("string")
    return pd.DataFrame(
        {
            "user": user,
            "check-in_time": pd.to_datetime(timestamp_text, errors="coerce"),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "trajectory_id": f"{user}/{trajectory_id}",
        }
    ).dropna(subset=["check-in_time", "latitude", "longitude"])


def _find_data_root(path: Path) -> Path | None:
    """Return a directory containing GeoLife user directories."""
    candidates = [
        path,
        path / "Data",
        path / "Geolife Trajectories 1.3" / "Data",
        path / "Geolife Trajectories 1.3",
    ]
    for candidate in candidates:
        if (candidate / "000" / "Trajectory").is_dir():
            return candidate
        data = candidate / "Data"
        if (data / "000" / "Trajectory").is_dir():
            return data
    return None
