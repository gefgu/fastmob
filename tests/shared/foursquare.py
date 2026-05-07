from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FOURSQUARE_DOWNLOAD_URL = "http://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip"
FOURSQUARE_ZIP_PATH = DATA_DIR / "dataset_tsmc2014.zip"
FOURSQUARE_FILE_NAME = "TSMC2014_NYC.txt"
FOURSQUARE_DEFAULT_ROWS = 10_000
FOURSQUARE_TIMESTAMP_FORMAT = "%a %b %d %H:%M:%S %z %Y"

RAW_COLUMNS = [
    "user",
    "location_id",
    "venue_category_id",
    "venue_category_name",
    "latitude",
    "longitude",
    "timezone_offset",
    "check-in_time",
]
OUTPUT_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location_id"]


def ensure_foursquare_dataset() -> Path:
    """Return the extracted Foursquare NYC data file, downloading it if needed."""
    data_file = _find_data_file(DATA_DIR)
    if data_file is not None:
        return data_file

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not FOURSQUARE_ZIP_PATH.exists():
        urllib.request.urlretrieve(FOURSQUARE_DOWNLOAD_URL, FOURSQUARE_ZIP_PATH)

    with zipfile.ZipFile(FOURSQUARE_ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    data_file = _find_data_file(DATA_DIR)
    if data_file is None:
        raise FileNotFoundError(
            f"Could not find {FOURSQUARE_FILE_NAME} after extracting {FOURSQUARE_ZIP_PATH}. "
            f"Source URL: {FOURSQUARE_DOWNLOAD_URL}"
        )
    return data_file


def load_foursquare_pandas(
    *,
    mode: str = "slice",
    rows: int = FOURSQUARE_DEFAULT_ROWS,
    dataset_root: Path | None = None,
) -> pd.DataFrame:
    """Load Foursquare NYC check-ins into normalized trajectory columns."""
    if mode not in {"slice", "full"}:
        raise ValueError("Foursquare mode must be 'slice' or 'full'")
    if rows <= 0:
        raise ValueError("Foursquare rows must be greater than zero")

    data_file = _find_data_file(dataset_root) if dataset_root is not None else ensure_foursquare_dataset()
    if data_file is None:
        raise FileNotFoundError(f"Could not find {FOURSQUARE_FILE_NAME} under {dataset_root}")

    df = read_foursquare_file(data_file)
    if df.empty:
        raise FileNotFoundError(f"No Foursquare NYC check-ins found in {data_file}")

    df = _normalize_foursquare_frame(df)
    if mode == "slice":
        df = df.iloc[:rows].copy()
    return df.reset_index(drop=True)


def read_foursquare_file(path: Path) -> pd.DataFrame:
    """Parse a raw Foursquare NYC file into normalized skmob/skmob2 columns."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        encoding="ISO-8859-1",
        usecols=["user", "location_id", "latitude", "longitude", "check-in_time"],
    )
    return _coerce_foursquare_columns(df)


def read_foursquare_text(text: str) -> pd.DataFrame:
    """Parse raw Foursquare NYC text for focused parser tests."""
    df = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        usecols=["user", "location_id", "latitude", "longitude", "check-in_time"],
    )
    return _coerce_foursquare_columns(df)


def _coerce_foursquare_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "user": df["user"].astype("string"),
            "check-in_time": pd.to_datetime(
                df["check-in_time"],
                errors="coerce",
                format=FOURSQUARE_TIMESTAMP_FORMAT,
                utc=True,
            ).dt.tz_localize(None),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "location_id": df["location_id"].astype("string"),
        }
    )
    return out.dropna(subset=OUTPUT_COLUMNS)


def _normalize_foursquare_frame(df: pd.DataFrame) -> pd.DataFrame:
    # skmob sorts per-user trajectories by datetime only. Rows that share the
    # same user and timestamp therefore need a stable tie-breaker for parity.
    return (
        df.sort_values(["user", "check-in_time", "latitude", "longitude", "location_id"], kind="mergesort")
        .drop_duplicates(["user", "check-in_time"], keep="first")
        .reset_index(drop=True)
    )


def _find_data_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file() and path.name == FOURSQUARE_FILE_NAME:
        return path
    candidate = path / FOURSQUARE_FILE_NAME
    if candidate.is_file():
        return candidate
    matches = sorted(path.rglob(FOURSQUARE_FILE_NAME), key=lambda candidate_path: candidate_path.as_posix())
    return matches[0] if matches else None
