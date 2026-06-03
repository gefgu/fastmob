from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from skmob2.data._frames import FlowDataFrame, TrajDataFrame

DATASETS_DIR = Path(__file__).parent / "datasets"


class DatasetBuilder(ABC):
    """Base class for dataset-specific builders."""

    def __init__(self):
        self.dataset_info = get_dataset_info(self.__class__.__name__)

    @abstractmethod
    def prepare(self, full_path_files):
        pass


def load_dataset(name, drop_columns=False, auth=None, show_progress=False):
    """Load one of the original scikit-mobility datasets."""
    if type(name) is not str:
        raise ValueError("The argument `name` must be a string.")
    if type(drop_columns) is not bool:
        raise ValueError("The argument `drop_columns` must be a boolean.")
    if auth is not None:
        if len(auth) != 2:
            raise ValueError("The argument `auth` must have length 2.")
        if type(auth[0]) is not str or type(auth[1]) is not str:
            raise ValueError("The argument `auth` must be a pair of strings.")
    if type(show_progress) is not bool:
        raise ValueError("The argument `show_progress` must be a boolean.")

    short_name = name[:-3] if name.endswith(".py") else name
    try:
        module = importlib.import_module(f".datasets.{short_name}.{short_name}", package="skmob2.data")
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name.startswith("skmob2.data.datasets") or exc.name == short_name):
            raise ValueError("Dataset name not found. Please use `list_datasets()` to list all the available datasets.")
        raise

    dataset_class = getattr(module, short_name)
    dataset_instance = dataset_class()
    dataset_info = dataset_instance.dataset_info
    hash_value = None if dataset_info["hash"] == "" else dataset_info["hash"]

    if dataset_info["auth"] == "yes":
        if auth is None or len(auth) != 2:
            raise ValueError("`auth` should be a pair (username, password) used for the authentication.")
    else:
        auth = ()

    full_path_files = _skmob2_downloader(
        dataset_info["url"],
        hash_value,
        auth=auth,
        download_format=dataset_info["download_format"],
        show_progress=show_progress,
    )
    dataset = dataset_instance.prepare(full_path_files)

    if isinstance(dataset, TrajDataFrame) and drop_columns:
        dataset = TrajDataFrame(
            dataset.df[["uid", "lat", "lng", "datetime"]],
            crs=getattr(dataset, "crs", None),
            parameters=getattr(dataset, "parameters", None),
        )

    if isinstance(dataset, (TrajDataFrame, FlowDataFrame, pd.DataFrame)):
        dataset._info = dataset_info
    else:
        try:
            dataset._info = dataset_info
        except Exception:
            pass

    return dataset


def list_datasets(details=False, data_types=None):
    """List available original scikit-mobility datasets."""
    if type(details) is not bool:
        raise ValueError("The argument `details` must be a boolean.")
    if data_types is not None and type(data_types) is not str:
        if not all(isinstance(item, str) for item in data_types):
            raise ValueError("The argument `data_types` must be a list of strings.")

    names = sorted(
        path.name
        for path in DATASETS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("__") and (path / f"{path.name}.json").is_file()
    )
    if not details and data_types is None:
        return names

    if isinstance(data_types, str):
        data_types = [data_types]

    datasets = {}
    for name in names:
        info = get_dataset_info(name)
        if data_types is None or info.get("data_type") in data_types:
            datasets[name] = info

    if details:
        return datasets
    return list(datasets)


def get_dataset_info(name):
    """Return metadata for a dataset."""
    path_info = DATASETS_DIR / name / f"{name}.json"
    try:
        with path_info.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print("Missing .json file")
        return -1


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root) / "skmob2_data"
    return Path.home() / ".cache" / "skmob2_data"


def _skmob2_downloader(url, known_hash, download_format=None, auth=(), show_progress=False):
    del show_progress
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_name = _download_file_name(url)
    destination = cache_dir / file_name

    if not destination.exists():
        _download_url(url, destination, auth=auth)
    if known_hash is not None:
        _verify_sha256(destination, known_hash)

    if download_format == "zip":
        out_dir = cache_dir / f"{destination.stem}_unzipped"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(out_dir)
        files = _iter_files(out_dir)
    elif download_format == "tar":
        out_dir = cache_dir / f"{destination.name}_untarred"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(destination) as archive:
                archive.extractall(out_dir)
        files = _iter_files(out_dir)
    else:
        files = [destination]

    return [str(path) for path in files]


def _download_file_name(url: str) -> str:
    parsed_name = urllib.request.urlparse(url).path.rsplit("/", 1)[-1]
    return parsed_name or hashlib.sha256(url.encode("utf-8")).hexdigest()


def _download_url(url: str, destination: Path, auth=()):
    request = urllib.request.Request(url)
    if auth:
        import base64

        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _verify_sha256(path: Path, known_hash: str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != known_hash:
        raise ValueError(f"Hash mismatch for downloaded dataset {path.name}.")


def _iter_files(root: Path) -> Iterable[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
