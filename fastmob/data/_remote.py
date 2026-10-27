"""Helpers shared by the public-dataset loaders: a cache directory and partial HTTP reads."""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request
from pathlib import Path


def cache_dir(name: str, cache: str | os.PathLike | None = None) -> Path:
    """Return (and create) the writable directory a dataset loader stores downloads in."""
    root = Path(cache) if cache is not None else Path(tempfile.gettempdir()) / "fastmob_data"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so a zip archive can be read in part."""

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.pos = 0
        self.bytes_read = 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        if head.headers.get("Accept-Ranges") != "bytes":
            raise OSError(f"{url} does not support partial downloads")
        self.size = int(head.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = (0, self.pos, self.size)[whence]
        self.pos = base + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.pos + size, self.size)
        if self.pos >= end:
            return b""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        data = urllib.request.urlopen(request).read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data
