"""Compatibility re-exports for shared internal helpers.

General backend, Arrow, grouping, and column-detection utilities live in
``fastmob.utils._common``. This module remains so existing private imports
continue to work.
"""

from __future__ import annotations

from fastmob.utils import _common as _utils_common

__all__ = [name for name in dir(_utils_common) if not name.startswith("__")]
globals().update({name: getattr(_utils_common, name) for name in __all__})
