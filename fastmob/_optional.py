from __future__ import annotations


def require_optional(module_name: str, extra: str = "generation"):
    try:
        module = __import__(module_name)
    except ImportError as exc:
        raise ImportError(f"{module_name} is required: pip install fastmob[{extra}]") from exc
    return module
