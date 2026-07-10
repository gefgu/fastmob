from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[4]
BRIGHTKITE_PATH = REPO_ROOT / "tests" / "shared" / "data" / "loc-brightkite_totalCheckins.txt.gz"
BRIGHTKITE_COLUMNS = ["user", "check-in_time", "latitude", "longitude", "location id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call an importable fkmob function under samply.")
    parser.add_argument("target", help="Import target in module:function form.")
    parser.add_argument(
        "--input-kind",
        choices=("brightkite-pandas", "brightkite-polars", "none"),
        default="brightkite-pandas",
        help="Input builder for the target's first positional argument.",
    )
    parser.add_argument("--rows", type=positive_int, default=10000, help="Rows to pass from the selected dataset.")
    parser.add_argument("--repeat", type=positive_int, default=3, help="Number of calls after one warmup call.")
    parser.add_argument("--kwargs-json", default="{}", help="JSON object passed as keyword arguments to the target.")
    parser.add_argument("--data-path", type=Path, default=BRIGHTKITE_PATH, help="Brightkite data path.")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def import_target(target: str) -> Callable[..., Any]:
    if ":" not in target:
        raise ValueError("target must use module:function syntax")
    module_name, function_name = target.split(":", 1)
    if not module_name or not function_name:
        raise ValueError("target must include both module and function names")
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"{target!r} is not callable")
    return func


def parse_kwargs(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--kwargs-json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--kwargs-json must decode to a JSON object")
    return payload


def build_input(args: argparse.Namespace) -> Any:
    if args.input_kind == "none":
        return None
    if args.input_kind == "brightkite-pandas":
        return load_brightkite_pandas(args.data_path).head(args.rows).copy()
    if args.input_kind == "brightkite-polars":
        return load_brightkite_polars(args.data_path).head(args.rows)
    raise ValueError(f"unsupported input kind: {args.input_kind}")


def load_brightkite_pandas(data_path: Path) -> Any:
    import pandas as pd

    df = pd.read_csv(data_path, sep="\t", header=None, names=BRIGHTKITE_COLUMNS)
    df["check-in_time"] = pd.to_datetime(df["check-in_time"], errors="coerce")
    return df


def load_brightkite_polars(data_path: Path) -> Any:
    import polars as pl

    return pl.read_csv(
        data_path,
        separator="\t",
        has_header=False,
        new_columns=BRIGHTKITE_COLUMNS,
        try_parse_dates=True,
    )


def call_target(func: Callable[..., Any], input_value: Any, kwargs: dict[str, Any]) -> Any:
    if input_value is None:
        return func(**kwargs)
    return func(input_value, **kwargs)


def main() -> int:
    args = parse_args()
    func = import_target(args.target)
    kwargs = parse_kwargs(args.kwargs_json)
    input_value = build_input(args)

    input_description = "none" if input_value is None else f"{args.input_kind}:{len(input_value)} rows"
    print(f"target={args.target}")
    print(f"input={input_description}")
    print(f"repeat={args.repeat}")
    print("warmup=1")

    call_target(func, input_value, kwargs)

    last_result: Any = None
    start = time.perf_counter()
    for index in range(args.repeat):
        last_result = call_target(func, input_value, kwargs)
        print(f"completed={index + 1}")
    elapsed = time.perf_counter() - start

    result_type = type(last_result).__name__
    result_size = len(last_result) if hasattr(last_result, "__len__") else None
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"result_type={result_type}")
    if result_size is not None:
        print(f"result_len={result_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
