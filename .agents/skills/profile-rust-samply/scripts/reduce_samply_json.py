from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


RUST_FOCUS_MARKERS = (
    "skmob2",
    "skmob2::",
    "src/",
    "pyo3",
    "_core",
    "rust",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce Firefox Profiler JSON from samply to hot frames and stacks.")
    parser.add_argument("profile_json", type=Path, help="Path to a Firefox Profiler .json or .json.gz file.")
    parser.add_argument("--top", type=positive_int, default=20, help="Maximum entries per ranking.")
    parser.add_argument("--filter", dest="name_filter", help="Case-insensitive frame/stack substring filter.")
    parser.add_argument("--thread", help="Case-insensitive thread name substring filter.")
    parser.add_argument("-o", "--output", type=Path, help="Write reduced JSON to this path instead of stdout.")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def load_profile(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def sidecar_path(profile_path: Path) -> Path:
    if profile_path.name.endswith(".json.gz"):
        return profile_path.with_name(profile_path.name[:-3] + ".syms.json")
    if profile_path.name.endswith(".json"):
        return profile_path.with_name(profile_path.name + ".syms.json")
    return profile_path.with_suffix(profile_path.suffix + ".syms.json")


def load_symbol_sidecar(profile_path: Path) -> dict[str, Any] | None:
    path = sidecar_path(profile_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def schema_names(table: dict[str, Any]) -> list[str]:
    schema = table.get("schema") or {}
    ordered = sorted(schema.items(), key=lambda item: int(item[1]))
    return [name for name, _ in ordered]


def table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    if "schema" not in table and isinstance(table.get("length"), int):
        length = int(table["length"])
        columns = {name: values for name, values in table.items() if name != "length" and isinstance(values, list)}
        return [
            {name: values[index] for name, values in columns.items() if index < len(values)}
            for index in range(length)
        ]
    names = schema_names(table)
    rows = table.get("data") or []
    return [dict(zip(names, row)) for row in rows]


def lookup_string(strings: list[Any], value: Any) -> str:
    if isinstance(value, int) and 0 <= value < len(strings):
        return str(strings[value])
    if value is None:
        return ""
    return str(value)


def normalize_debug_id(value: Any) -> str:
    return str(value or "").replace("-", "").upper()


def build_symbol_maps(payload: dict[str, Any], sidecar: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    if sidecar is None:
        return {}
    strings = sidecar.get("string_table") or []
    sidecar_by_debug_name = {item.get("debug_name"): item for item in sidecar.get("data") or []}
    sidecar_by_debug_id = {normalize_debug_id(item.get("debug_id")): item for item in sidecar.get("data") or []}
    symbol_maps: dict[int, list[dict[str, Any]]] = {}
    for lib_index, lib in enumerate(payload.get("libs") or []):
        item = sidecar_by_debug_name.get(lib.get("debugName") or lib.get("name"))
        if item is None:
            item = sidecar_by_debug_id.get(normalize_debug_id(lib.get("breakpadId")))
        if item is None:
            continue
        symbols = []
        for symbol in item.get("symbol_table") or []:
            name = lookup_string(strings, symbol.get("symbol"))
            if not name:
                continue
            symbols.append(
                {
                    "rva": int(symbol.get("rva") or 0),
                    "size": int(symbol.get("size") or 0),
                    "name": name,
                }
            )
        symbol_maps[lib_index] = sorted(symbols, key=lambda item: item["rva"])
    return symbol_maps


def resolve_symbol(address: Any, lib_index: Any, symbol_maps: dict[int, list[dict[str, Any]]]) -> str:
    if not isinstance(address, int) or not isinstance(lib_index, int):
        return ""
    symbols = symbol_maps.get(lib_index) or []
    for symbol in symbols:
        start = symbol["rva"]
        end = start + max(symbol["size"], 1)
        if start <= address < end:
            return symbol["name"]
    return ""


def collect_threads(profile: dict[str, Any]) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []
    for thread in profile.get("threads") or []:
        threads.append({"process": profile.get("meta", {}).get("product") or "root", "thread": thread})
    for process in profile.get("processes") or []:
        process_name = process.get("processName") or process.get("name") or process.get("pid") or "process"
        for thread in process.get("threads") or []:
            threads.append({"process": str(process_name), "thread": thread})
    return threads


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def frame_name(
    frame: dict[str, Any],
    strings: list[Any],
    func_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    symbol_maps: dict[int, list[dict[str, Any]]],
) -> str:
    func_index = frame.get("func")
    if isinstance(func_index, int) and 0 <= func_index < len(func_rows):
        func = func_rows[func_index]
        resource_index = func.get("resource")
        lib_index = None
        if isinstance(resource_index, int) and 0 <= resource_index < len(resource_rows):
            lib_index = resource_rows[resource_index].get("lib")
        symbol = resolve_symbol(frame.get("address"), lib_index, symbol_maps)
        if symbol:
            return symbol
        name = lookup_string(strings, func.get("name"))
        file_name = lookup_string(strings, func.get("fileName"))
        if name and file_name:
            return f"{name} ({file_name})"
        if name:
            return name
    value = first_present(
        frame,
        (
            "location",
            "name",
            "functionName",
            "symbol",
            "nativeSymbol",
            "address",
        ),
    )
    return lookup_string(strings, value) or "<unknown frame>"


def reconstruct_stack(
    stack_index: Any,
    stack_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    strings: list[Any],
    func_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    symbol_maps: dict[int, list[dict[str, Any]]],
) -> list[str]:
    if not isinstance(stack_index, int):
        return []

    frames: list[str] = []
    seen: set[int] = set()
    current: int | None = stack_index
    while current is not None:
        if current in seen or current < 0 or current >= len(stack_rows):
            break
        seen.add(current)
        stack = stack_rows[current]
        frame_index = first_present(stack, ("frame", "frameIndex"))
        if isinstance(frame_index, int) and 0 <= frame_index < len(frame_rows):
            frames.append(frame_name(frame_rows[frame_index], strings, func_rows, resource_rows, symbol_maps))
        prefix = first_present(stack, ("prefix", "prefixStack"))
        current = prefix if isinstance(prefix, int) else None
    frames.reverse()
    return frames


def sample_weight(sample: dict[str, Any], weight_field: str | None) -> float:
    if weight_field is None:
        return 1.0
    value = sample.get(weight_field)
    if value is None:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def rank(counter: Counter[str], top: int) -> list[dict[str, Any]]:
    return [{"name": name, "weight": round(weight, 6)} for name, weight in counter.most_common(top) if weight > 0]


def rank_stacks(counter: Counter[tuple[str, ...]], top: int) -> list[dict[str, Any]]:
    return [
        {"stack": list(stack), "weight": round(weight, 6)}
        for stack, weight in counter.most_common(top)
        if weight > 0
    ]


def matches_filter(text: str, needle: str | None) -> bool:
    return needle is None or needle.lower() in text.lower()


def matches_rust_focus(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in RUST_FOCUS_MARKERS)


def reduce_thread(
    thread: dict[str, Any],
    *,
    process_name: str,
    top: int,
    name_filter: str | None,
    symbol_maps: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    strings = thread.get("stringArray") or thread.get("stringTable") or []
    frame_rows = table_rows(thread.get("frameTable") or {})
    func_rows = table_rows(thread.get("funcTable") or {})
    resource_rows = table_rows(thread.get("resourceTable") or {})
    stack_rows = table_rows(thread.get("stackTable") or {})
    sample_rows = table_rows(thread.get("samples") or {})
    sample_table = thread.get("samples") or {}
    sample_schema = schema_names(sample_table) or [name for name in sample_table if name != "length"]
    stack_field = "stack" if "stack" in sample_schema else "stackIndex" if "stackIndex" in sample_schema else None
    time_field = "time" if "time" in sample_schema else None
    weight_field = "weight" if "weight" in sample_schema else None

    inclusive: Counter[str] = Counter()
    leaf: Counter[str] = Counter()
    stacks: Counter[tuple[str, ...]] = Counter()
    rust_inclusive: Counter[str] = Counter()
    rust_leaf: Counter[str] = Counter()
    rust_stacks: Counter[tuple[str, ...]] = Counter()
    unknown_samples = 0
    times: list[float] = []

    for sample in sample_rows:
        if time_field is not None and sample.get(time_field) is not None:
            try:
                times.append(float(sample[time_field]))
            except (TypeError, ValueError):
                pass
        stack_index = sample.get(stack_field) if stack_field is not None else None
        stack = reconstruct_stack(stack_index, stack_rows, frame_rows, strings, func_rows, resource_rows, symbol_maps)
        if not stack:
            unknown_samples += 1
            continue
        stack_text = " > ".join(stack)
        if not matches_filter(stack_text, name_filter):
            continue
        weight = sample_weight(sample, weight_field)
        stacks[tuple(stack)] += weight
        leaf[stack[-1]] += weight
        for name in dict.fromkeys(stack):
            inclusive[name] += weight
            if matches_rust_focus(name):
                rust_inclusive[name] += weight
        if matches_rust_focus(stack[-1]):
            rust_leaf[stack[-1]] += weight
        if any(matches_rust_focus(name) for name in stack):
            rust_stacks[tuple(stack)] += weight

    return {
        "process": process_name,
        "thread": thread.get("name") or thread.get("tid") or "<unknown thread>",
        "sample_count": len(sample_rows),
        "duration_range": {
            "start": min(times) if times else None,
            "end": max(times) if times else None,
        },
        "weight_field": weight_field or "sample_count",
        "unknown_stack_samples": unknown_samples,
        "top_inclusive_frames": rank(inclusive, top),
        "top_leaf_frames": rank(leaf, top),
        "top_stacks": rank_stacks(stacks, top),
        "rust_focused": {
            "top_inclusive_frames": rank(rust_inclusive, top),
            "top_leaf_frames": rank(rust_leaf, top),
            "top_stacks": rank_stacks(rust_stacks, top),
        },
    }


def reduce_profile(
    payload: dict[str, Any],
    *,
    profile_path: Path,
    top: int,
    name_filter: str | None = None,
    thread_filter: str | None = None,
) -> dict[str, Any]:
    symbol_maps = build_symbol_maps(payload, load_symbol_sidecar(profile_path))
    reduced_threads = []
    for item in collect_threads(payload):
        thread = item["thread"]
        thread_name = str(thread.get("name") or thread.get("tid") or "")
        if thread_filter and thread_filter.lower() not in thread_name.lower():
            continue
        reduced_threads.append(
            reduce_thread(
                thread,
                process_name=item["process"],
                top=top,
                name_filter=name_filter,
                symbol_maps=symbol_maps,
            )
        )

    return {
        "metadata": {
            "profile_path": str(profile_path),
            "profile_name": payload.get("meta", {}).get("profileName"),
            "processes": sorted({thread["process"] for thread in reduced_threads}),
            "threads": [thread["thread"] for thread in reduced_threads],
            "thread_count": len(reduced_threads),
            "symbol_sidecar": str(sidecar_path(profile_path)) if sidecar_path(profile_path).exists() else None,
        },
        "threads": reduced_threads,
    }


def main() -> int:
    args = parse_args()
    payload = load_profile(args.profile_json)
    reduced = reduce_profile(
        payload,
        profile_path=args.profile_json,
        top=args.top,
        name_filter=args.name_filter,
        thread_filter=args.thread,
    )
    text = json.dumps(reduced, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
