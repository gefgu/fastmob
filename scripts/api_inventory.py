"""Inventory the public API and check that each symbol has a catalogue anchor.

Writes ``inventory.md`` and ``coverage.csv`` (the example-coverage ledger) to ``--out``.
Usage: .venv/bin/python scripts/api_inventory.py --out AI/docs_audit/runs/learn-reference/lr1
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import re
from pathlib import Path

import fastmob

# Symbols present only in uncommitted work, so not part of any release: excluded from the inventory.
UNRELEASED = ("radius_of_gyration_lazy",)
# Exposed by `fastmob.models.__all__` by accident: an internal helper of the Poisson fit, not public API.
NOT_PUBLIC = ("fastmob.models.gravity.ci",)
# Packages whose ``__all__`` lists the public API (so the inventory does not depend on what the docs already name).
PUBLIC_PACKAGES = (
    "fastmob.tessellation",
    "fastmob.core",
    "fastmob.data",
    "fastmob.integration",
    "fastmob.io",
    "fastmob.measures",
    "fastmob.models",
    "fastmob.network",
    "fastmob.preprocessing",
    "fastmob.privacy",
    "fastmob.social",
    "fastmob.tessellation",
    "fastmob.trajectory",
)
INTERNAL = ("fastmob._core", "fastmob.utils", "fastmob.core.dispatch", "fastmob.core.base")
FIELDS = [
    "canonical_symbol",
    "module",
    "aliases",
    "kind",
    "public_evidence",
    "example_path",
    "cell_id",
    "use_case",
    "output_explanation",
    "execution_log",
    "status",
    "exclusion_reason",
]


def doc_targets(ref: Path) -> list[str]:
    return [
        m.group(1)
        for p in sorted(ref.rglob("*.md"))
        for m in re.finditer(r"^::: ([\w.]+)", p.read_text(), re.MULTILINE)
    ]


def resolve(path: str):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def canonical(obj) -> str:
    return f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', getattr(obj, '__name__', ''))}"


def collect() -> dict[str, dict]:
    syms: dict[str, dict] = {}

    def add(name: str, obj, evidence: str, kind: str, module: str):
        key = canonical(obj)
        row = syms.setdefault(key, {"module": module, "kind": kind, "aliases": set(), "evidence": set()})
        row["aliases"].add(name)
        row["evidence"].add(evidence)

    for name in fastmob.__all__:
        obj = getattr(fastmob, name)
        if inspect.ismodule(obj):
            continue
        if obj.__module__.startswith(INTERNAL):
            continue
        add(f"fastmob.{name}", obj, "fastmob.__all__", "class" if inspect.isclass(obj) else "function", obj.__module__)
    for package in PUBLIC_PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if obj is None or inspect.ismodule(obj) or not callable(obj):
                continue
            if getattr(obj, "__module__", "").startswith(INTERNAL) or not getattr(obj, "__module__", "").startswith(
                "fastmob"
            ):
                continue
            add(
                f"{package}.{name}",
                obj,
                f"{package}.__all__",
                "class" if inspect.isclass(obj) else "function",
                obj.__module__,
            )
    from fastmob.core.base import BaseDataFrame

    add("fastmob.core.base.BaseDataFrame", BaseDataFrame, "public base class", "class", BaseDataFrame.__module__)
    for target in doc_targets(Path("docs/src/reference")):
        obj = resolve(target)
        names = getattr(obj, "__all__", None) or [n for n in dir(obj) if not n.startswith("_")]
        members = [(f"{target}.{n}", getattr(obj, n)) for n in names] if inspect.ismodule(obj) else [(target, obj)]
        for n, o in members:
            if inspect.ismodule(o) or not callable(o):
                continue
            if inspect.ismodule(obj) and getattr(o, "__module__", "").split(".")[0] != "fastmob":
                continue
            if getattr(o, "__module__", "").startswith(INTERNAL):
                continue
            add(n, o, "reference page", "class" if inspect.isclass(o) else "function", getattr(o, "__module__", ""))
    for key in [k for k in syms if any(u in k for u in UNRELEASED + NOT_PUBLIC)]:
        del syms[key]
    for key, row in list(syms.items()):
        if row["kind"] != "class":
            continue
        cls = resolve(min(row["aliases"]))
        for n, m in inspect.getmembers(cls):
            if n.startswith("_") or not callable(m):
                continue
            owner = getattr(m, "__qualname__", "")
            if not owner.startswith(cls.__name__ + ".") or not str(getattr(m, "__module__", None)).startswith(
                "fastmob"
            ):
                continue  # inherited (from a base class or a builtin): counted under its defining class
            syms.setdefault(
                f"{cls.__module__}.{owner}",
                {
                    "module": cls.__module__,
                    "kind": "method",
                    "aliases": {f"{cls.__name__}.{n}"},
                    "evidence": {"public class method"},
                },
            )
    return syms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--site", type=Path, default=Path("site"))
    out = ap.parse_args()
    args = out
    syms = collect()
    pages = {p: p.read_text(errors="ignore") for p in args.site.rglob("*.html")} if args.site.exists() else {}
    blob = "\n".join(pages.values())
    args.out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for key, r in sorted(syms.items()):
        status = "uncovered"
        candidates = {key, *r["aliases"]}
        if r["kind"] == "method":  # rendered under the class's own anchor, e.g. fastmob.network.RoadNetwork.build
            owner, method = key.rsplit(".", 1)
            candidates |= {f"{alias}.{method}" for alias in syms.get(owner, {}).get("aliases", ())}
        has_anchor = any(f'id="{c}"' in blob for c in candidates)
        if not has_anchor and pages:
            missing.append(key)
        rows.append(
            {
                "canonical_symbol": key,
                "module": r["module"],
                "aliases": ";".join(sorted(r["aliases"])),
                "kind": r["kind"],
                "public_evidence": ";".join(sorted(r["evidence"])),
                "status": status,
            }
        )
    previous = {}
    ledger = args.out / "coverage.csv"
    if ledger.exists():  # keep what notebook_coverage.py already filled in
        with ledger.open() as fh:
            previous = {r["canonical_symbol"]: r for r in csv.DictReader(fh)}
    for r in rows:
        old = previous.get(r["canonical_symbol"], {})
        r.update(
            {
                k: v
                for k, v in old.items()
                if v and k not in ("canonical_symbol", "module", "aliases", "kind", "public_evidence")
            }
        )
    with (args.out / "coverage.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    fn = [r for r in rows if r["kind"] in ("function", "method")]
    by_mod: dict[str, int] = {}
    for r in fn:
        by_mod[r["module"].split(".")[1] if "." in r["module"] else r["module"]] = (
            by_mod.get(r["module"].split(".")[1] if "." in r["module"] else r["module"], 0) + 1
        )
    lines = [
        "# Public API inventory",
        "",
        f"Unique canonical symbols: {len(rows)} ({len(fn)} functions+methods, {len(rows) - len(fn)} classes).",
        f"80% target: {-(-len(fn) * 8 // 10)} functions/methods.",
        f"Symbols without a catalogue anchor in the built site: {len(missing)}"
        if pages
        else "Site not built: anchors unchecked.",
        "",
        "## Functions and methods per package",
        *[f"- {m}: {c}" for m, c in sorted(by_mod.items())],
        "",
        "## Missing anchors",
        *[f"- `{m}`" for m in missing],
    ]
    (args.out / "inventory.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
