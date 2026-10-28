"""Regenerate the API catalogue tables of Reference pages from docstrings and example links.

A page opts in with ``<!-- api-table:start -->`` / ``<!-- api-table:end -->`` markers. Rows follow the
page's ``:::`` directives (module directives expand to their public members). The Example column links
to the Learn notebook section recorded in ``docs/reference_examples.json`` by ``notebook_coverage.py``.

Usage: .venv/bin/python scripts/update_reference_tables.py [page.md ...]
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

from api_inventory import INTERNAL, canonical, resolve

REF = Path("docs/src/reference")
EXAMPLE_MARK = "<!-- example-link -->"
START, END = "<!-- api-table:start -->", "<!-- api-table:end -->"
GROUPS = Path("docs/reference_groups.json")


def summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    text = re.sub(r"\s+", " ", doc.split("\n\n")[0]).strip()
    text = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", text).replace("``", "`")
    text = re.split(r"(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)\.(?:\s|$)", text)[0].rstrip(".") + "."
    text = text.split(" -- ")[0]
    text = text if len(text) <= 150 else text[:147].rsplit(" ", 1)[0] + "..."
    return text.replace("|", "\\|") or "Undocumented."


def entries(page: Path) -> list[tuple[str, object]]:
    return [e for _, rows in directive_entries(page) for e in rows]


def directive_entries(page: Path) -> list[tuple[str, list[tuple[str, object]]]]:
    """``(directive target, rows)`` for each ``:::`` directive of the page, in page order."""
    targets = re.findall(r"^::: ([\w.]+)", page.read_text(), re.MULTILINE)
    return [(t, entries_for(t, resolve(t))) for t in targets]


def _class_methods(path: str, cls) -> list[tuple[str, object]]:
    """Public methods defined by ``cls`` itself; inherited ones belong to their defining class."""
    return [
        (f"{path}.{name}", member)
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_")
        and callable(member)
        and str(getattr(member, "__module__", None)).startswith("fastmob")
        and getattr(member, "__qualname__", "").startswith(cls.__name__ + ".")
    ]


def entries_for(target: str, obj) -> list[tuple[str, object]]:
    if not inspect.ismodule(obj):
        return [(target, obj), *(_class_methods(target, obj) if inspect.isclass(obj) else [])]
    tail = obj.__name__.rsplit(".", 1)[-1]
    names = getattr(obj, "__all__", None) or [
        n
        for n in dir(obj)
        if not n.startswith("_") and (getattr(getattr(obj, n), "__module__", None) == obj.__name__ or n == tail)
    ]
    out = []
    for n in names:
        member = getattr(obj, n)
        if not callable(member) or inspect.ismodule(member) or getattr(member, "__module__", "").startswith(INTERNAL):
            continue
        out.append((f"{target}.{n}", member))
        if inspect.isclass(member):
            out += _class_methods(f"{target}.{n}", member)
    return out


def table(page: Path, examples: dict) -> str:
    """One table, or one table per task group when ``docs/reference_groups.json`` lists the page."""
    groups = json.loads(GROUPS.read_text()).get(page.relative_to(REF).as_posix()) if GROUPS.exists() else None
    if not groups:
        return rows_table(page, entries(page), examples, entries(page))
    per_target = {t.rsplit(".", 1)[-1]: rows for t, rows in directive_entries(page)}
    listed = [name for _, names in groups for name in names]
    if sorted(listed) != sorted(per_target):
        raise SystemExit(f"{page}: groups and directives differ: {sorted(set(listed) ^ set(per_target))}")
    all_rows = entries(page)
    return "\n\n".join(
        f"**{title}**\n\n" + rows_table(page, [r for n in names for r in per_target[n]], examples, all_rows)
        for title, names in groups
    )


def rows_table(page: Path, rows: list[tuple[str, object]], examples: dict, all_rows: list) -> str:
    base = Path("../" * (len(page.relative_to(REF).parts)))  # from the page to docs/src/reference/..
    lines = ["| API | Description | Example |", "| --- | --- | --- |"]
    for path, obj in rows:
        name = path.rsplit(".", 1)[-1]
        qualname = getattr(obj, "__qualname__", "")
        if not inspect.isclass(obj) and "." in qualname and "<locals>" not in qualname:
            name = ".".join(qualname.split(".")[-2:])  # a method row carries its class
        ex = examples.get(canonical(obj))
        if ex is None and inspect.isclass(obj):  # a class row shows the first example of its methods
            ex = next(
                (
                    examples[canonical(o)]
                    for p2, o in all_rows
                    if p2.startswith(path + ".") and canonical(o) in examples
                ),
                None,
            )
        link = f"[{ex['title'].split(':')[0]}]({base / ex['page']}#{ex['anchor']})" if ex else ""
        lines.append(f"| [`{name}`](#{path}) | {summary(obj)} | {link} |")
    return "\n".join(lines)


def example_link(page: Path, target: str, examples: dict) -> str:
    """Markdown line linking a ``:::`` entry to the notebook section that runs it, or ""."""
    obj = resolve(target)
    members = [m for _, m in entries_for(target, obj)]
    base = Path("../" * len(page.relative_to(REF).parts))
    links = []
    for m in members:
        ex = examples.get(canonical(m))
        if ex:
            links.append(f"[{ex['title'].split(':')[0]}]({base / ex['page']}#{ex['anchor']})")
    return f"{EXAMPLE_MARK}**Worked example:** {' · '.join(dict.fromkeys(links))}\n" if links else ""


def add_example_links(page: Path, text: str, examples: dict) -> str:
    text = re.sub(rf"^{re.escape(EXAMPLE_MARK)}.*\n\n?", "", text, flags=re.MULTILINE)

    def put(match: re.Match) -> str:
        line = example_link(page, match.group(2), examples)
        return match.group(1) + (f"\n{line}" if line else "")

    return re.sub(r"(^::: ([\w.]+)\n(?:    .*\n)*)", put, text, flags=re.MULTILINE)


def main() -> None:
    examples = (
        json.loads(Path("docs/reference_examples.json").read_text())
        if Path("docs/reference_examples.json").exists()
        else {}
    )
    pages = [Path(p) for p in sys.argv[1:]] or sorted(REF.rglob("*.md"))
    for page in pages:
        text = page.read_text()
        if START not in text:
            continue
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        page.write_text(add_example_links(page, f"{head}{START}\n{table(page, examples)}\n{END}{tail}", examples))
        print("updated", page)


if __name__ == "__main__":
    main()
