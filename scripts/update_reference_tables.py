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


def summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    text = re.sub(r"\s+", " ", doc.split("\n\n")[0]).strip()
    text = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", text).replace("``", "`")
    text = re.split(r"(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)\.(?:\s|$)", text)[0].rstrip(".") + "."
    text = text.split(" -- ")[0]
    text = text if len(text) <= 150 else text[:147].rsplit(" ", 1)[0] + "..."
    return text.replace("|", "\\|") or "Undocumented."


def entries(page: Path) -> list[tuple[str, object]]:
    return [e for t in re.findall(r"^::: ([\w.]+)", page.read_text(), re.MULTILINE) for e in entries_for(t, resolve(t))]


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
    base = Path("../" * (len(page.relative_to(REF).parts)))  # from the page to docs/src/reference/..
    lines = ["| API | Description | Example |", "| --- | --- | --- |"]
    for path, obj in entries(page):
        name = path.rsplit(".", 1)[-1]
        ex = examples.get(canonical(obj))
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
