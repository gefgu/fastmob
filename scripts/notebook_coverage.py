"""Fill the example-coverage ledger from the executed Learn notebooks.

A public function counts as covered when an executed code cell (one with saved output)
calls it. Rows record the notebook, cell, the heading that states the use case, and the first
sentence of the markdown cell that explains the output. ``docs/reference_examples.json`` maps
each covered symbol to the notebook section so Reference tables can link to it.

Usage: .venv/bin/python scripts/notebook_coverage.py --ledger AI/docs_audit/runs/learn-reference/lr1/coverage.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import nbformat

from api_inventory import canonical, resolve

DOCS = Path("docs/src")


def slug(title: str) -> str:
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", title).strip().lower())


def first_sentence(markdown: str) -> str:
    text = re.sub(r"\s+", " ", markdown.strip().split("\n\n")[0])
    return re.split(r"(?<=[.!?]) ", text)[0]


def _parse(source: str) -> ast.AST:
    return ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))))


def fastmob_imports(sources: list[str]) -> dict[str, str]:
    """Local names bound to fastmob objects by any cell of the notebook."""
    imported: dict[str, str] = {}
    for source in sources:
        for node in ast.walk(_parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fastmob"):
                imported.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
            elif isinstance(node, ast.Import):
                imported.update({a.asname or a.name: a.name for a in node.names if a.name.startswith("fastmob")})
    return imported


def fastmob_calls(source: str, imported: dict[str, str]) -> set[str]:
    """Canonical names of fastmob functions called in ``source``."""
    found = set()
    for node in ast.walk(_parse(source)):
        if not isinstance(node, ast.Call):
            continue
        parts, func = [], node.func
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name) and func.id in imported:
            try:
                obj = resolve(".".join([imported[func.id], *reversed(parts)]))
            except (ImportError, AttributeError):
                continue
            if callable(obj):
                found.add(canonical(obj))
    return found


def scan(notebook: Path) -> list[dict]:
    nb = nbformat.read(notebook, as_version=4)
    imported = fastmob_imports([c.source for c in nb.cells if c.cell_type == "code"])
    rows, heading = [], ""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            headings = re.findall(r"^#{2,3} (.+)$", cell.source, re.MULTILINE)
            heading = headings[-1] if headings else heading
        elif cell.cell_type == "code" and cell.get("outputs") and cell.get("execution_count"):
            explanation = next(
                (first_sentence(c.source) for c in nb.cells[i + 1 :] if c.cell_type == "markdown" and c.source.strip()),
                "",
            )
            for symbol in sorted(fastmob_calls(cell.source, imported)):
                rows.append(
                    {
                        "symbol": symbol,
                        "example_path": notebook.relative_to(DOCS).with_suffix(".md").as_posix(),
                        "cell_id": f"cell-{i}",
                        "use_case": heading,
                        "anchor": slug(heading),
                        "output_explanation": explanation,
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--logs", type=Path, help="folder of per-notebook execution logs")
    args = ap.parse_args()
    found = {}
    for notebook in sorted(DOCS.rglob("*.ipynb")):
        status = args.logs / f"{notebook.stem}.status" if args.logs else None
        if status is None or not status.exists() or not status.read_text().startswith("exit=0"):
            continue  # only notebooks with a successful isolated run count
        for row in scan(notebook):
            found.setdefault(row["symbol"], row)
    with args.ledger.open() as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        hit = found.get(r["canonical_symbol"])
        if hit and r["kind"] != "class":
            r.update(
                example_path=hit["example_path"],
                cell_id=hit["cell_id"],
                use_case=hit["use_case"],
                output_explanation=hit["output_explanation"],
                status="covered",
                execution_log=f"{args.logs}/{Path(hit['example_path']).stem}.log" if args.logs else "",
            )
    with args.ledger.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(rows)
    index = {
        sym: {"page": h["example_path"], "anchor": h["anchor"], "title": h["use_case"]} for sym, h in found.items()
    }
    Path("docs/reference_examples.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    fn = [r for r in rows if r["kind"] in ("function", "method")]
    covered = [r for r in fn if r["status"] == "covered"]
    print(f"covered {len(covered)}/{len(fn)} = {len(covered) / len(fn):.1%}")


if __name__ == "__main__":
    main()
