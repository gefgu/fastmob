"""Fill the example-coverage ledger from the executed Learn notebooks.

A public function counts as covered when an executed code cell with a displayed result calls it
directly (recorded at run time by ``scripts/notebook_trace``, not guessed from the source). Rows record the notebook, cell, the heading that states the use case, and the first
sentence of the markdown cell that explains the output. ``docs/reference_examples.json`` maps
each covered symbol to the notebook section so Reference tables can link to it.

Usage: .venv/bin/python scripts/notebook_coverage.py --ledger AI/docs_audit/runs/learn-reference/lr1/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nbformat

DOCS = Path("docs/src")
# Symbols that are not counted, or not covered, with the evidence for it.
EXCLUDED: dict[str, tuple[str, str]] = {}
UNCOVERED = {
    ".load.load_dataset": (
        "uncovered",
        "the download host (www-public.it-sudparis.eu) did not resolve on 2026-10-27, so no example can run",
    ),
}


def slug(title: str) -> str:
    return re.sub(r"[-\s]+", "-", re.sub(r"[^\w\s-]", "", title).strip().lower())


def first_sentence(markdown: str) -> str:
    text = re.sub(r"\s+", " ", markdown.strip().split("\n\n")[0])
    return re.split(r"(?<=[.!?]) ", text)[0]


def scan(notebook: Path, calls: list[tuple[str, int | None]]) -> list[dict]:
    """Calls made directly by executed cells that display a result, from the recorded call log."""
    nb = nbformat.read(notebook, as_version=4)
    by_count = {
        c.execution_count: (i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.execution_count
    }
    heading_at, heading = {}, ""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            found = re.findall(r"^#{2,3} (.+)$", cell.source, re.MULTILINE)
            heading = found[-1] if found else heading
        heading_at[i] = heading
    rows = []
    for symbol, count in calls:
        if count not in by_count:
            continue
        i, cell = by_count[count]
        if not cell.get("outputs") or any(o.get("output_type") == "error" for o in cell.outputs):
            continue
        explanation = next(
            (first_sentence(c.source) for c in nb.cells[i + 1 :] if c.cell_type == "markdown" and c.source.strip()), ""
        )
        rows.append(
            {
                "symbol": symbol,
                "example_path": notebook.relative_to(DOCS).with_suffix(".md").as_posix(),
                "cell_id": f"cell-{i}",
                "use_case": heading_at[i],
                "anchor": slug(heading_at[i]),
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
        calls_file = args.logs / f"{notebook.stem}.calls.json"
        if not calls_file.exists():
            continue  # no call log: the notebook was not run with the tracer
        for row in scan(notebook, [tuple(c) for c in json.loads(calls_file.read_text())]):
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
    for r in rows:
        for suffix, (status, reason) in {**EXCLUDED, **UNCOVERED}.items():
            if r["canonical_symbol"].endswith(suffix):
                r.update(status=status, exclusion_reason=reason)
    with args.ledger.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(rows)
    index = {
        sym: {"page": h["example_path"], "anchor": h["anchor"], "title": h["use_case"]} for sym, h in found.items()
    }
    Path("docs/reference_examples.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    fn = [r for r in rows if r["kind"] in ("function", "method") and r["status"] != "excluded"]
    covered = [r for r in fn if r["status"] == "covered"]
    print(f"covered {len(covered)}/{len(fn)} = {len(covered) / len(fn):.1%}")


if __name__ == "__main__":
    main()
