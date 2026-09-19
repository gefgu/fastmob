"""Convert documentation notebooks to Markdown pages with Binder links."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "src"
BINDER_MARKER = "<!-- Launch this notebook in Binder. -->"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
CLEAR_OUTPUT_FOR = ("%pip install", "!wget", "tdf =", "tdf\n", "tdf.info")


def _binder_link(notebook: Path) -> str:
    repository_path = notebook.relative_to(ROOT).as_posix()
    encoded_path = quote(repository_path, safe="")
    return (
        f"{BINDER_MARKER}\n\n"
        f"[![Launch Binder](https://mybinder.org/badge_logo.svg)]"
        f"(https://mybinder.org/v2/gh/gefgu/fastmob/main?labpath={encoded_path})\n"
    )


def _escape_notebook_brackets(contents: str) -> str:
    """Prevent bracket-heavy output from becoming false links, not source code."""
    links: list[str] = []

    def preserve(match: re.Match[str]) -> str:
        links.append(match.group(0))
        return f"\x00LINK{len(links) - 1}\x00"

    contents = ANSI_ESCAPE.sub("", contents).replace("\r", "")
    contents = MARKDOWN_LINK.sub(preserve, contents)
    escaped_lines: list[str] = []
    in_fenced_code = False
    for line in contents.splitlines(keepends=True):
        if line.lstrip().startswith("Length:") or "geolife.zip         100%" in line:
            continue
        if "saved [" in line or "datetime64[" in line:
            continue
        if "━" in line or "eta 0:00:00" in line:
            continue
        if line.lstrip().startswith("```"):
            in_fenced_code = not in_fenced_code
            escaped_lines.append(line)
        elif in_fenced_code:
            escaped_lines.append(line)
        else:
            escaped_lines.append(line.replace("[", "&#91;").replace("]", "&#93;"))
    contents = "".join(escaped_lines)
    for index, link in enumerate(links):
        contents = contents.replace(f"\x00LINK{index}\x00", link)
    return contents


def _clean_notebook(notebook: Path, destination: Path) -> None:
    document = nbformat.read(notebook, as_version=4)
    for cell in document.cells:
        source = "".join(cell.get("source", []))
        if cell.cell_type == "code" and any(source.startswith(prefix) for prefix in CLEAR_OUTPUT_FOR):
            cell.outputs = []
    nbformat.write(document, destination)


def main() -> None:
    notebooks = sorted(DOCS.rglob("*.ipynb"))
    for notebook in notebooks:
        with tempfile.TemporaryDirectory() as temporary_dir:
            clean_notebook = Path(temporary_dir) / notebook.name
            _clean_notebook(notebook, clean_notebook)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--output-dir",
                    str(notebook.parent),
                    str(clean_notebook),
                ],
                check=True,
                cwd=ROOT,
            )
        markdown = notebook.with_suffix(".md")
        contents = markdown.read_text(encoding="utf-8")
        contents = _escape_notebook_brackets(contents)
        if not contents.startswith(BINDER_MARKER):
            contents = _binder_link(notebook) + "\n" + contents
        markdown.write_text(contents, encoding="utf-8")
        print(f"Generated {markdown.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
