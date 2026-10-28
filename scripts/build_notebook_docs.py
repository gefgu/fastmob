"""Convert documentation notebooks to Markdown pages with Colab and Binder links."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import nbformat

import sync_docs_version

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "src"
BINDER_MARKER = "<!-- Launch this notebook in Binder. -->"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
CLEAR_OUTPUT_FOR = ("%pip install", "!wget", "tdf =", "tdf\n", "tdf.info")


def _launch_links(notebook: Path) -> str:
    repository_path = notebook.relative_to(ROOT).as_posix()
    encoded_path = quote(repository_path, safe="")
    return (
        f"{BINDER_MARKER}\n\n"
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/gefgu/fastmob/blob/main/{quote(repository_path)}) "
        f"[![Launch Binder](https://mybinder.org/badge_logo.svg)]"
        f"(https://mybinder.org/v2/gh/gefgu/fastmob/main?labpath={encoded_path})\n"
    )


def _escape_line(line: str) -> str:
    """Escape brackets in prose and HTML, but not inside inline code spans."""
    if line.lstrip().startswith("<"):
        return line.replace("[", "&#91;").replace("]", "&#93;")
    parts = re.split(r"(`[^`\n]*`)", line)
    return "".join(
        part
        if part.startswith("`") and part.endswith("`") and len(part) > 1
        else part.replace("[", "&#91;").replace("]", "&#93;")
        for part in parts
    )


def _fence_text_outputs(contents: str) -> str:
    """Turn nbconvert's indented text outputs (after a code fence) into fenced blocks."""
    out: list[str] = []
    lines = contents.split("\n")
    in_fence = False
    after_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            after_fence = not in_fence
            out.append(line)
            i += 1
            continue
        indented_text = bool(line.strip()) and line.startswith("    ") and not line.lstrip().startswith("<")
        if after_fence and not in_fence and indented_text:
            block = []
            while i < len(lines) and (
                (lines[i].startswith("    ") and not lines[i].lstrip().startswith("<")) or lines[i].strip() == ""
            ):
                block.append(lines[i])
                i += 1
            while block and block[-1].strip() == "":
                i -= 1
                block.pop()
            out.append("```text")
            out.extend(b[4:] if b.startswith("    ") else b for b in block)
            out.append("```")
            continue
        if line.strip() and not in_fence:
            after_fence = False if not indented_text else after_fence
        out.append(line)
        i += 1
    return "\n".join(out)


def _escape_notebook_brackets(contents: str) -> str:
    """Prevent bracket-heavy output from becoming false links, not source code."""
    links: list[str] = []

    def preserve(match: re.Match[str]) -> str:
        links.append(match.group(0))
        return f"\x00LINK{len(links) - 1}\x00"

    contents = _fence_text_outputs(ANSI_ESCAPE.sub("", contents).replace("\r", ""))
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
            escaped_lines.append(_escape_line(line))
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
    sync_docs_version.sync()
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
            contents = _launch_links(notebook) + "\n" + contents
        markdown.write_text(contents, encoding="utf-8")
        print(f"Generated {markdown.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
