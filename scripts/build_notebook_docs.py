"""Convert documentation notebooks to Markdown pages with Binder links."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "src"
BINDER_MARKER = "<!-- Launch this notebook in Binder. -->"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def _binder_link(notebook: Path) -> str:
    repository_path = notebook.relative_to(ROOT).as_posix()
    encoded_path = quote(repository_path, safe="")
    return (
        f"{BINDER_MARKER}\n\n"
        f"[![Launch Binder](https://mybinder.org/badge_logo.svg)]"
        f"(https://mybinder.org/v2/gh/gefgu/fastmob/main?labpath={encoded_path})\n"
    )


def _escape_notebook_brackets(contents: str) -> str:
    """Prevent bracket-heavy notebook output from becoming false links."""
    links: list[str] = []

    def preserve(match: re.Match[str]) -> str:
        links.append(match.group(0))
        return f"\x00LINK{len(links) - 1}\x00"

    contents = MARKDOWN_LINK.sub(preserve, contents)
    contents = contents.replace("[", "&#91;").replace("]", "&#93;")
    for index, link in enumerate(links):
        contents = contents.replace(f"\x00LINK{index}\x00", link)
    return contents


def main() -> None:
    notebooks = sorted(DOCS.rglob("*.ipynb"))
    for notebook in notebooks:
        subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "markdown",
                "--output-dir", str(notebook.parent), str(notebook),
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
