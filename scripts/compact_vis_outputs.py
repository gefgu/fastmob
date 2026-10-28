"""Shrink the fastmob-vis charts saved in a notebook, in place.

A chart's saved output is an ``<iframe srcdoc>`` page that embeds the whole ECharts library (about
1.2 MB), so a notebook with a dozen charts would be tens of megabytes. This replaces the embedded
library with a ``<script src>`` for the same ECharts version from a CDN, which leaves the chart option
and its formatters in place: the saved chart still draws wherever the notebook is opened with a network
connection, and ``build_notebook_docs.py`` still draws it for the docs from the option it contains.

Run it on a notebook after executing it (outputs are otherwise as the cells produced them):

    uv run python scripts/compact_vis_outputs.py docs/src/learn/visualize-results.ipynb
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ECHARTS_VERSION = "5.6.1"
CDN = f'<script src="https://cdn.jsdelivr.net/npm/echarts@{ECHARTS_VERSION}/dist/echarts.min.js"></script>'
IFRAME = re.compile(r'(<iframe srcdoc=")([^"]*)(")')
LIBRARY = re.compile(r"<script>\s*/\*\s*\* Licensed to the Apache Software Foundation.*?</script>", re.DOTALL)


def compact(page: str) -> str:
    def shrink(found: re.Match) -> str:
        document = html.unescape(found[2])
        document, replaced = LIBRARY.subn(CDN, document, count=1)
        return found[1] + html.escape(document, quote=True) + found[3] if replaced else found[0]

    return IFRAME.sub(shrink, page)


def main(path: Path) -> None:
    notebook = json.loads(path.read_text())
    saved = 0
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "text/html" in data:
                before = "".join(data["text/html"])
                after = compact(before)
                saved += len(before) - len(after)
                data["text/html"] = after.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"{path}: {saved / 1e6:.1f} MB removed")


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        main(Path(argument))
