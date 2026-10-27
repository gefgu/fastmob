"""Keep version strings in the docs in sync with the Cargo.toml release version.

``fastmob-py/Cargo.toml`` is the authoritative version. This rewrites the footer
copyright in ``zensical.toml``, the BibTeX ``version`` and the "Released" line on
the About page. The release date comes from the newest row of the release-notes
table, so a release needs its release-notes row (and ``vX.Y.Z.md`` page) first.

Run automatically by ``scripts/build_notebook_docs.py``; ``tests/docs`` fails if
the files are out of sync.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "src"
RELEASE_ROW = re.compile(r"\| \[v(?P<version>[\d.]+)\]\([^)]*\)<br><small>(?P<date>[\d-]+)</small>")
SUBSTITUTIONS = {
    ROOT / "zensical.toml": (re.compile(r"(Fastkit-Mobility )[\d.]+( &middot;)"), r"\g<1>{version}\2"),
    DOCS / "about" / "index.md": (re.compile(r"(version = \{)[\d.]+(\})"), r"\g<1>{version}\2"),
}
RELEASED_LINE = re.compile(r"(Released )[\d-]+( \(\[release notes\])")


def cargo_version() -> str:
    manifest = tomllib.loads((ROOT / "fastmob-py" / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest["package"]["version"]


def latest_release() -> tuple[str, str]:
    row = RELEASE_ROW.search((DOCS / "release_notes" / "index.md").read_text(encoding="utf-8"))
    if row is None:
        raise SystemExit("release_notes/index.md has no release table row")
    return row["version"], row["date"]


def sync(check: bool = False) -> list[Path]:
    """Rewrite out-of-date files; with ``check`` only report them."""
    version = cargo_version()
    notes_version, date = latest_release()
    if notes_version != version:
        raise SystemExit(f"Cargo version {version} has no newest release-notes row (found v{notes_version}).")
    changed: list[Path] = []
    for path, (pattern, template) in SUBSTITUTIONS.items():
        text = path.read_text(encoding="utf-8")
        new = pattern.sub(template.format(version=version), text)
        if path.name == "index.md":
            new = RELEASED_LINE.sub(rf"\g<1>{date}\2", new)
        if new != text:
            changed.append(path)
            if not check:
                path.write_text(new, encoding="utf-8")
    return changed


if __name__ == "__main__":
    stale = sync(check="--check" in sys.argv)
    for path in stale:
        print(f"{'stale' if '--check' in sys.argv else 'updated'}: {path.relative_to(ROOT)}")
    sys.exit(1 if stale and "--check" in sys.argv else 0)
