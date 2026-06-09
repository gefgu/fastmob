"""
MkDocs hook: replace [@key] citation markers in rendered HTML with formatted
reference text drawn from docs/references.bib.

This runs on on_page_content (after mkdocstrings has rendered all docstrings),
so it handles citations that appear inside docstring-generated HTML — something
mkdocs-bibtex's on_page_markdown hook cannot do.
"""

import re
import pybtex.database

BIB_FILE = "docs/references.bib"
CITATION_PATTERN = re.compile(r"\[@([\w-]+)\]")

_formatted: dict[str, str] = {}


def _format_entry(key: str, entry: pybtex.database.Entry) -> str:
    f = entry.fields
    persons = entry.persons

    authors = persons.get("author", []) or persons.get("editor", [])
    author_str = ", ".join(
        " ".join(filter(None, p.first_names + p.middle_names + p.last_names))
        for p in authors
    )

    year = f.get("year", "")
    title = f.get("title", "").strip("{}")
    journal = f.get("journal", "")
    volume = f.get("volume", "")
    number = f.get("number", "")
    pages = f.get("pages", "")
    booktitle = f.get("booktitle", "")
    publisher = f.get("publisher", "")
    url = f.get("url", "")

    parts: list[str] = []
    if author_str:
        parts.append(author_str)
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(f"<em>{title}</em>.")
    if journal:
        venue = journal
        if volume:
            venue += f" {volume}"
            if number:
                venue += f"({number})"
        if pages:
            venue += f", {pages.replace('--', '–')}"
        parts.append(venue + ".")
    elif booktitle:
        venue = f"In <em>{booktitle}</em>"
        if pages:
            venue += f", {pages.replace('--', '–')}"
        parts.append(venue + ".")
    elif publisher:
        parts.append(publisher + ".")
    if url:
        parts.append(f'<a href="{url}">{url}</a>.')

    return " ".join(parts)


def on_config(config):
    bib = pybtex.database.parse_file(BIB_FILE)
    _formatted.clear()
    for key, entry in bib.entries.items():
        _formatted[key] = _format_entry(key, entry)


def on_page_content(html, *, page, config, files):
    """Replace [@key] markers in rendered HTML with text from the .bib file."""
    if not _formatted:
        return html

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        return _formatted.get(key, m.group(0))

    return CITATION_PATTERN.sub(_replace, html)
