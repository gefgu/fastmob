"""Fast checks for published-documentation regressions.

The site builder validates rendering; these tests make broken local links and
placeholder prose visible during normal test runs as well.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs" / "src"
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
DATA_STRUCTURE_PAGES = {
    "traj-dataframe.md",
    "positionfixes.md",
    "staypoints.md",
    "locations.md",
    "triplegs.md",
    "trips.md",
    "tours.md",
    "flow-dataframe.md",
}


def _local_target(page: Path, target: str) -> Path | None:
    target = target.strip().split(maxsplit=1)[0].split("#", 1)[0]
    # Notebook code/output can contain tuple and array indexing such as ``[1]``
    # that resembles a Markdown link to the lightweight regex above.
    if not target or target.isdigit() or "://" in target or target.startswith(("mailto:", "#")):
        return None
    path = (page.parent / target).resolve()
    if path.is_dir():
        return path / "index.md"
    if not path.suffix:
        return path.with_suffix(".md")
    return path


def test_published_markdown_has_no_todo_placeholders():
    pages = list(DOCS.rglob("*.md"))
    offenders = [page.relative_to(ROOT) for page in pages if "TODO" in page.read_text(encoding="utf-8")]
    assert not offenders, f"Replace TODO prose in published docs: {offenders}"


def test_published_markdown_local_links_resolve():
    missing: list[str] = []
    for page in DOCS.rglob("*.md"):
        for target in LINK.findall(page.read_text(encoding="utf-8")):
            local = _local_target(page, target)
            if local is not None and not local.exists():
                missing.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local documentation links:\n" + "\n".join(missing)


def test_data_structures_landing_page_links_every_structure_page():
    """Keep the overview and the Data Structures navigation in sync."""
    landing = DOCS / "reference" / "data-structures.md"
    linked = {
        Path(target.split("#", 1)[0]).name
        for target in LINK.findall(landing.read_text(encoding="utf-8"))
        if target.startswith("data-structures/")
    }
    assert DATA_STRUCTURE_PAGES <= linked
    assert {page.name for page in (DOCS / "reference" / "data-structures").glob("*.md")} == DATA_STRUCTURE_PAGES


def test_data_structure_pages_document_defining_classes():
    """Re-exports lose method docstrings; render the class source instead."""
    expected = {
        "traj-dataframe.md": "fastmob.core.trajectory_dataframe.TrajDataFrame",
        "positionfixes.md": "fastmob.core.positionfixes_dataframe.Positionfixes",
        "staypoints.md": "fastmob.core.staypoints_dataframe.Staypoints",
        "locations.md": "fastmob.core.locations_dataframe.Locations",
        "triplegs.md": "fastmob.core.triplegs_dataframe.Triplegs",
        "trips.md": "fastmob.core.trips_dataframe.Trips",
        "tours.md": "fastmob.core.tours_dataframe.Tours",
        "flow-dataframe.md": "fastmob.core.flow_dataframe.FlowDataFrame",
    }
    for page, symbol in expected.items():
        assert f"::: {symbol}" in (DOCS / "reference" / "data-structures" / page).read_text(encoding="utf-8")


def test_measure_packages_use_explicit_api_directives_not___all__():
    """Package-wide directives expose ``__all__`` and hide re-exported APIs."""
    individual = (DOCS / "reference" / "measures" / "individual.md").read_text(encoding="utf-8")
    fitting = (DOCS / "reference" / "fitting.md").read_text(encoding="utf-8")

    assert "::: fastmob.measures.individual\n" not in individual
    assert "::: fastmob.measures.individual.jump_lengths" in individual
    assert "::: fastmob.measures.fitting\n" not in fitting
    assert "::: fastmob.measures.fitting.fit_visitation_law" in fitting
