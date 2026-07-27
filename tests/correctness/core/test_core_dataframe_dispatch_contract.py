from __future__ import annotations

from pathlib import Path


def test_core_dataframe_wrappers_do_not_call_to_numpy_directly():
    core_dir = Path(__file__).parents[3] / "fastmob" / "core"
    offenders = []
    for path in sorted(core_dir.glob("*_dataframe.py")):
        if ".to_numpy(" in path.read_text():
            offenders.append(path.name)
    assert offenders == []


def test_hierarchy_dataframe_wrappers_do_not_import_numpy_directly():
    core_dir = Path(__file__).parents[3] / "fastmob" / "core"
    hierarchy_files = ["triplegs_dataframe.py", "trips_dataframe.py", "tours_dataframe.py"]
    offenders = []
    for name in hierarchy_files:
        if "import numpy" in (core_dir / name).read_text():
            offenders.append(name)
    assert offenders == []
