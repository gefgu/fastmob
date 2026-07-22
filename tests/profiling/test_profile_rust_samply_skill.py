from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "profile-rust-samply"


def load_script(name: str):
    path = SKILL_DIR / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_firefox_profile() -> dict:
    strings = ["python", "root", "fastmob::radius_of_gyration", "src/radius_of_gyration.rs:rog_for_presorted_slices"]
    return {
        "meta": {"profileName": "tiny"},
        "threads": [
            {
                "name": "python-main",
                "stringTable": strings,
                "frameTable": {
                    "schema": {"location": 0, "line": 1},
                    "data": [[1, None], [2, 10], [3, 42]],
                },
                "stackTable": {
                    "schema": {"prefix": 1, "frame": 0},
                    "data": [[0, None], [1, 0], [2, 1]],
                },
                "samples": {
                    "schema": {"time": 1, "stack": 0, "weight": 2},
                    "data": [[2, 0.0, 2], [2, 1.0, 3], [None, 2.0, 1]],
                },
            }
        ],
    }


def test_reduce_samply_json_handles_schema_names_and_rust_focus(tmp_path):
    reducer = load_script("reduce_samply_json.py")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(tiny_firefox_profile()), encoding="utf-8")

    reduced = reducer.reduce_profile(reducer.load_profile(profile_path), profile_path=profile_path, top=10)
    thread = reduced["threads"][0]

    assert reduced["metadata"]["profile_name"] == "tiny"
    assert thread["sample_count"] == 3
    assert thread["unknown_stack_samples"] == 1
    assert thread["weight_field"] == "weight"
    assert thread["top_leaf_frames"][0] == {
        "name": "src/radius_of_gyration.rs:rog_for_presorted_slices",
        "weight": 5.0,
    }
    rust_inclusive_names = {entry["name"] for entry in thread["rust_focused"]["top_inclusive_frames"]}
    assert "src/radius_of_gyration.rs:rog_for_presorted_slices" in rust_inclusive_names
    assert thread["top_stacks"][0]["stack"] == [
        "root",
        "fastmob::radius_of_gyration",
        "src/radius_of_gyration.rs:rog_for_presorted_slices",
    ]


def test_reduce_samply_json_reads_gzip_like_plain_json(tmp_path):
    reducer = load_script("reduce_samply_json.py")
    plain_path = tmp_path / "profile.json"
    gzip_path = tmp_path / "profile.json.gz"
    payload = tiny_firefox_profile()
    plain_path.write_text(json.dumps(payload), encoding="utf-8")
    with gzip.open(gzip_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    plain = reducer.reduce_profile(reducer.load_profile(plain_path), profile_path=plain_path, top=5)
    compressed = reducer.reduce_profile(reducer.load_profile(gzip_path), profile_path=gzip_path, top=5)

    assert plain["threads"][0]["top_leaf_frames"] == compressed["threads"][0]["top_leaf_frames"]


def test_profile_importable_samply_helpers():
    profiler = load_script("profile_importable_samply.py")

    assert profiler.positive_int("3") == 3
    assert profiler.parse_kwargs('{"ndigits": 2}') == {"ndigits": 2}
    assert profiler.import_target("math:sqrt")(9) == 3
