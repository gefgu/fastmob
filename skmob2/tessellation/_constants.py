"""Constants used by tessellation tilers."""

from __future__ import annotations

try:
    from pyproj import CRS
except ImportError as exc:  # pragma: no cover - exercised when optional deps are missing.
    raise ImportError("pyproj is required for tessellation: pip install skmob2[tessellation]") from exc

UNIVERSAL_CRS = CRS.from_epsg(3857)
DEFAULT_CRS = CRS.from_epsg(4326)
TILE_ID = "tile_id"
LEGACY_TILE_ID = "tile_ID"

# https://github.com/uber/h3/blob/master/docs/core-library/restable.md
H3_UTILS = {
    "average_hexagon_area": {
        "0": 4250546.8477,
        "1": 607220.9782429,
        "2": 86745.8540347,
        "3": 12392.2648621,
        "4": 1770.3235517,
        "5": 252.9033645,
        "6": 36.1290521,
        "7": 5.1612932,
        "8": 0.7373276,
        "9": 0.1053325,
        "10": 0.0150475,
        "11": 0.0021496,
        "12": 0.0003071,
        "13": 0.0000439,
        "14": 0.0000063,
        "15": 9e-7,
    },
    "average_hexagon_edge_length": {
        "0": 1107.712591,
        "1": 418.6760055,
        "2": 158.2446558,
        "3": 59.81085794,
        "4": 22.6063794,
        "5": 8.544408276,
        "6": 3.229482772,
        "7": 1.220629759,
        "8": 0.461354684,
        "9": 0.174375668,
        "10": 0.065907807,
        "11": 0.024910561,
        "12": 0.009415526,
        "13": 0.003559893,
        "14": 0.001348575,
        "15": 0.000509713,
    },
    "num_uniq_idxs": {
        "0": 122,
        "1": 842,
        "2": 5882,
        "3": 41162,
        "4": 288122,
        "5": 2016842,
        "6": 14117882,
        "7": 98825162,
        "8": 691776122,
        "9": 4842432842,
        "10": 33897029882,
        "11": 237279209162,
        "12": 1660954464122,
        "13": 11626681248842,
        "14": 81386768741882,
        "15": 569707381193162,
    },
}
