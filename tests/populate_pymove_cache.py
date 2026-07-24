"""Populate the PyMove reference cache used by cached comparison tests.

Runs the real ``pymove.utils.integration.join_with_pois`` /
``join_with_pois_by_category`` / ``join_with_events`` once, against the
exact trajectory/POI/event fixtures baked into PyMove's own docstring
examples (see ``pymove/utils/integration.py``'s ``join_with_pois``/
``join_with_events`` docstrings) -- reusing what PyMove itself already uses
to demonstrate/test these functions, rather than inventing new fixtures.

Run this script inside .venv-pymove via the shell wrapper:
    bash scripts/populate_pymove_cache.sh

Or directly (after activating the pymove env):
    .venv-pymove/bin/python tests/populate_pymove_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from pymove import MoveDataFrame
from pymove.utils.integration import join_with_events, join_with_pois, join_with_pois_by_category

from tests.shared.pymove_cache import _REFERENCE_DIR

# --- fixtures transcribed verbatim from pymove/utils/integration.py's
# --- join_with_pois / join_with_pois_by_category docstring examples.
_POI_TRAJ = pd.DataFrame(
    {
        "lat": [39.984094, 39.984559, 40.002899, 40.016238, 40.013814, 40.009735],
        "lon": [116.319236, 116.326696, 116.321520, 116.307691, 116.306525, 116.315069],
        "datetime": [
            "2008-10-23 05:53:05",
            "2008-10-23 10:37:26",
            "2008-10-23 10:50:16",
            "2008-10-23 11:03:06",
            "2008-10-23 11:58:33",
            "2008-10-23 23:50:45",
        ],
        "id": [1, 1, 1, 1, 2, 2],
    }
)
_POIS = pd.DataFrame(
    {
        "lat": [39.984094, 39.991013, 40.010000],
        "lon": [116.319236, 116.326384, 116.312615],
        "id": [1, 2, 3],
        "type_poi": ["policia", "policia", "comercio"],
        "name_poi": ["distrito_pol_1", "policia_federal", "supermercado_aroldo"],
    }
)

# --- fixtures transcribed verbatim from join_with_events's docstring example.
_EVENT_TRAJ = pd.DataFrame(
    {
        "lat": [39.984094, 39.984559, 39.993527, 39.978575, 39.981668],
        "lon": [116.319236, 116.326696, 116.326483, 116.326975, 116.310769],
        "datetime": [
            "2008-10-23 05:53:05",
            "2008-10-23 10:37:26",
            "2008-10-24 00:02:14",
            "2008-10-24 00:22:01",
            "2008-10-24 01:57:57",
        ],
        "id": [1, 1, 2, 3, 3],
    }
)
_EVENTS = pd.DataFrame(
    {
        "lat": [39.984094, 39.991013, 40.010000],
        "lon": [116.319236, 116.326384, 116.312615],
        "id": [1, 2, 3],
        "datetime": pd.to_datetime(["2008-10-23 05:53:05", "2008-10-23 10:37:26", "2008-10-24 01:57:57"]),
        "event_type": ["show", "show", "feira"],
        "event_id": ["forro_tropykalia", "dia_do_municipio", "adocao_de_animais"],
    }
)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def _populate_poi_fixture() -> None:
    out_dir = _REFERENCE_DIR / "doc_example_poi"
    print("    input.parquet / pois.parquet")
    _save(_POI_TRAJ, out_dir / "input.parquet")
    _save(_POIS, out_dir / "pois.parquet")

    move_df = MoveDataFrame(_POI_TRAJ.copy())

    print("    join_with_pois.parquet")
    result = join_with_pois(move_df.copy(), _POIS.copy())
    _save(pd.DataFrame(result), out_dir / "join_with_pois.parquet")

    print("    join_with_pois_by_category.parquet")
    result_cat = join_with_pois_by_category(move_df.copy(), _POIS.copy())
    _save(pd.DataFrame(result_cat), out_dir / "join_with_pois_by_category.parquet")

    print(f"    -> {out_dir}")


def _populate_events_fixture() -> None:
    out_dir = _REFERENCE_DIR / "doc_example_events"
    print("    input.parquet / events.parquet")
    _save(_EVENT_TRAJ, out_dir / "input.parquet")
    _save(_EVENTS, out_dir / "events.parquet")

    move_df = MoveDataFrame(_EVENT_TRAJ.copy())

    print("    join_with_events.parquet")
    result = join_with_events(move_df.copy(), _EVENTS.copy(), time_window=900)
    _save(pd.DataFrame(result), out_dir / "join_with_events.parquet")

    print(f"    -> {out_dir}")


def main() -> None:
    print("\n==> doc_example_poi")
    _populate_poi_fixture()
    print("\n==> doc_example_events")
    _populate_events_fixture()
    print("\nAll done. Commit tests/shared/pymove_reference/ to git to track the snapshots.")


if __name__ == "__main__":
    main()
