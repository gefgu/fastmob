"""Synthetic-data speed benchmark for PyMove-style POI/event trajectory joins.

Exercises `fastmob.integration.join_with_pois`, `join_with_pois_by_category`,
and `join_with_events` -- capabilities with no fastmob-internal precedent to
benchmark against and whose real-library comparison (PyMove) needs a
dedicated, old-pandas `.venv-pymove` unsuitable for routine benchmark runs
(see pyproject.toml's `dev-pymove` comment). Uses synthetic trajectory/POI/
event data (no external download) at a few sizes to report fastmob's own
scaling.

Usage:
    python benchmarks/integration_poi_event_speed.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_env import detect_cpu_info, get_default_output_dir
from benchmarks.utils import write_json

TRAJ_SIZES = [1_000, 10_000, 100_000]
N_POIS = 500
N_CATEGORIES = 5
N_EVENTS = 500
TIME_WINDOW_S = 900.0


def _build_synthetic_data(n_traj: int, rng):
    import numpy as np
    import pandas as pd

    traj_lat = rng.uniform(40.0, 40.1, size=n_traj)
    traj_lng = rng.uniform(-74.0, -73.9, size=n_traj)
    start = pd.Timestamp("2020-01-01")
    traj_datetime = start + pd.to_timedelta(rng.integers(0, 86400, size=n_traj), unit="s")
    traj_df = pd.DataFrame({"lat": traj_lat, "lng": traj_lng, "datetime": traj_datetime})

    poi_lat = rng.uniform(40.0, 40.1, size=N_POIS)
    poi_lng = rng.uniform(-74.0, -73.9, size=N_POIS)
    poi_category = rng.integers(0, N_CATEGORIES, size=N_POIS)
    pois_df = pd.DataFrame(
        {
            "lat": poi_lat,
            "lng": poi_lng,
            "id": np.arange(N_POIS),
            "name_poi": [f"poi_{i}" for i in range(N_POIS)],
            "type_poi": [f"category_{c}" for c in poi_category],
        }
    )

    event_lat = rng.uniform(40.0, 40.1, size=N_EVENTS)
    event_lng = rng.uniform(-74.0, -73.9, size=N_EVENTS)
    event_datetime = start + pd.to_timedelta(rng.integers(0, 86400, size=N_EVENTS), unit="s")
    events_df = pd.DataFrame(
        {
            "lat": event_lat,
            "lng": event_lng,
            "datetime": event_datetime,
            "event_id": [f"event_{i}" for i in range(N_EVENTS)],
            "event_type": [f"type_{i % 3}" for i in range(N_EVENTS)],
        }
    )
    return traj_df, pois_df, events_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-sizes", type=int, nargs="+", default=TRAJ_SIZES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import numpy as np
    from fastmob.integration import join_with_events, join_with_pois, join_with_pois_by_category

    rng = np.random.default_rng(0)
    per_size_results = []
    for n_traj in args.traj_sizes:
        traj_df, pois_df, events_df = _build_synthetic_data(n_traj, rng)

        t0 = time.perf_counter()
        with_pois = join_with_pois(traj_df, pois_df)
        pois_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        with_pois_cat = join_with_pois_by_category(traj_df, pois_df)
        pois_by_category_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        with_events = join_with_events(traj_df, events_df, time_window_s=TIME_WINDOW_S)
        events_seconds = time.perf_counter() - t0

        result = {
            "n_traj": n_traj,
            "n_pois": N_POIS,
            "n_categories": N_CATEGORIES,
            "n_events": N_EVENTS,
            "join_with_pois_seconds": pois_seconds,
            "join_with_pois_by_category_seconds": pois_by_category_seconds,
            "join_with_events_seconds": events_seconds,
            "matched_event_fraction": float((with_events["dist_event"] != float("inf")).mean()),
        }
        assert len(with_pois) == n_traj
        assert len(with_pois_cat) == n_traj
        per_size_results.append(result)
        print(
            f"  n_traj={n_traj}: "
            f"join_with_pois={pois_seconds:.4f}s "
            f"join_with_pois_by_category={pois_by_category_seconds:.4f}s "
            f"join_with_events={events_seconds:.4f}s"
        )

    payload = {
        "metadata": {
            "benchmark": "integration.join_with_pois+join_with_pois_by_category+join_with_events",
            "dataset": "synthetic_uniform",
            "n_pois": N_POIS,
            "n_categories": N_CATEGORIES,
            "n_events": N_EVENTS,
            "time_window_s": TIME_WINDOW_S,
            "cpu_info": detect_cpu_info(),
        },
        "results": {"by_traj_size": per_size_results},
    }

    output_path = args.output or (get_default_output_dir() / "fastmob_integration_poi_event_speed.json")
    write_json(payload, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
