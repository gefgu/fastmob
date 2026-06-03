"""DensityEPR workload for external profilers.

Examples:
    uv run scalene run --memory --profile-only skmob2 \
      -o .profiles/scalene/density_epr_500a_100000l_1w.json \
      benchmarks/profile_density_epr_workload.py

    samply record --save-only --rate 1000 \
      -o .profiles/samply/density_epr_500a_100000l_1w.json.gz -- \
      uv run python benchmarks/profile_density_epr_workload.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from skmob2.models import DensityEPR


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "tests" / "shared" / "skmob_reference" / "models"
DEFAULT_START = "2020-01-01 08:00:00"
DEFAULT_N_LOCATIONS = 10_000


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DensityEPR profiling workload.")
    parser.add_argument("--n-agents", type=positive_int, default=500)
    parser.add_argument("--n-locations", type=positive_int, default=DEFAULT_N_LOCATIONS)
    parser.add_argument("--days", type=positive_int, default=7)
    parser.add_argument("--random-state", type=int, default=2)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    return parser.parse_args()


def expand_tessellation(tessellation: pd.DataFrame, size: int) -> pd.DataFrame:
    base = tessellation.reset_index(drop=True)
    if size <= len(base):
        return base.head(size).copy()

    frames = []
    repeats = (size + len(base) - 1) // len(base)
    for repeat in range(repeats):
        frame = base.copy()
        frame["tile_id"] = frame["tile_id"].astype(str) + f"_{repeat}"
        frame["lat"] = frame["lat"].astype(float) + repeat * 0.01
        frame["lng"] = frame["lng"].astype(float) + repeat * 0.01
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).head(size)


def main() -> None:
    args = parse_args()
    tessellation_path = args.reference_dir / "input.parquet"
    if not tessellation_path.exists():
        raise SystemExit(
            f"Model benchmark input not found at {tessellation_path}. "
            "Run 'bash scripts/populate_skmob_cache.sh --datasets models' first."
        )

    tessellation = expand_tessellation(pd.read_parquet(tessellation_path), args.n_locations)
    start = pd.Timestamp(args.start)
    end = start + pd.Timedelta(days=args.days)
    starting_locations = [i % len(tessellation) for i in range(args.n_agents)]

    DensityEPR().generate(
        start,
        end,
        tessellation,
        n_agents=args.n_agents,
        starting_locations=starting_locations,
        relevance_column="population",
        random_state=args.random_state,
        show_progress=False,
    )


if __name__ == "__main__":
    main()
