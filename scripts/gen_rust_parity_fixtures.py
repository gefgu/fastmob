#!/usr/bin/env python
"""Generate fastmob-rs parity fixtures from the fastmob Python path.

The Python implementation is the reference: whatever it produces is what
`fastmob-rs` must reproduce bit-for-bit. This writes small, self-contained
fixtures (the Brightkite cache itself is downloaded, not checked in) that
`fastmob-rs/tests/parity.rs` asserts against.

    python scripts/gen_rust_parity_fixtures.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "tests/shared/data/loc-brightkite_totalCheckins.txt.gz"
FIXTURES = REPO_ROOT / "fastmob-rs/tests/fixtures"

ROWS = 2000
# Brightkite is stored grouped by user, so a contiguous prefix of 2000 rows is
# a single user. Sampling every Nth row instead spans hundreds of users while
# keeping the checked-in fixture small.
STRIDE = 100


def read_rows(rows: int) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    with gzip.open(DATASET, "rt") as handle:
        for index, line in enumerate(handle):
            if len(out) >= rows:
                break
            if index % STRIDE:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                int(fields[0])
                lat = float(fields[2])
                lng = float(fields[3])
            except ValueError:
                continue
            if not (np.isfinite(lat) and np.isfinite(lng)):
                continue
            out.append((fields[0], fields[1], fields[2], fields[3]))
    return out


def build_case(name: str, records: list[tuple[str, str, str, str]], *, string_uids: bool) -> None:
    """Write one fixture: the input TSV plus the Python path's expected output."""
    from fastmob.measures.individual import jump_lengths, radius_of_gyration, waiting_times

    case_dir = FIXTURES / name
    case_dir.mkdir(parents=True, exist_ok=True)

    if string_uids:
        # Composite ids like "10_2980" are the shape that silently collapsed
        # every user into one group when parsed as an integer.
        records = [(f"{uid}_{i % 7}", dt, lat, lng) for i, (uid, dt, lat, lng) in enumerate(records)]

    with (case_dir / "input.tsv").open("w") as handle:
        handle.write("uid\tdatetime\tlat\tlng\n")
        for uid, dt, lat, lng in records:
            handle.write(f"{uid}\t{dt}\t{lat}\t{lng}\n")

    df = pl.DataFrame(
        {
            "uid": [r[0] for r in records] if string_uids else [int(r[0]) for r in records],
            "datetime": [r[1] for r in records],
            "lat": [float(r[2]) for r in records],
            "lng": [float(r[3]) for r in records],
        }
    ).with_columns(
        pl.col("datetime").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us", strict=False)
    ).drop_nulls("datetime")

    arranged = df.sort(["uid", "datetime"], maintain_order=True)
    jumps = np.asarray(jump_lengths(arranged, merge=True, presorted=True), dtype="<f8")
    rog = (
        radius_of_gyration(arranged, presorted=True)["radius_of_gyration"]
        .drop_nulls()
        .to_numpy()
        .astype("<f8")
    )

    waits = np.asarray(waiting_times(arranged, merge=True, presorted=True), dtype="<f8")

    jumps.tofile(case_dir / "jump_lengths.f64")
    rog.tofile(case_dir / "radius_of_gyration.f64")
    waits.tofile(case_dir / "waiting_times.f64")
    print(
        f"{name}: {len(arranged)} rows, {len(jumps)} jumps, "
        f"{len(rog)} radii, {len(waits)} waits"
    )


def main() -> int:
    records = read_rows(ROWS)
    build_case("brightkite_integer_uid", records, string_uids=False)
    build_case("brightkite_string_uid", records, string_uids=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
