#!/usr/bin/env python
"""Compare fastmob-rs against the fastmob Python path on Brightkite data.

Runs both implementations over the same rows and reports (a) wall time and
(b) whether the values agree bit-for-bit. The Rust side is driven through the
``brightkite_jumps_rog`` example, which writes its results as raw little-endian
f64 dumps for exact comparison.

    python scripts/bench_fastmob_rs.py --rows 100000 1000000 4000000
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "tests/shared/data/loc-brightkite_totalCheckins.txt.gz"


def load_brightkite(rows: int) -> pl.DataFrame:
    """Read the first ``rows`` usable check-ins.

    Mirrors ``fastmob-rs/benches/support/brightkite.rs`` line for line so both
    implementations see byte-identical input.
    """
    uids: list[int] = []
    datetimes: list[str] = []
    lats: list[float] = []
    lngs: list[float] = []

    with gzip.open(DATASET, "rt") as handle:
        for line in handle:
            if len(uids) >= rows:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            try:
                uid = int(fields[0])
                lat = float(fields[2])
                lng = float(fields[3])
            except ValueError:
                continue
            if not (np.isfinite(lat) and np.isfinite(lng)):
                continue
            uids.append(uid)
            datetimes.append(fields[1])
            lats.append(lat)
            lngs.append(lng)

    df = pl.DataFrame({"uid": uids, "datetime": datetimes, "lat": lats, "lng": lngs})
    # Explicit format: Brightkite stamps are `2010-10-16T06:02:04Z`, and modern
    # polars refuses to infer a format when the data carries a zone. Parsing to
    # a naive microsecond datetime matches what the Rust side produces.
    return df.with_columns(
        pl.col("datetime").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us", strict=False)
    ).drop_nulls("datetime")


def run_python(df: pl.DataFrame) -> dict:
    """Time the fastmob Python path over an already-arranged frame."""
    from fastmob.measures.individual import jump_lengths, radius_of_gyration

    started = time.perf_counter()
    arranged = df.sort(["uid", "datetime"], maintain_order=True)
    prepare_secs = time.perf_counter() - started

    started = time.perf_counter()
    jumps = np.asarray(jump_lengths(arranged, merge=True, presorted=True))
    jumps = jumps[jumps > 0.0]
    jump_secs = time.perf_counter() - started

    started = time.perf_counter()
    rog_frame = radius_of_gyration(arranged, presorted=True)
    rog = rog_frame["radius_of_gyration"].drop_nulls().to_numpy()
    rog_secs = time.perf_counter() - started

    return {
        "prepare_secs": prepare_secs,
        "jump_lengths_secs": jump_secs,
        "radius_of_gyration_secs": rog_secs,
        "total_secs": prepare_secs + jump_secs + rog_secs,
        "jumps": jumps,
        "rog": rog,
    }


def run_rust(rows: int, out_dir: Path) -> dict:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "-p",
            "fastmob-rs",
            "--example",
            "brightkite_jumps_rog",
            "--",
            str(rows),
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["jumps"] = np.fromfile(out_dir / "jumps.f64", dtype="<f8")
    result["rog"] = np.fromfile(out_dir / "rog.f64", dtype="<f8")
    return result


def compare(name: str, rust: np.ndarray, python: np.ndarray) -> str:
    """Report exact agreement, falling back to a description of the mismatch."""
    if rust.shape != python.shape:
        return f"{name}: SHAPE MISMATCH rust={rust.shape[0]} python={python.shape[0]}"
    if np.array_equal(rust, python):
        return f"{name}: exact match ({rust.shape[0]} values)"

    # Order can legitimately differ when two rows tie on (uid, datetime); a
    # sorted comparison distinguishes that from a real numeric divergence.
    if np.array_equal(np.sort(rust), np.sort(python)):
        return f"{name}: same multiset, different order ({rust.shape[0]} values)"
    diff = np.abs(np.sort(rust) - np.sort(python))
    return f"{name}: MISMATCH max_abs_diff={diff.max():.3e} n_differing={(diff > 0).sum()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 1_000_000, 4_000_000])
    args = parser.parse_args()

    table: list[tuple] = []
    checks: list[str] = []
    for rows in args.rows:
        df = load_brightkite(rows)
        with tempfile.TemporaryDirectory() as tmp:
            rust = run_rust(rows, Path(tmp))
            python = run_python(df)

            checks.append(f"[{rows:,} rows] " + compare("jump_lengths", rust["jumps"], python["jumps"]))
            checks.append(f"[{rows:,} rows] " + compare("radius_of_gyration", rust["rog"], python["rog"]))
            table.append(
                (
                    rows,
                    rust["prepare_secs"] * 1e3,
                    python["prepare_secs"] * 1e3,
                    rust["total_secs"] * 1e3,
                    python["total_secs"] * 1e3,
                    python["total_secs"] / rust["total_secs"],
                )
            )

    header = f"| {'rows':>9} | {'rs prep':>9} | {'py prep':>9} | {'rs total':>9} | {'py total':>9} | {'speedup':>8} |"
    print()
    print(header)
    print("|" + "|".join(["-" * (len(part) + 2) for part in header.split("|")[1:-1]]) + "|")
    for rows, rs_prep, py_prep, rs_total, py_total, speedup in table:
        print(
            f"| {rows:>9,} | {rs_prep:>8.1f}ms | {py_prep:>8.1f}ms | "
            f"{rs_total:>8.1f}ms | {py_total:>8.1f}ms | {speedup:>7.2f}x |"
        )
    print()
    for check in checks:
        print(check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
