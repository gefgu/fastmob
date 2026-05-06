# %%
"""
Brightkite radius-of-gyration benchmark.

This script is organized as editor cells (``# %%``) so you can open it
in VS Code or Jupyter as a multi-cell workflow. It expects the Brightkite
CSV to be at `tests/shared/data/brightkite.csv`.

Cell 1: load dataset into Polars (full ~4M rows)
Cell 2: run the `skmob2` benchmark
Cell 3: run the original `skmob` benchmark (pandas conversion)
"""

# %%
import time
from pathlib import Path
import polars as pl

DATA_PATH = Path("tests/shared/data/brightkite.csv")

if not DATA_PATH.exists():
    raise SystemExit(
        f"Dataset not found at {DATA_PATH}. Download or place the brightkite.csv file there."
    )

print("Reading CSV into Polars (this may take a while)...")
df = pl.read_csv(str(DATA_PATH))
print(f"Loaded dataframe with {len(df)} rows and columns: {list(df.columns)[:10]}")

# %%
# skmob2 benchmark
from skmob2.measures.spatial.radius_of_gyration import radius_of_gyration as rog_skmob2  # noqa: E402

print("\nWarming up skmob2...")
_ = rog_skmob2(df)

times = []
for i in range(5):
    time.sleep(0.5)
    start = time.perf_counter()
    _ = rog_skmob2(df)
    end = time.perf_counter()
    duration = end - start
    times.append(duration)
    print(f"skmob2 Round {i+1}: {duration:.4f} seconds")

print(f"skmob2 Average: {sum(times)/len(times):.4f} s")
print(f"skmob2 Minimum: {min(times):.4f} s")

# %%
# original scikit-mobility benchmark (pandas conversion)
print("\nConverting to pandas for scikit-mobility (may use a lot of memory)...")
pdf = df.to_pandas()

try:
    from skmob.measures.individual import radius_of_gyration as rog_skmob
except Exception:  # pragma: no cover - runtime import issues depend on environment
    raise SystemExit(
        "Unable to import scikit-mobility. Install scikit-mobility in your environment to run the comparison: pip install scikit-mobility"
    )

print("Warming up scikit-mobility...")
_ = rog_skmob(pdf)

times = []
for i in range(5):
    time.sleep(0.5)
    start = time.perf_counter()
    _ = rog_skmob(pdf)
    end = time.perf_counter()
    duration = end - start
    times.append(duration)
    print(f"scikit-mobility Round {i+1}: {duration:.4f} seconds")

print(f"scikit-mobility Average: {sum(times)/len(times):.4f} s")
print(f"scikit-mobility Minimum: {min(times):.4f} s")
