#!/usr/bin/env bash
# Re-run fkmob memory benchmarks with memray (1 iteration each) and regenerate plots.
# Speed benchmarks are skipped if their JSON already exists.
# skmob baseline data is left untouched; comparison plots are generated where available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"
PYTHON="$REPO_ROOT/.venv/bin/python"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "=== Building fkmob._core ==="
unset CONDA_PREFIX 2>/dev/null || true
maturin develop -q

# ── Results dir ───────────────────────────────────────────────────────────────
RESULTS_DIR=$(
    "$PYTHON" -c \
    "import sys; sys.path.insert(0, 'benchmarks'); from benchmark_env import get_default_output_dir; print(get_default_output_dir())"
)
mkdir -p "$RESULTS_DIR"
echo "=== Results dir: ${RESULTS_DIR#$REPO_ROOT/} ==="

# ── Phase 1: fkmob memory benchmarks (memray, 1 iteration, raw order) ───────
# evaluation is excluded: its memory image is commented out in benchmarks.md
# models are excluded: handled by a separate session
SUITES=(individual collective preprocessing privacy)

echo ""
echo "=== Phase 1: fkmob memory benchmarks (memray, 1 iteration) ==="
for SUITE in "${SUITES[@]}"; do
    for BACKEND in pandas polars; do
        echo ""
        echo "--- $SUITE / $BACKEND / memory ---"
        "$PYTHON" "benchmarks/$SUITE/speed_suite.py" \
            --library fkmob \
            --backend "$BACKEND" \
            --profile memory \
            --input-order raw \
            --iterations 1 \
            --sleep 0 \
            --output-dir "$RESULTS_DIR"
    done
done

# ── Phase 2: Plot generation ──────────────────────────────────────────────────
# Passes --input-order both so sorted speed plots are regenerated alongside raw.
# Memory plots are always raw (no sorted memory images referenced in benchmarks.md).
# skmob comparison JSON files are used automatically when present; otherwise
# plot_all_comparisons.sh falls back to standalone fkmob-only plots.
echo ""
echo "=== Phase 2: Generating plots ==="
bash "$SCRIPT_DIR/plot_all_comparisons.sh" \
    --env-dir "$RESULTS_DIR" \
    --input-order both

# ── Phase 3: Copy plots to docs/src/assets/benchmarks/ ───────────────────────
echo ""
echo "=== Phase 3: Copying plots to docs/src/assets/benchmarks/ ==="
ASSETS_DIR="$REPO_ROOT/docs/src/assets/benchmarks"
mkdir -p "$ASSETS_DIR/sorted"

# Copy raw plots (all PNGs at the top level of plots/)
find "$RESULTS_DIR/plots" -maxdepth 1 -name "*.png" -exec cp -f {} "$ASSETS_DIR/" \;

# Copy sorted plots
if [ -d "$RESULTS_DIR/plots/sorted" ]; then
    find "$RESULTS_DIR/plots/sorted" -maxdepth 1 -name "*.png" -exec cp -f {} "$ASSETS_DIR/sorted/" \;
fi

echo ""
RAW_COUNT=$(find "$ASSETS_DIR" -maxdepth 1 -name "*.png" | wc -l)
SORTED_COUNT=$(find "$ASSETS_DIR/sorted" -maxdepth 1 -name "*.png" | wc -l)
echo "=== Done! $RAW_COUNT PNG(s) in assets/, $SORTED_COUNT in assets/sorted/ ==="
