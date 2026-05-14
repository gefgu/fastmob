#!/usr/bin/env bash
# Generate marketing comparison plots for available skmob2 speed benchmarks.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

RESULTS_DIR="$REPO_ROOT/tests/benchmarks/results"
OUTPUT_DIR="$RESULTS_DIR/plots"
PLOT_SCRIPT="$REPO_ROOT/tests/benchmarks/plot_benchmark_comparisons.py"
FAILURES=()

run_comparison() {
    local suite="$1"
    local backend="$2"
    shift 2
    local original_json="$RESULTS_DIR/skmob_${suite}_speed_prebuilt_tdf.json"
    local optimized_json="$RESULTS_DIR/skmob2_${suite}_speed_${backend}.json"

    if [ ! -f "$original_json" ]; then
        echo "Skipping ${suite}/${backend}: missing $(basename "$original_json")"
        return 0
    fi
    if [ ! -f "$optimized_json" ]; then
        echo "Skipping ${suite}/${backend}: missing $(basename "$optimized_json")"
        return 0
    fi

    echo
    echo "==> Plotting ${suite}/${backend}"
    if "$PYTHON" "$PLOT_SCRIPT" \
        --original-json "$original_json" \
        --optimized-json "$optimized_json" \
        --suite "$suite" \
        --backend "$backend" \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
        echo "==> ${suite}/${backend}: ok"
    else
        local status=$?
        echo "==> ${suite}/${backend}: failed with exit code ${status}"
        FAILURES+=("${suite}/${backend} (${status})")
    fi
}

run_model_comparison() {
    local original_json="$RESULTS_DIR/skmob_models_speed.json"
    local optimized_json="$RESULTS_DIR/skmob2_models_speed.json"

    if [ ! -f "$original_json" ]; then
        echo "Skipping models: missing $(basename "$original_json")"
        return 0
    fi
    if [ ! -f "$optimized_json" ]; then
        echo "Skipping models: missing $(basename "$optimized_json")"
        return 0
    fi

    echo
    echo "==> Plotting models"
    if "$PYTHON" "$PLOT_SCRIPT" \
        --original-json "$original_json" \
        --optimized-json "$optimized_json" \
        --suite models \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
        echo "==> models: ok"
    else
        local status=$?
        echo "==> models: failed with exit code ${status}"
        FAILURES+=("models (${status})")
    fi
}

for suite in spatial privacy visits; do
    for backend in pandas polars; do
        run_comparison "$suite" "$backend" "$@"
    done
done

run_model_comparison "$@"

echo
echo "==> Plot generation complete. Images are in ${OUTPUT_DIR#$REPO_ROOT/}"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "==> Failures:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
