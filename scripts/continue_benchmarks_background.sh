#!/usr/bin/env bash
# Continue the long Python 3.12 benchmark run in the background.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-$REPO_ROOT/.venv-py312/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/results/py312_intel_core_i7_10700_16c}"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastmob-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

DRIVER_LOG="$LOG_DIR/background_continue_driver.log"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$DRIVER_LOG"
}

run_if_missing() {
    local label="$1"
    local expected="$2"
    shift 2

    if [ -f "$expected" ]; then
        log "skip $label: $(basename "$expected") already exists"
        return 0
    fi

    log "start $label"
    "$@" > "$LOG_DIR/${label}.log" 2>&1
    local status=$?
    printf '%s\n' "$status" > "$LOG_DIR/${label}.status"
    log "finish $label: exit $status"
    return 0
}

run_always() {
    local label="$1"
    shift

    log "start $label"
    "$@" > "$LOG_DIR/${label}.log" 2>&1
    local status=$?
    printf '%s\n' "$status" > "$LOG_DIR/${label}.status"
    log "finish $label: exit $status"
    return 0
}

if [ ! -x "$PYTHON" ]; then
    log "ERROR: Python executable not found: $PYTHON"
    exit 2
fi

log "background continuation started"
log "python: $PYTHON"
log "results: $RESULTS_DIR"

run_if_missing \
    fastmob_preprocessing_memory_pandas \
    "$RESULTS_DIR/fastmob_preprocessing_memory_pandas.json" \
    "$PYTHON" -u benchmarks/preprocessing/speed_suite.py \
        --library fastmob --backend pandas --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_preprocessing_memory_polars \
    "$RESULTS_DIR/fastmob_preprocessing_memory_polars.json" \
    "$PYTHON" -u benchmarks/preprocessing/speed_suite.py \
        --library fastmob --backend polars --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_evaluation_memory_pandas \
    "$RESULTS_DIR/fastmob_evaluation_memory_pandas.json" \
    "$PYTHON" -u benchmarks/evaluation/speed_suite.py \
        --backend pandas --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_evaluation_memory_polars \
    "$RESULTS_DIR/fastmob_evaluation_memory_polars.json" \
    "$PYTHON" -u benchmarks/evaluation/speed_suite.py \
        --backend polars --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_models_memory \
    "$RESULTS_DIR/fastmob_models_memory.json" \
    "$PYTHON" -u benchmarks/models/speed_suite.py \
        --library fastmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_models_large_scale_memory \
    "$RESULTS_DIR/fastmob_models_memory_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py \
        --library fastmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_models_location_large_scale_memory \
    "$RESULTS_DIR/fastmob_models_memory_location_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py \
        --library fastmob --mode location --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_privacy_memory_pandas \
    "$RESULTS_DIR/fastmob_privacy_memory_pandas.json" \
    "$PYTHON" -u benchmarks/privacy/speed_suite.py \
        --library fastmob --backend pandas --profile memory --output-dir "$RESULTS_DIR"

run_if_missing \
    fastmob_privacy_memory_polars \
    "$RESULTS_DIR/fastmob_privacy_memory_polars.json" \
    "$PYTHON" -u benchmarks/privacy/speed_suite.py \
        --library fastmob --backend polars --profile memory --output-dir "$RESULTS_DIR"

run_always \
    plot_all_comparisons \
    bash scripts/plot_all_comparisons.sh \
        --env-dir "$RESULTS_DIR" --input-order both

touch "$RESULTS_DIR/background_continue.done"
log "background continuation complete"
