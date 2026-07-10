#!/usr/bin/env bash
# Run original scikit-mobility baselines and regenerate comparison plots.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-$REPO_ROOT/.venv-skmob/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/results/py312_intel_core_i7_10700_16c}"
LOG_DIR="$RESULTS_DIR/logs"
DRIVER_LOG="$LOG_DIR/legacy_skmob_comparisons_driver.log"

mkdir -p "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fkmob-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

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

log "legacy skmob comparison run started"
log "python: $PYTHON"
log "results: $RESULTS_DIR"

run_if_missing skmob_individual_speed_prebuilt_tdf "$RESULTS_DIR/skmob_individual_speed_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/individual/speed_suite.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_individual_memory_prebuilt_tdf "$RESULTS_DIR/skmob_individual_memory_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/individual/speed_suite.py --library skmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing skmob_collective_speed_prebuilt_tdf "$RESULTS_DIR/skmob_collective_speed_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/collective/speed_suite.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_collective_memory_prebuilt_tdf "$RESULTS_DIR/skmob_collective_memory_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/collective/speed_suite.py --library skmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing skmob_preprocessing_speed_prebuilt_tdf "$RESULTS_DIR/skmob_preprocessing_speed_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/preprocessing/speed_suite.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_preprocessing_memory_prebuilt_tdf "$RESULTS_DIR/skmob_preprocessing_memory_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/preprocessing/speed_suite.py --library skmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing skmob_privacy_speed_prebuilt_tdf "$RESULTS_DIR/skmob_privacy_speed_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/privacy/speed_suite.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_privacy_memory_prebuilt_tdf "$RESULTS_DIR/skmob_privacy_memory_prebuilt_tdf.json" \
    "$PYTHON" -u benchmarks/privacy/speed_suite.py --library skmob --profile memory --output-dir "$RESULTS_DIR"

run_if_missing skmob_models_speed "$RESULTS_DIR/skmob_models_speed.json" \
    "$PYTHON" -u benchmarks/models/speed_suite.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_models_memory "$RESULTS_DIR/skmob_models_memory.json" \
    "$PYTHON" -u benchmarks/models/speed_suite.py --library skmob --profile memory --output-dir "$RESULTS_DIR"
run_if_missing skmob_models_speed_large_scale "$RESULTS_DIR/skmob_models_speed_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py --library skmob --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_models_speed_location_large_scale "$RESULTS_DIR/skmob_models_speed_location_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py --library skmob --mode location --profile speed --output-dir "$RESULTS_DIR"
run_if_missing skmob_models_memory_large_scale "$RESULTS_DIR/skmob_models_memory_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py --library skmob --profile memory --output-dir "$RESULTS_DIR"
run_if_missing skmob_models_memory_location_large_scale "$RESULTS_DIR/skmob_models_memory_location_large_scale.json" \
    "$PYTHON" -u benchmarks/speed_models_large_scale.py --library skmob --mode location --profile memory --output-dir "$RESULTS_DIR"

run_always plot_all_comparisons_with_legacy \
    bash scripts/plot_all_comparisons.sh --env-dir "$RESULTS_DIR" --input-order both

touch "$RESULTS_DIR/legacy_skmob_comparisons.done"
log "legacy skmob comparison run complete"
