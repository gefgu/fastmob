#!/usr/bin/env bash
# Run only the original scikit-mobility benchmark suites.
#
# Intended for long unattended runs, for example:
#
#   nohup bash scripts/run_skmob_benchmarks.sh > benchmarks/results/logs/skmob_overnight.out 2>&1 &
#
# Extra arguments are forwarded to each suite, for example:
#
#   bash scripts/run_skmob_benchmarks.sh --sizes 1000 10000 100000 --iterations 3 --sleep 0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKMOB_VENV="${SKMOB_VENV:-$REPO_ROOT/.venv-skmob}"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
LOG_DIR="$RESULTS_DIR/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
INDIVIDUAL_LOG="$LOG_DIR/skmob_individual_${TIMESTAMP}.log"
COLLECTIVE_LOG="$LOG_DIR/skmob_collective_${TIMESTAMP}.log"
PREPROCESSING_LOG="$LOG_DIR/skmob_preprocessing_${TIMESTAMP}.log"

if [ ! -x "$SKMOB_VENV/bin/python" ]; then
    echo "ERROR: skmob virtual environment not found at $SKMOB_VENV"
    echo "Create it with the legacy scikit-mobility stack before running this script."
    exit 1
fi

mkdir -p "$LOG_DIR"

run_skmob_suite() {
    local suite_path="$1"
    local log_path="$2"
    shift 2

    echo "==> Running $suite_path"
    echo "    log: $log_path"
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/skmob2-matplotlib}" \
    PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning}" \
        "$SKMOB_VENV/bin/python" "$suite_path" --library skmob --output-dir "$RESULTS_DIR" "$@" \
        2>&1 | tee "$log_path"
}

run_skmob_suite benchmarks/individual/speed_suite.py "$INDIVIDUAL_LOG" "$@"
run_skmob_suite benchmarks/collective/speed_suite.py "$COLLECTIVE_LOG" "$@"
run_skmob_suite benchmarks/preprocessing/speed_suite.py "$PREPROCESSING_LOG" "$@"

echo "==> Done"
echo "Individual log:    $INDIVIDUAL_LOG"
echo "Collective log:    $COLLECTIVE_LOG"
echo "Preprocessing log: $PREPROCESSING_LOG"
echo "Results:           $RESULTS_DIR"
