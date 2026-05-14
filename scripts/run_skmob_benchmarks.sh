#!/usr/bin/env bash
# Run only the original scikit-mobility benchmark suites.
#
# Intended for long unattended runs, for example:
#
#   nohup bash scripts/run_skmob_benchmarks.sh > tests/benchmarks/results/logs/skmob_overnight.out 2>&1 &
#
# Extra arguments are forwarded to both suites, for example:
#
#   bash scripts/run_skmob_benchmarks.sh --sizes 1000 10000 100000 --iterations 3 --sleep 0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKMOB_VENV="${SKMOB_VENV:-$REPO_ROOT/.venv-skmob}"
RESULTS_DIR="$REPO_ROOT/tests/benchmarks/results"
LOG_DIR="$RESULTS_DIR/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SPATIAL_LOG="$LOG_DIR/skmob_spatial_${TIMESTAMP}.log"
VISITS_LOG="$LOG_DIR/skmob_visits_${TIMESTAMP}.log"

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

run_skmob_suite tests/benchmarks/speed_spatial_suite.py "$SPATIAL_LOG" "$@"
run_skmob_suite tests/benchmarks/speed_visits_suite.py "$VISITS_LOG" "$@"

echo "==> Done"
echo "Spatial log: $SPATIAL_LOG"
echo "Visits log:  $VISITS_LOG"
echo "Results:     $RESULTS_DIR"
