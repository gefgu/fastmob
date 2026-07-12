#!/usr/bin/env bash
# Periodically record status for the legacy skmob speed benchmark run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/benchmarks/results/py312_intel_core_i7_10700_16c}"
LOG_DIR="$RESULTS_DIR/logs"
MONITOR_LOG="$LOG_DIR/legacy_skmob_speed_monitor.log"
DRIVER_LOG="$LOG_DIR/legacy_skmob_speed_driver.log"
PID_FILE="$RESULTS_DIR/legacy_skmob_speed.pid"
DONE_FILE="$RESULTS_DIR/legacy_skmob_speed.done"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"

mkdir -p "$LOG_DIR"

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$MONITOR_LOG"
}

while true; do
    log "status check"

    if [ -f "$PID_FILE" ]; then
        pid="$(cat "$PID_FILE")"
        if ps -p "$pid" >/dev/null 2>&1; then
            cmd="$(ps -p "$pid" -o cmd= 2>/dev/null || true)"
            log "runner pid $pid is active: $cmd"
        else
            log "runner pid $pid is not active"
        fi
    else
        log "runner pid file missing: $PID_FILE"
    fi

    active_children="$(pgrep -af 'speed_suite.py --library skmob|speed_models_large_scale.py --library skmob|plot_all_comparisons' || true)"
    if [ -n "$active_children" ]; then
        log "active benchmark command(s):"
        printf '%s\n' "$active_children" | sed 's/^/  /' | tee -a "$MONITOR_LOG"
    else
        log "no active skmob speed benchmark child command found"
    fi

    if [ -f "$DRIVER_LOG" ]; then
        log "driver tail:"
        tail -n 8 "$DRIVER_LOG" | sed 's/^/  /' | tee -a "$MONITOR_LOG"
    else
        log "driver log missing: $DRIVER_LOG"
    fi

    legacy_count="$(find "$RESULTS_DIR" -maxdepth 1 -type f -name 'skmob_*.json' | wc -l)"
    comparison_count="$(find "$RESULTS_DIR/plots" -maxdepth 1 -type f -name 'fastmob_vs_skmob*.png' 2>/dev/null | wc -l)"
    log "legacy skmob JSON count: $legacy_count"
    log "comparison plot count: $comparison_count"

    if [ -f "$DONE_FILE" ]; then
        log "done marker found: $DONE_FILE"
        break
    fi

    sleep "$INTERVAL_SECONDS"
done

log "monitor exiting"
