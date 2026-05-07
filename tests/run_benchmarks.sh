#!/usr/bin/env bash
# Run standalone benchmark suites across available profiles and comparison modes.
#
# Usage:
#   bash tests/run_benchmarks.sh
#   bash tests/run_benchmarks.sh --sizes 1000 --iterations 1 --sleep 0
#
# Any extra arguments are forwarded to each standalone benchmark suite.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_VENV="$REPO_ROOT/.venv"
SKMOB_VENV="$REPO_ROOT/.venv-skmob"
MOVINGPANDAS_VENV="${MOVINGPANDAS_VENV:-$MAIN_VENV}"
LOG_DIR="$REPO_ROOT/tests/benchmarks/results/logs"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PROFILES=("speed" "memory")
SKMOB_TIMING_MODES=("prebuilt_tdf" "workflow_tdf")
FAILURES=()

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
    exit 1
fi

mkdir -p "$LOG_DIR"

run_python() {
    local venv="$1"
    shift
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="$venv" \
        PATH="$venv/bin:$PATH" \
        "$venv/bin/python" "$@"
}

can_import() {
    local venv="$1"
    local module="$2"
    run_python "$venv" -c "import ${module}" >/dev/null 2>&1
}

run_job() {
    local label="$1"
    local venv="$2"
    shift 2

    local log_path="$LOG_DIR/${RUN_ID}_${label}.log"
    echo
    echo "==> ${label}"
    echo "    log: ${log_path#$REPO_ROOT/}"
    echo "    command: ${venv#$REPO_ROOT/}/bin/python $*"

    if run_python "$venv" "$@" 2>&1 | tee "$log_path"; then
        echo "==> ${label}: ok"
    else
        local status=$?
        echo "==> ${label}: failed with exit code ${status}"
        FAILURES+=("${label} (${status})")
    fi
}

echo "==> Building skmob2._core in .venv ..."
run_job "build_skmob2_core" "$MAIN_VENV" -m maturin develop

echo
echo "==> Running skmob2 pandas/Polars benchmark suites in .venv ..."
for profile in "${PROFILES[@]}"; do
    run_job "skmob2_spatial_${profile}_both" \
        "$MAIN_VENV" \
        tests/benchmarks/speed_spatial_suite.py \
        "$@" \
        --library skmob2 \
        --backend both \
        --profile "$profile"
    run_job "skmob2_visits_${profile}_both" \
        "$MAIN_VENV" \
        tests/benchmarks/speed_visits_suite.py \
        "$@" \
        --library skmob2 \
        --backend both \
        --profile "$profile"
done

if [ -x "$SKMOB_VENV/bin/python" ]; then
    echo
    echo "==> Running original skmob benchmark suites in .venv-skmob ..."
    for profile in "${PROFILES[@]}"; do
        for timing_mode in "${SKMOB_TIMING_MODES[@]}"; do
            run_job "skmob_spatial_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                tests/benchmarks/speed_spatial_suite.py \
                "$@" \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode"
            run_job "skmob_visits_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                tests/benchmarks/speed_visits_suite.py \
                "$@" \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode"
        done
    done
else
    echo "WARNING: .venv-skmob not found; skipping skmob comparison benchmarks."
    echo "         Create it with the legacy skmob stack before running skmob comparisons."
fi

if [ -x "$MOVINGPANDAS_VENV/bin/python" ] && can_import "$MOVINGPANDAS_VENV" movingpandas; then
    label="${MOVINGPANDAS_VENV#$REPO_ROOT/}"
    echo
    echo "==> Running MovingPandas benchmark suites in ${label} ..."
    for profile in "${PROFILES[@]}"; do
        run_job "movingpandas_spatial_${profile}" \
            "$MOVINGPANDAS_VENV" \
            tests/benchmarks/speed_spatial_suite.py \
            "$@" \
            --library movingpandas \
            --profile "$profile"
        run_job "movingpandas_visits_${profile}" \
            "$MOVINGPANDAS_VENV" \
            tests/benchmarks/speed_visits_suite.py \
            "$@" \
            --library movingpandas \
            --profile "$profile"
    done
else
    echo "WARNING: movingpandas is not importable in ${MOVINGPANDAS_VENV#$REPO_ROOT/}; skipping MovingPandas benchmarks."
    echo "         Install it with: source .venv/bin/activate && uv pip install -e '.[dev-movingpandas]'"
fi

echo
echo "==> Benchmark run complete. Logs are in ${LOG_DIR#$REPO_ROOT/} with prefix ${RUN_ID}_"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "==> Failures:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi

echo "==> All requested benchmark jobs completed successfully."
