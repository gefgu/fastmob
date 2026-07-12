#!/usr/bin/env bash
# Run standalone benchmark suites across available profiles and comparison modes.
#
# Usage:
#   bash scripts/run_benchmarks.sh
#   bash scripts/run_benchmarks.sh --sizes 1000 --iterations 1 --sleep 0
#
# Any extra arguments are forwarded to each standalone benchmark suite.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_VENV="${FKMOB_BENCH_VENV:-$REPO_ROOT/.venv-py312}"
SKMOB_VENV="$REPO_ROOT/.venv-skmob"
MOVINGPANDAS_VENV="${MOVINGPANDAS_VENV:-$MAIN_VENV}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PROFILES=("speed" "memory")
SKMOB_TIMING_MODES=("prebuilt_tdf" "workflow_tdf")
FAILURES=()

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
    echo "ERROR: Python 3.12 benchmark environment not found at ${MAIN_VENV#$REPO_ROOT/}."
    echo "       Run 'bash scripts/setup_env.sh --python 3.12 --venv .venv-py312' first."
    exit 1
fi

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

detect_env_slug() {
    local venv="$1"
    run_python "$venv" - <<'PYEOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_env import detect_cpu_info, build_env_slug
print(build_env_slug(detect_cpu_info()))
PYEOF
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

ENV_SLUG="$(detect_env_slug "$MAIN_VENV" 2>/dev/null || echo "unknown_env")"
OUTPUT_DIR="$REPO_ROOT/benchmarks/results/${ENV_SLUG}"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "==> Environment: ${ENV_SLUG}"
echo "==> Results dir: ${OUTPUT_DIR#$REPO_ROOT/}"

echo "==> Building fastmob._core in ${MAIN_VENV#$REPO_ROOT/} ..."
run_job "build_fastmob_core" "$MAIN_VENV" -m maturin develop --uv

echo
echo "==> Running fastmob pandas/Polars benchmark suites in ${MAIN_VENV#$REPO_ROOT/} ..."
for profile in "${PROFILES[@]}"; do
    run_job "fastmob_individual_${profile}_both" \
        "$MAIN_VENV" \
        benchmarks/individual/speed_suite.py \
        "$@" \
        --library fastmob \
        --backend both \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_collective_${profile}_both" \
        "$MAIN_VENV" \
        benchmarks/collective/speed_suite.py \
        "$@" \
        --library fastmob \
        --backend both \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_preprocessing_${profile}_both" \
        "$MAIN_VENV" \
        benchmarks/preprocessing/speed_suite.py \
        "$@" \
        --library fastmob \
        --backend both \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_evaluation_${profile}_both" \
        "$MAIN_VENV" \
        benchmarks/evaluation/speed_suite.py \
        "$@" \
        --backend both \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_models_${profile}" \
        "$MAIN_VENV" \
        benchmarks/models/speed_suite.py \
        "$@" \
        --library fastmob \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_models_large_scale_${profile}" \
        "$MAIN_VENV" \
        benchmarks/speed_models_large_scale.py \
        "$@" \
        --library fastmob \
        --mode trajectory \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
    run_job "fastmob_models_location_large_scale_${profile}" \
        "$MAIN_VENV" \
        benchmarks/speed_models_large_scale.py \
        "$@" \
        --library fastmob \
        --mode location \
        --profile "$profile" \
        --output-dir "$OUTPUT_DIR"
done

for profile in "${PROFILES[@]}"; do
    run_job "fastmob_privacy_${profile}_both" \
        "$MAIN_VENV" \
        benchmarks/privacy/speed_suite.py \
        --library fastmob \
        --backend both \
        --profile "$profile" \
        --input-order both \
        --output-dir "$OUTPUT_DIR"
done

if [ -x "$SKMOB_VENV/bin/python" ]; then
    echo
    echo "==> Running original skmob benchmark suites in .venv-skmob ..."
    for profile in "${PROFILES[@]}"; do
        for timing_mode in "${SKMOB_TIMING_MODES[@]}"; do
            run_job "skmob_individual_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                benchmarks/individual/speed_suite.py \
                "$@" \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode" \
                --output-dir "$OUTPUT_DIR"
            run_job "skmob_collective_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                benchmarks/collective/speed_suite.py \
                "$@" \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode" \
                --output-dir "$OUTPUT_DIR"
            run_job "skmob_preprocessing_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                benchmarks/preprocessing/speed_suite.py \
                "$@" \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode" \
                --output-dir "$OUTPUT_DIR"
        done
        run_job "skmob_models_${profile}" \
            "$SKMOB_VENV" \
            benchmarks/models/speed_suite.py \
            "$@" \
            --library skmob \
            --profile "$profile" \
            --output-dir "$OUTPUT_DIR"
        run_job "skmob_models_large_scale_${profile}" \
            "$SKMOB_VENV" \
            benchmarks/speed_models_large_scale.py \
            "$@" \
            --library skmob \
            --mode trajectory \
            --profile "$profile" \
            --output-dir "$OUTPUT_DIR"
        run_job "skmob_models_location_large_scale_${profile}" \
            "$SKMOB_VENV" \
            benchmarks/speed_models_large_scale.py \
            "$@" \
            --library skmob \
            --mode location \
            --profile "$profile" \
            --output-dir "$OUTPUT_DIR"
    done
    for timing_mode in "${SKMOB_TIMING_MODES[@]}"; do
        for profile in "${PROFILES[@]}"; do
            run_job "skmob_privacy_${profile}_${timing_mode}" \
                "$SKMOB_VENV" \
                benchmarks/privacy/speed_suite.py \
                --library skmob \
                --profile "$profile" \
                --timing-mode "$timing_mode" \
                --input-order both \
                --output-dir "$OUTPUT_DIR"
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
        run_job "movingpandas_individual_${profile}" \
            "$MOVINGPANDAS_VENV" \
            benchmarks/individual/speed_suite.py \
            "$@" \
            --library movingpandas \
            --profile "$profile" \
            --output-dir "$OUTPUT_DIR"
        run_job "movingpandas_preprocessing_${profile}" \
            "$MOVINGPANDAS_VENV" \
            benchmarks/preprocessing/speed_suite.py \
            "$@" \
            --library movingpandas \
            --profile "$profile" \
            --output-dir "$OUTPUT_DIR"
    done
else
    echo "WARNING: movingpandas is not importable in ${MOVINGPANDAS_VENV#$REPO_ROOT/}; skipping MovingPandas benchmarks."
    echo "         Install it with: source .venv/bin/activate && uv pip install -e '.[dev-movingpandas]'"
fi

echo
echo "==> Benchmark run complete. Results are in ${OUTPUT_DIR#$REPO_ROOT/}, logs in ${LOG_DIR#$REPO_ROOT/}/ with prefix ${RUN_ID}_"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "==> Failures:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi

echo "==> All requested benchmark jobs completed successfully."
