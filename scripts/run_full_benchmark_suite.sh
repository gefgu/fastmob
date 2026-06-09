#!/usr/bin/env bash
# Full benchmark suite: skmob2 (py312) → skmob (venv-skmob) → plots.
#
# Usage:
#   bash scripts/run_full_benchmark_suite.sh
#   bash scripts/run_full_benchmark_suite.sh --phase skmob2   # only Phase 1
#   bash scripts/run_full_benchmark_suite.sh --phase skmob    # only Phase 2
#   bash scripts/run_full_benchmark_suite.sh --phase plots    # only Phase 3

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_VENV="$REPO_ROOT/.venv-py312"
SKMOB_VENV="$REPO_ROOT/.venv-skmob"
OUTPUT_DIR="$REPO_ROOT/benchmarks/results/py312_intel_core_i7_10700_16c"
LOG_DIR="$OUTPUT_DIR/logs"
STATUS_FILE="$LOG_DIR/run_status.txt"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
FAILURES=()
PHASE="all"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --phase=*) PHASE="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

run_python() {
    local venv="$1"; shift
    env -u CONDA_PREFIX \
        VIRTUAL_ENV="$venv" \
        PATH="$venv/bin:$PATH" \
        "$venv/bin/python" "$@"
}

update_status() {
    local phase="$1" job="$2" status="$3"
    printf 'PHASE=%s\nJOB=%s\nSTATUS=%s\nLAST_UPDATED=%s\n' \
        "$phase" "$job" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$STATUS_FILE"
}

run_job() {
    local label="$1" venv="$2" phase="$3"
    shift 3
    local log_path="$LOG_DIR/${RUN_ID}_${label}.log"
    echo
    echo "==> ${label}"
    echo "    log: ${log_path#$REPO_ROOT/}"
    update_status "$phase" "$label" "running"
    if run_python "$venv" "$@" 2>&1 | tee "$log_path"; then
        echo "==> ${label}: ok"
        update_status "$phase" "$label" "ok"
    else
        local status=$?
        echo "==> ${label}: FAILED (exit ${status})"
        update_status "$phase" "$label" "failed"
        FAILURES+=("${label} (${status})")
    fi
}

json_exists() { [ -f "$OUTPUT_DIR/$1" ]; }

both_orders_exist() {
    json_exists "$1" && json_exists "$2"
}

# ── Phase 0: Build ──────────────────────────────────────────────────────────

if [ "$PHASE" = "all" ] || [ "$PHASE" = "skmob2" ]; then
    echo
    echo "==> Phase 0: Building skmob2._core in ${MAIN_VENV#$REPO_ROOT/} ..."
    run_job "build_skmob2_core" "$MAIN_VENV" "build" -m maturin develop --uv
fi

# ── Phase 1: skmob2 benchmarks (always fresh) ────────────────────────────────

if [ "$PHASE" = "all" ] || [ "$PHASE" = "skmob2" ]; then
    echo
    echo "==> Phase 1: skmob2 benchmarks (fresh) in ${MAIN_VENV#$REPO_ROOT/} ..."

    for profile in speed memory; do
        run_job "skmob2_preprocessing_${profile}_both" "$MAIN_VENV" "skmob2" \
            benchmarks/preprocessing/speed_suite.py \
            --library skmob2 --backend both --input-order both \
            --profile "$profile" --output-dir "$OUTPUT_DIR"

        run_job "skmob2_individual_${profile}_both" "$MAIN_VENV" "skmob2" \
            benchmarks/individual/speed_suite.py \
            --library skmob2 --backend both --input-order both \
            --profile "$profile" --output-dir "$OUTPUT_DIR"

        run_job "skmob2_collective_${profile}_both" "$MAIN_VENV" "skmob2" \
            benchmarks/collective/speed_suite.py \
            --library skmob2 --backend both --input-order both \
            --profile "$profile" --output-dir "$OUTPUT_DIR"

        run_job "skmob2_evaluation_${profile}_both" "$MAIN_VENV" "skmob2" \
            benchmarks/evaluation/speed_suite.py \
            --library skmob2 --backend both \
            --profile "$profile" --output-dir "$OUTPUT_DIR"

        run_job "skmob2_models_${profile}" "$MAIN_VENV" "skmob2" \
            benchmarks/models/speed_suite.py \
            --library skmob2 --n-agents 500 --n-locations 10000 \
            --profile "$profile" --output-dir "$OUTPUT_DIR"

        run_job "skmob2_privacy_${profile}_both" "$MAIN_VENV" "skmob2" \
            benchmarks/privacy/speed_suite.py \
            --library skmob2 --backend both --input-order both \
            --repeat-dataset 10 \
            --profile "$profile" --output-dir "$OUTPUT_DIR"
    done
fi

# ── Phase 2: skmob benchmarks (existence-checked) ────────────────────────────

if [ "$PHASE" = "all" ] || [ "$PHASE" = "skmob" ]; then
    if [ ! -x "$SKMOB_VENV/bin/python" ]; then
        echo "WARNING: .venv-skmob not found; skipping skmob comparison benchmarks."
    else
        echo
        echo "==> Phase 2: skmob benchmarks in .venv-skmob ..."

        for timing_mode in prebuilt_tdf workflow_tdf; do
            for profile in speed memory; do
                # individual
                raw="skmob_individual_${profile}_${timing_mode}.json"
                sorted="skmob_individual_${profile}_sorted_${timing_mode}.json"
                if ! both_orders_exist "$raw" "$sorted"; then
                    run_job "skmob_individual_${profile}_${timing_mode}" "$SKMOB_VENV" "skmob" \
                        benchmarks/individual/speed_suite.py \
                        --library skmob --timing-mode "$timing_mode" \
                        --input-order both \
                        --profile "$profile" --output-dir "$OUTPUT_DIR"
                else
                    echo "==> skmob_individual_${profile}_${timing_mode}: skipped (both ${raw} and ${sorted} exist)"
                fi

                # collective
                raw="skmob_collective_${profile}_${timing_mode}.json"
                sorted="skmob_collective_${profile}_sorted_${timing_mode}.json"
                if ! both_orders_exist "$raw" "$sorted"; then
                    run_job "skmob_collective_${profile}_${timing_mode}" "$SKMOB_VENV" "skmob" \
                        benchmarks/collective/speed_suite.py \
                        --library skmob --timing-mode "$timing_mode" \
                        --input-order both \
                        --profile "$profile" --output-dir "$OUTPUT_DIR"
                else
                    echo "==> skmob_collective_${profile}_${timing_mode}: skipped (both ${raw} and ${sorted} exist)"
                fi

                # preprocessing
                raw="skmob_preprocessing_${profile}_${timing_mode}.json"
                sorted="skmob_preprocessing_${profile}_sorted_${timing_mode}.json"
                if ! both_orders_exist "$raw" "$sorted"; then
                    run_job "skmob_preprocessing_${profile}_${timing_mode}" "$SKMOB_VENV" "skmob" \
                        benchmarks/preprocessing/speed_suite.py \
                        --library skmob --timing-mode "$timing_mode" \
                        --input-order both \
                        --profile "$profile" --output-dir "$OUTPUT_DIR"
                else
                    echo "==> skmob_preprocessing_${profile}_${timing_mode}: skipped (both ${raw} and ${sorted} exist)"
                fi

                # privacy — always re-run (repeat-dataset changed to 10)
                run_job "skmob_privacy_${profile}_${timing_mode}" "$SKMOB_VENV" "skmob" \
                    benchmarks/privacy/speed_suite.py \
                    --library skmob --timing-mode "$timing_mode" \
                    --input-order both --repeat-dataset 10 \
                    --profile "$profile" --output-dir "$OUTPUT_DIR"
            done
        done
    fi
fi

# ── Phase 3: Plot generation ─────────────────────────────────────────────────

if [ "$PHASE" = "all" ] || [ "$PHASE" = "plots" ]; then
    echo
    echo "==> Phase 3: Generating comparison plots ..."
    update_status "plots" "plot_all_comparisons" "running"
    if PYTHON="$MAIN_VENV/bin/python" bash scripts/plot_all_comparisons.sh --input-order both 2>&1 | tee "$LOG_DIR/${RUN_ID}_plots.log"; then
        update_status "plots" "plot_all_comparisons" "ok"
        echo "==> Plots: ok"
    else
        update_status "plots" "plot_all_comparisons" "failed"
        FAILURES+=("plot_all_comparisons")
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo
echo "==> Run complete. Results: ${OUTPUT_DIR#$REPO_ROOT/}"
echo "    Logs: ${LOG_DIR#$REPO_ROOT/}/${RUN_ID}_*.log"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "==> FAILURES:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "==> All jobs completed successfully."
