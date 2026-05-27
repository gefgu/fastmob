#!/usr/bin/env bash
# Generate marketing comparison plots for available skmob2 speed benchmarks.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

_BASE_RESULTS_DIR="$REPO_ROOT/tests/benchmarks/results"
PLOT_SCRIPT="$REPO_ROOT/tests/benchmarks/plot_benchmark_comparisons.py"
FAILURES=()
INPUT_ORDER="raw"
PLOT_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input-order)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --input-order requires one of: raw, sorted, both"
                exit 1
            fi
            INPUT_ORDER="$2"
            shift 2
            ;;
        --input-order=*)
            INPUT_ORDER="${1#*=}"
            shift
            ;;
        *)
            PLOT_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$INPUT_ORDER" != "raw" ] && [ "$INPUT_ORDER" != "sorted" ] && [ "$INPUT_ORDER" != "both" ]; then
    echo "ERROR: --input-order must be one of: raw, sorted, both"
    exit 1
fi

input_orders() {
    if [ "$INPUT_ORDER" = "both" ]; then
        printf '%s\n' raw sorted
    else
        printf '%s\n' "$INPUT_ORDER"
    fi
}

order_part() {
    if [ "$1" = "raw" ]; then
        printf ''
    else
        printf '%s_' "$1"
    fi
}

# Parse --env-dir from arguments; pass remaining args to the plot script
ENV_DIR=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-dir)
            ENV_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}"

# Auto-discover the most recently modified env subfolder when --env-dir is not given
if [ -z "$ENV_DIR" ]; then
    ENV_DIR="$(find "$_BASE_RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d \
        -not -name "plots" -not -name "logs" \
        | while read -r d; do
            ls "$d"/*.json >/dev/null 2>&1 && echo "$d"
          done \
        | xargs -I{} stat --format="%Y {}" {} 2>/dev/null \
        | sort -rn \
        | head -1 \
        | awk '{print $2}')"
    if [ -z "$ENV_DIR" ]; then
        # Fallback: legacy flat results dir
        ENV_DIR="$_BASE_RESULTS_DIR"
        echo "WARNING: No env subfolders found with JSON files; using flat results dir."
    else
        echo "==> Auto-discovered env dir: ${ENV_DIR#$REPO_ROOT/}"
    fi
fi

RESULTS_DIR="${ENV_DIR}"
OUTPUT_DIR="${ENV_DIR}/plots"

run_comparison() {
    local suite="$1"
    local backend="$2"
    local input_order="$3"
    shift 3
    local part
    part="$(order_part "$input_order")"
    local original_json="$RESULTS_DIR/skmob_${suite}_speed_${part}prebuilt_tdf.json"
    local optimized_json="$RESULTS_DIR/skmob2_${suite}_speed_${part}${backend}.json"

    if [ ! -f "$original_json" ]; then
        echo "Skipping ${suite}/${backend}/${input_order}: missing $(basename "$original_json")"
        return 0
    fi
    if [ ! -f "$optimized_json" ]; then
        echo "Skipping ${suite}/${backend}/${input_order}: missing $(basename "$optimized_json")"
        return 0
    fi

    echo
    echo "==> Plotting ${suite}/${backend}/${input_order}"
    if "$PYTHON" "$PLOT_SCRIPT" \
        --original-json "$original_json" \
        --optimized-json "$optimized_json" \
        --suite "$suite" \
        --backend "$backend" \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
        echo "==> ${suite}/${backend}/${input_order}: ok"
    else
        local status=$?
        echo "==> ${suite}/${backend}/${input_order}: failed with exit code ${status}"
        FAILURES+=("${suite}/${backend}/${input_order} (${status})")
    fi
}

run_model_comparison() {
    local suite="$1"
    local original_json="$2"
    local optimized_json="$3"
    local label="$4"
    shift 4

    if [ ! -f "$original_json" ]; then
        echo "Skipping ${label}: missing $(basename "$original_json")"
        return 0
    fi
    if [ ! -f "$optimized_json" ]; then
        echo "Skipping ${label}: missing $(basename "$optimized_json")"
        return 0
    fi

    echo
    echo "==> Plotting ${label}"
    if "$PYTHON" "$PLOT_SCRIPT" \
        --original-json "$original_json" \
        --optimized-json "$optimized_json" \
        --suite "$suite" \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
        echo "==> ${label}: ok"
    else
        local status=$?
        echo "==> ${label}: failed with exit code ${status}"
        FAILURES+=("${label} (${status})")
    fi
}

for suite in spatial privacy visits; do
    for backend in pandas polars; do
        while IFS= read -r input_order; do
            run_comparison "$suite" "$backend" "$input_order" "${PLOT_ARGS[@]}"
        done < <(input_orders)
    done
done

echo
echo "==> Plotting models (unified: location-only + agent-based, small + large scale)"
if "$PYTHON" "$PLOT_SCRIPT" \
    --suite models \
    --original-json "$RESULTS_DIR/skmob_models_speed.json" \
    --optimized-json "$RESULTS_DIR/skmob2_models_speed.json" \
    --large-json-skmob2 "$RESULTS_DIR/skmob2_models_speed_large_scale.json" \
    --large-json-skmob  "$RESULTS_DIR/skmob_models_speed_large_scale.json" \
    --large-loc-json-skmob2 "$RESULTS_DIR/skmob2_models_speed_location_large_scale.json" \
    --large-loc-json-skmob  "$RESULTS_DIR/skmob_models_speed_location_large_scale.json" \
    --output-dir "$OUTPUT_DIR" \
    "${PLOT_ARGS[@]}"; then
    echo "==> models: ok"
else
    status=$?
    echo "==> models: failed with exit code ${status}"
    FAILURES+=("models (${status})")
fi

run_memory_comparison() {
    local suite="$1"
    local backend="$2"
    shift 2
    local original_json="$RESULTS_DIR/skmob_${suite}_memory_prebuilt_tdf.json"
    local optimized_json="$RESULTS_DIR/skmob2_${suite}_memory_${backend}.json"

    if [ ! -f "$original_json" ]; then
        echo "Skipping ${suite}/${backend}/memory: missing $(basename "$original_json")"
        return 0
    fi
    if [ ! -f "$optimized_json" ]; then
        echo "Skipping ${suite}/${backend}/memory: missing $(basename "$optimized_json")"
        return 0
    fi

    echo
    echo "==> Plotting ${suite}/${backend}/memory"
    if "$PYTHON" "$PLOT_SCRIPT" \
        --original-json "$original_json" \
        --optimized-json "$optimized_json" \
        --suite "$suite" \
        --backend "$backend" \
        --profile memory \
        --output-dir "$OUTPUT_DIR" \
        "$@"; then
        echo "==> ${suite}/${backend}/memory: ok"
    else
        local status=$?
        echo "==> ${suite}/${backend}/memory: failed with exit code ${status}"
        FAILURES+=("${suite}/${backend}/memory (${status})")
    fi
}

for suite in spatial visits; do
    for backend in pandas polars; do
        run_memory_comparison "$suite" "$backend" "${PLOT_ARGS[@]}"
    done
done

echo
echo "==> Plotting models memory (unified: location-only + agent-based, small + large scale)"
if "$PYTHON" "$PLOT_SCRIPT" \
    --suite models \
    --profile memory \
    --original-json "$RESULTS_DIR/skmob_models_memory.json" \
    --optimized-json "$RESULTS_DIR/skmob2_models_memory.json" \
    --large-json-skmob2 "$RESULTS_DIR/skmob2_models_memory_large_scale.json" \
    --large-loc-json-skmob2 "$RESULTS_DIR/skmob2_models_memory_location_large_scale.json" \
    --output-dir "$OUTPUT_DIR" \
    "${PLOT_ARGS[@]}"; then
    echo "==> models memory: ok"
else
    status=$?
    echo "==> models memory: failed with exit code ${status}"
    FAILURES+=("models memory (${status})")
fi

echo
echo "==> Plot generation complete. Images are in ${OUTPUT_DIR#$REPO_ROOT/}"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "==> Failures:"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
