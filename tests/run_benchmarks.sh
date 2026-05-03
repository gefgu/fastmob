#!/usr/bin/env bash
# Run benchmark tests using pytest-benchmark.
# Results are printed as a table by default.
#
# Usage:
#   bash tests/run_benchmarks.sh                                  # table output
#   bash tests/run_benchmarks.sh --benchmark-json=results.json    # save JSON
#   bash tests/run_benchmarks.sh --benchmark-save=baseline        # save named snapshot
#   bash tests/run_benchmarks.sh -k 1k                            # only 1k-row size
#   bash tests/run_benchmarks.sh -k radius_of_gyration            # one workload family
#
# Compare saved snapshots:
#   pytest-benchmark compare baseline 0001
#
# Any extra arguments are forwarded directly to pytest.
# When comparison benchmarks are selected, this script runs each comparison in
# the virtual environment with compatible dependencies:
#   .venv        -> skmob2 pandas/Polars and movingpandas, when installed there
#   .venv-skmob  -> skmob, which requires the legacy Shapely/geopandas stack

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MAIN_VENV="$REPO_ROOT/.venv"
SKMOB_VENV="$REPO_ROOT/.venv-skmob"
MOVINGPANDAS_VENV="${MOVINGPANDAS_VENV:-$MAIN_VENV}"

if [ ! -f "$MAIN_VENV/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
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

pytest_args_for_env() {
    local label="$1"
    shift
    local arg value path stem
    for arg in "$@"; do
        case "$arg" in
            --benchmark-json=*)
                value="${arg#--benchmark-json=}"
                if [[ "$value" == *.json ]]; then
                    stem="${value%.json}"
                    printf '%s\n' "--benchmark-json=${stem}.${label}.json"
                else
                    printf '%s\n' "--benchmark-json=${value}.${label}"
                fi
                ;;
            --benchmark-save=*)
                value="${arg#--benchmark-save=}"
                printf '%s\n' "--benchmark-save=${value}-${label}"
                ;;
            *)
                printf '%s\n' "$arg"
                ;;
        esac
    done
}

run_benchmarks() {
    local label="$1"
    local venv="$2"
    local marker="$3"
    shift 3
    local args=()
    mapfile -t args < <(pytest_args_for_env "$label" "$@")
    run_python "$venv" -m pytest tests/benchmarks/ -v -m "$marker" "${args[@]}"
}

if ! run_python "$MAIN_VENV" -m pip --version >/dev/null 2>&1; then
    run_python "$MAIN_VENV" -m ensurepip --upgrade
fi

echo "==> Building skmob2._core in .venv ..."
run_python "$MAIN_VENV" -m maturin develop

echo "==> Running skmob2 benchmarks in .venv ..."
run_benchmarks "skmob2" "$MAIN_VENV" "not skmob and not movingpandas" "$@"

if [ -x "$SKMOB_VENV/bin/python" ]; then
    echo "==> Running skmob comparison benchmarks in .venv-skmob ..."
    run_benchmarks "skmob" "$SKMOB_VENV" "skmob" "$@"
else
    echo "WARNING: .venv-skmob not found; skipping skmob comparison benchmarks."
    echo "         Create it with the legacy skmob stack before running skmob comparisons."
fi

if [ -x "$MOVINGPANDAS_VENV/bin/python" ] && can_import "$MOVINGPANDAS_VENV" movingpandas; then
    label="${MOVINGPANDAS_VENV#$REPO_ROOT/}"
    echo "==> Running movingpandas comparison benchmarks in ${label} ..."
    run_benchmarks "movingpandas" "$MOVINGPANDAS_VENV" "movingpandas" "$@"
else
    echo "WARNING: movingpandas is not importable in ${MOVINGPANDAS_VENV#$REPO_ROOT/}; skipping movingpandas benchmarks."
    echo "         Install it with: source .venv/bin/activate && uv pip install -e '.[dev-movingpandas]'"
fi
