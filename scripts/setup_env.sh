#!/usr/bin/env bash
# Create the standard contributor environment.
#
# Third-party comparison libraries deliberately use their own environments:
#   bash scripts/setup_benchmark_env.sh skmob
#   bash scripts/setup_benchmark_env.sh pymove --python 3.10
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_SPEC=""
VENV_DIR=".venv"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python) PYTHON_SPEC="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *)
            echo "ERROR: unknown argument: $1"
            echo "Usage: bash scripts/setup_env.sh [--python VERSION] [--venv PATH]"
            exit 2
            ;;
    esac
done

echo "==> Creating contributor environment at ${VENV_DIR} ..."
[ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
if [ -n "$PYTHON_SPEC" ]; then
    uv venv --python "$PYTHON_SPEC" "$VENV_DIR"
else
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
unset CONDA_PREFIX
uv sync --active --group dev

echo "Done. Activate it with: source ${VENV_DIR}/bin/activate"
