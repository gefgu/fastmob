#!/usr/bin/env bash
# Create an isolated environment for one external-library comparison suite.
# These stacks are excluded from pyproject.toml on purpose: several require
# mutually incompatible NumPy, pandas, or Python versions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/setup_benchmark_env.sh {skmob|movingpandas|ptrail|transbigdata|pymove} [--python VERSION] [--venv PATH]"
    exit 2
fi

ENV_NAME="$1"
shift
REQUIREMENTS="benchmarks/environments/${ENV_NAME}.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: unknown benchmark environment: ${ENV_NAME}"
    exit 2
fi

PYTHON_SPEC=""
VENV_DIR=".venv-${ENV_NAME}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --python) PYTHON_SPEC="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *) echo "ERROR: unknown argument: $1"; exit 2 ;;
    esac
done

if [ "$ENV_NAME" = "pymove" ] && [ -z "$PYTHON_SPEC" ]; then
    PYTHON_SPEC="3.10"
fi

echo "==> Creating isolated ${ENV_NAME} environment at ${VENV_DIR} ..."
[ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
if [ -n "$PYTHON_SPEC" ]; then
    uv venv --python "$PYTHON_SPEC" "$VENV_DIR"
else
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
unset CONDA_PREFIX
uv pip install -e .
uv pip install -r "$REQUIREMENTS"

echo "Done. Activate it with: source ${VENV_DIR}/bin/activate"
