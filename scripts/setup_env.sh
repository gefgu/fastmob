#!/usr/bin/env bash
# Create a uv virtual environment and install all packages needed to run tests.
# Run from anywhere — the script resolves the repo root automatically.
#
# Usage:
#   bash scripts/setup_env.sh                    # core dev deps only
#   bash scripts/setup_env.sh --python 3.12 --venv .venv-py312
#   bash scripts/setup_env.sh --skmob            # also install scikit-mobility (optional)
#   bash scripts/setup_env.sh --movingpandas     # also install movingpandas + geopandas (optional)
#   bash scripts/setup_env.sh --ptrail           # also install ptrail (optional)
#
# After completion, activate the environment with:
#   source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL_SKMOB=false
INSTALL_MOVINGPANDAS=false
INSTALL_PTRAIL=false
PYTHON_SPEC=""
VENV_DIR=".venv"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --skmob)
            INSTALL_SKMOB=true
            shift
            ;;
        --movingpandas)
            INSTALL_MOVINGPANDAS=true
            shift
            ;;
        --ptrail)
            INSTALL_PTRAIL=true
            shift
            ;;
        --python)
            PYTHON_SPEC="$2"
            shift 2
            ;;
        --venv)
            VENV_DIR="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            exit 2
            ;;
    esac
done

echo "==> Creating virtual environment at ${VENV_DIR} ..."
[ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
if [ -n "$PYTHON_SPEC" ]; then
    uv venv --python "$PYTHON_SPEC" "$VENV_DIR"
else
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
unset CONDA_PREFIX

echo "==> Installing maturin and polars ..."
# FIX 2: Added polars. Because the venv is active, we can safely drop the `--python` flags.
uv pip install "maturin>=1.13,<2.0" polars

echo "==> Building Rust extension (maturin develop) ..."
if ! maturin develop; then
    maturin develop --uv
fi

echo "==> Installing dev dependencies ..."
uv pip install -e ".[dev]"

if $INSTALL_SKMOB; then
    echo "==> Installing scikit-mobility (skmob comparison tests) ..."
    uv pip install -e ".[dev-skmob]" || {
        echo "WARNING: scikit-mobility failed to install."
        echo "         skmob-comparison tests will be skipped automatically."
    }
else
    echo ""
    echo "NOTE: scikit-mobility not installed. skmob-comparison tests will be skipped."
    echo "      To install it, re-run with: bash scripts/setup_env.sh --skmob"
fi

if $INSTALL_MOVINGPANDAS; then
    echo "==> Installing movingpandas + geopandas (movingpandas comparison benchmarks) ..."
    uv pip install -e ".[dev-movingpandas]" || {
        echo "WARNING: movingpandas or geopandas failed to install."
        echo "         movingpandas-comparison benchmarks will be skipped automatically."
    }
else
    echo ""
    echo "NOTE: movingpandas not installed. movingpandas-comparison benchmarks will be skipped."
    echo "      To install it, re-run with: bash scripts/setup_env.sh --movingpandas"
fi

if $INSTALL_PTRAIL; then
    echo "==> Installing ptrail (PTRAIL comparison tests) ..."
    uv pip install -e ".[dev-ptrail]" || {
        echo "WARNING: ptrail failed to install."
        echo "         ptrail-comparison tests will be skipped automatically."
    }
else
    echo ""
    echo "NOTE: ptrail not installed. ptrail-comparison tests will be skipped."
    echo "      To install it, re-run with: bash scripts/setup_env.sh --ptrail"
fi

echo ""
echo "Done. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
