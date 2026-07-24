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
#   bash scripts/setup_env.sh --venv .venv-transbigdata --transbigdata  # dedicated env, see pyproject.toml dev-transbigdata comment
#   bash scripts/setup_env.sh --venv .venv-pymove --python 3.10 --pymove  # dedicated env, see CLAUDE.md
#
# After completion, activate the environment with:
#   source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL_SKMOB=false
INSTALL_MOVINGPANDAS=false
INSTALL_PTRAIL=false
INSTALL_TRANSBIGDATA=false
INSTALL_PYMOVE=false
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
        --transbigdata)
            INSTALL_TRANSBIGDATA=true
            shift
            ;;
        --pymove)
            INSTALL_PYMOVE=true
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

if $INSTALL_TRANSBIGDATA; then
    echo "==> Installing transbigdata + osmnx (TransBigData comparison tests) ..."
    uv pip install -e ".[dev-transbigdata]" || {
        echo "WARNING: transbigdata or osmnx failed to install."
        echo "         transbigdata-comparison tests will be skipped automatically."
    }
else
    echo ""
    echo "NOTE: transbigdata not installed. transbigdata-comparison tests will be skipped."
    echo "      To install it, re-run with: bash scripts/setup_env.sh --transbigdata"
fi

if $INSTALL_PYMOVE; then
    echo "==> Installing pymove (PyMove comparison tests) ..."
    uv pip install -e ".[dev-pymove]" && {
        # pymove's own metadata allows pandas>=1.5 (this project's base
        # dependency), but its code needs pandas<1.4; that pin can't live in
        # the dev-pymove extra without conflicting with the base dependency
        # at resolve time, so force it here as a plain (non-editable)
        # install. A newer dask also refuses to import under pandas<2.0, so
        # pin it down to match. See pyproject.toml's dev-pymove comment.
        echo "==> Re-pinning pandas/dask for pymove's actual code (see pyproject.toml dev-pymove) ..."
        uv pip install "pandas>=1.1.0,<1.4.0" "dask[dataframe]<2022.2"
    } || {
        echo "WARNING: pymove failed to install."
        echo "         pymove-comparison tests will be skipped automatically."
    }
else
    echo ""
    echo "NOTE: pymove not installed. pymove-comparison tests will be skipped."
    echo "      To install it, re-run with: bash scripts/setup_env.sh --pymove"
    echo "      (needs a dedicated venv: bash scripts/setup_env.sh --venv .venv-pymove --python 3.10 --pymove)"
fi

echo ""
echo "Done. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
