#!/usr/bin/env bash
# Create a uv virtual environment and install all packages needed to run tests.
# Run from anywhere — the script resolves the repo root automatically.
#
# Usage:
#   bash tests/setup_env.sh                    # core dev deps only
#   bash tests/setup_env.sh --skmob            # also install scikit-mobility (optional)
#   bash tests/setup_env.sh --movingpandas     # also install movingpandas + geopandas (optional)
#
# After completion, activate the environment with:
#   source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL_SKMOB=false
INSTALL_MOVINGPANDAS=false
for arg in "$@"; do
    [[ "$arg" == "--skmob" ]] && INSTALL_SKMOB=true
    [[ "$arg" == "--movingpandas" ]] && INSTALL_MOVINGPANDAS=true
done

echo "==> Creating virtual environment at .venv ..."
[ -d .venv ] && rm -rf .venv
uv venv .venv

# FIX 1: Activate the venv explicitly so Maturin ignores your active Conda environments
source .venv/bin/activate

echo "==> Installing maturin and polars ..."
# FIX 2: Added polars. Because the venv is active, we can safely drop the `--python` flags.
uv pip install "maturin>=1.13,<2.0" polars

echo "==> Building Rust extension (maturin develop) ..."
maturin develop

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
    echo "      To install it, re-run with: bash tests/setup_env.sh --skmob"
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
    echo "      To install it, re-run with: bash tests/setup_env.sh --movingpandas"
fi

echo ""
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"