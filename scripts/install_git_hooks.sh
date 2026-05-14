#!/usr/bin/env bash
# Configure this checkout to use pre-commit hooks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash scripts/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate

if [ "$(git config --get core.hooksPath || true)" = ".githooks" ]; then
    git config --unset core.hooksPath
fi

pre-commit install

echo "Git hooks installed with pre-commit."
