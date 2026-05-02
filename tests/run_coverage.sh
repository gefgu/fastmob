#!/usr/bin/env bash
# Run correctness tests with Python coverage for the skmob2 package.
# By default excludes the skmob-comparison tests (requires skmob installed).
# Pass -m skmob to include them, or -m "" to run everything.
#
# Usage:
#   bash tests/run_coverage.sh              # synthetic tests only
#   bash tests/run_coverage.sh -m skmob     # skmob-comparison tests only
#   bash tests/run_coverage.sh -m ""        # all correctness tests
#   bash tests/run_coverage.sh --cov-report=html
#
# Any extra arguments are forwarded directly to pytest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: virtual environment not found. Run 'bash tests/setup_env.sh' first."
    exit 1
fi

source .venv/bin/activate
unset CONDA_PREFIX

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade
fi

python -m maturin develop

# Default: skip skmob-comparison tests unless the caller overrides -m
MARKER_ARGS=("-m" "not skmob")
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "-m" ]]; then
        MARKER_ARGS=()
    fi
    EXTRA_ARGS+=("$arg")
done

python -m pytest tests/correctness/ \
    --cov=skmob2 \
    --cov-branch \
    --cov-report=term-missing \
    "${MARKER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
