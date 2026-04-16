#!/usr/bin/env bash
# Run correctness tests.
# By default excludes the skmob-comparison tests (requires skmob installed).
# Pass -m skmob to include them, or -m "" to run everything.
#
# Usage:
#   bash tests/run_correctness.sh              # synthetic tests only
#   bash tests/run_correctness.sh -m skmob     # skmob-comparison tests only
#   bash tests/run_correctness.sh -m ""        # all correctness tests
#   bash tests/run_correctness.sh -v           # verbose output
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
maturin develop

# Default: skip skmob-comparison tests unless the caller overrides -m
MARKER_ARGS=("-m" "not skmob")
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == "-m" ]]; then
        MARKER_ARGS=()
    fi
    EXTRA_ARGS+=("$arg")
done

pytest tests/correctness/ "${MARKER_ARGS[@]}" "${EXTRA_ARGS[@]}"
