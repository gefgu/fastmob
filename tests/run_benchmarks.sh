#!/usr/bin/env bash
# Run benchmark tests using pytest-benchmark.
# Results are printed as a table by default.
#
# Usage:
#   bash tests/run_benchmarks.sh                                  # table output
#   bash tests/run_benchmarks.sh --benchmark-json=results.json    # save JSON
#   bash tests/run_benchmarks.sh --benchmark-save=baseline        # save named snapshot
#   bash tests/run_benchmarks.sh -k 1k                            # only 1k-row size
#
# Compare saved snapshots:
#   pytest-benchmark compare baseline 0001
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

python -m pytest tests/benchmarks/ -v "$@"
