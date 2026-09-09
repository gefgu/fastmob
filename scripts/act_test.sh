#!/usr/bin/env bash
# Run CI.yml's `test` job locally, in a container matching the GitHub Actions
# environment, via https://github.com/nektos/act. Catches lint/test failures
# before pushing, without waiting in the Actions queue.
#
# Usage:
#   bash scripts/act_test.sh [extra act args...]
#
# Requires: docker, act.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH. act runs jobs inside Docker containers."
    exit 1
fi

if ! command -v act >/dev/null 2>&1; then
    echo "ERROR: act not found on PATH."
    echo "       Install it: https://nektosact.com/installation/index.html"
    echo "       (e.g. 'curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | bash')"
    exit 1
fi

act push -W .github/workflows/CI.yml -j test "$@"
