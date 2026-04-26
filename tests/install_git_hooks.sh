#!/usr/bin/env bash
# Configure this checkout to use the tracked Git hooks in .githooks/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

git config core.hooksPath .githooks

echo "Git hooks installed: core.hooksPath=.githooks"
