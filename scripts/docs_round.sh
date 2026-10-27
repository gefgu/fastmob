#!/usr/bin/env bash
# Build the docs and capture screenshots for one critic round.
# Usage: scripts/docs_round.sh ROUND PAGE_URL [PAGE_URL ...]   (PAGE_URL "" = home)
set -euo pipefail
round="$1"; shift
uv run --group docs zensical build --strict >/dev/null
for p in "$@"; do
  name="$(echo "${p%/}" | tr / _)"; name="${name:-home}"
  uv run --with playwright python scripts/capture_docs_pages.py "AI/docs_audit/runs/pages/r${round}" "$p" >/dev/null 2>&1
  echo "captured $name r${round}"
done
