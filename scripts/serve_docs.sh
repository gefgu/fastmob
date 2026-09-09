#!/usr/bin/env bash
set -euo pipefail

uv run --group docs python scripts/build_notebook_docs.py
exec uv run --group docs zensical serve --dev-addr "${DOCS_DEV_ADDR:-127.0.0.1:5178}"
