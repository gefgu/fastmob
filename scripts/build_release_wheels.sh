#!/usr/bin/env bash
# Build a local mirror of CI's Linux wheel/sdist jobs, before pushing a release tag.
#
# This exists to catch build breakage (feature-flag issues, manifest-path
# problems, missing native deps) on your own machine, so the GitHub-hosted
# run triggered by a version tag is likely to succeed on the first try
# instead of burning queued macOS/Windows minutes on a retry.
#
# macOS and Windows wheels are intentionally not built here: they require
# Apple/Microsoft-signed toolchains that don't exist on this machine, so
# those stay CI-only.
#
# Usage:
#   bash scripts/build_release_wheels.sh
#
# Requires: docker (manylinux/musllinux wheels), zig (aarch64 cross-compile).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MATURIN_IMAGE="ghcr.io/pyo3/maturin"
FEATURE_ARGS=(--no-default-features --features stvd-emd)

mkdir -p dist

echo "==> Building sdist ..."
uv run --with maturin maturin sdist --out dist

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH. Install Docker to build manylinux/musllinux wheels locally."
    exit 1
fi

echo "==> Building manylinux x86_64 wheel (docker: ${MATURIN_IMAGE}) ..."
docker run --rm -v "$REPO_ROOT":/io "$MATURIN_IMAGE" \
    build --release --out dist "${FEATURE_ARGS[@]}"

echo "==> Building musllinux x86_64 wheel (docker: ${MATURIN_IMAGE}) ..."
docker run --rm -v "$REPO_ROOT":/io "$MATURIN_IMAGE" \
    build --release --out dist --manylinux musllinux_1_2 "${FEATURE_ARGS[@]}"

if command -v zig >/dev/null 2>&1; then
    echo "==> Cross-compiling aarch64 Linux wheel (zig) ..."
    # .cargo/config.toml pins target-cpu=native for this machine's own benchmarks;
    # that's meaningless (and breaks the zig/clang backend) when cross-compiling
    # to a different architecture, so clear it for this one invocation.
    if RUSTFLAGS="" uv run --with maturin maturin build --release \
        --target aarch64-unknown-linux-gnu --zig -i 3.12 \
        --out dist "${FEATURE_ARGS[@]}"; then
        :
    else
        echo "WARNING: aarch64 zig cross-build failed; continuing without it."
        echo "         CI still builds aarch64 wheels independently via a manylinux/QEMU container,"
        echo "         so this only affects local pre-flight coverage, not the real release."
    fi
else
    echo "WARNING: zig not found on PATH, skipping local aarch64 wheel build."
    echo "         Install with 'pip install ziglang' (or your system package manager) to enable it."
fi

echo "==> Done. Artifacts in dist/:"
ls -lh dist
