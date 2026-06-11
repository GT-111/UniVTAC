#!/usr/bin/env bash
# Build and install tacex_uipc (C++/CUDA extension).
# Must be run from the repo root *after* `uv venv && source .venv/bin/activate && uv sync`.
#
# Usage:
#   bash scripts/build_tacex_uipc.sh
#
# Override defaults via environment:
#   CUDA_HOME=/usr/local/cuda-12.8
#   CMAKE_CUDA_ARCHITECTURES=90   # e.g., H100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- defaults (overridable) ----
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"   # RTX 4090 D = sm_89
CC="${CC:-/usr/bin/gcc-11}"
CXX="${CXX:-/usr/bin/g++-11}"
CMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-$CXX}"
VCPKG_ROOT="${VCPKG_ROOT:-$HOME/Toolchain/vcpkg}"
VCPKG_DOWNLOADS="${VCPKG_DOWNLOADS:-$VCPKG_ROOT/downloads}"

# ---- pre-download tinyxml (SourceForge may be unreachable) ----
TINYXML_URL="http://archive.ubuntu.com/ubuntu/pool/universe/t/tinyxml/tinyxml_2.6.2.orig.tar.gz"
TINYXML_DST="$VCPKG_DOWNLOADS/tinyxml_2_6_2.tar.gz"
if [ ! -f "$TINYXML_DST" ]; then
    echo "[build_uipc] Downloading tinyxml from Ubuntu archive..."
    mkdir -p "$VCPKG_DOWNLOADS"
    wget -q "$TINYXML_URL" -O "$TINYXML_DST" || curl -sL "$TINYXML_URL" -o "$TINYXML_DST"
fi

# ---- export env vars for cmake / vcpkg ----
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
export VCPKG_DOWNLOADS
export VCPKG_OVERLAY_PORTS="$REPO_ROOT/third_party/TacEx/source/tacex_uipc/overlay-ports"
export CUDA_HOME CUDACXX CC CXX CMAKE_CUDA_HOST_COMPILER CMAKE_CUDA_ARCHITECTURES

echo "[build_uipc] Building tacex_uipc (CUDA arch: $CMAKE_CUDA_ARCHITECTURES)..."
cd "$REPO_ROOT"
rm -rf third_party/TacEx/source/tacex_uipc/build
uv sync --group uipc --no-build-isolation -v

echo "[build_uipc] Done."
