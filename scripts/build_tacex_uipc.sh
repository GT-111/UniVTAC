#!/usr/bin/env bash
# Build tacex_uipc (C++/CUDA extension wrapping libuipc FEM library)
#
# Prerequisites:
#   - vcpkg installed at ~/Toolchain/vcpkg
#   - CUDA Toolkit 12.8 at /usr/local/cuda-12.8
#   - Virtual environment activated
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/build_tacex_uipc.sh
#
# Override defaults:
#   CMAKE_CUDA_ARCHITECTURES=90 bash scripts/build_tacex_uipc.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TACEX_UIPC_DIR="$PROJECT_ROOT/third_party/TacEx/source/tacex_uipc"

# --- vcpkg toolchain ---
VCPKG_ROOT="${VCPKG_ROOT:-$HOME/Toolchain/vcpkg}"
if [ ! -f "$VCPKG_ROOT/vcpkg" ]; then
    echo "[ERROR] vcpkg not found at $VCPKG_ROOT"
    echo "Install with:"
    echo "  mkdir -p ~/Toolchain && cd ~/Toolchain"
    echo "  git clone https://github.com/microsoft/vcpkg.git"
    echo "  cd vcpkg && ./bootstrap-vcpkg.sh"
    exit 1
fi
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
echo "[INFO] vcpkg toolchain: $CMAKE_TOOLCHAIN_FILE"

# --- CUDA architecture ---
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-89}"
echo "[INFO] CUDA architectures: $CMAKE_CUDA_ARCHITECTURES"

# --- Build jobs ---
UIPC_BUILD_JOBS="${UIPC_BUILD_JOBS:-$(nproc)}"
echo "[INFO] Build jobs: $UIPC_BUILD_JOBS"

# --- Build and install tacex_uipc (editable) ---
echo "[INFO] Building tacex_uipc from $TACEX_UIPC_DIR ..."
cd "$TACEX_UIPC_DIR"
uv pip install -e .

# --- Install uipc Python wrapper ---
echo "[INFO] Installing uipc Python package ..."
cd "$TACEX_UIPC_DIR/libuipc/python"
uv pip install -e .

# --- Copy pyuipc .so to where uipc expects it ---
PYUIPC_SRC=$(find "$TACEX_UIPC_DIR/build" -name "pyuipc*.so" -type f 2>/dev/null | head -1)
UIPC_PKG_DIR=$(python -c "import uipc; from pathlib import Path; print(Path(uipc.__file__).parent)")
UIPC_MODULES_DIR="$UIPC_PKG_DIR/modules/Release/bin"
mkdir -p "$UIPC_MODULES_DIR"
cp "$PYUIPC_SRC" "$UIPC_MODULES_DIR/"

# --- Copy libuipc .so dependencies ---
for lib in $(find "$TACEX_UIPC_DIR/build" -name "libuipc*.so" -type f 2>/dev/null); do
    cp "$lib" "$UIPC_MODULES_DIR/"
done

echo "[INFO] tacex_uipc build complete."
echo "[INFO] Verify: python -c 'import uipc; import tacex_uipc; print(\"OK\")'"
