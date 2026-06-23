#!/usr/bin/env bash
# Build UniVTAC Docker images locally.
#
# Image layers (bottom-up): base → isaac-sim → isaac-lab → tacex → univtac
#
# Usage:
#   docker/build.sh                                 # build all layers
#   docker/build.sh univtac                         # build just the app layer (+ deps)
#   docker/build.sh --tag 0.2.0                     # build all with a specific tag
#   docker/build.sh --push --tag 0.2.0              # build + push to GHCR
#   docker/build.sh --cuda-arch 90                  # build for sm_90 (H100/B200)
#
# BuildKit cache: --mount=type=cache in Dockerfiles persists across builds.
# First build downloads Isaac Sim (~15GB, ~30 min). Subsequent builds hit cache.
#
# Environment:
#   OMNI_KIT_ACCEPT_EULA=YES                         # required for Isaac Sim layer
#
set -euo pipefail

TAG="latest"
TARGET=""
PUSH=false
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-89}"
ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-NO}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)          TAG="$2"; shift 2 ;;
    --push)         PUSH=true; shift ;;
    --cuda-arch)    CUDA_ARCH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^[^#]/{ s/^# \?//p; }' "$0"
      exit 0 ;;
    -*)             echo "Unknown flag: $1"; exit 1 ;;
    *)              TARGET="$1"; shift ;;
  esac
done

REGISTRY="ghcr.io/univtac"

# Share host uv cache as build context (read-only mount for hardlink seeding).
# Falls back gracefully when the cache doesn't exist (CI, fresh machines).
UV_CACHE="${UV_CACHE_DIR:-$HOME/.cache/uv}"
CACHE_FLAGS=()
if [ -d "$UV_CACHE" ]; then
  echo "[INFO] Host uv cache: $UV_CACHE ($(du -sh "$UV_CACHE" 2>/dev/null | cut -f1))"
  CACHE_FLAGS=(--build-context "uv-cache=$UV_CACHE")
fi

# All layers in dependency order
LAYERS=(base isaac-sim isaac-lab tacex univtac)

build_layer() {
  local name="$1"
  local image_name="${name//_/-}"
  local dockerfile="docker/Dockerfile.${name}"
  local image_tag="${REGISTRY}/${image_name}:${TAG}"
  local build_args=(--build-arg "OMNI_KIT_ACCEPT_EULA=${ACCEPT_EULA}")

  # Pass CUDA arch to tacex layer
  if [[ "$name" == "tacex" ]]; then
    build_args+=(--build-arg "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
  fi

  echo "========================================="
  echo "Building: ${image_tag}"
  echo "  Dockerfile: ${dockerfile}"
  echo "  CUDA arch:  sm_${CUDA_ARCH}"
  echo "========================================="
  docker build -t "${image_tag}" -f "${dockerfile}" "${build_args[@]}" "${CACHE_FLAGS[@]}" .

  if $PUSH; then
    echo "Pushing: ${image_tag}"
    docker push "${image_tag}"
  fi
}

if [[ -n "$TARGET" ]]; then
  # Build target + all its dependencies
  for layer in "${LAYERS[@]}"; do
    build_layer "$layer"
    [[ "$layer" == "$TARGET" ]] && break
  done
else
  # Build all layers
  for layer in "${LAYERS[@]}"; do
    build_layer "$layer"
  done
fi

echo ""
echo "Done. Images built:"
for layer in "${LAYERS[@]}"; do
  image_name="${layer//_/-}"
  echo "  ${REGISTRY}/${image_name}:${TAG}"
done
