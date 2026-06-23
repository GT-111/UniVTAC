#!/usr/bin/env bash
# Push UniVTAC Docker images to GHCR.
#
# Usage:
#   docker/push.sh                    # push all layers with :latest tag
#   docker/push.sh --tag 0.2.0        # push all layers with a specific tag
#   docker/push.sh univtac            # push just the app layer
#
set -euo pipefail

TAG="latest"
TARGET=""
REGISTRY="ghcr.io/univtac"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)   TAG="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^[^#]/{ s/^# \?//p; }' "$0"
      exit 0 ;;
    -*)      echo "Unknown flag: $1"; exit 1 ;;
    *)       TARGET="$1"; shift ;;
  esac
done

LAYERS=(base isaac-sim isaac-lab tacex univtac)

push_layer() {
  local name="$1"
  local image_name="${name//_/-}"
  local image_tag="${REGISTRY}/${image_name}:${TAG}"
  echo "Pushing: ${image_tag}"
  docker push "${image_tag}"
}

if [[ -n "$TARGET" ]]; then
  push_layer "$TARGET"
else
  for layer in "${LAYERS[@]}"; do
    push_layer "$layer"
  done
fi

echo "Done."
