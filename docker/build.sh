#!/usr/bin/bash
# Build the LingBot-VA DROID post-training image with plain `docker build`
# (no docker-compose needed). Run from anywhere.
#
#   bash docker/build.sh
#
# MAX_JOBS bounds the flash-attn compile; raise it on a big-RAM DGX to build faster:
#   MAX_JOBS=16 bash docker/build.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lingbot-va-droid:latest}"
MAX_JOBS="${MAX_JOBS:-4}"

echo "Building ${IMAGE} (MAX_JOBS=${MAX_JOBS}) ..."
docker build \
    -f "${REPO_ROOT}/docker/Dockerfile" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    -t "${IMAGE}" \
    "${REPO_ROOT}"
echo "Done: ${IMAGE}"
