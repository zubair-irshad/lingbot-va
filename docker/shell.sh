#!/usr/bin/bash
# Drop into an interactive shell in the container (code + scratch dirs mounted, GPUs
# attached, wandb env passed through). From here you can `wandb login`, build the
# dataset, and launch training manually so you can watch it.
#
#   export DATASET_DIR=/datasets/zubair/droid_lerobot
#   export OUTPUT_DIR=/datasets/zubair/outputs
#   bash docker/shell.sh
#
# It's just `docker/run.sh` with CMD=bash.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD=bash exec bash "${REPO_ROOT}/docker/run.sh"
