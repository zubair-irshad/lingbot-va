#!/usr/bin/bash
# Run LingBot-VA DROID post-training in the container with plain `docker run`
# (no docker-compose needed). Reads docker/.env for wandb creds + paths.
#
#   bash docker/run.sh                       # 8-GPU FSDP, no offload, wandb on
#   bash docker/run.sh num_steps=5000        # extra key=value config overrides
#   CMD=bash bash docker/run.sh              # drop into an interactive shell instead
#
# Env (or docker/.env): WANDB_API_KEY, WANDB_TEAM_NAME, WANDB_PROJECT,
#   NGPU (default 8), CONFIG_NAME (default droid_train),
#   DATASET_DIR, CKPT_DIR, OUTPUT_DIR (host paths bind-mounted into the container),
#   GPUS (default "all"; e.g. '"device=0,1"'), IMAGE.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load docker/.env if present.
if [ -f "${REPO_ROOT}/docker/.env" ]; then
    set -a; . "${REPO_ROOT}/docker/.env"; set +a
fi

IMAGE="${IMAGE:-lingbot-va-droid:latest}"
NGPU="${NGPU:-8}"
CONFIG_NAME="${CONFIG_NAME:-droid_train}"
GPUS="${GPUS:-all}"
DATASET_DIR="$(cd "${DATASET_DIR:-${REPO_ROOT}/data/droid_lerobot}" 2>/dev/null && pwd || echo "${REPO_ROOT}/data/droid_lerobot")"
CKPT_DIR="$(cd "${CKPT_DIR:-${REPO_ROOT}/checkpoints}" 2>/dev/null && pwd || echo "${REPO_ROOT}/checkpoints")"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "${OUTPUT_DIR}"

# Default command: full-speed 8-GPU training (no CPU offload) with wandb.
DEFAULT_CMD="bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true $*"
CMD="${CMD:-${DEFAULT_CMD}}"

exec docker run --rm -it \
    --gpus "${GPUS}" \
    --shm-size=64g --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e NGPU="${NGPU}" \
    -e CONFIG_NAME="${CONFIG_NAME}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}" \
    -e WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-droid}" \
    -e WANDB_RUN_NAME="${WANDB_RUN_NAME:-}" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -v "${DATASET_DIR}:/workspace/lingbot-va/data/droid_lerobot" \
    -v "${CKPT_DIR}:/workspace/lingbot-va/checkpoints" \
    -v "${OUTPUT_DIR}:/workspace/lingbot-va/outputs" \
    "${IMAGE}" \
    ${CMD}
