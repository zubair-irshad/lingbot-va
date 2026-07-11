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
# Writable locations. On a shared DGX, point these at your scratch space, e.g.
#   DATASET_DIR=/datasets/zubair/droid_lerobot  OUTPUT_DIR=/datasets/zubair/outputs
# They are mounted at their OWN absolute path inside the container so the paths you
# pass to build_droid_lerobot.py (--out) and the trainer resolve verbatim.
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/droid_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints}"
DROID_DIR="${DROID_DIR:-${REPO_ROOT}/1.0.1}"       # raw DROID episodes (read-only)
# Mount the live repo code over the image copy so host `git pull` takes effect with
# no rebuild. Set MOUNT_CODE=0 to use the code baked into the image instead.
MOUNT_CODE="${MOUNT_CODE:-1}"

mkdir -p "${OUTPUT_DIR}" "${DATASET_DIR}" 2>/dev/null || true

# Bind DATASET_DIR / OUTPUT_DIR / CKPT_DIR at their own absolute paths (so e.g.
# --out /datasets/zubair/droid_lerobot works verbatim), plus the default in-repo
# location for backwards compatibility.
MOUNTS=(
    -v "${CKPT_DIR}:/workspace/lingbot-va/checkpoints"
    -v "$(cd "${DATASET_DIR}" && pwd):$(cd "${DATASET_DIR}" && pwd)"
    -v "$(cd "${OUTPUT_DIR}" && pwd):$(cd "${OUTPUT_DIR}" && pwd)"
    -v "$(cd "${DATASET_DIR}" && pwd):/workspace/lingbot-va/data/droid_lerobot"
)
[ -d "${DROID_DIR}" ] && MOUNTS+=(-v "$(cd "${DROID_DIR}" && pwd):/workspace/lingbot-va/1.0.1:ro")
[ "${MOUNT_CODE}" = "1" ] && MOUNTS+=(-v "${REPO_ROOT}:/workspace/lingbot-va")

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
    -e LINGBOT_DATASET="$(cd "${DATASET_DIR}" && pwd)" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${MOUNTS[@]}" \
    "${IMAGE}" \
    ${CMD}
