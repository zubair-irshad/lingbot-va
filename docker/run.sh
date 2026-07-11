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

# Create the writable dirs (fail loudly if we truly can't — e.g. no perms).
for d in "${DATASET_DIR}" "${OUTPUT_DIR}"; do
    mkdir -p "${d}" || { echo "ERROR: cannot create ${d} (check permissions)"; exit 1; }
done
# Resolve to absolute paths WITHOUT requiring cd (realpath -m handles non-existing too).
abspath() { realpath -m "$1" 2>/dev/null || python3 -c "import os,sys;print(os.path.abspath(sys.argv[1]))" "$1"; }
DATASET_DIR="$(abspath "${DATASET_DIR}")"
OUTPUT_DIR="$(abspath "${OUTPUT_DIR}")"
CKPT_DIR="$(abspath "${CKPT_DIR}")"
DROID_DIR="$(abspath "${DROID_DIR}")"

# Bind DATASET_DIR / OUTPUT_DIR / CKPT_DIR at their own absolute paths (so e.g.
# --out /datasets/zubair/droid_lerobot works verbatim), plus the default in-repo
# location for backwards compatibility.
# Host scratch root (default /datasets), mounted at its own path if it exists — so
# /datasets/zubair/... is always reachable in the container even when DATASET_DIR /
# OUTPUT_DIR weren't exported. Set SCRATCH_DIR="" to disable.
SCRATCH_DIR="${SCRATCH_DIR-/datasets}"
[ -n "${SCRATCH_DIR}" ] && [ -d "${SCRATCH_DIR}" ] && SCRATCH_DIR="$(abspath "${SCRATCH_DIR}")" || SCRATCH_DIR=""

# under_scratch <path> -> true if <path> is inside SCRATCH_DIR (already covered by its mount)
under_scratch() { [ -n "${SCRATCH_DIR}" ] && case "$1/" in "${SCRATCH_DIR}/"*) return 0;; esac; return 1; }

MOUNTS=(-v "${CKPT_DIR}:/workspace/lingbot-va/checkpoints")
[ -n "${SCRATCH_DIR}" ] && MOUNTS+=(-v "${SCRATCH_DIR}:${SCRATCH_DIR}")
# Self-mount DATASET_DIR/OUTPUT_DIR only if NOT already covered by the scratch mount
# (avoids overlapping/nested binds).
under_scratch "${DATASET_DIR}" || MOUNTS+=(-v "${DATASET_DIR}:${DATASET_DIR}")
under_scratch "${OUTPUT_DIR}"  || MOUNTS+=(-v "${OUTPUT_DIR}:${OUTPUT_DIR}")
# Back-compat: also expose the dataset at the in-repo default path.
MOUNTS+=(-v "${DATASET_DIR}:/workspace/lingbot-va/data/droid_lerobot")
[ -d "${DROID_DIR}" ] && MOUNTS+=(-v "${DROID_DIR}:/workspace/lingbot-va/1.0.1:ro")
[ "${MOUNT_CODE}" = "1" ] && MOUNTS+=(-v "${REPO_ROOT}:/workspace/lingbot-va")

# Default command: full-speed 8-GPU training (no CPU offload) with wandb.
DEFAULT_CMD="bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true $*"
CMD="${CMD:-${DEFAULT_CMD}}"

# Run as the host user so files written to bind-mounts (checkpoints, datasets) are
# owned by YOU, not root — so they're accessible outside the container. Set
# RUN_AS_ROOT=1 to run as root instead (e.g. if you need apt/pip install inside).
USER_ARGS=()
if [ "${RUN_AS_ROOT:-0}" != "1" ]; then
    USER_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache)
fi

exec docker run --rm -it \
    --gpus "${GPUS}" \
    --shm-size=64g --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${USER_ARGS[@]}" \
    -e NGPU="${NGPU}" \
    -e CONFIG_NAME="${CONFIG_NAME}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}" \
    -e WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-lingbot-va-droid}" \
    -e WANDB_RUN_NAME="${WANDB_RUN_NAME:-}" \
    -e LINGBOT_DATASET="${DATASET_DIR}" \
    -e TOKENIZERS_PARALLELISM=false \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${MOUNTS[@]}" \
    "${IMAGE}" \
    ${CMD}
