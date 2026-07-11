#!/usr/bin/bash
# DROID post-training with forward-inverse dynamics consistency.
#
# Uses the existing FSDP trainer (wan_va/train.py) with CONFIG_NAME=droid_train,
# which sets dynamics_consistency=True (forward video-from-action + inverse
# action-from-video, shared params).
#
# Usage:
#   NGPU=1 bash script/run_droid_posttrain.sh                 # single freest GPU
#   NGPU=8 bash script/run_droid_posttrain.sh                 # on the A100 DGX
#   DATASET=/datasets/zubair/droid_lerobot NGPU=8 bash script/run_droid_posttrain.sh
# Any extra key=value pairs become --key value config overrides forwarded to train.py.
set -x
umask 022   # world-readable outputs (checkpoints accessible outside the container)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29502"}
CONFIG_NAME=${CONFIG_NAME:-"droid_train"}
SAVE_ROOT=${SAVE_ROOT:-"$REPO_ROOT/outputs/droid_train"}

# When single-GPU, pin the freest device unless the caller set CUDA_VISIBLE_DEVICES.
if [ "${NGPU}" = "1" ] && [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | nl -v0 -w1 -s' ' | sort -k2 -nr | head -n1 | awk '{print $1}')
    export CUDA_VISIBLE_DEVICES
fi
echo "NGPU=${NGPU} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

overrides=()
have_dataset_override=0
for kv in "$@"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    [ "${key}" = "dataset_path" ] && have_dataset_override=1
    overrides+=("--${key}" "${val}")
done

# Pass the dataset dir EXPLICITLY as a config override (don't rely on env
# propagating through torchrun subprocesses). Precedence: caller's dataset_path=
# arg > $DATASET > $LINGBOT_DATASET.
DATASET="${DATASET:-${LINGBOT_DATASET:-}}"
if [ "${have_dataset_override}" = "0" ] && [ -n "${DATASET}" ]; then
    overrides+=(--dataset_path "${DATASET}")
    echo "Using dataset_path=${DATASET}"
fi

export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train --config-name ${CONFIG_NAME} --save-root "${SAVE_ROOT}" "${overrides[@]}"
