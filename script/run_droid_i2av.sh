#!/usr/bin/bash
# DROID zero-shot image->video+action (i2av) inference on a single GPU.
#
# Usage:
#   bash script/run_droid_i2av.sh [input_img_path=<dir>] [prompt="..."] [num_chunks_to_infer=N]
#
# Extra key=value pairs after the script name are forwarded as --key value to the
# server (currently: input_img_path, prompt, num_chunks_to_infer). Example:
#   bash script/run_droid_i2av.sh input_img_path=example/droid/autolab_bowls \
#        prompt="stack two bowls on top of each other"
set -x
umask 007

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate the project venv if present.
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
CONFIG_NAME=${CONFIG_NAME:-"droid_i2av"}
SAVE_ROOT=${SAVE_ROOT:-"$REPO_ROOT/outputs/droid_i2av"}

# Auto-pick the GPU with the most free memory (unless the caller pinned it).
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | nl -v0 -w1 -s' ' | sort -k2 -nr | head -n1 | awk '{print $1}')
fi
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Turn positional key=value args into --key value overrides (array preserves spaces).
overrides=()
for kv in "$@"; do
    key="${kv%%=*}"
    val="${kv#*=}"
    overrides+=("--${key}" "${val}")
done

export TOKENIZERS_PARALLELISM=false
PYTORCH_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.wan_va_server --config-name ${CONFIG_NAME} --save_root "${SAVE_ROOT}" "${overrides[@]}"
