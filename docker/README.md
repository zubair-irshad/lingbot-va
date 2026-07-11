# DGX (8×A100) Docker post-training for LingBot-VA on DROID

Runs the single-scene (or multi-scene) DROID post-training with **forward-inverse
dynamics consistency** (`droid_train` config) inside a container, with wandb logging.
On the DGX, GPUs are dedicated so FSDP CPU-offload is turned **off** for full speed.

## 1. Build

```bash
cd docker
docker compose build           # ~cuda12.6 base + torch cu126 + flash-attn compile
```

## 2. Configure

```bash
cp .env.example .env
# edit .env: WANDB_API_KEY / WANDB_TEAM_NAME / WANDB_PROJECT, and the host paths:
#   DATASET_DIR  -> your data/droid_lerobot (built by build_droid_lerobot.py)
#   CKPT_DIR     -> dir containing lingbot-va-base/
#   OUTPUT_DIR   -> where checkpoints/logs are written
```

### 2a. Download the base checkpoint (~23 GB) on the DGX

Put it in `<repo>/checkpoints/lingbot-va-base` (the default the configs look for; or set
`CKPT_DIR` in `.env` to wherever you download it):

```bash
pip install "huggingface_hub[cli]"    # if not already available
hf download robbyant/lingbot-va-base --local-dir checkpoints/lingbot-va-base
# (older CLI: huggingface-cli download robbyant/lingbot-va-base --local-dir checkpoints/lingbot-va-base)
```

### 2b. Get the DROID episodes + build the dataset

The raw DROID 1.0.1 episodes must be present under `<repo>/1.0.1/…`. Then build the
single-scene LeRobot latent dataset into `<repo>/data/droid_lerobot` (the default
`DATASET_DIR`):

```bash
python droid_helpers/build_droid_lerobot.py \
  --scene "1.0.1/<lab>/success/<date>/<time>" \
  --out data/droid_lerobot --model checkpoints/lingbot-va-base \
  --fps 12 --height 256 --width 256
```

You can run 2a/2b either on the host (then bind-mount) or inside the container
(`docker compose run --rm posttrain bash`, then run the commands there).

## 3. Train (8 GPUs, FSDP, no offload, wandb on)

```bash
docker compose run --rm posttrain
# equivalent to, inside the container:
#   NGPU=8 CONFIG_NAME=droid_train bash script/run_droid_posttrain.sh \
#       fsdp_cpu_offload=false enable_wandb=true
```

Override any config key as trailing `key=value` args, e.g. a longer run with a
custom loss balance:

```bash
docker compose run --rm posttrain \
  bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true \
       num_steps=5000 forward_dynamics_weight=1.0 inverse_dynamics_weight=1.0 \
       batch_size=1 gradient_accumulation_steps=4
```

Checkpoints land in `${OUTPUT_DIR}/droid_train/checkpoints/checkpoint_step_*/transformer/`
(diffusers format — load the same way as the base model).

## GPU prerequisites
The container needs the **NVIDIA Container Toolkit** on the host (standard on any DGX):
```bash
nvidia-ctk --version           # should exist on the DGX
```
The compose file reserves all GPUs via the `deploy.resources` block (the modern CDI path,
equivalent to `docker run --gpus all`). If your DGX uses the legacy runtime instead, run:
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $PWD/../data/droid_lerobot:/workspace/lingbot-va/data/droid_lerobot \
  -v $PWD/../checkpoints:/workspace/lingbot-va/checkpoints \
  -v $PWD/../outputs:/workspace/lingbot-va/outputs \
  -e WANDB_API_KEY=... -e WANDB_TEAM_NAME=... -e WANDB_PROJECT=lingbot-va-droid \
  lingbot-va-droid:latest \
  bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true
```
(The image itself is verified: torch cu126 + flash-attn 2.8.3 + lerobot all import correctly.)

## Notes
- `shm_size: 64gb` + `ipc: host` are set for dataloader workers and FSDP collectives.
- `MAX_JOBS=4` bounds the flash-attn compile (~15-20 min); raise it on the DGX to build faster.
- The image compiles flash-attn against the CUDA 12.6 base (matches torch cu126), so
  the `attn_mode="flex"` training path and `"flashattn"` inference both work.
- Set `WANDB_RUN_NAME` in `.env` to name the run (defaults to the config name).
- For a memory-constrained node, override `fsdp_cpu_offload=true` (as used on the dev A6000s).
