# DGX (8×A100) Docker post-training for LingBot-VA on DROID

Runs the single-scene (or multi-scene) DROID post-training with **forward-inverse
dynamics consistency** (`droid_train` config) inside a container, with wandb logging.
On the DGX, GPUs are dedicated so FSDP CPU-offload is turned **off** for full speed.

There are two ways to build/run: **plain `docker` scripts** (recommended — no
docker-compose needed, works on the DGX) or **docker compose** (if you have the v2
plugin). The scripts are the simplest path.

## 1. Build

```bash
# from the repo root
bash docker/build.sh                 # ~cuda12.6 base + torch cu126 + flash-attn compile
MAX_JOBS=16 bash docker/build.sh     # faster flash-attn compile on a big-RAM DGX
```

(compose alternative: `cd docker && docker compose build`)

## 2. Configure

```bash
cp docker/.env.example docker/.env
# edit docker/.env: WANDB_API_KEY / WANDB_TEAM_NAME / WANDB_PROJECT, and the host paths:
#   DATASET_DIR  -> your data/droid_lerobot (built by build_droid_lerobot.py)
#   CKPT_DIR     -> dir containing lingbot-va-base/
#   OUTPUT_DIR   -> where checkpoints/logs are written
# (defaults resolve to <repo>/data/droid_lerobot, <repo>/checkpoints, <repo>/outputs)
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

Run 2a/2b inside the container via `CMD=bash bash docker/run.sh`. The raw DROID
episodes are mounted read-only from the host `DROID_DIR` (default `<repo>/1.0.1`) to
`/workspace/lingbot-va/1.0.1`, and the dataset you build is written to the mounted
`DATASET_DIR`, so it persists on the host for the training run. (If `1.0.1/` isn't at
the repo root on the host, set `DROID_DIR=/abs/path/to/1.0.1` before `docker/run.sh`.)

## 3. Train (8 GPUs, FSDP, no offload, wandb on)

```bash
bash docker/run.sh                          # default: 8-GPU FSDP, no offload, wandb on
```

Override any config key as trailing `key=value` args, e.g. a longer run with a
custom loss balance:

```bash
bash docker/run.sh num_steps=5000 forward_dynamics_weight=1.0 \
     inverse_dynamics_weight=1.0 batch_size=1 gradient_accumulation_steps=4
```

Drop into an interactive shell in the container (to build the dataset, debug, etc.):

```bash
CMD=bash bash docker/run.sh
```

Pick specific GPUs / fewer GPUs:

```bash
GPUS='"device=0,1,2,3"' NGPU=4 bash docker/run.sh
```

(compose alternative: `cd docker && docker compose run --rm posttrain`)

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
