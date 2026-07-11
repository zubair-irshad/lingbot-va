# DGX (8×A100) Docker post-training for LingBot-VA on DROID

Runs the single-scene (or multi-scene) DROID post-training with **forward-inverse
dynamics consistency** (`droid_train` config) inside a container, with wandb logging.
On the DGX, GPUs are dedicated so FSDP CPU-offload is turned **off** for full speed.

The recommended flow is: **build the image → open an interactive shell in the
container → `wandb login` → build the dataset → launch training yourself** (so you can
watch it live). No `docker-compose` is required.

---

## Step 0 — one-time: build the image (on the DGX)

```bash
cd ~/lingbot-va
git pull                                   # make sure you have the latest scripts

MAX_JOBS=16 bash docker/build.sh           # cuda12.6 + torch cu126 + flash-attn compile
                                           # (~10-15 min; MAX_JOBS speeds the flash-attn build)
```

`run.sh`/`shell.sh` **mount the live repo code** into the container (`MOUNT_CODE=1`, the
default), so after this initial build a plain `git pull` on the host is enough to pick up
code changes — **no rebuild needed** unless dependencies change. Set `MOUNT_CODE=0` to use
the code baked into the image instead.

## Step 1 — download the base checkpoint (~23 GB), once

Put it where the config looks (`<repo>/checkpoints/lingbot-va-base`), or anywhere and set
`CKPT_DIR`:

```bash
pip install "huggingface_hub[cli]"         # host or container, either is fine
hf download robbyant/lingbot-va-base --local-dir checkpoints/lingbot-va-base
```

## Step 2 — point at your scratch space + open a shell in the container

On the shared DGX, write datasets/outputs to your scratch (e.g. `/datasets/zubair`).
`shell.sh` creates those dirs, mounts them at their own absolute path inside the
container, attaches all GPUs, and passes wandb env through.

```bash
export DATASET_DIR=/datasets/zubair/droid_lerobot
export OUTPUT_DIR=/datasets/zubair/outputs
bash docker/shell.sh                       # <-- drops you into a shell in the container
```

You are now `root@<container>:/workspace/lingbot-va#`.

## Step 3 — inside the container: wandb login

```bash
wandb login                                # paste your API key when prompted
# (or non-interactively: export WANDB_API_KEY=... before docker/shell.sh)
```

## Step 4 — inside the container: build the dataset

The raw DROID episodes are mounted read-only at `./1.0.1`. Build the LeRobot latent
dataset into your scratch `--out` (a `lerobot/` subdir is created under it):

```bash
python droid_helpers/build_droid_lerobot.py \
  --scene "1.0.1/AUTOLab/success/2023-07-14/Fri_Jul_14_16:20:36_2023" \
  --out /datasets/zubair/droid_lerobot \
  --model checkpoints/lingbot-va-base \
  --fps 12 --height 256 --width 256
```

Notes:
- The builder **never deletes** anything. If `.../droid_lerobot/lerobot` already exists it
  errors — pass a fresh `--out` (e.g. `droid_lerobot_v2`) or remove that subdir yourself.
- fps=12 (source 60 fps) → 20 actions per latent frame, matching the model's action stream.

## Step 5 — inside the container: launch training (8 GPUs, watch it live)

```bash
NGPU=8 bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true
```

- `fsdp_cpu_offload=false` → full-speed GPU-resident training (the dev-box A6000 path used
  `true`; the DGX has the memory to keep everything on-GPU).
- `enable_wandb=true` → logs to the project/team from your env (`WANDB_PROJECT`,
  `WANDB_TEAM_NAME`); set `WANDB_RUN_NAME` to name the run.
- The trainer reads the dataset from `LINGBOT_DATASET` (set automatically to `DATASET_DIR`
  by `shell.sh`/`run.sh`), so it finds `/datasets/zubair/droid_lerobot` with no extra flags.

Override any config key as trailing `key=value` args, e.g. a longer run:

```bash
NGPU=8 bash script/run_droid_posttrain.sh fsdp_cpu_offload=false enable_wandb=true \
  num_steps=5000 forward_dynamics_weight=1.0 inverse_dynamics_weight=1.0 \
  batch_size=1 gradient_accumulation_steps=4
```

Checkpoints land in `${OUTPUT_DIR}/droid_train/checkpoints/checkpoint_step_*/transformer/`
(diffusers format — load the same way as the base model).

---

## One-shot alternative (non-interactive)

If you'd rather not use the shell and already ran `wandb login` (or set `WANDB_API_KEY`),
`run.sh` builds nothing but launches training directly:

```bash
export DATASET_DIR=/datasets/zubair/droid_lerobot OUTPUT_DIR=/datasets/zubair/outputs
bash docker/run.sh                                   # 8-GPU, no offload, wandb on
bash docker/run.sh num_steps=5000                    # with overrides
GPUS='"device=0,1,2,3"' NGPU=4 bash docker/run.sh    # fewer GPUs
```

`docker compose` alternative (only if the v2 plugin is installed):
`cd docker && docker compose build && docker compose run --rm posttrain`.

---

## Environment / knobs

Set via `export`, or persist in `docker/.env` (auto-loaded by `run.sh`/`shell.sh`;
copy `docker/.env.example`):

| Var | Default | Meaning |
|---|---|---|
| `DATASET_DIR` | `<repo>/data/droid_lerobot` | LeRobot latent dataset (your scratch on the DGX) |
| `OUTPUT_DIR` | `<repo>/outputs` | checkpoints + logs |
| `CKPT_DIR` | `<repo>/checkpoints` | dir containing `lingbot-va-base/` |
| `DROID_DIR` | `<repo>/1.0.1` | raw DROID episodes (mounted read-only) |
| `NGPU` | `8` | GPUs for FSDP |
| `GPUS` | `all` | which GPUs (`'"device=0,1"'` for a subset) |
| `MOUNT_CODE` | `1` | mount live repo code (host `git pull` applies, no rebuild) |
| `WANDB_API_KEY` / `WANDB_TEAM_NAME` / `WANDB_PROJECT` / `WANDB_RUN_NAME` | — | wandb config |
| `MAX_JOBS` | `4` | flash-attn compile parallelism (build-time; raise on DGX) |

## GPU prerequisites

The container needs the **NVIDIA Container Toolkit** on the host (standard on any DGX):

```bash
nvidia-ctk --version           # should exist on the DGX
```

`run.sh`/`shell.sh` use `--gpus`. If your DGX only has the legacy runtime, prepend
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` to a manual `docker run` (see git history
for the full form). The image itself is verified: torch cu126 + flash-attn 2.8.3 + lerobot
all import correctly.

## Notes
- `shm-size=64g` + `--ipc=host` are set for dataloader workers and FSDP collectives.
- flash-attn is compiled against the CUDA 12.6 base (matches torch cu126), so both the
  `attn_mode="flex"` training path and `"flashattn"` inference work.
- Memory-constrained node? Override `fsdp_cpu_offload=true` (as used on the dev A6000s).
