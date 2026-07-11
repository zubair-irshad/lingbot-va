#!/usr/bin/env python3
"""Action-conditioned video rollout: predict video FROM ground-truth DROID actions.

This is the inference counterpart to the action-conditioned post-training. It loads
a (fine-tuned) LingBot-VA checkpoint and, for each chunk, injects the scene's REAL
actions as the clean action condition (via the server's KV-cache path, which already
feeds `obs['state']` through `preprocess_action`) and then samples ONLY the video.
The decoded video should track the real episode when the model has overfit the scene.

It reuses `VA_Server` from `wan_va/wan_va_server.py` wholesale — we only drive it with
real observations + GT actions instead of the model's own predicted actions.

Usage:
    python droid_helpers/run_action_conditioned_infer.py \
        --scene "1.0.1/AUTOLab/success/2023-07-14/Fri_Jul_14_16:20:36_2023" \
        --model outputs/droid_train/checkpoints/checkpoint_step_40 \
        --config droid --num-chunks 5 --out outputs/droid_action_cond

Note: pass --model pointing at a dir that contains a `transformer/` subdir. VAE /
text_encoder / tokenizer are taken from the base checkpoint via the config's
`wan22_pretrained_model_name_or_path` (fine-tuning only updates the transformer),
so use --base to point at the base if the fine-tuned dir lacks those subdirs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wan_va"))

CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
ACTION_DIM = 30


def load_metadata(scene_dir):
    m = glob.glob(os.path.join(scene_dir, "metadata_*.json"))
    with open(m[0]) as f:
        return json.load(f)


def resolve_mp4(scene_dir, serial):
    d = os.path.join(scene_dir, "recordings", "MP4")
    direct = os.path.join(d, f"{serial}.mp4")
    if os.path.exists(direct):
        return direct
    return [x for x in glob.glob(os.path.join(d, f"*{serial}*.mp4")) if "stereo" not in x][0]


def read_frames(mp4, ids, h, w):
    cap = cv2.VideoCapture(mp4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for fi in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(fi, total - 1)))
        ok, fr = cap.read()
        if not ok:
            fr = out[-1] if out else np.zeros((h, w, 3), np.uint8)
        out.append(cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out).astype(np.uint8)


def load_gt_actions(scene_dir, action_per_frame):
    """Return raw actions arranged as [action_dim, chunk_frames, action_per_frame]
    per chunk, matching what the server's preprocess_action expects (C, F, H)."""
    with h5py.File(os.path.join(scene_dir, "trajectory.h5"), "r") as h5:
        cart = np.asarray(h5["action"]["cartesian_position"][:], dtype=np.float32)
        grip = np.asarray(h5["action"]["gripper_position"][:], dtype=np.float32).reshape(-1, 1)
    T = cart.shape[0]
    a = np.zeros((T, ACTION_DIM), dtype=np.float32)
    a[:, 0:6] = cart
    a[:, 28:29] = grip
    return a, T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--model", required=True, help="dir with fine-tuned transformer/")
    p.add_argument("--base", default="checkpoints/lingbot-va-base",
                   help="base ckpt for vae/text_encoder/tokenizer if --model lacks them")
    p.add_argument("--config", default="droid")
    p.add_argument("--num-chunks", type=int, default=5)
    p.add_argument("--out", default="outputs/droid_action_cond")
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")

    from configs import VA_CONFIGS
    from distributed.util import init_distributed
    from wan_va_server import VA_Server
    from utils import init_logger
    init_logger()

    cfg = VA_CONFIGS[args.config]
    # If the fine-tuned dir has only transformer/, symlink base's vae/text_encoder/
    # tokenizer next to it so from_pretrained resolves them.
    model_dir = os.path.abspath(args.model)
    for sub in ("vae", "text_encoder", "tokenizer"):
        tgt = os.path.join(model_dir, sub)
        if not os.path.exists(tgt):
            src = os.path.join(os.path.abspath(args.base), sub)
            os.symlink(src, tgt)
            print(f"[link] {tgt} -> {src}")
    cfg.wan22_pretrained_model_name_or_path = model_dir
    cfg.enable_offload = True

    init_distributed(1, 0, 0)
    cfg.rank = 0; cfg.local_rank = 0; cfg.world_size = 1
    cfg.save_root = os.path.abspath(args.out)
    os.makedirs(cfg.save_root, exist_ok=True)

    meta = load_metadata(args.scene)
    task = meta.get("current_task", "")
    ext1, wrist = str(meta["ext1_cam_serial"]), str(meta["wrist_cam_serial"])
    serial_for_key = {CAM_KEYS[0]: ext1, CAM_KEYS[1]: wrist, CAM_KEYS[2]: wrist}

    gt_action, T = load_gt_actions(args.scene, cfg.action_per_frame)
    stride = max(1, round(60 / args.fps))
    fcs = cfg.frame_chunk_size
    apf = cfg.action_per_frame

    server = VA_Server(cfg)
    server.video_processor = VideoProcessor(vae_scale_factor=1)
    server._reset(prompt=task)

    # Frame 0 observations for each cam (resized to config h/w).
    init_obs = {k: read_frames(resolve_mp4(args.scene, s), [0], cfg.height, cfg.width)[0]
                for k, s in serial_for_key.items()}
    obs = {"obs": [init_obs]}

    pred_latents = []
    for c in range(args.num_chunks):
        frame_st = c * fcs
        actions, latents = server._infer(obs, frame_st_id=frame_st)
        pred_latents.append(latents)

        # Build the GT action block for this chunk: [action_dim, fcs, apf] and feed
        # it (plus the next real observation) to the KV-cache path as the clean
        # action condition for the following chunk.
        raw0 = frame_st * stride
        idxs = [min(raw0 + i, T - 1) for i in range(fcs * apf)]
        chunk_actions = gt_action[idxs].reshape(fcs, apf, ACTION_DIM).transpose(2, 0, 1)  # C,F,H

        next_frame = (c + 1) * fcs
        next_raw = [min(next_frame * stride + i * stride, T - 1) for i in range(fcs)]
        key_frames = []
        for rf in next_raw:
            key_frames.append({k: read_frames(resolve_mp4(args.scene, s), [rf], cfg.height, cfg.width)[0]
                               for k, s in serial_for_key.items()})
        server.infer(dict(obs=key_frames, compute_kv_cache=True, imagine=False, state=chunk_actions))

    server.transformer.clear_cache(server.cache_name)
    server.streaming_vae.clear_cache()
    del server.transformer, server.text_encoder
    import gc; gc.collect()
    torch.cuda.empty_cache()
    if cfg.enable_offload:
        server.vae = server.vae.to(server.device).to(server.dtype)

    # Decode chunk-by-chunk (the full concat overflows the shared GPU at conv3d).
    frames = []
    for lat in pred_latents:
        with torch.no_grad():
            v = server.decode_one_video(lat, "np")[0]
        frames.extend(list(v))
        torch.cuda.empty_cache()
    video = frames
    out_mp4 = os.path.join(cfg.save_root, "action_conditioned.mp4")
    export_to_video(video, out_mp4, fps=10)
    print(f"[done] action-conditioned rollout -> {out_mp4}  (task: {task!r})")


if __name__ == "__main__":
    main()
