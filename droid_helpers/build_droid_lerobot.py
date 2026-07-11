#!/usr/bin/env python3
"""Build a single-scene DROID LeRobot latent dataset for LingBot-VA post-training.

Pipeline (matches README "Custom Dataset Preparation" + the contract in
`wan_va/dataset/lerobot_latent_dataset.py`):

  1. Read the DROID episode (`trajectory.h5` + 3 MP4 cams + `metadata_*.json`).
  2. Build a 30-dim action per raw frame: EEF cartesian (6D) + gripper (1D) packed
     into the primary-arm slots (channels 0..6 + 28), the rest zero — this is the
     RAW parquet action; the dataset loader later re-indexes via
     inverse_used_action_channel_ids and normalizes with the config quantiles.
  3. Create a LeRobot v2.1 dataset (via LeRobotDataset.create + add_frame +
     save_episode) with the 3 cam videos resized to ~256x256, an `action` column,
     and an `observation.state` column.
  4. Inject `action_config` into meta/episodes.jsonl (single full-episode segment).
  5. Extract VAE latents per cam (streaming Wan2.2 VAE, same normalization as the
     server's `_encode_obs`) + the T5 text embedding, and save
     `latents/chunk-000/<cam_key>/episode_000000_<start>_<end>.pth` with exactly
     the keys the loader reads.
  6. Write empty_emb.pt alongside (via make_empty_emb) if requested.

The camera mapping mirrors prepare_droid_scene.py:
  ext1 -> cam_high ; wrist -> cam_left_wrist AND cam_right_wrist (duplicated).

Usage:
    python droid_helpers/build_droid_lerobot.py \
        --scene "1.0.1/AUTOLab/success/2023-07-14/Fri_Jul_14_16:20:36_2023" \
        --out    data/droid_lerobot \
        --model  checkpoints/lingbot-va-base \
        --fps 10 --height 256 --width 256
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
ACTION_DIM = 30  # matches va_droid_cfg.action_dim


# --------------------------------------------------------------------------- #
# DROID reading
# --------------------------------------------------------------------------- #
def load_metadata(scene_dir: str) -> dict:
    matches = glob.glob(os.path.join(scene_dir, "metadata_*.json"))
    if not matches:
        raise FileNotFoundError(f"no metadata_*.json in {scene_dir}")
    with open(matches[0]) as f:
        return json.load(f)


def resolve_mp4(scene_dir: str, serial: str) -> str:
    mp4_dir = os.path.join(scene_dir, "recordings", "MP4")
    direct = os.path.join(mp4_dir, f"{serial}.mp4")
    if os.path.exists(direct):
        return direct
    matches = [m for m in glob.glob(os.path.join(mp4_dir, f"*{serial}*.mp4"))
               if "stereo" not in os.path.basename(m)]
    if not matches:
        raise FileNotFoundError(f"no MP4 for serial {serial} in {mp4_dir}")
    return matches[0]


def read_video_frames(mp4_path: str, frame_ids, height: int, width: int) -> np.ndarray:
    """Return [N, H, W, 3] uint8 RGB frames at the given source indices, resized."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for fi in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(fi, total - 1)))
        ok, frame = cap.read()
        if not ok:
            # pad with the last good frame
            frame = out[-1][..., ::-1] if out else np.zeros((height, width, 3), np.uint8)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out).astype(np.uint8)


def load_droid_actions(scene_dir: str):
    """Return (raw_action [T,30], state [T,30], T). EEF 6D + gripper in slots 0..6+28."""
    h5_path = os.path.join(scene_dir, "trajectory.h5")
    with h5py.File(h5_path, "r") as h5:
        # action/cartesian_position: [T,6] tx,ty,tz,rx,ry,rz (Euler XYZ)
        cart = np.asarray(h5["action"]["cartesian_position"][:], dtype=np.float32)
        grip = np.asarray(h5["action"]["gripper_position"][:], dtype=np.float32).reshape(-1, 1)
        # observation robot state EEF for the state column
        obs_cart = np.asarray(
            h5["observation"]["robot_state"]["cartesian_position"][:], dtype=np.float32)
        obs_grip = np.asarray(
            h5["observation"]["robot_state"]["gripper_position"][:], dtype=np.float32).reshape(-1, 1)
    T = cart.shape[0]

    def pack(cart6, grip1):
        a = np.zeros((cart6.shape[0], ACTION_DIM), dtype=np.float32)
        a[:, 0:6] = cart6           # primary-arm EEF pose -> channels 0..5
        a[:, 28:29] = grip1         # primary-arm gripper -> channel 28
        return a

    return pack(cart, grip), pack(obs_cart, obs_grip), T


# --------------------------------------------------------------------------- #
# LeRobot dataset creation
# --------------------------------------------------------------------------- #
def build_lerobot(out_dir: str, scene_dir: str, meta: dict, ori_fps: int,
                  height: int, width: int, frame_ids: np.ndarray,
                  raw_action: np.ndarray, state: np.ndarray, T: int):
    """Write a LeRobot v2.1 dataset at RAW frame resolution.

    The latent dataset loader reads the parquet `action` column across the full
    raw frame range [start_frame, end_frame) and re-derives the latent frames via
    `frame_ids`, so LeRobot must contain ALL T raw frames (not the subsampled
    ones). Returns the SUBSAMPLED cam videos (indexed by frame_ids) for latent
    extraction.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobotDataset.create requires the root to NOT exist yet.
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    task = meta.get("current_task", "manipulation task")

    features = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    for key in CAM_KEYS:
        features[key] = {"dtype": "video", "shape": (height, width, 3),
                         "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(
        repo_id="droid/scene0",
        fps=ori_fps,
        features=features,
        root=out_dir,
        robot_type="franka",
        use_videos=True,
        video_backend="pyav",
    )

    # Post-training uses the 3 DISTINCT DROID cameras (duplicating the wrist cam
    # looked poor). Layout: external1 -> cam_high, wrist -> cam_left_wrist,
    # external2 -> cam_right_wrist.
    ext1 = str(meta["ext1_cam_serial"])
    ext2 = str(meta["ext2_cam_serial"])
    wrist = str(meta["wrist_cam_serial"])
    serial_for_key = {CAM_KEYS[0]: ext1, CAM_KEYS[1]: wrist, CAM_KEYS[2]: ext2}

    # Read ALL raw frames per cam (resized). The parquet needs one row per raw frame.
    all_ids = np.arange(T, dtype=int)
    cam_full = {key: read_video_frames(resolve_mp4(scene_dir, serial), all_ids, height, width)
                for key, serial in serial_for_key.items()}

    for i in range(T):
        frame = {
            "action": raw_action[i].astype(np.float32),
            "observation.state": state[i].astype(np.float32),
        }
        for key in CAM_KEYS:
            frame[key] = cam_full[key][i]
        ds.add_frame(frame, task=task)
    ds.save_episode()

    # Subsampled videos (indexed by frame_ids) for VAE latent extraction.
    cam_videos = {key: cam_full[key][frame_ids] for key in CAM_KEYS}
    return ds, task, cam_videos


def inject_action_config(out_dir: str, num_frames: int, task: str):
    """Add an `action_config` field (single full-episode segment) to episodes.jsonl."""
    ep_path = os.path.join(out_dir, "meta", "episodes.jsonl")
    lines = []
    with open(ep_path) as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    for ep in lines:
        ep["action_config"] = [{
            "start_frame": 0,
            "end_frame": int(ep.get("length", num_frames)),
            "action_text": task,
        }]
    with open(ep_path, "w") as f:
        for ep in lines:
            f.write(json.dumps(ep) + "\n")
    return int(lines[0].get("length", num_frames))


# --------------------------------------------------------------------------- #
# Latent extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_latents(out_dir: str, model_dir: str, cam_videos: dict, task: str,
                    frame_ids: np.ndarray, ori_fps: int, fps: int,
                    height: int, width: int, start_frame: int, end_frame: int,
                    device: str):
    from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer

    dtype = torch.bfloat16
    vae = load_vae(os.path.join(model_dir, "vae"), torch_dtype=dtype, torch_device=device)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device)
    latents_std = torch.tensor(vae.config.latents_std, device=device)

    def normalize(mu):
        m = latents_mean.view(1, -1, 1, 1, 1)
        s = (1.0 / latents_std).view(1, -1, 1, 1, 1)
        return ((mu.float() - m) * s).to(mu)

    # Text embedding (same truncate+pad as the server's _get_t5_prompt_embeds).
    tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer"))
    text_encoder = load_text_encoder(os.path.join(model_dir, "text_encoder"),
                                     torch_dtype=dtype, torch_device=device)
    max_seq_len = 512
    ti = tokenizer([task], padding="max_length", max_length=max_seq_len, truncation=True,
                   add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    seq_len = int(ti.attention_mask.gt(0).sum(dim=1).long()[0].item())
    text_emb = text_encoder(ti.input_ids.to(device), ti.attention_mask.to(device)).last_hidden_state[0]
    text_emb = text_emb[:seq_len]
    text_emb = torch.cat([text_emb, text_emb.new_zeros(max_seq_len - text_emb.size(0), text_emb.size(1))], dim=0)
    text_emb = text_emb.to(dtype=dtype, device="cpu")

    latent_dir_root = os.path.join(out_dir, "latents", "chunk-000")
    for key, vid in cam_videos.items():
        # vid: [N,H,W,3] uint8 -> [1,3,N,H,W] in [-1,1]. N = 4k+1 so the VAE's 4x
        # temporal compression yields (N-1)/4+1 latent frames.
        v = torch.from_numpy(vid).float().permute(3, 0, 1, 2).unsqueeze(0)
        v = v / 255.0 * 2.0 - 1.0
        mu = vae.encode(v.to(device).to(dtype)).latent_dist.mode()  # [1, C, f, h, w]
        mu = normalize(mu)
        latent = mu[0].permute(1, 2, 3, 0).contiguous()   # [f, h, w, C]
        lf, lh, lw, C = latent.shape
        latent_flat = latent.reshape(lf * lh * lw, C).to(torch.bfloat16).cpu()

        out = {
            "latent": latent_flat,
            "latent_num_frames": int(lf),
            "latent_height": int(lh),
            "latent_width": int(lw),
            "video_num_frames": int(vid.shape[0]),
            "video_height": int(height),
            "video_width": int(width),
            "text_emb": text_emb,
            "text": task,
            "frame_ids": [int(x) for x in frame_ids.tolist()],
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "fps": int(fps),
            "ori_fps": int(ori_fps),
        }
        d = os.path.join(latent_dir_root, key)
        os.makedirs(d, exist_ok=True)
        fpath = os.path.join(d, f"episode_000000_{start_frame}_{end_frame}.pth")
        torch.save(out, fpath)
        print(f"[latent] {key}: latent[{lf},{lh},{lw},{C}] video[{vid.shape[0]}f {height}x{width}] -> {fpath}")

    return text_emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="checkpoints/lingbot-va-base")
    # fps=12 with ori_fps=60 -> frame_stride=5 -> actions-per-latent-frame = 5*4 = 20,
    # matching va_droid_cfg.action_per_frame (required by the model's action stream).
    p.add_argument("--fps", type=int, default=12, help="target sampling fps")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--write-empty-emb", action="store_true", default=True)
    args = p.parse_args()

    scene_dir = args.scene.rstrip("/")
    out_dir = os.path.abspath(args.out)
    meta = load_metadata(scene_dir)
    ori_fps = 60  # DROID MP4s are 60fps
    raw_action, state, T = load_droid_actions(scene_dir)

    # Subsample frame indices at target fps. The loader requires (a) frame_ids[0]
    # >= start_frame (act_shift>=0) and (b) a constant stride == frame_ids[1]-[0].
    stride = max(1, round(ori_fps / args.fps))
    frame_ids = np.arange(0, T, stride, dtype=int)
    # loader: latent_frame_num = (len(frame_ids)-1)//4 + 1, and it needs the video
    # frame count divisible into 4-frame VAE temporal groups. Trim to (4k+1) frames.
    keep = ((len(frame_ids) - 1) // 4) * 4 + 1
    frame_ids = frame_ids[:keep]
    start_frame, end_frame = 0, T
    print(f"[build] scene={scene_dir}\n[build] task={meta.get('current_task')!r} T={T} "
          f"stride={stride} -> {len(frame_ids)} latent frames (fps~{args.fps})")

    ds, task, cam_videos = build_lerobot(out_dir, scene_dir, meta, ori_fps,
                                         args.height, args.width, frame_ids,
                                         raw_action, state, T)
    length = inject_action_config(out_dir, T, task)
    print(f"[build] LeRobot dataset written to {out_dir} (episode length={length})")

    extract_latents(out_dir, args.model, cam_videos, task, frame_ids, ori_fps,
                    args.fps, args.height, args.width, start_frame, end_frame, args.device)

    if args.write_empty_emb:
        from droid_helpers.make_empty_emb import encode_empty
        emb = encode_empty(args.model, 512, args.device, torch.bfloat16)
        torch.save(emb, os.path.join(out_dir, "empty_emb.pt"))
        print(f"[build] empty_emb.pt saved {tuple(emb.shape)}")

    print(f"[build] DONE. Set va_droid_train_cfg.dataset_path = {out_dir}")


if __name__ == "__main__":
    main()
