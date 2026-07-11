#!/usr/bin/env python3
"""Headless CUT3R inference on DROID scenes (no viser server).

Subsamples frames from a DROID camera MP4 (or an image folder), runs the
recurrent CUT3R forward pass once, and dumps per-frame depth / confidence /
camera pose plus a fused point-cloud PLY and a JSON sanity summary.

Reuses prepare_input / prepare_output from demo.py so behaviour matches the
official demo exactly, minus the blocking PointCloudViewer.

Example:
    python droid_helpers/run_cut3r_droid.py \
        --model_path src/cut3r_512_dpt_4_64.pth --size 512 --num_frames 24 \
        --seq_path "1.0.1/IRIS/success/2023-06-02/Fri_Jun__2_11:21:20_2023/recordings/MP4/29838012.mp4" \
        --output_dir droid_out/IRIS
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

# Repo root on path so `add_ckpt_path` and `demo` import cleanly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Headless CUT3R on a DROID scene.")
    p.add_argument("--model_path", default="src/cut3r_512_dpt_4_64.pth")
    p.add_argument("--seq_path", required=True, help="MP4 file or image folder.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--num_frames", type=int, default=24,
                   help="Uniformly subsample this many frames from the video.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis_threshold", type=float, default=1.5,
                   help="Confidence percentile-free threshold used only for the "
                        "point count reported / exported to PLY.")
    return p.parse_args()


def extract_frames(video_path: str, num_frames: int, dst: str) -> list[str]:
    """Uniformly sample `num_frames` frames from a video into `dst`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), num=min(num_frames, total)).astype(int)
    idxs = sorted(set(idxs.tolist()))
    paths = []
    for out_i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            break
        fp = os.path.join(dst, f"frame_{out_i:04d}.jpg")
        cv2.imwrite(fp, frame)
        paths.append(fp)
    cap.release()
    return paths


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray):
    """Minimal binary-free PLY writer. pts/cols: (N,3) float / uint8."""
    n = pts.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")


def main():
    args = parse_args()
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    add_path_to_dust3r(args.model_path)

    # Delayed imports (need ckpt dir on path first).
    from demo import prepare_input, prepare_output
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- gather frame paths -------------------------------------------------
    tmp = None
    if os.path.isdir(args.seq_path):
        import glob
        img_paths = sorted(glob.glob(os.path.join(args.seq_path, "*")))
        step = max(1, len(img_paths) // args.num_frames)
        img_paths = img_paths[::step][: args.num_frames]
    else:
        tmp = tempfile.mkdtemp()
        img_paths = extract_frames(args.seq_path, args.num_frames, tmp)
    if not img_paths:
        print(f"no frames from {args.seq_path}")
        return
    print(f"[droid] {len(img_paths)} frames from {args.seq_path}")

    # ---- build views --------------------------------------------------------
    views = prepare_input(img_paths, [True] * len(img_paths), args.size,
                           revisit=1, update=True)

    print(f"[droid] loading model {args.model_path} ...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    outputs, state_args = inference(views, model, device)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    print(f"[droid] inference {dt:.2f}s total, {dt/len(views):.3f}s/frame")

    # prepare_output also writes depth/conf/color/camera to output_dir.
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, revisit=1, use_pose=True)

    # ---- sanity stats + fused PLY ------------------------------------------
    all_pts, all_cols = [], []
    per_frame = []
    for i, (pw, col, cf) in enumerate(zip(pts3ds_other, colors, conf)):
        pw = pw.reshape(-1, 3).cpu().numpy()
        col = (col.reshape(-1, 3).cpu().numpy() * 255).astype(np.uint8)
        cf = cf.reshape(-1).cpu().numpy()
        mask = cf > args.vis_threshold
        depth = pw[:, 2]
        per_frame.append({
            "frame": i,
            "conf_mean": float(cf.mean()),
            "conf_kept_frac": float(mask.mean()),
            "depth_min": float(np.nanmin(depth)),
            "depth_med": float(np.nanmedian(depth)),
            "depth_max": float(np.nanmax(depth)),
        })
        all_pts.append(pw[mask])
        all_cols.append(col[mask])

    fused_pts = np.concatenate(all_pts, 0)
    fused_cols = np.concatenate(all_cols, 0)
    # subsample PLY to keep it light
    if fused_pts.shape[0] > 400_000:
        sel = np.linspace(0, fused_pts.shape[0] - 1, 400_000).astype(int)
        fused_pts, fused_cols = fused_pts[sel], fused_cols[sel]
    ply_path = os.path.join(args.output_dir, "fused_pointcloud.ply")
    write_ply(ply_path, fused_pts, fused_cols)

    # camera trajectory extent (translation of recovered c2w poses)
    t = cam_dict["t"]  # (B,3)
    traj_extent = (t.max(0) - t.min(0)).tolist()

    summary = {
        "seq_path": args.seq_path,
        "num_frames": len(views),
        "size": args.size,
        "inference_total_s": round(dt, 3),
        "inference_per_frame_s": round(dt / len(views), 4),
        "focal_mean_px": float(np.mean(cam_dict["focal"])),
        "camera_traj_extent_xyz": [round(v, 4) for v in traj_extent],
        "fused_points_kept": int(fused_pts.shape[0]),
        "per_frame": per_frame,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[droid] wrote {ply_path} ({fused_pts.shape[0]} pts) and summary.json")
    print(f"[droid] focal~{summary['focal_mean_px']:.1f}px  "
          f"traj extent(m) {summary['camera_traj_extent_xyz']}")

    if tmp:
        import shutil
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
