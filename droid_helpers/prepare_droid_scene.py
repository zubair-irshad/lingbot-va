#!/usr/bin/env python3
"""Prepare a DROID scene's init frames for LingBot-VA i2av inference.

Reads a DROID episode directory (the one containing `trajectory.h5`,
`metadata_*.json`, and `recordings/MP4/*.mp4`), extracts frame 0 of the three
cameras, and writes them as the three PNGs that the server's `load_init_obs()`
(`wan_va/wan_va_server.py`) expects, named after `obs_cam_keys` in the DROID config:

    observation.images.cam_high.png         <- exterior cam 1 (ext1)
    observation.images.cam_left_wrist.png   <- wrist cam
    observation.images.cam_right_wrist.png   <- wrist cam (duplicated)

Camera mapping decision (see plan): DROID has 2 exterior + 1 wrist cam; we map
ext1 -> cam_high and duplicate the wrist cam into both wrist slots to keep the
released 3-cam Franka layout that `lingbot-va-base` was trained with.

Prints the scene's task text (use it as the `prompt`).

Usage:
    python droid_helpers/prepare_droid_scene.py \
        --scene "1.0.1/AUTOLab/success/2023-07-14/Fri_Jul_14_16:20:36_2023" \
        --out   example/droid/autolab_bowls
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2


CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def load_metadata(scene_dir: str) -> dict:
    matches = glob.glob(os.path.join(scene_dir, "metadata_*.json"))
    if not matches:
        raise FileNotFoundError(f"no metadata_*.json in {scene_dir}")
    with open(matches[0]) as f:
        return json.load(f)


def first_frame(mp4_path: str):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {mp4_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame 0 from {mp4_path}")
    # OpenCV reads BGR; the server does Image.open(...).convert("RGB"), so store RGB.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resolve_mp4(scene_dir: str, serial: str) -> str:
    """MP4 for a camera serial (handles stereo suffix / naming variants)."""
    mp4_dir = os.path.join(scene_dir, "recordings", "MP4")
    direct = os.path.join(mp4_dir, f"{serial}.mp4")
    if os.path.exists(direct):
        return direct
    matches = [m for m in glob.glob(os.path.join(mp4_dir, f"*{serial}*.mp4"))
               if "stereo" not in os.path.basename(m)]
    if not matches:
        raise FileNotFoundError(f"no MP4 for serial {serial} in {mp4_dir}")
    return matches[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="DROID episode dir (has trajectory.h5)")
    p.add_argument("--out", required=True, help="output dir for the 3 init PNGs")
    p.add_argument("--distinct-cams", action="store_true",
                   help="Use the 2 external + 1 wrist cameras as 3 DISTINCT feeds "
                        "(ext1->cam_high, ext2->cam_left_wrist, wrist->cam_right_wrist) "
                        "instead of duplicating the single wrist cam.")
    p.add_argument("--two-cams", action="store_true",
                   help="2-camera mode: one external + one wrist "
                        "(ext1->cam_high, wrist->cam_left_wrist). Matches the "
                        "droid_2cam config.")
    args = p.parse_args()

    scene_dir = args.scene.rstrip("/")
    meta = load_metadata(scene_dir)
    task = meta.get("current_task", "")

    ext1 = str(meta["ext1_cam_serial"])
    ext2 = str(meta["ext2_cam_serial"])
    wrist = str(meta["wrist_cam_serial"])

    if args.two_cams:
        # One external + one wrist (matches va_droid_2cam_cfg.obs_cam_keys).
        serial_for_key = {
            CAM_KEYS[0]: ext1,   # exterior 1 -> cam_high
            CAM_KEYS[1]: wrist,  # wrist      -> cam_left_wrist
        }
    elif args.distinct_cams:
        # 2 external + 1 wrist as three distinct feeds.
        serial_for_key = {
            CAM_KEYS[0]: ext1,   # exterior 1 -> cam_high
            CAM_KEYS[1]: ext2,   # exterior 2 -> cam_left_wrist slot
            CAM_KEYS[2]: wrist,  # wrist      -> cam_right_wrist slot
        }
    else:
        # ext1 -> cam_high ; wrist -> both wrist slots (duplicated)
        serial_for_key = {
            CAM_KEYS[0]: ext1,
            CAM_KEYS[1]: wrist,
            CAM_KEYS[2]: wrist,
        }

    os.makedirs(args.out, exist_ok=True)
    for key, serial in serial_for_key.items():
        mp4 = resolve_mp4(scene_dir, serial)
        rgb = first_frame(mp4)
        out_png = os.path.join(args.out, f"{key}.png")
        # cv2.imwrite expects BGR
        cv2.imwrite(out_png, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[prep] {key} <- {os.path.basename(mp4)} ({rgb.shape[1]}x{rgb.shape[0]}) -> {out_png}")

    print(f"\n[prep] scene task (use as --prompt):\n  {task!r}")
    print(f"[prep] init frames ready in: {args.out}")
    print(f"[prep] run inference with:\n"
          f"  CONFIG_NAME=droid_i2av bash script/run_droid_i2av.sh "
          f"input_img_path={args.out} prompt=\"{task}\"")


if __name__ == "__main__":
    main()
