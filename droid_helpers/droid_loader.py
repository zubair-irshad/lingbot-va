"""Step 1 — DROID episode loader + calibration verification.

- parses pnp_cam2base_multiview.json (two external cams per episode)
- opens each camera's .svo file via pyzed, grabs rectified left/right + intrinsics
- loads trajectory.h5 (joints, cartesian, gripper, actions)
- writes verify_calib_<serial>.png: EE from FK projected onto each view
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from PIL.Image import init
import cv2
import h5py
import numpy as np

from ..utils.geometry import parse_extrinsic, project_point
from ..utils.viz import draw_ee_projection, save


def _open_svo_first_frame(svo_path: str):
    """Return (left_rgb, right_rgb, intrinsics_dict) from a ZED .svo file."""
    import pyzed.sl as sl  # optional dep

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.optional_settings_path = "/usr/local/zed/settings"
    init.depth_mode = sl.DEPTH_MODE.NONE  # TRI Stereo handles depth in stage 02
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed for {svo_path}: {status}")

    left, right = sl.Mat(), sl.Mat()
    if zed.grab() != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED grab failed for {svo_path}")
    zed.retrieve_image(left, sl.VIEW.LEFT)
    zed.retrieve_image(right, sl.VIEW.RIGHT)

    calib = zed.get_camera_information().camera_configuration.calibration_parameters
    # ZED SDK returns baseline in millimeters — DROID's own collector converts
    # via `0.001 * calib.get_camera_baseline()`. Use that API, not the 4x4
    # stereo_transform (whose flattening index was previously wrong).

    K = {
        "fx": float(calib.left_cam.fx),
        "fy": float(calib.left_cam.fy),
        "cx": float(calib.left_cam.cx),
        "cy": float(calib.left_cam.cy),
        "baseline": 0.001 * float(calib.get_camera_baseline()),  # meters
        "width": int(calib.left_cam.image_size.width),
        "height": int(calib.left_cam.image_size.height),
    }
    left_rgb = left.get_data()[:, :, :3].copy()
    right_rgb = right.get_data()[:, :, :3].copy()
    zed.close()
    return left_rgb, right_rgb, K


def load_episode(droid_root: str | Path, episode_id: str, calib_path: str | Path) -> tuple[dict, dict]:
    droid_root = Path(droid_root)

    with open(calib_path) as f:
        calibs = json.load(f)
    if episode_id not in calibs:
        raise KeyError(f"episode_id {episode_id!r} not in {calib_path}")
    entry = calibs[episode_id]
    rel_path = entry["relative_path"]
    episode_dir = droid_root / rel_path

    cameras: dict[str, dict] = {}
    for key, val in entry.items():
        if key == "relative_path":
            continue
        serial = key
        T_c2b = parse_extrinsic(val)

        # DROID canonical layout (verified): recordings/SVO/{serial}.svo
        svo_path = episode_dir / "recordings" / "SVO" / f"{serial}.svo"
        if not svo_path.exists():
            # Fallback glob in case a repack renames the file
            matches = glob.glob(str(episode_dir / "recordings" / "SVO" /
                                    f"*{serial}*.svo"))
            if not matches:
                raise FileNotFoundError(f"no .svo for serial {serial} in {svo_path.parent}")
            svo_path = Path(matches[0])
        left, right, K = _open_svo_first_frame(str(svo_path))
        cameras[serial] = {
            "left_rgb": left,
            "right_rgb": right,
            "intrinsics": K,
            "T_cam2base": T_c2b,
        }

    # DROID trajectory.h5 schema (verified against rerun-io/python-example-droid-dataset):
    #   observation/robot_state/{joint_positions, gripper_position,
    #                            joint_velocities, joint_torques_computed,
    #                            motor_torques_measured}
    #   action/{cartesian_position, cartesian_velocity, gripper_position,
    #           gripper_velocity, target_cartesian_position,
    #           target_gripper_position, joint_velocity}   (all leaf Datasets)
    # EE-in-base per step comes from action/cartesian_position[i] as 6-vec
    # [tx,ty,tz,rx,ry,rz] with Euler XYZ. robot_state has NO cartesian_position.
    h5_path = episode_dir / "trajectory.h5"
    with h5py.File(h5_path, "r") as h5:
        obs = h5["observation"]["robot_state"]
        action_grp = h5["action"]
        actions = {k: action_grp[k][:] for k in action_grp.keys()
                   if isinstance(action_grp[k], h5py.Dataset)}
        robot_state = {k: obs[k][:] for k in obs.keys()
                       if isinstance(obs[k], h5py.Dataset)}
        traj = {
            "joint_positions":    robot_state["joint_positions"],
            "gripper_position":   robot_state["gripper_position"],
            # EE 6-vec in base frame, per step — authoritative source of EE pose.
            "cartesian_position": actions["cartesian_position"],
            "actions":            actions,
            "robot_state":        robot_state,
        }
    return cameras, traj


def verify_calibration(cameras: dict, traj: dict, save_dir: str | Path) -> dict:
    """Project the frame-0 EE from FK onto each camera. Green dot = gripper tip."""
    save_dir = Path(save_dir)
    report: dict[str, tuple[int, int]] = {}
    ee_base = traj["cartesian_position"][0, :3]
    for serial, cam in cameras.items():
        u, v, _ = project_point(ee_base, cam["T_cam2base"], cam["intrinsics"])
        vis = draw_ee_projection(cam["left_rgb"], (u, v), f"EE @ ({u},{v})")
        save(save_dir / f"verify_calib_{serial}.png", vis)
        report[serial] = (u, v)
        print(f"  cam {serial}: EE -> ({u}, {v})")
    return report
