"""Stage 01 — load a DROID episode, verify calibration, checkpoint to disk."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from manipverse import ROOT
from manipverse.io import load_episode, verify_calibration
from _common import get_config, get_scene, save_pickle, scene_option, artifact


@click.command()
@scene_option
def main(scene_id: int) -> None:
    cfg = get_config()
    scene = get_scene(scene_id)
    droid_root = Path(cfg["paths"]["episodes"]) / "1.0.1"
    if not droid_root.is_absolute():
        droid_root = ROOT / droid_root
    calib_path = ROOT / "data" / "calibration" / "pnp_cam2base_multiview.json"

    # episode_id keys pnp_cam2base_multiview.json — default to basename of rel path
    episode_id = scene.get("episode_id") or scene["relative_path"].split("/")[-1]

    print(f"[01] loading episode {episode_id} ({scene['task']})")
    cameras, traj = load_episode(droid_root, episode_id, calib_path)

    verify_dir = artifact(scene_id, "01_load", ".").parent
    verify_calibration(cameras, traj, verify_dir)

    save_pickle({"cameras": cameras, "traj": traj, "scene": scene},
                artifact(scene_id, "01_load", "episode.pkl"))
    print(f"[01] done -> {artifact(scene_id, '01_load', 'episode.pkl')}")


if __name__ == "__main__":
    main()
