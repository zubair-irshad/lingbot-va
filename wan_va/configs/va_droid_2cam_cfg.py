# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""DROID 2-camera config: one external + one wrist camera.

DROID's native rig is 2 exterior + 1 wrist; the cleanest 2-cam view for LingBot-VA
is one exterior (cam_high) + the wrist (cam_left_wrist). The model is a shared
backbone that positions each camera's latents by RoPE grid ids, so the latent
width simply becomes width//16 * 2 instead of *3 — no weight surgery needed.

Runs zero-shot off the base checkpoint (`wan22_pretrained_model_name_or_path`).
"""
import os
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

va_droid_2cam_cfg = EasyDict(__name__='Config: VA droid 2cam')
va_droid_2cam_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_droid_2cam_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_CKPT", os.path.join(_REPO_ROOT, "checkpoints", "lingbot-va-base"))

va_droid_2cam_cfg.attn_window = 30
va_droid_2cam_cfg.frame_chunk_size = 4
va_droid_2cam_cfg.env_type = 'none'
va_droid_2cam_cfg.enable_offload = True

va_droid_2cam_cfg.height = 224
va_droid_2cam_cfg.width = 320
va_droid_2cam_cfg.action_dim = 30
va_droid_2cam_cfg.action_per_frame = 20
# One external + one wrist camera.
va_droid_2cam_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist'
]
va_droid_2cam_cfg.guidance_scale = 5
va_droid_2cam_cfg.action_guidance_scale = 1

va_droid_2cam_cfg.num_inference_steps = 5
va_droid_2cam_cfg.video_exec_step = -1
va_droid_2cam_cfg.action_num_inference_steps = 10

va_droid_2cam_cfg.snr_shift = 5.0
va_droid_2cam_cfg.action_snr_shift = 1.0

va_droid_2cam_cfg.used_action_channel_ids = list(range(0, 7)) + list(range(
    28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [len(va_droid_2cam_cfg.used_action_channel_ids)
                                   ] * va_droid_2cam_cfg.action_dim
for i, j in enumerate(va_droid_2cam_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_droid_2cam_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_droid_2cam_cfg.action_norm_method = 'quantiles'
va_droid_2cam_cfg.norm_stat = {
    "q01": [
        0.3051295876502991, -0.22647984325885773, 0.19957000017166138,
        -0.022680532187223434, -0.05553057789802551, -0.2693849802017212,
        -0.29341773986816405, 0.2935442328453064, -0.4431332051753998,
        0.21256473660469055, -0.7962440848350525, -0.40816226601600647,
        -0.28359392285346985, -0.44507765769958496
    ] + [0.] * 16,
    "q99": [
        0.7572150230407715, 0.47736290097236633, 0.6428080797195435,
        0.9835678935050964, 0.9927203059196472, 0.28041139245033264,
        0.47529348731040877, 0.7564866304397571, 0.04082797020673729,
        0.5355993628501885, 0.9976375699043274, 0.8973174452781656,
        0.6016915678977965, 0.5027598619461056
    ] + [0.] * 14 + [1.0, 1.0],
}
