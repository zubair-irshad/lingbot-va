# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""DROID single-arm Franka config.

DROID episodes are single-arm Franka (FR3) with 3 cameras: two exterior ZED cams
and one wrist ZED cam. We reuse the released Franka action layout (action_dim=30,
EEF 6D + gripper packed into the first 7 used channels, mirrored block for the
second arm which is unused/zero-padded here) so the released `lingbot-va-base`
checkpoint can be run zero-shot without touching the action head.

Camera mapping (decided): DROID exterior cam1 -> cam_high, DROID wrist cam ->
cam_left_wrist AND cam_right_wrist (duplicated) to keep the 3-cam Franka layout.
"""
import os
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

# Repo root (…/wan_va/configs/ -> repo root). Keeps paths portable across machines
# (dev box, DGX container at /workspace/lingbot-va). Override with LINGBOT_CKPT.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

va_droid_cfg = EasyDict(__name__='Config: VA droid')
va_droid_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

# Base checkpoint (bundles transformer/vae/text_encoder/tokenizer). Defaults to
# <repo>/checkpoints/lingbot-va-base; override with env LINGBOT_CKPT.
va_droid_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_CKPT", os.path.join(_REPO_ROOT, "checkpoints", "lingbot-va-base"))

va_droid_cfg.attn_window = 30
va_droid_cfg.frame_chunk_size = 4
va_droid_cfg.env_type = 'none'

# Single-GPU inference: offload VAE + text encoder to CPU to fit a 14B model
# (~18GB VRAM for i2av per the README).
va_droid_cfg.enable_offload = True

va_droid_cfg.height = 224
va_droid_cfg.width = 320
va_droid_cfg.action_dim = 30
va_droid_cfg.action_per_frame = 20
va_droid_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_droid_cfg.guidance_scale = 5
va_droid_cfg.action_guidance_scale = 1

va_droid_cfg.num_inference_steps = 5
va_droid_cfg.video_exec_step = -1
va_droid_cfg.action_num_inference_steps = 10

va_droid_cfg.snr_shift = 5.0
va_droid_cfg.action_snr_shift = 1.0

# Same channel packing as the Franka config: EEF (0:7) + gripper (28) for the
# "left"/primary arm, then the mirrored (7:14)+(29) block for the (unused) second arm.
va_droid_cfg.used_action_channel_ids = list(range(0, 7)) + list(range(
    28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [len(va_droid_cfg.used_action_channel_ids)
                                   ] * va_droid_cfg.action_dim
for i, j in enumerate(va_droid_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_droid_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_droid_cfg.action_norm_method = 'quantiles'
# Reuse the Franka normalization statistics (same EEF+gripper layout). These get
# refined from data during post-training; for zero-shot they are a reasonable prior.
va_droid_cfg.norm_stat = {
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
