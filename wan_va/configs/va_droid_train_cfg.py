# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""DROID single-scene post-training config (forward-inverse dynamics consistency).

Fine-tunes on one DROID scene with `dynamics_consistency=True`: both the video and
action streams stay noised in the single shared-parameter pass, so the model's
native forward path (video-from-action) and native inverse path (action-from-video,
via the existing `action_proj_out` head) are trained jointly. Shared parameters act
as an implicit anti-drift regularizer. No architecture change — only the action
conditioning / loss weighting. See train.py for the mechanics.
"""
from easydict import EasyDict
from .va_droid_cfg import va_droid_cfg, _REPO_ROOT
import os

va_droid_train_cfg = EasyDict(__name__='Config: VA droid train')
va_droid_train_cfg.update(va_droid_cfg)

# LeRobot latent dataset built by build_droid_lerobot.py. Defaults to
# <repo>/data/droid_lerobot; override with env LINGBOT_DATASET.
va_droid_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_DATASET", os.path.join(_REPO_ROOT, "data", "droid_lerobot"))
va_droid_train_cfg.empty_emb_path = os.path.join(va_droid_train_cfg.dataset_path, 'empty_emb.pt')
va_droid_train_cfg.enable_wandb = False
va_droid_train_cfg.load_worker = 4
va_droid_train_cfg.save_interval = 200
va_droid_train_cfg.gc_interval = 50
va_droid_train_cfg.cfg_prob = 0.1

# ---- Forward-inverse dynamics consistency (SC2) ----
# Keep BOTH the video and action streams noised in the single shared-parameter
# pass, jointly training:
#   * forward dynamics  (latent_loss):  render video FROM the clean-action condition
#   * inverse dynamics  (action_loss):  recover actions FROM the clean-video condition
# Because both modes share transformer weights, the forward mode is implicitly
# regularized to render frames from which the inverse mode can recover the commanded
# actions — an anti-drift regularizer a forward-only objective lacks. This supersedes
# the legacy forward-only `action_condition` path (auto-disabled when this is on).
va_droid_train_cfg.dynamics_consistency = True
va_droid_train_cfg.forward_dynamics_weight = 1.0
va_droid_train_cfg.inverse_dynamics_weight = 1.0

# Legacy forward-only knobs (kept for reference; inactive while dynamics_consistency=True).
va_droid_train_cfg.action_condition = False
va_droid_train_cfg.action_consistency_weight = 1.0
va_droid_train_cfg.action_cond_snr_floor = 0.02

# Fit the 14B full fine-tune on limited/shared GPU memory by offloading FSDP
# params + grads + optimizer state to CPU (slower per step, but runs on-box). On
# the 8xA100 DGX, override fsdp_cpu_offload=false for full-speed GPU-resident
# training. (Note: 8-bit optimizers are NOT compatible with FSDP2 DTensor sharding.)
va_droid_train_cfg.use_8bit_optimizer = False
va_droid_train_cfg.fsdp_cpu_offload = True

# ---- Training parameters (tuned for single-scene overfit) ----
va_droid_train_cfg.learning_rate = 1e-4
va_droid_train_cfg.beta1 = 0.9
va_droid_train_cfg.beta2 = 0.95
va_droid_train_cfg.weight_decay = 1e-1
va_droid_train_cfg.warmup_steps = 10
va_droid_train_cfg.batch_size = 1
va_droid_train_cfg.gradient_accumulation_steps = 4
va_droid_train_cfg.num_steps = 2000
