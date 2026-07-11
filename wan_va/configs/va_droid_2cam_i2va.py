# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_droid_2cam_cfg import va_droid_2cam_cfg

va_droid_2cam_i2va_cfg = EasyDict(__name__='Config: VA droid 2cam i2va')
va_droid_2cam_i2va_cfg.update(va_droid_2cam_cfg)

va_droid_2cam_i2va_cfg.input_img_path = 'example/droid'
# Long-horizon rollout by default.
va_droid_2cam_i2va_cfg.num_chunks_to_infer = 40
va_droid_2cam_i2va_cfg.prompt = 'stack two bowls on top of each other'
va_droid_2cam_i2va_cfg.infer_mode = 'i2va'
