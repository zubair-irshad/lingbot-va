# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_droid_cfg import va_droid_cfg

va_droid_i2va_cfg = EasyDict(__name__='Config: VA droid i2va')
va_droid_i2va_cfg.update(va_droid_cfg)

# Set by prepare_droid_scene.py; points at a dir with the 3 {cam_key}.png init frames.
va_droid_i2va_cfg.input_img_path = 'example/droid'
va_droid_i2va_cfg.num_chunks_to_infer = 10
va_droid_i2va_cfg.prompt = 'stack two bowls on top of each other'
va_droid_i2va_cfg.infer_mode = 'i2va'
