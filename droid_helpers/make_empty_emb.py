#!/usr/bin/env python3
"""Generate empty_emb.pt — the T5 embedding of the empty string "".

The LeRobot latent dataset (`wan_va/dataset/lerobot_latent_dataset.py`) loads this
tensor and substitutes it for the real text embedding with probability `cfg_prob`
(classifier-free guidance dropout during training). Shape must match the per-item
`text_emb` stored in the latent .pth files: [L, D] where D=4096 (UMT5) and L is the
padded max sequence length used at latent-extraction time.

Reuses the same tokenizer + text encoder loaders as the server
(`wan_va/modules/utils.py`) and the `_get_t5_prompt_embeds` truncation logic from
`wan_va/wan_va_server.py`.

Usage:
    python droid_helpers/make_empty_emb.py \
        --model checkpoints/lingbot-va-base \
        --out   <dataset_dir>/empty_emb.pt \
        --max-seq-len 512
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wan_va.modules.utils import load_text_encoder, load_tokenizer  # noqa: E402


def encode_empty(model_dir: str, max_seq_len: int, device: str, dtype) -> torch.Tensor:
    tokenizer = load_tokenizer(os.path.join(model_dir, "tokenizer"))
    text_encoder = load_text_encoder(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=dtype,
        torch_device=device,
    )

    text_inputs = tokenizer(
        [""],
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_len = int(mask.gt(0).sum(dim=1).long()[0].item())

    with torch.no_grad():
        emb = text_encoder(ids.to(device), mask.to(device)).last_hidden_state  # [1, L, D]
    emb = emb[0].to(dtype=dtype, device="cpu")

    # Match the server: keep only valid tokens, then zero-pad back to max_seq_len.
    emb = emb[:seq_len]
    emb = torch.cat([emb, emb.new_zeros(max_seq_len - emb.size(0), emb.size(1))], dim=0)
    return emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/lingbot-va-base")
    p.add_argument("--out", required=True)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dtype = torch.bfloat16
    emb = encode_empty(args.model, args.max_seq_len, args.device, dtype)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(emb, args.out)
    print(f"[empty_emb] saved {tuple(emb.shape)} {emb.dtype} -> {args.out}")


if __name__ == "__main__":
    main()
