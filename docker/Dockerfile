# LingBot-VA DROID post-training image for an 8xA100 DGX.
#
# CUDA 12.6 devel base (has nvcc) so flash-attn compiles cleanly against the
# torch cu126 wheels — matching the versions verified on the dev box.
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0" \
    PYTORCH_ALLOC_CONF=expandable_segments:True

# System deps: python 3.10, ffmpeg (video I/O), git, build toolchain for flash-attn.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git ffmpeg libgl1 libglib2.0-0 ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging

WORKDIR /workspace/lingbot-va

# 1) Torch stack from the cu126 index (kept as a separate layer for caching).
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu126

# 2) Python deps.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt \
    && pip install --no-deps lerobot==0.3.3

# 3) flash-attn (needs torch present; no build isolation). CUDA 12.6 base matches
#    the torch build, so the version check passes here (unlike the dev box).
#    MAX_JOBS bounds the nvcc parallelism so the compile doesn't get OOM-killed;
#    raise it on the DGX (lots of RAM) to speed the ~15-20min build up
#    (docker build --build-arg MAX_JOBS=16 ...).
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
RUN pip install flash-attn==2.8.3.post1 --no-build-isolation

# 4) Project code. Mount data/checkpoints at run time instead of baking them in.
COPY . /workspace/lingbot-va

ENV PYTHONPATH=/workspace/lingbot-va:/workspace/lingbot-va/wan_va

CMD ["bash"]
