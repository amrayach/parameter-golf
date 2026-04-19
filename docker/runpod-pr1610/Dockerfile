ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2204
FROM ${BASE_IMAGE}

ARG VENV_DIR=/opt/pg-venv
ARG TORCH_VERSION=2.9.1+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_CUDA_VERSION=12.8
ARG FLASH_ATTN_SPEC=flash-attn>=2.8,<2.9
ARG FLASH_ATTN_WHEEL_URL=
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG MAX_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    VENV_DIR=${VENV_DIR} \
    PATH=${VENV_DIR}/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VENV_DIR}"

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir --upgrade --force-reinstall \
        "torch==${TORCH_VERSION}" \
        --extra-index-url "${TORCH_INDEX_URL}" && \
    python -m pip install --no-cache-dir --upgrade \
        numpy \
        tqdm \
        huggingface-hub \
        setuptools \
        typing-extensions==4.15.0 \
        datasets \
        "fsspec>=2023.1.0,<=2026.2.0" \
        tiktoken \
        sentencepiece \
        zstandard \
        brotli \
        python-minifier \
        psutil

# Prefer a prebuilt wheel when one is provided. Otherwise do the one-time
# source build here so fresh RunPod pods never rebuild flash-attn again.
RUN if [[ -n "${FLASH_ATTN_WHEEL_URL}" ]]; then \
        python -m pip install --no-cache-dir --force-reinstall --no-deps "${FLASH_ATTN_WHEEL_URL}"; \
    else \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
        python -m pip install --no-cache-dir --force-reinstall --no-build-isolation --no-deps "${FLASH_ATTN_SPEC}"; \
    fi

RUN python - <<PY
import brotli
import sentencepiece
import torch
import triton
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

expected_torch = "${TORCH_VERSION}"
expected_cuda = "${TORCH_CUDA_VERSION}"
if torch.__version__ != expected_torch:
    raise SystemExit(f"torch version mismatch: expected {expected_torch}, got {torch.__version__}")
if torch.version.cuda != expected_cuda:
    raise SystemExit(f"CUDA runtime mismatch: expected {expected_cuda}, got {torch.version.cuda}")

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"triton {triton.__version__}")
print("flash_attn_interface: OK")
print("python runtime deps: OK")
PY

WORKDIR /workspace
# Rebuild and repin the published image digest before the next RunPod session.
CMD ["sleep", "infinity"]
