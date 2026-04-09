#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

echo "==> RunPod environment preparation"
echo "repo: ${REPO_ROOT}"
echo "python: $(${PYTHON_BIN} --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Visible GPUs"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "nvidia-smi not found; this is not a CUDA pod." >&2
  exit 1
fi

${PYTHON_BIN} -m pip install --upgrade pip
${PYTHON_BIN} -m pip install --no-cache-dir huggingface-hub sentencepiece brotli

${PYTHON_BIN} - <<'PY'
import importlib

required = [
    "brotli",
    "sentencepiece",
    "torch",
    "huggingface_hub",
    "triton",
]
for module_name in required:
    importlib.import_module(module_name)

try:
    import flash_attn_interface  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "flash_attn_interface is missing or broken. "
        "Use the official Parameter Golf RunPod template or another FA3-enabled image."
    ) from exc

import brotli
import sentencepiece
import torch
import triton

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
print("triton", triton.__version__)
print("sentencepiece", sentencepiece.__version__)
print("brotli", getattr(brotli, "__version__", "unknown"))
PY

echo "==> Environment looks usable for 07c1"
