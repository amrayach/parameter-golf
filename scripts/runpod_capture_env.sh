#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_capture_env.sh [label]

Writes a reproducibility snapshot under logs/ containing:
  - timestamp, hostname, kernel
  - nvidia-smi summary
  - Python / torch / CUDA metadata
  - flash_attn_interface import path
  - brotli and sentencepiece versions
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

LABEL="${1:-manual}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/logs"
OUT_FILE="${OUT_DIR}/${LABEL}.env.txt"
mkdir -p "${OUT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo
  echo "[nvidia-smi]"
  if command -v nvidia-smi >/dev/null 2>&1; then
    # cuda_version is shown in the nvidia-smi header, not as a per-GPU query field.
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo
    echo "[nvidia-smi-header]"
    nvidia-smi | sed -n '1,3p'
  else
    echo "nvidia-smi missing"
  fi
  echo
  echo "[python]"
  python3 - <<'PY'
import platform
import sys

print(f"python={platform.python_version()}")

try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cudnn={torch.backends.cudnn.version()}")
except Exception as exc:
    print(f"torch_error={exc}")

try:
    import flash_attn_interface
    print(f"flash_attn_interface_file={flash_attn_interface.__file__}")
    print(f"flash_attn_interface_version={getattr(flash_attn_interface, '__version__', 'unknown')}")
except Exception as exc:
    print(f"flash_attn_interface_error={exc}")

try:
    import brotli
    print(f"brotli={getattr(brotli, '__version__', 'unknown')}")
except Exception as exc:
    print(f"brotli_error={exc}")

try:
    import sentencepiece
    print(f"sentencepiece={sentencepiece.__version__}")
except Exception as exc:
    print(f"sentencepiece_error={exc}")
PY
} | tee "${OUT_FILE}"

echo "==> Wrote ${OUT_FILE}"
