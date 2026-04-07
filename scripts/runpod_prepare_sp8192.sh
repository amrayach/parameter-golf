#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VARIANT="${VARIANT:-sp8192}"
MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"
DATA_ROOT="${DATA_ROOT:-/root/pgdata}"
CACHE_ROOT="${CACHE_ROOT:-/root/pgcache}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-1}"

case "${VARIANT}" in
  sp8192)
    TRAIN_SHARDS=128
    TOKENIZER_MODEL="fineweb_8192_bpe.model"
    TOKENIZER_VOCAB="fineweb_8192_bpe.vocab"
    DATASET_DIR="fineweb10B_sp8192"
    ;;
  sp1024)
    TRAIN_SHARDS=195
    TOKENIZER_MODEL="fineweb_1024_bpe.model"
    TOKENIZER_VOCAB="fineweb_1024_bpe.vocab"
    DATASET_DIR="fineweb10B_sp1024"
    ;;
  *)
    echo "Unsupported VARIANT=${VARIANT}; expected sp8192 or sp1024" >&2
    exit 1
    ;;
esac

cd "${REPO_ROOT}"

echo "==> RunPod data preparation"
echo "repo: ${REPO_ROOT}"
echo "variant: ${VARIANT}"
echo "matched repo: ${MATCHED_FINEWEB_REPO_ID}"
echo "data root: ${DATA_ROOT}"
echo "cache root: ${CACHE_ROOT}"

mkdir -p "${DATA_ROOT}/datasets" "${DATA_ROOT}/tokenizers" "${CACHE_ROOT}/huggingface/hub"

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export MATCHED_FINEWEB_REPO_ID

rm -rf data/datasets data/tokenizers
ln -s "${DATA_ROOT}/datasets" data/datasets
ln -s "${DATA_ROOT}/tokenizers" data/tokenizers

rm -f data/manifest.json data/datasets/manifest.json

DATASET_PATH="${REPO_ROOT}/data/datasets/${DATASET_DIR}"
MODEL_PATH="${REPO_ROOT}/data/tokenizers/${TOKENIZER_MODEL}"
VOCAB_PATH="${REPO_ROOT}/data/tokenizers/${TOKENIZER_VOCAB}"

train_count=0
val_count=0
if [ -d "${DATASET_PATH}" ]; then
  train_count="$(find "${DATASET_PATH}" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
  val_count="$(find "${DATASET_PATH}" -maxdepth 1 -name 'fineweb_val_*.bin' | wc -l | tr -d ' ')"
fi

if [ "${train_count}" != "${TRAIN_SHARDS}" ] || [ "${val_count}" != "1" ] || [ ! -f "${MODEL_PATH}" ] || [ ! -f "${VOCAB_PATH}" ]; then
  echo "==> Dataset/tokenizer incomplete; downloading ${VARIANT}"
  "${PYTHON_BIN}" data/cached_challenge_fineweb.py --variant "${VARIANT}" --train-shards "${TRAIN_SHARDS}"
else
  echo "==> Dataset/tokenizer already present; skipping download"
fi

echo "==> Runtime dependency check"
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib
for name in ("brotli", "sentencepiece", "torch", "huggingface_hub", "triton", "flash_attn_interface"):
    importlib.import_module(name)
PY
then
  if [ "${INSTALL_RUNTIME_DEPS}" = "1" ]; then
    echo "==> Installing missing Python packages"
    "${PYTHON_BIN}" -m pip install --no-cache-dir huggingface-hub sentencepiece brotli
  else
    echo "Missing runtime dependencies and INSTALL_RUNTIME_DEPS=0" >&2
    exit 1
  fi
fi

echo "==> Verification"
"${PYTHON_BIN}" - <<PY
import brotli
import sentencepiece
import torch
import triton
from pathlib import Path
repo = Path(${REPO_ROOT@Q})
dataset = repo / "data" / "datasets" / ${DATASET_DIR@Q}
tok_dir = repo / "data" / "tokenizers"
print("train", len(list(dataset.glob("fineweb_train_*.bin"))))
print("val", len(list(dataset.glob("fineweb_val_*.bin"))))
print("tok_model", (tok_dir / ${TOKENIZER_MODEL@Q}).exists())
print("tok_vocab", (tok_dir / ${TOKENIZER_VOCAB@Q}).exists())
print("datasets_link", (repo / "data" / "datasets").resolve())
print("tokenizers_link", (repo / "data" / "tokenizers").resolve())
print("hf_cache", Path(${CACHE_ROOT@Q}) / "huggingface" / "hub")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
print("triton", triton.__version__)
print("sentencepiece", sentencepiece.__version__)
print("brotli", getattr(brotli, "__version__", "unknown"))
PY
