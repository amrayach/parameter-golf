#!/usr/bin/env bash
set -euo pipefail

# Fill these before use.
REPO_PATH="${REPO_PATH:-/path/to/parameter-golf}"
SCRIPT_PATH="${SCRIPT_PATH:-train_gpt.py}"
PARTITION="${PARTITION:-H100}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
CPUS_PER_GPU="${CPUS_PER_GPU:-12}"
RUN_CMD="${RUN_CMD:-torchrun --standalone --nproc_per_node=8 ${SCRIPT_PATH}}"

srun \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=8 \
  --gpus=8 \
  --cpus-per-gpu="${CPUS_PER_GPU}" \
  --gpu-bind=none \
  --time="${TIME_LIMIT}" \
  bash -lc "cd \"${REPO_PATH}\" && ${RUN_CMD}"
