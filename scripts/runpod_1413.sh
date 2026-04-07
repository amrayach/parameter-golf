#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_1413.sh [seed]

Runs the faithful PR #1413 SP8192 legal score-first TTT lane on a single
8xH100 SXM RunPod pod.

Environment overrides:
  PR_REF=pr1413
  PR_FETCH_SPEC=pull/1413/head:pr1413
  RECORD_REL=records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828
  FETCH_PAYLOAD=1
  PREPARE_SP8192=1
  TORCHRUN_BIN=torchrun
  NPROC_PER_NODE=8
  ARCHIVE_ROOT=/workspace
  RUN_TAG=pr1413
  RUN_STAMP=<yyyymmdd_hhmmss>
  RUN_ID=<custom_run_id>
  DATA_DIR=<repo>/data
  NCCL_NET=Socket
  QK_GAIN_INIT=5.0
  TTT_ENABLED=1
  TTT_LR=0.005
  TTT_EPOCHS=3
  LOOP_START=4
  LOOP_END=5
  PARALLEL_RESIDUAL_START=-1
  SKIP_TRAINING=0
  NGRAM_TILT_ENABLED=0
  NGRAM_BASE_BETA=2.0
  NGRAM_AGREE_BONUS=0.1
  NGRAM_WITHIN_THRESHOLD=0.25
  NGRAM_WITHIN_BETA=0.0
  NGRAM_WORD_THRESHOLD=0.8
  NGRAM_WORD_BETA=0.0
  NGRAM_OPEN_TABLE_BITS=26
  NGRAM_ORDER_STRIDE=2
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

SEED_VALUE="${1:-${SEED:-0}}"
if [[ ! "${SEED_VALUE}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED_VALUE}" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR_REF="${PR_REF:-pr1413}"
PR_FETCH_SPEC="${PR_FETCH_SPEC:-pull/1413/head:${PR_REF}}"
RECORD_REL="${RECORD_REL:-records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828}"
RECORD_DIR="${REPO_ROOT}/${RECORD_REL}"
FETCH_PAYLOAD="${FETCH_PAYLOAD:-1}"
PREPARE_SP8192="${PREPARE_SP8192:-1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/workspace}"
RUN_TAG="${RUN_TAG:-pr1413}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_TAG}_s${SEED_VALUE}_${RUN_STAMP}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ARCHIVE_ROOT}/${RUN_TAG}_archive_${RUN_STAMP}/seed${SEED_VALUE}}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

cd "${REPO_ROOT}"

if [[ "${FETCH_PAYLOAD}" == "1" ]]; then
  git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/openai/parameter-golf.git
  git fetch upstream "${PR_FETCH_SPEC}"
  git checkout "${PR_REF}" -- "${RECORD_REL}"
fi

if [[ ! -f "${RECORD_DIR}/train_gpt.py" ]]; then
  echo "Missing PR #1413 payload at ${RECORD_DIR}" >&2
  exit 1
fi

if [[ "${PREPARE_SP8192}" == "1" ]]; then
  bash scripts/runpod_prepare_sp8192.sh
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; this is not a CUDA pod." >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "Need at least ${NPROC_PER_NODE} visible GPUs, found ${GPU_COUNT}" >&2
  exit 1
fi

mkdir -p "${ARCHIVE_DIR}" "${ARCHIVE_DIR}/logs" "${RECORD_DIR}/logs"

source_files=("README.md" "submission.json" "train_gpt.py")
for optional_source in ngram_tilt.py fused_expert_kernel.cpp variant_manifest.json; do
  if [[ -f "${RECORD_DIR}/${optional_source}" ]]; then
    source_files+=("${optional_source}")
  fi
done
cp -v "${source_files[@]/#/${RECORD_DIR}/}" "${ARCHIVE_DIR}/"

{
  echo "RUN_ID=${RUN_ID}"
  echo "SEED=${SEED_VALUE}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "RUN_STAMP=${RUN_STAMP}"
  echo "ARCHIVE_DIR=${ARCHIVE_DIR}"
  echo "DATA_DIR=${DATA_DIR}"
  echo "RECORD_REL=${RECORD_REL}"
  echo "REPO_HEAD=$(git rev-parse HEAD)"
  echo "PR_REF=${PR_REF}"
  echo "PR_REF_HEAD=$(git rev-parse "${PR_REF}")"
  echo "NCCL_NET=${NCCL_NET:-Socket}"
  echo "QK_GAIN_INIT=${QK_GAIN_INIT:-5.0}"
  echo "TTT_ENABLED=${TTT_ENABLED:-1}"
  echo "TTT_LR=${TTT_LR:-0.005}"
  echo "TTT_EPOCHS=${TTT_EPOCHS:-3}"
  echo "LOOP_START=${LOOP_START:-4}"
  echo "LOOP_END=${LOOP_END:-5}"
  echo "PARALLEL_RESIDUAL_START=${PARALLEL_RESIDUAL_START:--1}"
  echo "SKIP_TRAINING=${SKIP_TRAINING:-0}"
  echo "NGRAM_TILT_ENABLED=${NGRAM_TILT_ENABLED:-0}"
  echo "NGRAM_BASE_BETA=${NGRAM_BASE_BETA:-2.0}"
  echo "NGRAM_AGREE_BONUS=${NGRAM_AGREE_BONUS:-0.1}"
  echo "NGRAM_WITHIN_THRESHOLD=${NGRAM_WITHIN_THRESHOLD:-0.25}"
  echo "NGRAM_WITHIN_BETA=${NGRAM_WITHIN_BETA:-0.0}"
  echo "NGRAM_WORD_THRESHOLD=${NGRAM_WORD_THRESHOLD:-0.8}"
  echo "NGRAM_WORD_BETA=${NGRAM_WORD_BETA:-0.0}"
  echo "NGRAM_OPEN_TABLE_BITS=${NGRAM_OPEN_TABLE_BITS:-26}"
  echo "NGRAM_ORDER_STRIDE=${NGRAM_ORDER_STRIDE:-2}"
  echo "TORCHRUN_BIN=${TORCHRUN_BIN}"
  echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
} > "${ARCHIVE_DIR}/run_meta.env"

sha256sum "${source_files[@]/#/${ARCHIVE_DIR}/}" > "${ARCHIVE_DIR}/source_sha256.txt"

echo "==> PR #1413 RunPod launch"
echo "repo_root: ${REPO_ROOT}"
echo "record_dir: ${RECORD_DIR}"
echo "archive_dir: ${ARCHIVE_DIR}"
echo "run_id: ${RUN_ID}"
echo "seed: ${SEED_VALUE}"
echo "gpu_count: ${GPU_COUNT}"

(
  cd "${RECORD_DIR}"
  env \
    PYTHONUNBUFFERED=1 \
    DATA_DIR="${DATA_DIR}" \
    NCCL_NET="${NCCL_NET:-Socket}" \
    RUN_ID="${RUN_ID}" \
    SEED="${SEED_VALUE}" \
    QK_GAIN_INIT="${QK_GAIN_INIT:-5.0}" \
    TTT_ENABLED="${TTT_ENABLED:-1}" \
    TTT_LR="${TTT_LR:-0.005}" \
    TTT_EPOCHS="${TTT_EPOCHS:-3}" \
    LOOP_START="${LOOP_START:-4}" \
    LOOP_END="${LOOP_END:-5}" \
    PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:--1}" \
    SKIP_TRAINING="${SKIP_TRAINING:-0}" \
    NGRAM_TILT_ENABLED="${NGRAM_TILT_ENABLED:-0}" \
    NGRAM_BASE_BETA="${NGRAM_BASE_BETA:-2.0}" \
    NGRAM_AGREE_BONUS="${NGRAM_AGREE_BONUS:-0.1}" \
    NGRAM_WITHIN_THRESHOLD="${NGRAM_WITHIN_THRESHOLD:-0.25}" \
    NGRAM_WITHIN_BETA="${NGRAM_WITHIN_BETA:-0.0}" \
    NGRAM_WORD_THRESHOLD="${NGRAM_WORD_THRESHOLD:-0.8}" \
    NGRAM_WORD_BETA="${NGRAM_WORD_BETA:-0.0}" \
    NGRAM_OPEN_TABLE_BITS="${NGRAM_OPEN_TABLE_BITS:-26}" \
    NGRAM_ORDER_STRIDE="${NGRAM_ORDER_STRIDE:-2}" \
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
) |& tee "${ARCHIVE_DIR}/console.txt"

cp -v \
  "${RECORD_DIR}/logs/${RUN_ID}.txt" \
  "${ARCHIVE_DIR}/logs/"

artifact_files=(
  "${ARCHIVE_DIR}/logs/${RUN_ID}.txt"
  "${ARCHIVE_DIR}/final_model.int6.ptz"
)
# final_model.pt is only written when training runs (not for SKIP_TRAINING=1).
# Copy it when present; skip gracefully otherwise.
if [[ -f "${RECORD_DIR}/final_model.pt" ]]; then
  cp -v "${RECORD_DIR}/final_model.pt" "${ARCHIVE_DIR}/"
  artifact_files+=("${ARCHIVE_DIR}/final_model.pt")
else
  echo "Note: final_model.pt not found in ${RECORD_DIR} (expected for eval-only runs)"
fi
cp -v "${RECORD_DIR}/final_model.int6.ptz" "${ARCHIVE_DIR}/"
if [[ -f "${RECORD_DIR}/libfused_ngram.so" ]]; then
  cp -v "${RECORD_DIR}/libfused_ngram.so" "${ARCHIVE_DIR}/"
  artifact_files+=("${ARCHIVE_DIR}/libfused_ngram.so")
fi

sha256sum "${artifact_files[@]}" > "${ARCHIVE_DIR}/artifact_sha256.txt"

echo "==> Preserved outputs"
echo "  ${ARCHIVE_DIR}/console.txt"
echo "  ${ARCHIVE_DIR}/logs/${RUN_ID}.txt"
[[ -f "${ARCHIVE_DIR}/final_model.pt" ]] && echo "  ${ARCHIVE_DIR}/final_model.pt"
echo "  ${ARCHIVE_DIR}/final_model.int6.ptz"
