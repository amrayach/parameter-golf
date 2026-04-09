#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_07c1.sh smoke
  bash scripts/runpod_07c1.sh base <seed>
  bash scripts/runpod_07c1.sh ttt <seed>

Modes:
  smoke  Cheap 1xGPU structural validation on the 07c1 script.
  base   Exact 8xGPU non-TTT launch matching Pegasus 07c1 baseline settings.
  ttt    Exact 8xGPU TTT launch matching Pegasus 07c1 fixed rerun settings.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

MODE="$1"
SEED_ARG="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"

cd "${REPO_ROOT}"
mkdir -p logs

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "Missing script: ${SCRIPT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "Missing data path: ${DATA_PATH}" >&2
  echo "Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1" >&2
  exit 1
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "Missing tokenizer path: ${TOKENIZER_PATH}" >&2
  exit 1
fi

COMMON_ENV=(
  "PYTHONUNBUFFERED=1"
  "MKL_NUM_THREADS=1"
  "NUMEXPR_NUM_THREADS=1"
  "OMP_NUM_THREADS=1"
  "PIP_ROOT_USER_ACTION=ignore"
  "DATA_PATH=${DATA_PATH}"
  "TOKENIZER_PATH=${TOKENIZER_PATH}"
  "VOCAB_SIZE=1024"
  "MATRIX_LR=0.024"
  "MATRIX_LR_LATE=0.019"
  "SCALAR_LR=0.020"
  "SCALAR_LR_LATE=0.038"
  "TIED_EMBED_LR=0.022"
  "MUON_MOMENTUM=0.985"
  "WARMDOWN_ITERS=4000"
  "GPTQ_RESERVE_MS=0"
  "NUM_LAYERS=12"
  "BIGRAM_VOCAB_SIZE=5120"
  "VE_DIM=128"
  "WINDOW_SIZE=512"
  "WINDOW_ATTN_LAYERS=2,4,6,8,10"
  "QK_GAIN_INIT=2.5"
)

run_and_log() {
  local run_id="$1"
  shift
  local console_log="logs/${run_id}.console.log"
  echo "==> mode: ${MODE}"
  echo "==> run_id: ${run_id}"
  echo "==> console_log: ${console_log}"
  env "$@" 2>&1 | tee "${console_log}"
}

case "${MODE}" in
  smoke)
    RUN_ID="${RUN_ID:-runpod_07c1_smoke}"
    SEED="${SEED:-1337}"
    run_and_log "${RUN_ID}" \
      "${COMMON_ENV[@]}" \
      "RUN_ID=${RUN_ID}" \
      "SEED=${SEED}" \
      "LOCAL_RANK=0" \
      "RANK=0" \
      "WORLD_SIZE=1" \
      "LOCAL_SEQS_PER_GPU=${LOCAL_SEQS_PER_GPU:-4}" \
      "LOCAL_SEQ_LEN=${LOCAL_SEQ_LEN:-2048}" \
      "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048}" \
      "EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-2048}" \
      "VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-65536}" \
      "ITERATIONS=${ITERATIONS:-40}" \
      "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-120}" \
      "TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10}" \
      "VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}" \
      "EVAL_STRIDE=${EVAL_STRIDE:-64}" \
      "TTT_ENABLED=${TTT_ENABLED:-0}" \
      "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
    ;;

  base|ttt)
    if [[ -z "${SEED_ARG}" ]]; then
      echo "${MODE} mode requires a seed argument" >&2
      usage
      exit 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia-smi not found; this is not a CUDA pod." >&2
      exit 1
    fi

    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    if [[ "${GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
      echo "Need at least ${NPROC_PER_NODE} visible GPUs, found ${GPU_COUNT}" >&2
      exit 1
    fi

    if [[ "${MODE}" == "base" ]]; then
      RUN_ID="${RUN_ID:-07c1_runpod_base_s${SEED_ARG}}"
      TTT_ENABLED_VALUE=0
    else
      RUN_ID="${RUN_ID:-07c1_runpod_ttt_s${SEED_ARG}}"
      TTT_ENABLED_VALUE=1
    fi

    run_and_log "${RUN_ID}" \
      "${COMMON_ENV[@]}" \
      "RUN_ID=${RUN_ID}" \
      "SEED=${SEED_ARG}" \
      "TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-589824}" \
      "MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}" \
      "SEQ_LENS_PER_GPU=${SEQ_LENS_PER_GPU:-2048,2048,2048,2048,2048,6144,6144,6144}" \
      "SEQS_PER_GPU=${SEQS_PER_GPU:-36,36,36,36,36,10,10,10}" \
      "TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048}" \
      "EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-6144}" \
      "EVAL_STRIDE=${EVAL_STRIDE:-64}" \
      "TTT_ENABLED=${TTT_ENABLED_VALUE}" \
      "MASTER_PORT=${MASTER_PORT:-29500}" \
      "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_PATH}"
    ;;

  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac
