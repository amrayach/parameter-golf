#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_1413_batch.sh [seed] [run ...]

Runs the prepared local PR #1413 experiment suite sequentially.

Examples:
  bash scripts/runpod_1413_batch.sh 0
  bash scripts/runpod_1413_batch.sh 0 A C
  bash scripts/runpod_1413_batch.sh 42 B D E

Run labels:
  A  faithful #1413 local mirror
  B  parallel residuals only
  C  LOOP_START=3 only
  D  parallel residuals + LOOP_START=3
  E  eval-only n-gram tilt replay on the current stack checkpoint

Prerequisites:
  1. Run `python3 scripts/prepare_pr1413_variants.py --force` locally.
  2. Sync the repo to the RunPod pod.
  3. Run this script from the repo root on the pod.
EOF
}

if [[ $# -gt 0 && ( "${1}" == "-h" || "${1}" == "--help" ) ]]; then
  usage
  exit 0
fi

SEED_VALUE="${1:-0}"
if [[ ! "${SEED_VALUE}" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: ${SEED_VALUE}" >&2
  usage
  exit 1
fi
shift || true

if [[ $# -eq 0 ]]; then
  RUNS=(A B C D E)
else
  RUNS=("$@")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RECORD_REL="records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_LocalBase"
STACK_RECORD_REL="records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep"
BASE_RECORD_DIR="${REPO_ROOT}/${BASE_RECORD_REL}"
STACK_RECORD_DIR="${REPO_ROOT}/${STACK_RECORD_REL}"

for required_dir in "${BASE_RECORD_DIR}" "${STACK_RECORD_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Missing prepared experiment folder: ${required_dir}" >&2
    echo "Run: python3 scripts/prepare_pr1413_variants.py --force" >&2
    exit 1
  fi
done

# Share a single archive timestamp across all runs in this batch so outputs
# land in one directory tree instead of being scattered across 5 timestamps.
export RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Only run SP8192 data/deps prep once (the first invocation).  Subsequent
# runs reuse the already-prepared environment.
FIRST_RUN=1

run_case() {
  local label="$1"
  shift
  echo "==> Starting run ${label} (seed ${SEED_VALUE})"
  local prep_flag=1
  local run_id=""
  local archive_root="/workspace/pr1413_archive_${RUN_STAMP}/seed${SEED_VALUE}"
  local archive_dir=""
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == RUN_ID=* ]]; then
      run_id="${arg#RUN_ID=}"
      break
    fi
  done
  if [[ -n "${run_id}" ]]; then
    archive_dir="${archive_root}/${run_id}"
  else
    archive_dir="${archive_root}/${label}"
  fi
  if [[ "${FIRST_RUN}" -eq 0 ]]; then
    prep_flag=0
  fi
  FIRST_RUN=0
  env PREPARE_SP8192="${prep_flag}" ARCHIVE_DIR="${archive_dir}" "$@" bash scripts/runpod_1413.sh "${SEED_VALUE}"
}

for run in "${RUNS[@]}"; do
  case "${run}" in
    A)
      run_case A \
        FETCH_PAYLOAD=0 \
        RECORD_REL="${BASE_RECORD_REL}" \
        RUN_ID="pr1413_ctrl_s${SEED_VALUE}"
      ;;
    B)
      run_case B \
        FETCH_PAYLOAD=0 \
        RECORD_REL="${STACK_RECORD_REL}" \
        RUN_ID="pr1413_par7_s${SEED_VALUE}" \
        PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:-7}"
      ;;
    C)
      run_case C \
        FETCH_PAYLOAD=0 \
        RECORD_REL="${BASE_RECORD_REL}" \
        RUN_ID="pr1413_loop35_s${SEED_VALUE}" \
        LOOP_START="${LOOP_START:-3}" \
        LOOP_END="${LOOP_END:-5}"
      ;;
    D)
      run_case D \
        FETCH_PAYLOAD=0 \
        RECORD_REL="${STACK_RECORD_REL}" \
        RUN_ID="pr1413_combo_s${SEED_VALUE}" \
        PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:-7}" \
        LOOP_START="${LOOP_START:-3}" \
        LOOP_END="${LOOP_END:-5}"
      ;;
    E)
      if [[ ! -f "${STACK_RECORD_DIR}/final_model.int6.ptz" ]]; then
        echo "Run E requires an existing stack checkpoint at ${STACK_RECORD_DIR}/final_model.int6.ptz" >&2
        echo "Run D first, or copy the intended best checkpoint into ${STACK_RECORD_DIR}." >&2
        exit 1
      fi
      if ! command -v g++ >/dev/null 2>&1; then
        echo "Run E requires g++ for ngram kernel compilation. Install with: apt-get install -y g++" >&2
        exit 1
      fi
      run_case E \
        FETCH_PAYLOAD=0 \
        RECORD_REL="${STACK_RECORD_REL}" \
        RUN_ID="pr1413_ngram_eval_s${SEED_VALUE}" \
        SKIP_TRAINING=1 \
        PARALLEL_RESIDUAL_START="${PARALLEL_RESIDUAL_START:-7}" \
        LOOP_START="${LOOP_START:-3}" \
        LOOP_END="${LOOP_END:-5}" \
        NGRAM_TILT_ENABLED=1 \
        NGRAM_BASE_BETA="${NGRAM_BASE_BETA:-2.0}" \
        NGRAM_AGREE_BONUS="${NGRAM_AGREE_BONUS:-0.1}" \
        NGRAM_WITHIN_THRESHOLD="${NGRAM_WITHIN_THRESHOLD:-0.25}" \
        NGRAM_WITHIN_BETA="${NGRAM_WITHIN_BETA:-0.0}" \
        NGRAM_WORD_THRESHOLD="${NGRAM_WORD_THRESHOLD:-0.8}" \
        NGRAM_WORD_BETA="${NGRAM_WORD_BETA:-0.0}" \
        NGRAM_OPEN_TABLE_BITS="${NGRAM_OPEN_TABLE_BITS:-26}" \
        NGRAM_ORDER_STRIDE="${NGRAM_ORDER_STRIDE:-2}"
      ;;
    *)
      echo "Unknown run label: ${run}" >&2
      usage
      exit 1
      ;;
  esac
done
