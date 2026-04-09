#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pod_candidate_seed_sweep.sh <candidate>

Candidate presets:
  r1       Eval-only n-gram tilt on preserved D checkpoints
  r4       Eval-only n-gram tilt + freeze4 on preserved D checkpoints
  requant  Re-quantize from existing final_model.pt with new OWC settings (no training)
  r6       Cautious Muon train + eval
  r7       OWC train + eval
  r8       CDQuant + OWC train + eval
  r9       Full stack train + eval

Optional env:
  SEEDS=0,42,1234,1337,2025
  RUN_STAMP=<stamp>
  RESULTS_DIR=/workspace/candidate_verify_<candidate>_<stamp>
  STACK_RECORD_REL=records/.../train_bundle
  CHECKPOINT_SOURCE_ROOT=/workspace/pr1413_combo_seed_checkpoints
  REQUIRE_LEGAL=1
  OWC_GAMMA_STEPS=10
  OWC_SCOPE=all
  CDQUANT_ITERS=3
  EXPORT_PACKING=torchsave

Notes:
  - Eval-only presets (`r1`, `r4`) require staged D checkpoints under
    ${CHECKPOINT_SOURCE_ROOT}/seed<seed>/final_model.int6.ptz.
  - Train-side presets abort the sweep on the first over-cap artifact when
    REQUIRE_LEGAL=1, because the candidate would not be submission-valid.
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

CANDIDATE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_STACK_RECORD_REL="records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep"
STACK_RECORD_REL="${STACK_RECORD_REL:-${DEFAULT_STACK_RECORD_REL}}"
STACK_RECORD_DIR="${REPO_ROOT}/${STACK_RECORD_REL}"
CHECKPOINT_SOURCE_ROOT="${CHECKPOINT_SOURCE_ROOT:-/workspace/pr1413_combo_seed_checkpoints}"
SEEDS_CSV="${SEEDS:-0,42,1234,1337,2025}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/candidate_verify_${CANDIDATE}_${RUN_STAMP}}"
REQUIRE_LEGAL="${REQUIRE_LEGAL:-1}"
SIZE_CAP=16000000
OWC_GAMMA_STEPS_VALUE="${OWC_GAMMA_STEPS:-10}"
OWC_SCOPE_VALUE="${OWC_SCOPE:-all}"
CDQUANT_ITERS_VALUE="${CDQUANT_ITERS:-3}"
EXPORT_PACKING_VALUE="${EXPORT_PACKING:-torchsave}"

[[ -f "${STACK_RECORD_DIR}/train_gpt.py" ]] || {
  echo "Missing stack payload at ${STACK_RECORD_DIR}" >&2
  exit 1
}

IFS=',' read -r -a SEEDS <<<"${SEEDS_CSV}"

COMMON_PARALLEL_RESIDUAL_START=7
COMMON_LOOP_START=3
COMMON_LOOP_END=5

MODE=""
TRAIN_DESC=""
TRAIN_NAME=""
EVAL_DESC=""
EVAL_NAME=""
TRAIN_ENV=()
EVAL_ENV=()

case "${CANDIDATE}" in
  r1)
    MODE="eval_only"
    EVAL_NAME="r1_eval"
    EVAL_DESC="SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  r4)
    MODE="eval_only"
    EVAL_NAME="r4_eval"
    EVAL_DESC="SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_FREEZE_BLOCKS=4"
    EVAL_ENV=(SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_ENABLED=1 TTT_FREEZE_BLOCKS=4)
    ;;
  requant)
    # Re-quantize from existing final_model.pt with new OWC settings.
    # Requires final_model.pt in the STACK_RECORD_DIR (copy from a prior training run).
    # Skips training entirely — only runs serialize (GPTQ + export) then eval.
    MODE="train_then_eval"
    TRAIN_NAME="requant_export"
    TRAIN_DESC="REQUANT_ONLY=1 OWC_ENABLED=1 OWC_GAMMA_STEPS=${OWC_GAMMA_STEPS_VALUE} OWC_SCOPE=${OWC_SCOPE_VALUE}"
    TRAIN_ENV=(REQUANT_ONLY=1 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" TTT_ENABLED=1)
    EVAL_NAME="requant_eval"
    EVAL_DESC="REQUANT OWC_SCOPE=${OWC_SCOPE_VALUE} gamma=${OWC_GAMMA_STEPS_VALUE} + NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  r6)
    MODE="train_then_eval"
    TRAIN_NAME="r6_train"
    TRAIN_DESC="CAUTIOUS_MUON=1 EXPORT_PACKING=${EXPORT_PACKING_VALUE}"
    TRAIN_ENV=(SKIP_TRAINING=0 CAUTIOUS_MUON=1 EXPORT_PACKING="${EXPORT_PACKING_VALUE}" TTT_ENABLED=1)
    EVAL_NAME="r6_eval"
    EVAL_DESC="CAUTIOUS_MUON=1 EXPORT_PACKING=${EXPORT_PACKING_VALUE} + NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 CAUTIOUS_MUON=1 EXPORT_PACKING="${EXPORT_PACKING_VALUE}" NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  r7)
    MODE="train_then_eval"
    TRAIN_NAME="r7_train"
    TRAIN_DESC="OWC_ENABLED=1 OWC_GAMMA_STEPS=${OWC_GAMMA_STEPS_VALUE} OWC_SCOPE=${OWC_SCOPE_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE}"
    TRAIN_ENV=(SKIP_TRAINING=0 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" TTT_ENABLED=1)
    EVAL_NAME="r7_eval"
    EVAL_DESC="OWC_ENABLED=1 OWC_GAMMA_STEPS=${OWC_GAMMA_STEPS_VALUE} OWC_SCOPE=${OWC_SCOPE_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE} + NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  r8)
    MODE="train_then_eval"
    TRAIN_NAME="r8_train"
    TRAIN_DESC="CDQUANT_ENABLED=1 CDQUANT_ITERS=${CDQUANT_ITERS_VALUE} OWC_ENABLED=1 OWC_GAMMA_STEPS=${OWC_GAMMA_STEPS_VALUE} OWC_SCOPE=${OWC_SCOPE_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE}"
    TRAIN_ENV=(SKIP_TRAINING=0 CDQUANT_ENABLED=1 CDQUANT_ITERS="${CDQUANT_ITERS_VALUE}" OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" TTT_ENABLED=1)
    EVAL_NAME="r8_eval"
    EVAL_DESC="CDQuant+OWC OWC_SCOPE=${OWC_SCOPE_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE} + NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 CDQUANT_ENABLED=1 CDQUANT_ITERS="${CDQUANT_ITERS_VALUE}" OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  r9)
    MODE="train_then_eval"
    TRAIN_NAME="r9_train"
    TRAIN_DESC="CAUTIOUS_MUON=1 OWC_ENABLED=1 OWC_GAMMA_STEPS=${OWC_GAMMA_STEPS_VALUE} OWC_SCOPE=${OWC_SCOPE_VALUE} CDQUANT_ENABLED=1 CDQUANT_ITERS=${CDQUANT_ITERS_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE}"
    TRAIN_ENV=(SKIP_TRAINING=0 CAUTIOUS_MUON=1 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" CDQUANT_ENABLED=1 CDQUANT_ITERS="${CDQUANT_ITERS_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" TTT_ENABLED=1)
    EVAL_NAME="r9_eval"
    EVAL_DESC="full_stack OWC_SCOPE=${OWC_SCOPE_VALUE} EXPORT_PACKING=${EXPORT_PACKING_VALUE} + NGRAM_TILT_ENABLED=1"
    EVAL_ENV=(SKIP_TRAINING=1 CAUTIOUS_MUON=1 OWC_ENABLED=1 OWC_GAMMA_STEPS="${OWC_GAMMA_STEPS_VALUE}" OWC_SCOPE="${OWC_SCOPE_VALUE}" CDQUANT_ENABLED=1 CDQUANT_ITERS="${CDQUANT_ITERS_VALUE}" EXPORT_PACKING="${EXPORT_PACKING_VALUE}" NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
    ;;
  *)
    echo "Unknown candidate: ${CANDIDATE}" >&2
    usage
    exit 1
    ;;
esac

mkdir -p "${RESULTS_DIR}"
echo -e "seed\trun_name\tstatus\tbpb\twall_seconds\ttotal_bytes\tmargin_bytes\tenv_vars" > "${RESULTS_DIR}/summary.tsv"

extract_bpb() {
  local logfile="$1"
  local bpb=""
  bpb=$(grep -oP 'legal_ttt_exact.*?val_bpb:\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  if [[ -z "$bpb" ]]; then
    bpb=$(grep -oP 'ttt_sliding:done.*?val_bpb=\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  fi
  if [[ -z "$bpb" ]]; then
    bpb=$(grep -oP 'quantized_sliding_window.*?val_bpb:\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  fi
  echo "${bpb:-FAILED}"
}

extract_total_bytes() {
  local logfile="$1"
  grep -oP 'Total submission size quantized\+brotli: \K[0-9]+' "$logfile" 2>/dev/null | tail -1
}

extract_total_bytes_from_console() {
  local console_file="$1"
  grep -oP 'Total submission size quantized\+brotli: \K[0-9]+' "$console_file" 2>/dev/null | tail -1
}

run_one() {
  local seed="$1"
  local run_name="$2"
  local env_desc="$3"
  shift 3

  local run_dir="${RESULTS_DIR}/seed${seed}/${run_name}"
  local archive_dir="${run_dir}/archive"
  mkdir -p "${archive_dir}"

  echo ""
  echo "========================================================================"
  echo "  RUN: ${run_name}"
  echo "  SEED: ${seed}"
  echo "  ENV: ${env_desc}"
  echo "  TIME: $(date -Iseconds)"
  echo "========================================================================"

  local t0 rc t1 wall
  t0=$(date +%s)
  rc=0
  (
    set -e
    env \
      FETCH_PAYLOAD=0 \
      RECORD_REL="${STACK_RECORD_REL}" \
      RUN_ID="${run_name}" \
      SEED="${seed}" \
      PREPARE_SP8192=0 \
      ARCHIVE_DIR="${archive_dir}" \
      PARALLEL_RESIDUAL_START="${COMMON_PARALLEL_RESIDUAL_START}" \
      LOOP_START="${COMMON_LOOP_START}" \
      LOOP_END="${COMMON_LOOP_END}" \
      "$@" \
      bash scripts/runpod_1413.sh "${seed}"
  ) > "${run_dir}/output.log" 2>&1 || rc=$?
  t1=$(date +%s)
  wall=$((t1 - t0))

  local bpb total_bytes margin status
  if [[ $rc -ne 0 ]]; then
    status="FAILED"
    bpb="FAILED"
    total_bytes=""
    margin=""
    echo "  FAILED (rc=${rc}) after ${wall}s"
  else
    status="OK"
    bpb=$(extract_bpb "${run_dir}/output.log")
    total_bytes=$(extract_total_bytes "${run_dir}/output.log")
    if [[ -n "${total_bytes}" ]]; then
      margin=$((SIZE_CAP - total_bytes))
      echo "  DONE: BPB=${bpb} wall=${wall}s total_bytes=${total_bytes} margin=${margin}"
    else
      margin=""
      echo "  DONE: BPB=${bpb} wall=${wall}s"
    fi
  fi

  echo -e "${seed}\t${run_name}\t${status}\t${bpb}\t${wall}\t${total_bytes}\t${margin}\t${env_desc}" >> "${RESULTS_DIR}/summary.tsv"
  [[ $rc -eq 0 ]]
}

orig_ckpt="${STACK_RECORD_DIR}/final_model.int6.ptz"
orig_pt="${STACK_RECORD_DIR}/final_model.pt"
ckpt_backup="${RESULTS_DIR}/.record_final_model.int6.ptz"
pt_backup="${RESULTS_DIR}/.record_final_model.pt"

[[ -f "${orig_ckpt}" ]] && cp "${orig_ckpt}" "${ckpt_backup}"
[[ -f "${orig_pt}" ]] && cp "${orig_pt}" "${pt_backup}"

cleanup() {
  [[ -f "${ckpt_backup}" ]] && cp "${ckpt_backup}" "${orig_ckpt}" 2>/dev/null || true
  if [[ -f "${pt_backup}" ]]; then
    cp "${pt_backup}" "${orig_pt}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

install_seed_checkpoint() {
  local seed="$1"
  local src_dir="${CHECKPOINT_SOURCE_ROOT}/seed${seed}"
  [[ -f "${src_dir}/final_model.int6.ptz" ]] || {
    echo "Missing staged checkpoint for seed ${seed}: ${src_dir}/final_model.int6.ptz" >&2
    return 1
  }
  cp "${src_dir}/final_model.int6.ptz" "${orig_ckpt}"
}

seed_source_total_bytes() {
  local seed="$1"
  local src_console="${CHECKPOINT_SOURCE_ROOT}/seed${seed}/console.txt"
  if [[ -f "${src_console}" ]]; then
    extract_total_bytes_from_console "${src_console}"
  fi
}

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Candidate Seed Sweep                                              ║"
echo "║  Candidate: ${CANDIDATE}"
echo "║  Results: ${RESULTS_DIR}"
echo "╚══════════════════════════════════════════════════════════════════════╝"

first_run=1
for seed in "${SEEDS[@]}"; do
  if [[ "${MODE}" == "eval_only" ]]; then
    install_seed_checkpoint "${seed}"
    if [[ "${first_run}" == "1" ]]; then
      env PREPARE_SP8192=1 FETCH_PAYLOAD=0 RECORD_REL="${STACK_RECORD_REL}" \
        bash scripts/runpod_prepare_sp8192.sh > "${RESULTS_DIR}/env_prep.log" 2>&1 || true
      first_run=0
    fi
    run_id="${CANDIDATE}_s${seed}"
    run_one "${seed}" "${run_id}" "${EVAL_DESC}" "${EVAL_ENV[@]}"
    source_total_bytes=$(seed_source_total_bytes "${seed}" || true)
    if [[ -n "${source_total_bytes}" ]]; then
      echo "  source_total_bytes(seed${seed})=${source_total_bytes}" | tee -a "${RESULTS_DIR}/notes.txt"
    fi
  else
    [[ "${first_run}" == "1" ]] && prep_flag=1 || prep_flag=0
    first_run=0
    train_run_id="${TRAIN_NAME}_s${seed}"
    eval_run_id="${EVAL_NAME}_s${seed}"

    if ! run_one "${seed}" "${train_run_id}" "${TRAIN_DESC}" PREPARE_SP8192="${prep_flag}" "${TRAIN_ENV[@]}"; then
      echo "Training failed for seed ${seed}; aborting sweep." >&2
      exit 1
    fi

    train_log="${RESULTS_DIR}/seed${seed}/${train_run_id}/output.log"
    train_total_bytes=$(extract_total_bytes "${train_log}")
    if [[ "${REQUIRE_LEGAL}" == "1" && -n "${train_total_bytes}" && "${train_total_bytes}" -gt "${SIZE_CAP}" ]]; then
      echo "Seed ${seed} produced over-cap artifact (${train_total_bytes} > ${SIZE_CAP}); aborting sweep." >&2
      exit 1
    fi

    train_ckpt="${RESULTS_DIR}/seed${seed}/${train_run_id}/archive/final_model.int6.ptz"
    [[ -f "${train_ckpt}" ]] || {
      echo "Missing train checkpoint for seed ${seed}: ${train_ckpt}" >&2
      exit 1
    }
    cp "${train_ckpt}" "${orig_ckpt}"
    run_one "${seed}" "${eval_run_id}" "${EVAL_DESC}" "${EVAL_ENV[@]}"
  fi
done

echo ""
echo "==> Sweep complete. Summary:"
if command -v column >/dev/null 2>&1; then
  column -t -s$'\t' "${RESULTS_DIR}/summary.tsv"
else
  cat "${RESULTS_DIR}/summary.tsv"
fi
