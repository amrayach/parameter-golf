#!/usr/bin/env bash
set -uo pipefail

# RunPod batch orchestration for PR #1413 optimization interventions.
# Usage: bash scripts/run_pod_batch.sh [tier1|tier2|all]
#
# Tier 1: eval-only runs reusing D checkpoint (~5 min each)
# Tier 2: full training runs with new interventions (~12 min each)
#
# Prerequisite: run `python3 scripts/prepare_pr1413_variants.py --force`
# and sync the repo to the RunPod pod before launching.

usage() {
  cat <<'EOF'
Usage: bash scripts/run_pod_batch.sh [tier1|tier2|all]

  tier1  Eval-only sweeps (R1-R5) reusing D checkpoint
  tier2  Full training runs (R6-R9) with new interventions
  all    Both tiers sequentially (default)
EOF
}

TIER="${1:-all}"
case "${TIER}" in
  tier1|tier2|all) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown tier: ${TIER}" >&2; usage; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_RECORD_REL="records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep"
STACK_RECORD_DIR="${REPO_ROOT}/${STACK_RECORD_REL}"

if [[ ! -f "${STACK_RECORD_DIR}/train_gpt.py" ]]; then
  echo "Missing stack payload. Run: python3 scripts/prepare_pr1413_variants.py --force" >&2
  exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_STAMP
RESULTS_DIR="/workspace/optimization_batch_${RUN_STAMP}"
mkdir -p "${RESULTS_DIR}"

# Common env vars for all runs
COMMON_SEED=0
COMMON_PARALLEL_RESIDUAL_START=7
COMMON_LOOP_START=3
COMMON_LOOP_END=5

# TSV header
echo -e "timestamp\trun_name\tbpb\twall_seconds\tenv_vars" > "${RESULTS_DIR}/summary.tsv"

extract_bpb() {
  # Extract the best BPB from a run log.
  # Format reference (from console.txt):
  #   legal_ttt_exact val_loss:2.79177797 val_bpb:1.08078425 eval_time:324216ms
  #   ttt_sliding:done val_loss=2.791778 val_bpb=1.080784 elapsed=324.0s
  #   quantized_sliding_window val_loss:2.79649863 val_bpb:1.08261177 eval_time:91365ms
  #   quantized val_loss:2.83976191 val_bpb:1.09936033 eval_time:9661ms
  local logfile="$1"
  local bpb=""
  # Prefer legal_ttt_exact BPB (uses colon separator)
  bpb=$(grep -oP 'legal_ttt_exact.*?val_bpb:\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  if [[ -z "$bpb" ]]; then
    # ttt_sliding internal log (uses equals separator)
    bpb=$(grep -oP 'ttt_sliding:done.*?val_bpb=\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  fi
  if [[ -z "$bpb" ]]; then
    bpb=$(grep -oP 'quantized_sliding_window.*?val_bpb:\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  fi
  if [[ -z "$bpb" ]]; then
    # Plain "quantized" with space to avoid matching quantized_sliding_window
    bpb=$(grep -oP 'quantized val_loss:.*?val_bpb:\K[0-9]+\.[0-9]+' "$logfile" 2>/dev/null | tail -1)
  fi
  echo "${bpb:-FAILED}"
}

run_experiment() {
  local run_name="$1"
  shift
  local env_desc="$1"
  shift
  # Remaining args are env var assignments

  local run_dir="${RESULTS_DIR}/${run_name}"
  mkdir -p "${run_dir}"
  local t0
  t0=$(date +%s)

  echo ""
  echo "========================================================================"
  echo "  RUN: ${run_name}"
  echo "  ENV: ${env_desc}"
  echo "  TIME: $(date -Iseconds)"
  echo "========================================================================"

  local archive_dir="${run_dir}/archive"
  mkdir -p "${archive_dir}"

  # Run in subshell with set -e for fast-fail; || rc=$? lets the outer batch continue
  local rc=0
  (
    set -e
    env \
      FETCH_PAYLOAD=0 \
      RECORD_REL="${STACK_RECORD_REL}" \
      RUN_ID="${run_name}" \
      SEED="${COMMON_SEED}" \
      PREPARE_SP8192=0 \
      ARCHIVE_DIR="${archive_dir}" \
      PARALLEL_RESIDUAL_START="${COMMON_PARALLEL_RESIDUAL_START}" \
      LOOP_START="${COMMON_LOOP_START}" \
      LOOP_END="${COMMON_LOOP_END}" \
      "$@" \
      bash scripts/runpod_1413.sh "${COMMON_SEED}"
  ) > "${run_dir}/output.log" 2>&1 || rc=$?

  local t1
  t1=$(date +%s)
  local wall=$((t1 - t0))

  if [[ $rc -ne 0 ]]; then
    echo "  FAILED (rc=${rc}) after ${wall}s — see ${run_dir}/output.log"
    echo -e "$(date -Iseconds)\t${run_name}\tFAILED\t${wall}\t${env_desc}" >> "${RESULTS_DIR}/summary.tsv"
    return 0  # Don't abort batch on individual run failure
  fi

  local bpb
  bpb=$(extract_bpb "${run_dir}/output.log")
  echo "  DONE: BPB=${bpb} wall=${wall}s"
  echo -e "$(date -Iseconds)\t${run_name}\t${bpb}\t${wall}\t${env_desc}" >> "${RESULTS_DIR}/summary.tsv"

  # Copy any produced artifacts
  for art in final_model.int6.ptz final_model.pt; do
    if [[ -f "${archive_dir}/${art}" ]]; then
      cp -v "${archive_dir}/${art}" "${run_dir}/" 2>/dev/null || true
    fi
  done

  return 0
}

get_bpb_from_summary() {
  local run_name="$1"
  awk -F'\t' -v name="$run_name" '$2 == name && $3 != "FAILED" { print $3 }' "${RESULTS_DIR}/summary.tsv" | tail -1
}

# ==========================================================================
#  TIER 1: Eval-only runs (reuse D checkpoint via SKIP_TRAINING=1)
# ==========================================================================
run_tier1() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  TIER 1: Eval-only sweeps (R1-R5)                                  ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"

  # Ensure env prep runs once
  echo "==> Preparing SP8192 environment..."
  env PREPARE_SP8192=1 FETCH_PAYLOAD=0 RECORD_REL="${STACK_RECORD_REL}" \
    bash scripts/runpod_prepare_sp8192.sh 2>&1 | tee "${RESULTS_DIR}/env_prep.log" || true

  # R1: E baseline
  run_experiment "R1_e_baseline" \
    "SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1" \
    SKIP_TRAINING=1 \
    NGRAM_TILT_ENABLED=1 \
    TTT_ENABLED=1

  # R2: E + RMSDecay low
  run_experiment "R2_e_rmsdecay_low" \
    "SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_OPTIMIZER=rmsdecay TTT_DECAY=0.001" \
    SKIP_TRAINING=1 \
    NGRAM_TILT_ENABLED=1 \
    TTT_ENABLED=1 \
    TTT_OPTIMIZER=rmsdecay \
    TTT_DECAY=0.001

  # R3: E + RMSDecay high
  run_experiment "R3_e_rmsdecay_high" \
    "SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_OPTIMIZER=rmsdecay TTT_DECAY=0.005" \
    SKIP_TRAINING=1 \
    NGRAM_TILT_ENABLED=1 \
    TTT_ENABLED=1 \
    TTT_OPTIMIZER=rmsdecay \
    TTT_DECAY=0.005

  # R4: E + freeze 4
  run_experiment "R4_e_freeze4" \
    "SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_FREEZE_BLOCKS=4" \
    SKIP_TRAINING=1 \
    NGRAM_TILT_ENABLED=1 \
    TTT_ENABLED=1 \
    TTT_FREEZE_BLOCKS=4

  # R5: dynamic combo of best eval-time settings
  local best_decay_run=""
  local best_decay_bpb="9999"
  local r2_bpb r3_bpb r4_bpb r1_bpb

  r1_bpb=$(get_bpb_from_summary "R1_e_baseline")
  r2_bpb=$(get_bpb_from_summary "R2_e_rmsdecay_low")
  r3_bpb=$(get_bpb_from_summary "R3_e_rmsdecay_high")
  r4_bpb=$(get_bpb_from_summary "R4_e_freeze4")

  # Pick best decay
  local r5_ttt_optimizer="sgd"
  local r5_ttt_decay="0"
  local r5_freeze=""
  local r5_desc="SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1"

  if [[ -n "$r2_bpb" && -n "$r3_bpb" ]]; then
    if awk "BEGIN { exit !($r2_bpb < $r3_bpb) }"; then
      r5_ttt_optimizer="rmsdecay"; r5_ttt_decay="0.001"
    else
      r5_ttt_optimizer="rmsdecay"; r5_ttt_decay="0.005"
    fi
    r5_desc="${r5_desc} TTT_OPTIMIZER=${r5_ttt_optimizer} TTT_DECAY=${r5_ttt_decay}"
  fi

  # Check if freeze helped vs baseline
  if [[ -n "$r4_bpb" && -n "$r1_bpb" ]]; then
    if awk "BEGIN { exit !($r4_bpb < $r1_bpb) }"; then
      r5_freeze="4"
      r5_desc="${r5_desc} TTT_FREEZE_BLOCKS=4"
    fi
  fi

  local r5_extra_env=()
  r5_extra_env+=(SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_ENABLED=1)
  r5_extra_env+=(TTT_OPTIMIZER="${r5_ttt_optimizer}" TTT_DECAY="${r5_ttt_decay}")
  if [[ -n "$r5_freeze" ]]; then
    r5_extra_env+=(TTT_FREEZE_BLOCKS="${r5_freeze}")
  fi

  echo "  R5 dynamic config: ${r5_desc}"
  run_experiment "R5_e_combo" "${r5_desc}" "${r5_extra_env[@]}"

  echo ""
  echo "==> Tier 1 complete. Summary:"
  column -t -s$'\t' "${RESULTS_DIR}/summary.tsv"
}

# ==========================================================================
#  Helper: determine best eval-time config from tier 1
# ==========================================================================
best_eval_env() {
  # Returns the env vars for the best tier 1 eval config.
  # Always includes NGRAM_TILT_ENABLED=1.
  # Falls back to TTT_OPTIMIZER=rmsdecay TTT_DECAY=0.001 if nothing worked.
  local best_run="" best_bpb="9999"
  for rn in R1_e_baseline R2_e_rmsdecay_low R3_e_rmsdecay_high R4_e_freeze4 R5_e_combo; do
    local bpb
    bpb=$(get_bpb_from_summary "$rn")
    if [[ -n "$bpb" && "$bpb" != "FAILED" ]]; then
      if awk "BEGIN { exit !($bpb < $best_bpb) }"; then
        best_bpb="$bpb"
        best_run="$rn"
      fi
    fi
  done

  local env_str=""
  if [[ -n "$best_run" ]]; then
    # Parse the winning run's env from summary.tsv
    env_str=$(awk -F'\t' -v name="$best_run" '$2 == name { print $5 }' "${RESULTS_DIR}/summary.tsv" | tail -1)
  fi

  # Fallback if empty: safe default eval config
  if [[ -z "$env_str" ]]; then
    env_str="SKIP_TRAINING=1 NGRAM_TILT_ENABLED=1 TTT_OPTIMIZER=rmsdecay TTT_DECAY=0.001"
  fi

  # Ensure NGRAM_TILT_ENABLED=1 is always present
  if [[ "$env_str" != *NGRAM_TILT_ENABLED=1* ]]; then
    env_str="NGRAM_TILT_ENABLED=1 ${env_str}"
  fi

  echo "$env_str"
}


# ==========================================================================
#  TIER 2: Training runs with new interventions
# ==========================================================================
run_tier2() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║  TIER 2: Training runs with interventions (R6-R9)                  ║"
  echo "╚══════════════════════════════════════════════════════════════════════╝"

  # Get best eval-time config from tier 1 (if available)
  local eval_env_str
  eval_env_str=$(best_eval_env 2>/dev/null || echo "NGRAM_TILT_ENABLED=1")
  echo "==> Best eval-time config: ${eval_env_str}"

  # Backup original checkpoint so tier 2 runs don't permanently overwrite it
  local orig_ckpt="${STACK_RECORD_DIR}/final_model.int6.ptz"
  local ckpt_backup="${RESULTS_DIR}/.original_checkpoint.int6.ptz"
  if [[ -f "${orig_ckpt}" ]]; then
    cp "${orig_ckpt}" "${ckpt_backup}"
    echo "==> Backed up original checkpoint to ${ckpt_backup}"
  fi

  # Trap: restore checkpoint on any exit (crash, signal, or normal completion)
  _ckpt_cleanup() {
    if [[ -f "${ckpt_backup}" && -f "${orig_ckpt}" ]]; then
      cp "${ckpt_backup}" "${orig_ckpt}" 2>/dev/null || true
      echo "==> Checkpoint restored from backup (trap)"
    fi
  }
  trap _ckpt_cleanup EXIT

  # Helper: restore original checkpoint before each training run
  restore_checkpoint() {
    if [[ -f "${ckpt_backup}" ]]; then
      cp "${ckpt_backup}" "${orig_ckpt}"
    fi
  }

  # Helper: install a training run's checkpoint for its eval pass
  install_checkpoint() {
    local src="$1"
    if [[ -f "$src" ]]; then
      cp "$src" "${orig_ckpt}"
      return 0
    fi
    return 1
  }

  # --- R6: Cautious Muon training, then eval with best eval-time config ---
  echo ""
  echo "--- R6: Cautious Muon (training) ---"
  restore_checkpoint
  run_experiment "R6_d_cautious_muon_train" \
    "CAUTIOUS_MUON=1" \
    SKIP_TRAINING=0 \
    CAUTIOUS_MUON=1 \
    TTT_ENABLED=1

  # Eval pass: install this run's checkpoint for eval
  if install_checkpoint "${RESULTS_DIR}/R6_d_cautious_muon_train/archive/final_model.int6.ptz"; then
    local r6_eval_args
    r6_eval_args=(SKIP_TRAINING=1 TTT_ENABLED=1 CAUTIOUS_MUON=1)
    for pair in $eval_env_str; do
      r6_eval_args+=("$pair")
    done
    run_experiment "R6_d_cautious_muon_eval" \
      "CAUTIOUS_MUON=1 + best_eval(${eval_env_str})" \
      "${r6_eval_args[@]}"
  else
    echo "  R6 training failed, skipping eval pass"
  fi

  # --- R7: OWC training, then eval ---
  echo ""
  echo "--- R7: OWC (training) ---"
  restore_checkpoint
  run_experiment "R7_d_owc_train" \
    "OWC_ENABLED=1 OWC_GAMMA_STEPS=10" \
    SKIP_TRAINING=0 \
    OWC_ENABLED=1 \
    OWC_GAMMA_STEPS=10 \
    TTT_ENABLED=1

  if install_checkpoint "${RESULTS_DIR}/R7_d_owc_train/archive/final_model.int6.ptz"; then
    local r7_eval_args
    r7_eval_args=(SKIP_TRAINING=1 TTT_ENABLED=1 OWC_ENABLED=1 OWC_GAMMA_STEPS=10)
    for pair in $eval_env_str; do
      r7_eval_args+=("$pair")
    done
    run_experiment "R7_d_owc_eval" \
      "OWC_ENABLED=1 + best_eval(${eval_env_str})" \
      "${r7_eval_args[@]}"
  else
    echo "  R7 training failed, skipping eval pass"
  fi

  # --- R8: CDQuant + OWC (training) ---
  # No separate timing probe — the training script's own GPTQ budget guard
  # will catch if CDQuant exceeds the 12s reserve.
  echo ""
  echo "--- R8: CDQuant + OWC (training) ---"
  restore_checkpoint
  run_experiment "R8_d_cdquant_owc_train" \
    "CDQUANT_ENABLED=1 CDQUANT_ITERS=3 OWC_ENABLED=1 OWC_GAMMA_STEPS=10" \
    SKIP_TRAINING=0 \
    CDQUANT_ENABLED=1 \
    CDQUANT_ITERS=3 \
    OWC_ENABLED=1 \
    OWC_GAMMA_STEPS=10 \
    TTT_ENABLED=1

  if install_checkpoint "${RESULTS_DIR}/R8_d_cdquant_owc_train/archive/final_model.int6.ptz"; then
    local r8_eval_args
    r8_eval_args=(SKIP_TRAINING=1 TTT_ENABLED=1 CDQUANT_ENABLED=1 CDQUANT_ITERS=3 OWC_ENABLED=1 OWC_GAMMA_STEPS=10)
    for pair in $eval_env_str; do
      r8_eval_args+=("$pair")
    done
    run_experiment "R8_d_cdquant_owc_eval" \
      "CDQuant+OWC + best_eval(${eval_env_str})" \
      "${r8_eval_args[@]}"
  else
    echo "  R8 training failed, skipping eval pass"
  fi

  # --- R9: Full stack (Cautious Muon + OWC + CDQuant + best eval) ---
  echo ""
  echo "--- R9: Full stack (training) ---"
  restore_checkpoint
  run_experiment "R9_d_full_stack_train" \
    "CAUTIOUS_MUON=1 OWC_ENABLED=1 OWC_GAMMA_STEPS=10 CDQUANT_ENABLED=1 CDQUANT_ITERS=3" \
    SKIP_TRAINING=0 \
    CAUTIOUS_MUON=1 \
    OWC_ENABLED=1 \
    OWC_GAMMA_STEPS=10 \
    CDQUANT_ENABLED=1 \
    CDQUANT_ITERS=3 \
    TTT_ENABLED=1

  if install_checkpoint "${RESULTS_DIR}/R9_d_full_stack_train/archive/final_model.int6.ptz"; then
    local r9_eval_args
    r9_eval_args=(SKIP_TRAINING=1 TTT_ENABLED=1 CAUTIOUS_MUON=1 OWC_ENABLED=1 OWC_GAMMA_STEPS=10 CDQUANT_ENABLED=1 CDQUANT_ITERS=3)
    for pair in $eval_env_str; do
      r9_eval_args+=("$pair")
    done
    run_experiment "R9_d_full_stack_eval" \
      "full_stack + best_eval(${eval_env_str})" \
      "${r9_eval_args[@]}"
  else
    echo "  R9 training failed, skipping eval pass"
  fi

  # Restore original checkpoint so record dir is clean
  restore_checkpoint
  echo ""
  echo "==> Tier 2 complete."
}

# ==========================================================================
#  MAIN: Final ranking
# ==========================================================================

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Parameter Golf Optimization Batch                                 ║"
echo "║  Results: ${RESULTS_DIR}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "Start: $(date -Iseconds)"
echo ""

case "${TIER}" in
  tier1) run_tier1 ;;
  tier2) run_tier2 ;;
  all)   run_tier1; run_tier2 ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  FINAL RANKING                                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Sort by BPB (column 3), exclude header and FAILED runs
{
  head -1 "${RESULTS_DIR}/summary.tsv"
  tail -n +2 "${RESULTS_DIR}/summary.tsv" | grep -v FAILED | sort -t$'\t' -k3 -n
  echo "---"
  tail -n +2 "${RESULTS_DIR}/summary.tsv" | grep FAILED || true
} | tee "${RESULTS_DIR}/final_ranking.txt" | column -t -s$'\t'

echo ""
echo "End: $(date -Iseconds)"
echo "Results saved to: ${RESULTS_DIR}"
echo "Summary: ${RESULTS_DIR}/summary.tsv"
echo "Ranking: ${RESULTS_DIR}/final_ranking.txt"
