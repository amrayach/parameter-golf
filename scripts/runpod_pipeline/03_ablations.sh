#!/usr/bin/env bash
# Stage 3: Corrector ablations — eval-only on persisted seed-0 checkpoint
# Runs 1a (alpha=0.3, orders=8), 1b (alpha=0.3, orders=5,8,12), 1c (alpha=0.1, orders=5,8,12)
# Writes runs/ablation_summary.json with BPB deltas and recommended path
# Kill criterion: if all three deltas < 0.001 BPB, sets recommended_path=fallback
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
TRAIN_SCRIPT="records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
RUNS_DIR="${REPO_DIR}/runs"
CKPT_DIR="/workspace/checkpoints/seed0"
GATE_A_SUMMARY="${RUNS_DIR}/gate_a_summary.json"
PYTHON="/opt/pg-venv/bin/python"

mkdir -p "${RUNS_DIR}"
exec > >(tee -a "${RUNS_DIR}/03_ablations.log") 2>&1

echo "=== Stage 3: Corrector ablations === $(date)"

cd "${REPO_DIR}"

# Prerequisites
[ -f "${GATE_A_SUMMARY}" ] || {
    echo "ERROR: ${GATE_A_SUMMARY} not found — run 02_gate_a.sh first" >&2; exit 1
}
[ -f "${CKPT_DIR}/final_model.int6.ptz" ] || {
    echo "ERROR: ${CKPT_DIR}/final_model.int6.ptz not found — run 02_gate_a.sh first" >&2; exit 1
}

BASELINE_BPB=$("${PYTHON}" -c "import json; print(json.load(open('${GATE_A_SUMMARY}'))['bpb'])")
echo "Baseline BPB (Gate A seed 0): ${BASELINE_BPB}"

# Remove stale partial results so idempotent re-run is clean
RESULTS_FILE="${RUNS_DIR}/ablation_results.json"
[ -f "${RESULTS_FILE}" ] && {
    echo "Removing stale ${RESULTS_FILE} for clean re-run"
    rm "${RESULTS_FILE}"
}

run_ablation() {
    local label="$1"
    local alpha="$2"
    local orders="$3"
    local ablation_dir="${RUNS_DIR}/ablation_${label}"
    local log_file="${RUNS_DIR}/ablation_${label}_log.txt"

    echo ""
    echo "--- Ablation ${label}: CORRECTOR_ALPHA=${alpha} CORRECTOR_ORDERS=${orders} ---"
    mkdir -p "${ablation_dir}"

    EVAL_ONLY_QUANTIZED_PATH="${CKPT_DIR}/final_model.int6.ptz" \
    ARTIFACT_DIR="${ablation_dir}" \
    CORRECTOR_ALPHA="${alpha}" \
    CORRECTOR_ORDERS="${orders}" \
    PHASED_TTT_ENABLED=1 \
    PHASED_TTT_PREFIX_DOCS=2000 \
    PYTHONUNBUFFERED=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NCCL_DEBUG=WARN \
        /opt/pg-venv/bin/torchrun \
        --standalone --nproc_per_node=8 \
        "${TRAIN_SCRIPT}" \
        2>&1 | tee "${log_file}"

    "${PYTHON}" - "${log_file}" "${BASELINE_BPB}" "${label}" "${alpha}" "${orders}" "${RESULTS_FILE}" <<'PY'
import re, sys, json, pathlib

log_path   = pathlib.Path(sys.argv[1])
baseline   = float(sys.argv[2])
label      = sys.argv[3]
alpha      = sys.argv[4]
orders     = sys.argv[5]
results_f  = pathlib.Path(sys.argv[6])
log_text   = log_path.read_text()

m = re.search(
    r"quantized_ttt_phased val_loss:[0-9.]+ val_bpb:([0-9.]+) eval_time:([0-9]+)ms",
    log_text
)
if not m:
    print(f"ERROR: ablation {label}: quantized_ttt_phased line not found in {log_path}", flush=True)
    sys.exit(1)

bpb     = float(m.group(1))
eval_ms = int(m.group(2))
delta   = baseline - bpb  # positive = improvement

ms = re.search(r"Total submission size quantized\+\S+: ([0-9]+) bytes", log_text)
artifact_bytes = int(ms.group(1)) if ms else 0

print(f"  bpb={bpb:.8f}  delta={delta:+.6f}  eval={eval_ms/1000:.1f}s  artifact={artifact_bytes}B")

results = json.loads(results_f.read_text()) if results_f.exists() else {}
results[label] = {
    "config": {"CORRECTOR_ALPHA": alpha, "CORRECTOR_ORDERS": orders},
    "bpb": bpb, "delta": delta, "eval_ms": eval_ms,
    "artifact_bytes": artifact_bytes, "log": str(log_path)
}
results_f.write_text(json.dumps(results, indent=2))
PY
    echo "  ablation ${label}: recorded"
}

run_ablation "1a" "0.3" "8"
run_ablation "1b" "0.3" "5,8,12"
run_ablation "1c" "0.1" "5,8,12"

echo ""
echo "=== Ablation summary ==="
"${PYTHON}" - "${RUNS_DIR}" "${BASELINE_BPB}" <<'PY'
import json, pathlib, sys

runs_dir     = pathlib.Path(sys.argv[1])
baseline_bpb = float(sys.argv[2])
results_f    = runs_dir / "ablation_results.json"

if not results_f.exists():
    print("ERROR: ablation_results.json not found — at least one run failed", flush=True)
    sys.exit(1)

results = json.loads(results_f.read_text())
if len(results) < 3:
    print(f"WARNING: only {len(results)}/3 ablations completed", flush=True)

best_label = max(results, key=lambda k: results[k]["delta"])
best_delta  = results[best_label]["delta"]
all_below   = all(v["delta"] < 0.001 for v in results.values())
# Three-way fork per AGENT_SYNC.md:22:
#   >= 0.002         -> primary (run Gate B)
#   0.001 <= x < 0.002 -> hold_human_decision (print both options, do not auto-proceed)
#   all < 0.001      -> fallback
if all_below:
    recommended = "fallback"
elif best_delta >= 0.002:
    recommended = "primary"
else:
    recommended = "hold_human_decision"

summary = {
    "baseline_bpb":        baseline_bpb,
    "runs":                results,
    "best_config":         best_label,
    "best_config_details": results[best_label]["config"],
    "best_delta":          best_delta,
    "recommended_path":    recommended
}
out = runs_dir / "ablation_summary.json"
out.write_text(json.dumps(summary, indent=2))

print(f"{'Label':<6} {'Alpha':>6} {'Orders':<10} {'BPB':>12} {'Delta':>10}")
print("-" * 52)
for lbl, r in sorted(results.items()):
    marker = " ←" if lbl == best_label else "  "
    print(f"{lbl:<6} {r['config']['CORRECTOR_ALPHA']:>6} {r['config']['CORRECTOR_ORDERS']:<10} "
          f"{r['bpb']:>12.8f} {r['delta']:>+10.6f}{marker}")

print(f"\nKill criterion (all deltas < 0.001): {'YES' if all_below else 'NO'}")
if recommended == "primary":
    print(f"Recommended path: PRIMARY  (best delta {best_delta:+.6f} >= 0.002 threshold)")
elif recommended == "hold_human_decision":
    print(f"Recommended path: HOLD/HUMAN DECISION  (best delta {best_delta:+.6f} in 0.001–0.002 band)")
else:
    print(f"Recommended path: FALLBACK  (all deltas < 0.001)")
print(f"\nSummary → {out}")
PY

echo ""
echo "03_ablations: DONE"
echo "Next: bash scripts/runpod_pipeline/04_decide_and_proceed.sh"
