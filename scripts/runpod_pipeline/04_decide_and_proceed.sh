#!/usr/bin/env bash
# Stage 4 decision point: reads ablation_summary.json and prints recommendation.
# Does NOT auto-trigger Stage 4. User makes the call after reviewing output.
set -euo pipefail

REPO_DIR="/workspace/parameter-golf"
RUNS_DIR="${REPO_DIR}/runs"
SUMMARY_FILE="${RUNS_DIR}/ablation_summary.json"
PYTHON="/opt/pg-venv/bin/python"

[ -f "${SUMMARY_FILE}" ] || {
    echo "ERROR: ${SUMMARY_FILE} not found — run 03_ablations.sh first" >&2; exit 1
}

"${PYTHON}" - "${SUMMARY_FILE}" "scripts/runpod_pipeline" <<'PY'
import json, sys

summary    = json.load(open(sys.argv[1]))
script_dir = sys.argv[2]
runs       = summary["runs"]
best       = summary["best_config"]
rec        = summary["recommended_path"]

sep = "=" * 62
print(f"\n{sep}")
print("  STAGE 4 DECISION POINT")
print(sep)
print(f"\n  Baseline BPB (Gate A, seed 0): {summary['baseline_bpb']:.8f}")
print(f"\n  Ablation results:")
print(f"  {'Label':<6} {'Alpha':>6} {'Orders':<10} {'BPB':>12} {'Delta':>10}")
print(f"  {'-'*50}")
for lbl, r in sorted(runs.items()):
    marker = " ← BEST" if lbl == best else ""
    print(f"  {lbl:<6} {r['config']['CORRECTOR_ALPHA']:>6} {r['config']['CORRECTOR_ORDERS']:<10} "
          f"{r['bpb']:>12.8f} {r['delta']:>+10.6f}{marker}")

all_below  = all(v["delta"] < 0.001 for v in runs.values())
best_delta = summary["best_delta"]
c          = summary["best_config_details"]

print(f"\n  Kill criterion (all deltas < 0.001): {'YES' if all_below else 'NO'}")
print(f"  Best delta: {best_delta:+.6f}  (threshold: 0.002 primary / 0.001 marginal)")
print(f"  Recommended path: {rec.upper()}")

if rec == "primary":
    print(f"\n  ✓  CLEAR SIGNAL — proceed with Stage 4 primary (Gate B)")
    print(f"     CORRECTOR_ALPHA={c['CORRECTOR_ALPHA']}  CORRECTOR_ORDERS={c['CORRECTOR_ORDERS']}")
    print(f"     Expected delta:  {best_delta:+.6f} BPB")
    print(f"\n     Command:")
    print(f"       BEST_ALPHA={c['CORRECTOR_ALPHA']} BEST_ORDERS='{c['CORRECTOR_ORDERS']}' \\")
    print(f"       bash {script_dir}/04a_gate_b.sh")

elif rec == "hold_human_decision":
    print(f"\n  ⚠   MARGINAL SIGNAL — human decision required")
    print(f"     Best delta {best_delta:+.6f} is in the 0.001–0.002 hold band.")
    print(f"     This is above the kill floor but below the clear-proceed threshold.")
    print(f"     At ~$0.36/min idle, decide within ~3 minutes or terminate and regroup.")
    print(f"\n     Option A — Accept marginal, run Gate B (~$14 more):")
    print(f"       BEST_ALPHA={c['CORRECTOR_ALPHA']} BEST_ORDERS='{c['CORRECTOR_ORDERS']}' \\")
    print(f"       bash {script_dir}/04a_gate_b.sh")
    print(f"\n     Option B — Reject marginal, run fallback requant (~$4 more):")
    print(f"       bash {script_dir}/04b_fallback_level1a.sh")
    print(f"\n     Option C — Terminate pod now, submit Gate A baseline reproduction.")

else:
    print(f"\n  ✗  CORRECTOR KILLED — all 3 configs showed < 0.001 BPB gain")
    print(f"     The corrector adds no meaningful signal.")
    print(f"     Submit clean #1610 reproduction + corrector-ablation negative result.")
    print(f"\n     Command (export-side fallback, optional):")
    print(f"       bash {script_dir}/04b_fallback_level1a.sh")

print(f"\n{sep}\n")
PY
