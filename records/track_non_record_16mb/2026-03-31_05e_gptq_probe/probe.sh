#!/usr/bin/env bash
# Session 05e: GPTQ Probe — two-phase: 50-step train → GPTQ replay
# All artifacts preserved in working directory for inspection.
# Must run from REPO ROOT (data paths are repo-root-relative).
set -euo pipefail

SCRIPT=records/track_non_record_16mb/2026-03-31_05e_gptq_probe/train_gpt.py
export PYTHONUNBUFFERED=1
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"
pip install --no-cache-dir sentencepiece zstandard 2>/dev/null

PHASE1_LOG=$(mktemp /tmp/probe_phase1_XXXXXX.log)
PHASE2_LOG=$(mktemp /tmp/probe_phase2_XXXXXX.log)

echo "ENABLE_TORCH_COMPILE=$ENABLE_TORCH_COMPILE"
echo "=== Phase 1: 50-step training to produce checkpoint ==="
ITERATIONS=50 \
MAX_WALLCLOCK_SECONDS=120 \
EVAL_STRIDE=0 \
VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 \
  python -u "$SCRIPT" 2>&1 | tee "$PHASE1_LOG"

# Verify checkpoint exists
if [ ! -f final_model.pt ]; then
    echo "FATAL: final_model.pt not produced by Phase 1"
    exit 1
fi
echo "Phase 1 done. Checkpoint: $(ls -lh final_model.pt)"

NAIVE_BPB=$(grep -oP 'final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$PHASE1_LOG" | tail -1)
echo "Phase 1 naive roundtrip BPB: ${NAIVE_BPB:-unknown}"

echo ""
echo "=== Phase 2: GPTQ replay on checkpoint ==="
EXPORT_ONLY_CHECKPOINT=./final_model.pt \
GPTQ_PROBE=1 \
EXPORT_TAG=gptq \
EXPORT_SKIP_SLIDING_EVAL=1 \
  python -u "$SCRIPT" 2>&1 | tee "$PHASE2_LOG"

GPTQ_BPB=$(grep -oP 'final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:\K[0-9.]+' "$PHASE2_LOG" | tail -1)
echo "Phase 2 GPTQ roundtrip BPB: ${GPTQ_BPB:-unknown}"

echo ""
echo "=== Probe complete ==="
echo "Artifacts:"
ls -lh final_model.pt final_model.int6.ptz final_model_gptq.int6.ptz gptq_layer_diagnostics_gptq.json 2>/dev/null || true
echo ""
echo "=== Decision ==="
echo "  Naive roundtrip BPB:  ${NAIVE_BPB:-unknown}"
echo "  GPTQ roundtrip BPB:   ${GPTQ_BPB:-unknown}"
if [ -f gptq_layer_diagnostics_gptq.json ]; then
    python3 -c "
import json
with open('gptq_layer_diagnostics_gptq.json') as f:
    d = json.load(f)
s = d['summary']
n = s['num_layers']
w = s['worse_than_naive_rowmax']
print(f'  Layers: {n}, worse_than_naive_rowmax: {w}')
naive = '${NAIVE_BPB:-}'
gptq = '${GPTQ_BPB:-}'
bpb_ok = False
if naive and gptq:
    try:
        bpb_ok = float(gptq) < float(naive)
        print(f'  GPTQ < naive BPB: {bpb_ok} ({gptq} vs {naive})')
    except ValueError:
        print('  BPB comparison: could not parse values')
if w > n // 2:
    print('  VERDICT: PARK GPTQ — majority of layers worse than naive')
elif w < 10 and bpb_ok:
    print('  VERDICT: GPTQ UNBLOCKED — fewer than 10 layers worse AND GPTQ BPB < naive BPB')
elif w < 10:
    print('  VERDICT: AMBIGUOUS — layer count looks good but GPTQ BPB did not beat naive')
else:
    print(f'  VERDICT: AMBIGUOUS — {w}/{n} layers worse, try on full 8xH100 checkpoint')
"
fi
rm -f "$PHASE1_LOG" "$PHASE2_LOG"
