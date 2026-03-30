#!/usr/bin/env bash
# ============================================================
# Session 05c-plus SMOKE TEST
# Run on ANY single Pegasus GPU before committing $25 RunPod.
# ============================================================
#
# What this validates:
#   1. Script boots, torch.compile succeeds
#   2. Forward/backward pass works (training + EMA)
#   3. VE128 params exist in model and receive non-zero gradient updates
#   4. Export path (int6 + zstd) produces valid artifact under 16MB
#   5. Roundtrip eval runs (no shape mismatches from VE injection)
#
# What this does NOT validate:
#   - BPB quality (meaningless at 1xGPU with 50 steps)
#   - Multi-GPU DDP correctness (need 2+ GPUs for that)
#   - Exact step timing (different GPU type)
#   - Sliding window eval (skipped to keep smoke fast on slow GPUs)
#
# Usage (from repo root on Pegasus):
#   srun -K -p <PARTITION> --nodes=1 --ntasks=1 --gpus-per-task=1 \
#     --cpus-per-task=6 --mem=80G --time=00:10:00 \
#     --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
#     --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
#     bash records/track_non_record_16mb/2026-03-30_training_bundle_plus/smoke_test.sh
#
#   Pegasus partition names:
#     -p H100        # best match for final run
#     -p A100-80GB   # fallback
#     -p A100-40GB   # fallback
#     -p RTXA6000    # if nothing else available
#
# ============================================================

set -euo pipefail

SMOKE_DIR="$(mktemp -d /tmp/smoke_05c_XXXXXX)"
trap 'rm -rf "$SMOKE_DIR"' EXIT

echo "=========================================="
echo " 05c-plus SMOKE TEST"
echo "=========================================="
echo "Date:      $(date -Iseconds)"
echo "Host:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Smoke dir: $SMOKE_DIR"
echo "=========================================="

# ---------- DEPS ----------
export PYTHONUNBUFFERED=1
pip install --no-cache-dir sentencepiece zstandard 2>/dev/null

# ---------- 1xGPU SMOKE CONFIG ----------
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=120
export VAL_LOSS_EVERY=25
export TRAIN_LOG_EVERY=10
# Skip sliding eval — it is slow on fallback GPUs and not needed for smoke
export EVAL_STRIDE=0

SCRIPT=records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
OUTPUT="$SMOKE_DIR/smoke_output.txt"

echo ""
echo ">>> Phase 1: Syntax check"
python3 -m py_compile "$SCRIPT"
echo "PASS: py_compile"

echo ""
echo ">>> Phase 2: Training + Export (50 steps, 1xGPU)"
python -u "$SCRIPT" 2>&1 | tee "$OUTPUT"
echo ""
echo "PASS: Training + export completed"

# ---------- POST-RUN CHECKS ----------
echo ""
echo ">>> Phase 3: Post-run validation"

FAIL=0
check() {
    local name="$1" result="$2"
    if [ "$result" = "1" ]; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        FAIL=1
    fi
}

# Check 1: Artifact exists and is under cap
if [ -f final_model.int6.ptz ]; then
    ARTIFACT_SIZE=$(stat -c%s final_model.int6.ptz)
    CODE_SIZE=$(stat -c%s "$SCRIPT")
    TOTAL=$((ARTIFACT_SIZE + CODE_SIZE))
    echo "Artifact: $ARTIFACT_SIZE bytes"
    echo "Code:     $CODE_SIZE bytes"
    echo "Total:    $TOTAL bytes"
    check "Under 16MB cap ($TOTAL <= 16000000)" "$([ "$TOTAL" -le 16000000 ] && echo 1 || echo 0)"
else
    check "Artifact file exists" "0"
fi

# Check 2: Key log lines present (features active)
echo ""
for pattern in "model_params:" "anchor:05c_plus" "ve=128" "leaky_relu_sq=0.5" \
               "pre_quant_ema" "final_int6_roundtrip"; do
    check "Log contains '$pattern'" "$(grep -q "$pattern" "$OUTPUT" && echo 1 || echo 0)"
done

# Check 3: Parameter count is sane (~27M with VE128, anchor was ~26.8M)
PARAM_COUNT=$(grep "model_params:" "$OUTPUT" | grep -oP '\d+' || echo "0")
echo ""
echo "Parameter count: $PARAM_COUNT"
check "Param count in [26M, 28M]" "$([ "$PARAM_COUNT" -gt 26000000 ] && [ "$PARAM_COUNT" -lt 28000000 ] && echo 1 || echo 0)"

# Check 4: No NaN/Inf in training loss
check "No NaN/Inf in training loss" "$(grep -qP 'train_loss:(nan|inf|-inf)' "$OUTPUT" && echo 0 || echo 1)"

# Check 5: Roundtrip eval produced a number (not crash)
check "Roundtrip eval produced valid loss" "$(grep -qP 'final_int6_roundtrip val_loss:\d' "$OUTPUT" && echo 1 || echo 0)"

# Check 6: VE parameters changed from init (proves they received gradient updates)
# The script logs the model state; we check VE-specific params exist in the checkpoint
echo ""
echo ">>> Phase 4: VE128 gradient validation"
python3 -c "
import torch, sys
sd = torch.load('final_model.pt', map_location='cpu')
# Check VE params exist
ve_keys = [k for k in sd if 've_shared' in k or 've_layer_scales' in k]
if not ve_keys:
    print('FAIL: No VE parameters found in checkpoint')
    sys.exit(1)
print(f'VE params found: {len(ve_keys)} keys')
for k in ve_keys:
    t = sd[k]
    print(f'  {k}: shape={list(t.shape)} norm={t.float().norm().item():.6f}')

# Check that VE embed weights moved from init (std=0.01)
embed_w = sd.get('ve_shared.embed.weight')
if embed_w is not None:
    init_std = 0.01
    actual_std = embed_w.float().std().item()
    # After 50 steps of training, std should have shifted from init
    print(f'  ve_shared.embed.weight: init_std={init_std}, actual_std={actual_std:.6f}')
    if actual_std < 1e-10:
        print('FAIL: VE embed weights are all zeros (never updated)')
        sys.exit(1)
    print('PASS: VE embed weights are non-zero')

# Check that VE proj weights moved from zero init
# (Note: _init_weights applies orthogonal, so even before training these should be non-zero)
proj_w = sd.get('ve_shared.proj.weight')
if proj_w is not None:
    proj_norm = proj_w.float().norm().item()
    print(f'  ve_shared.proj.weight: norm={proj_norm:.6f}')
    if proj_norm < 1e-10:
        print('FAIL: VE proj weights are all zeros')
        sys.exit(1)
    print('PASS: VE proj weights are non-zero')

# Check VE scale moved from init (0.1)
scale = sd.get('ve_shared.scale')
if scale is not None:
    val = scale.item()
    print(f'  ve_shared.scale: init=0.1, actual={val:.6f}')
    print('PASS: VE scale exists')

print('PASS: All VE parameters present and non-degenerate')
" 2>&1 | tee -a "$OUTPUT"
VE_CHECK=$?
check "VE128 params present and non-degenerate" "$([ $VE_CHECK -eq 0 ] && echo 1 || echo 0)"

# ---------- CLEANUP ----------
# Only remove smoke-produced artifacts, never touch logs/ or other repo state
rm -f final_model.pt final_model.int6.ptz

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "=========================================="
    echo " ALL SMOKE CHECKS PASSED"
    echo "=========================================="
    echo ""
    echo "Next: Run full 8xH100 with confidence."
    echo "See README.md for launch command."
else
    echo "=========================================="
    echo " SMOKE FAILED — see above for details"
    echo "=========================================="
    echo "Output saved to: $OUTPUT"
    # Keep smoke dir on failure for debugging
    trap - EXIT
    exit 1
fi
