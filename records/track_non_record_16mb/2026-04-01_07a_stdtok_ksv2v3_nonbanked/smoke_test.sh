#!/usr/bin/env bash
set -uo pipefail

SMOKE_DIR="$(mktemp -d /tmp/smoke_07a_XXXXXX)"
SCRIPT="records/track_non_record_16mb/2026-04-01_07a_stdtok_ksv2v3_nonbanked/train_gpt.py"
OUTPUT="$SMOKE_DIR/smoke_output.txt"

echo "=========================================="
echo " 07a SMOKE TEST"
echo "=========================================="
echo "Date:      $(date -Iseconds)"
echo "Host:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Smoke dir: $SMOKE_DIR (kept on failure, cleaned on success)"
echo "=========================================="

export PYTHONUNBUFFERED=1
pip install --no-cache-dir sentencepiece brotli 2>/dev/null

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export ITERATIONS="${ITERATIONS:-50}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-25}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-10}"
export EVAL_STRIDE=0
export WINDOW_SIZE=-1
export WINDOW_ATTN_LAYERS=""
export SMOKE_SAVE_VE_INIT=1

for f in final_model.pt final_model.int6.br ve_init_snapshot.pt; do
    if [ -f "$f" ]; then
        rm -f "$f"
    fi
done

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

echo ""
echo ">>> Phase 1: Syntax check"
if python3 -m py_compile "$SCRIPT"; then
    echo "PASS: py_compile"
else
    echo "FAIL: py_compile"
    FAIL=1
fi

echo ""
echo ">>> Phase 2: Training + Export"
if python -u "$SCRIPT" 2>&1 | tee "$OUTPUT"; then
    echo "PASS: Training + export completed"
else
    echo "FAIL: Training script exited with error"
    FAIL=1
fi

echo ""
echo ">>> Phase 3: Post-run validation"
if [ -f final_model.int6.br ]; then
    ARTIFACT_SIZE=$(stat -c%s final_model.int6.br)
    CODE_SIZE=$(stat -c%s "$SCRIPT")
    TOTAL=$((ARTIFACT_SIZE + CODE_SIZE))
    echo "Artifact: $ARTIFACT_SIZE bytes"
    echo "Code:     $CODE_SIZE bytes"
    echo "Total:    $TOTAL bytes"
    check "Under 16MB cap ($TOTAL <= 16000000)" "$([ "$TOTAL" -le 16000000 ] && echo 1 || echo 0)"
else
    check "Artifact file exists" "0"
fi

if [ -f "$OUTPUT" ]; then
    for pattern in \
        "anchor:07a_stdtok_ksv2v3_nonbanked" \
        "features:layers=12" \
        "resid_lambdas=1" \
        "pre_quant_ema" \
        "final_int6_roundtrip"
    do
        check "Log contains '$pattern'" "$(grep -q "$pattern" "$OUTPUT" && echo 1 || echo 0)"
    done
fi

if [ -f "$OUTPUT" ]; then
    check "No NaN/Inf in training loss" "$(grep -qP 'train_loss:(nan|inf|-inf)' "$OUTPUT" && echo 0 || echo 1)"
    check "Roundtrip eval produced valid loss" "$(grep -qP 'final_int6_roundtrip val_loss:\d' "$OUTPUT" && echo 1 || echo 0)"
fi

echo ""
echo ">>> Phase 4: VE weight-change validation"
if [ -f final_model.pt ] && [ -f ve_init_snapshot.pt ]; then
    python3 -c "
import sys
import torch

init = torch.load('ve_init_snapshot.pt', map_location='cpu')
final = torch.load('final_model.pt', map_location='cpu')
ok = True
for key in sorted(init):
    if key not in final:
        ok = False
        print(f'FAIL: missing {key}')
        continue
    delta = (final[key].float() - init[key].float()).norm().item()
    print(f'{key}: delta_norm={delta:.6f}')
    if delta < 1e-8:
        ok = False
        print(f'FAIL: {key} unchanged')
if not ok:
    sys.exit(1)
" 2>&1 | tee -a "$OUTPUT"
    check "VE weights changed from init" "$([ ${PIPESTATUS[0]} -eq 0 ] && echo 1 || echo 0)"
else
    check "final_model.pt exists for VE check" "$([ -f final_model.pt ] && echo 1 || echo 0)"
    check "ve_init_snapshot.pt exists for VE check" "$([ -f ve_init_snapshot.pt ] && echo 1 || echo 0)"
fi

rm -f final_model.pt final_model.int6.br ve_init_snapshot.pt

echo ""
if [ "$FAIL" -eq 0 ]; then
    rm -rf "$SMOKE_DIR"
    echo "ALL SMOKE CHECKS PASSED"
else
    echo "SMOKE FAILED — output kept at $OUTPUT"
    exit 1
fi
