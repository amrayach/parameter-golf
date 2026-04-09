#!/usr/bin/env bash
set -euo pipefail

# RunPod pod setup for PR #1413 optimization batch.
# Run this ONCE after SSH-ing into the pod.
#
# Usage:
#   bash scripts/setup_runpod.sh              # if already in /workspace/parameter-golf
#   bash /workspace/parameter-golf/scripts/setup_runpod.sh   # from anywhere
#
# Prerequisites:
#   1. Pod deployed with the official Parameter Golf template
#   2. SSH access configured
#   3. D checkpoint scp'd to pod (see CHECKPOINT section below)

REPO_URL="git@github.com:amrayach/parameter-golf.git"
BRANCH="07c1-base-runpod-strict-submission"
STACK_RECORD="records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Parameter Golf — RunPod Setup                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Clone or update repo ──────────────────────────────────────────────
echo "==> [1/6] Syncing repository..."
cd /workspace
if [ -d "parameter-golf" ]; then
    cd parameter-golf
    git fetch origin
    git checkout "${BRANCH}"
    git reset --hard "origin/${BRANCH}"
    echo "    Repo updated to latest ${BRANCH}"
else
    git clone --branch "${BRANCH}" "${REPO_URL}" parameter-golf
    cd parameter-golf
    echo "    Repo cloned on branch ${BRANCH}"
fi
echo "    HEAD: $(git rev-parse --short HEAD) — $(git log -1 --format=%s)"
echo ""

# ── 2. Download SP8192 dataset ───────────────────────────────────────────
echo "==> [2/6] Downloading SP8192 dataset (this takes a few minutes)..."
if [ -d "data/datasets/fineweb10B_sp8192" ]; then
    EXISTING_SHARDS=$(ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)
    if [ "${EXISTING_SHARDS}" -ge 80 ]; then
        echo "    Dataset already present (${EXISTING_SHARDS} train shards). Skipping download."
    else
        echo "    Partial dataset found (${EXISTING_SHARDS} shards). Re-downloading..."
        python3 data/cached_challenge_fineweb.py --variant sp8192
    fi
else
    python3 data/cached_challenge_fineweb.py --variant sp8192
fi
echo ""

# ── 3. Verify dataset ───────────────────────────────────────────────────
echo "==> [3/6] Verifying dataset..."
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "    Train shards: ${TRAIN_COUNT}"
echo "    Val shards:   ${VAL_COUNT}"
if [ "${TRAIN_COUNT}" -lt 80 ]; then
    echo "    WARNING: Expected >=80 train shards, got ${TRAIN_COUNT}"
fi
ls data/tokenizers/fineweb_*8192* 2>/dev/null | head -2 | sed 's/^/    Tokenizer: /'
echo ""

# ── 4. Verify GPU setup ─────────────────────────────────────────────────
echo "==> [4/6] Verifying GPU setup..."
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "    GPU count: ${GPU_COUNT}"
nvidia-smi -L 2>/dev/null | head -2 | sed 's/^/    /'
if [ "${GPU_COUNT}" -ne 8 ]; then
    echo "    WARNING: Expected 8 GPUs, got ${GPU_COUNT}. Challenge requires 8xH100 SXM."
fi
echo ""

# ── 5. Verify LZMA wrapper and batch script ──────────────────────────────
echo "==> [5/6] Verifying experiment artifacts..."
MISSING=0
for f in "${STACK_RECORD}/train_gpt.py" "scripts/run_pod_batch.sh" "scripts/runpod_1413.sh" "scripts/runpod_prepare_sp8192.sh"; do
    if [ -f "$f" ]; then
        echo "    OK: $f"
    else
        echo "    MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done
echo ""

# ── 6. Checkpoint status ────────────────────────────────────────────────
echo "==> [6/6] Checking D checkpoint..."
CKPT_PATH="${STACK_RECORD}/final_model.int6.ptz"
if [ -f "${CKPT_PATH}" ]; then
    CKPT_SIZE=$(du -h "${CKPT_PATH}" | cut -f1)
    echo "    OK: ${CKPT_PATH} (${CKPT_SIZE})"
else
    echo "    NOT FOUND: ${CKPT_PATH}"
    echo ""
    echo "    The D checkpoint is needed for Tier 1 (eval-only) runs."
    echo "    To copy it from your local machine:"
    echo ""
    echo "      # Option A: scp the single checkpoint (16 MB)"
    echo "      scp -P <PORT> artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/final_model.int6.ptz \\"
    echo "        root@<POD_IP>:/workspace/parameter-golf/${STACK_RECORD}/final_model.int6.ptz"
    echo ""
    echo "      # Option B: scp the full archive (1.4 GB) and extract"
    echo "      scp -P <PORT> artifacts/runpod_pull/pr1413_archive_20260407_213205.tar.gz \\"
    echo "        root@<POD_IP>:/workspace/"
    echo "      cd /workspace/parameter-golf"
    echo "      tar xzf /workspace/pr1413_archive_20260407_213205.tar.gz \\"
    echo "        --strip-components=3 -C ${STACK_RECORD}/ \\"
    echo "        pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/final_model.int6.ptz"
    echo ""
    echo "    After copying, you can skip to Tier 2 if you prefer (it trains from scratch)."
    MISSING=$((MISSING + 1))
fi
echo ""

# ── 7. Python import check ──────────────────────────────────────────────
echo "==> Quick Python check..."
python3 -c "import torch; print(f'    PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs')"
python3 -c "import sentencepiece; print('    sentencepiece OK')"
python3 -c "from flash_attn_interface import flash_attn_func; print('    flash_attn_interface (FA3) OK')" 2>/dev/null || \
    echo "    WARNING: flash_attn_interface not available (FA3 may not be installed)"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════════════╗"
if [ "${MISSING}" -eq 0 ]; then
    echo "║  SETUP COMPLETE — Ready to run                                    ║"
else
    echo "║  SETUP COMPLETE — ${MISSING} warning(s), see above                         ║"
fi
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Commands:"
echo "  cd /workspace/parameter-golf"
echo "  bash scripts/run_pod_batch.sh tier1   # eval-only sweeps, ~25 min"
echo "  bash scripts/run_pod_batch.sh tier2   # training runs, ~50 min"
echo "  bash scripts/run_pod_batch.sh all     # everything, ~75 min"
