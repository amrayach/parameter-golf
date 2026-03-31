#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CPU_SMOKE=1
export CPU_SMOKE_ITERATIONS="${CPU_SMOKE_ITERATIONS:-2}"
export CPU_SMOKE_SEQ_LEN="${CPU_SMOKE_SEQ_LEN:-128}"
export CPU_SMOKE_MAX_WALLCLOCK_SECONDS="${CPU_SMOKE_MAX_WALLCLOCK_SECONDS:-30}"

python -u records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py "$@"
